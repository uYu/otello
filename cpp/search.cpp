#include "othello.hpp"
#include "nnue.hpp"
#include "search_table.hpp"
#include <memory>
#include <limits>

namespace othello {
namespace {
constexpr int weights[64] = {
120,-25, 20,  5,  5, 20,-25,120,
-25,-45, -5, -5, -5, -5,-45,-25,
 20, -5, 15,  3,  3, 15, -5, 20,
  5, -5,  3,  3,  3,  3, -5,  5,
  5, -5,  3,  3,  3,  3, -5,  5,
 20, -5, 15,  3,  3, 15, -5, 20,
-25,-45, -5, -5, -5, -5,-45,-25,
120,-25, 20,  5,  5, 20,-25,120};
constexpr int inf = 2000000;
int terminal(const Position& p) {
    int diff = count(p.own()) - count(p.opponent());
    return diff > 0 ? 100000 + diff * 100 : diff < 0 ? -100000 + diff * 100 : 0;
}
Bits adjacent(Bits b) { Bits r = 0; for(int d=0; d<8; ++d) r |= shift(b,d); return r; }
struct Timeout {};
struct SearchCache {
    detail::TranspositionTable table;
    std::unique_ptr<nnue::Network> evaluator;
    void prepare(const nnue::Network* network){
        // Compare model contents, not addresses: handles may be destroyed and
        // reallocated at the same address, and C++ callers may edit a network.
        bool same=network&&evaluator?network->sameParameters(*evaluator):!network&&!evaluator;
        if(!same){table.clear();evaluator=network?network->snapshot():nullptr;}
        table.newSearch();
    }
};
SearchCache& searchCache(){thread_local SearchCache cache;return cache;}
struct MoveList {
    std::array<int,64> moves{};
    int size=0;
    int front() const{return moves[0];}
    const int* begin() const{return moves.data();}
    const int* end() const{return moves.data()+size;}
};
class Search {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(), deadline;
    detail::TranspositionTable& table;
    const nnue::Network* network;
    std::array<nnue::Accumulator,130> accumulators{};
    std::array<Position,130> positions{};
    std::array<bool,130> ready{};
    std::array<std::array<int,64>,2> history{};
    std::array<std::array<std::array<int,64>,65>,2> continuation{};
    std::uint64_t nodes=0,ttHits=0,ttCutoffs=0,betaCutoffs=0,pvsResearches=0,aspirationRetries=0,evaluations=0;
    std::uint64_t accumulatorUpdates=0;
    bool useDeadline;
    void checkTime(){if(useDeadline&&std::chrono::steady_clock::now()>=deadline)throw Timeout{};}
    void ensureAccumulator(int ply){
        if(ready[ply])return;
        if(ply==0)accumulators[0]=network->refresh(positions[0]);
        else{
            ensureAccumulator(ply-1);
            accumulators[ply]=network->update(accumulators[ply-1],positions[ply-1],positions[ply]);
            ++accumulatorUpdates;
        }
        ready[ply]=true;
    }
    int eval(const Position& p,int ply){
        ++evaluations;if(!network)return evaluate(p);
        if(network->spatial())return network->evaluate(p);
        ensureAccumulator(ply);return network->evaluate(accumulators[ply],p.turn);
    }
    void descend(const Position& after,int ply){
        if(network&&!network->spatial()){positions[ply+1]=after;ready[ply+1]=false;}
    }
    MoveList ordered(const Position& p,Bits moves,int preferred,int previous){
        MoveList list;std::array<int,64> scores{};
        int side=p.turn==1?0:1;
        while(moves){
            int m=first(moves);moves&=moves-1;
            int score=m==preferred?1000000:((corners>>m&1)?100000:0)+history[side][m]+continuation[side][previous][m]+weights[m]*16;
            // Stable insertion sort: short lists, one priority calculation per
            // move, and no heap allocation in the recursive search.
            int i=list.size++;
            while(i>0&&score>scores[i-1]){list.moves[i]=list.moves[i-1];scores[i]=scores[i-1];--i;}
            list.moves[i]=m;scores[i]=score;
        }
        return list;
    }
    static void updateHistory(int& value,int bonus){
        constexpr int limit=16384;bonus=std::clamp(bonus,-limit,limit);
        value+=bonus-value*std::abs(bonus)/limit;
    }
    int negamax(const Position& p,int depth,int alpha,int beta,int ply,int previous){
        ++nodes;if(useDeadline&&(nodes&(network&&network->architecture==nnue::Architecture::Residual?15:255))==0)checkTime();
        Bits moves=legal(p);
        if(!moves){
            if(!legal(p.opponent(),p.own()))return terminal(p);
            Position passed=play(p,-1);descend(passed,ply);
            return -negamax(passed,depth,-beta,-alpha,ply+1,64);
        }
        if(depth<=0)return eval(p,ply);
        int preferred=-1;
        // A bound may cut off a search; it must never be treated as an exact score.
        if(const auto* entry=table.probe(p)){
            ++ttHits;preferred=entry->move;
            // A previous call can have searched deeper. Use it for ordering,
            // but preserve the API's exact fixed-depth candidate scores.
            if(entry->depth==depth&&(entry->bound==detail::Bound::Exact||(entry->bound==detail::Bound::Lower&&entry->score>=beta)
                ||(entry->bound==detail::Bound::Upper&&entry->score<=alpha))){++ttCutoffs;return entry->score;}
        }
        const int originalAlpha=alpha;int best=-inf,bestMove=-1,searched=0;
        const auto list=ordered(p,moves,preferred,previous);
        for(int m:list){
            Position child=play(p,m);descend(child,ply);int value;
            if(searched==0)value=-negamax(child,depth-1,-beta,-alpha,ply+1,m);
            else{
                // PVS: test whether a later move improves alpha before a full-window search.
                value=-negamax(child,depth-1,-alpha-1,-alpha,ply+1,m);
                if(value>alpha&&value<beta){++pvsResearches;value=-negamax(child,depth-1,-beta,-alpha,ply+1,m);}
            }
            ++searched;
            if(value>best){best=value;bestMove=m;}alpha=std::max(alpha,value);
            if(alpha>=beta){
                ++betaCutoffs;int side=p.turn==1?0:1,bonus=std::min(1024,depth*depth*8);
                updateHistory(history[side][m],bonus);updateHistory(continuation[side][previous][m],bonus);
                for(int i=0;i<searched-1;++i){int earlier=list.moves[i];updateHistory(history[side][earlier],-bonus/4);updateHistory(continuation[side][previous][earlier],-bonus/4);}
                break;
            }
        }
        table.store(p,depth,best,bestMove,best<=originalAlpha?detail::Bound::Upper:best>=beta?detail::Bound::Lower:detail::Bound::Exact);
        return best;
    }
    int rootMove(const Position& p,int move,int depth,int previousScore){
        Position child=play(p,move);descend(child,0);
        int delta=48,alpha=-inf,beta=inf;
        if(depth>=3&&std::abs(previousScore)<100000){alpha=previousScore-delta;beta=previousScore+delta;}
        for(;;){
            if(depth>1)checkTime();
            int value=-negamax(child,depth-1,-beta,-alpha,1,move);
            if(value>alpha&&value<beta)return value;
            ++aspirationRetries;delta=std::min(inf,delta*2);
            if(value<=alpha)alpha=std::max(-inf,value-delta);
            if(value>=beta)beta=std::min(inf,value+delta);
        }
    }
public:
    Search(int ms,const nnue::Network* net,detail::TranspositionTable& tt):deadline(start+std::chrono::milliseconds(ms)),table(tt),network(net),useDeadline(ms>0){}
    SearchResult run(const Position& p,SearchOptions opt){
        SearchResult result;Bits moves=legal(p);
        if(!moves){
            if(over(p)){result.score=terminal(p);result.exact=true;return result;}
            result=run(play(p,-1),opt);result.move=-1;result.score=-result.score;result.candidates.clear();return result;
        }
        if(network){positions[0]=p;ready[0]=false;}
        auto list=ordered(p,moves,-1,64);
        result.move=list.front();auto fallback=play(p,result.move);descend(fallback,0);result.score=-eval(fallback,1);
        if(opt.level==0){
            auto rng=opt.seed?opt.seed:1;rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
            for(int m:list){auto child=play(p,m);descend(child,0);result.candidates.push_back({m,-eval(child,1)});}
            std::stable_sort(result.candidates.begin(),result.candidates.end(),[](auto a,auto b){return a.score>b.score;});
            auto picked=result.candidates[rng%std::min<std::size_t>(3,result.candidates.size())];result.move=picked.move;result.score=picked.score;result.depth=1;
        }else{
            std::array<int,64> previousScores{};
            for(int depth=1;depth<=std::min(opt.maxDepth,p.empty());++depth){
                try{
                    std::vector<Candidate> candidates;
                    candidates.reserve(list.size);
                    for(int m:list)candidates.push_back({m,rootMove(p,m,depth,previousScores[m])});
                    // Publish only fully completed iterations. All root candidate scores are exact
                    // at the reported depth, including after aspiration fail-high/fail-low retries.
                    std::stable_sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.score>b.score;});
                    result.move=candidates.front().move;result.score=candidates.front().score;result.depth=depth;result.candidates=candidates;
                    list.size=0;for(auto c:candidates){list.moves[list.size++]=c.move;previousScores[c.move]=c.score;}
                    if(depth>=p.empty()){result.exact=true;break;}
                }catch(const Timeout&){break;}
            }
        }
        result.nodes=nodes;result.ttHits=ttHits;result.ttCutoffs=ttCutoffs;result.betaCutoffs=betaCutoffs;
        result.pvsResearches=pvsResearches;result.aspirationRetries=aspirationRetries;result.evaluations=evaluations;
        result.accumulatorUpdates=accumulatorUpdates;
        result.elapsed=int(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());return result;
    }
};
}
int evaluate(const Position& p) {
    Bits own=p.own(),opp=p.opponent(),empty=~(own|opp);
    int positional=0;
    for(int i=0;i<64;++i) positional += int((own>>i)&1)*weights[i] - int((opp>>i)&1)*weights[i];
    int mobility=count(legal(own,opp))-count(legal(opp,own));
    int frontier=count(opp & adjacent(empty))-count(own & adjacent(empty));
    int corner=count(own&corners)-count(opp&corners);
    int discs=count(own)-count(opp);
    return positional*2 + mobility*16 + frontier*7 + corner*180 + discs*(p.empty()<16?10:1);
}
SearchResult search(const Position& p, SearchOptions opt) {
    if(opt.milliseconds<0 || opt.milliseconds>60000 || opt.maxDepth<1 || opt.maxDepth>64 || opt.level<0 || opt.level>2)
        throw std::invalid_argument("Invalid search options");
    auto& cache=searchCache();cache.prepare(opt.network);
    return std::make_unique<Search>(opt.milliseconds,opt.network,cache.table)->run(p,opt);
}
void clearSearchCache(){searchCache().table.clear();}
}

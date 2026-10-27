#include "othello.hpp"
#include "nnue.hpp"
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
struct Entry { Bits own=0, opp=0; int depth=-1, score=0, move=-1, flag=0; };
class Search {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(), deadline;
    std::vector<Entry> table;
    const nnue::Network* network;
    std::array<nnue::Accumulator,130> accumulators{};
    std::array<std::array<int,64>,2> history{};
    std::array<std::array<std::array<int,64>,65>,2> continuation{};
    std::uint64_t nodes=0,ttHits=0,ttCutoffs=0,betaCutoffs=0,pvsResearches=0,aspirationRetries=0,evaluations=0;
    bool useDeadline;
    void checkTime(){if(useDeadline&&std::chrono::steady_clock::now()>=deadline)throw Timeout{};}
    std::size_t key(const Position& p) const{return std::size_t(p.hash)&(table.size()-1);}
    int eval(const Position& p,int ply){++evaluations;return network?network->evaluate(accumulators[ply],p.turn):evaluate(p);}
    void descend(const Position& before,const Position& after,int ply){
        if(network)accumulators[ply+1]=network->update(accumulators[ply],before,after);
    }
    std::vector<int> ordered(const Position& p,Bits moves,int preferred,int previous){
        std::vector<int> list;while(moves){list.push_back(first(moves));moves&=moves-1;}
        int side=p.turn==1?0:1;
        auto priority=[&](int m){return m==preferred?1000000:((corners>>m&1)?100000:0)+history[side][m]+continuation[side][previous][m]+weights[m]*16;};
        std::stable_sort(list.begin(),list.end(),[&](int a,int b){return priority(a)>priority(b);});return list;
    }
    static void updateHistory(int& value,int bonus){
        constexpr int limit=16384;bonus=std::clamp(bonus,-limit,limit);
        value+=bonus-value*std::abs(bonus)/limit;
    }
    int negamax(const Position& p,int depth,int alpha,int beta,int ply,int previous){
        ++nodes;if(useDeadline&&(nodes&255)==0)checkTime();
        Bits moves=legal(p);
        if(!moves){
            if(!legal(p.opponent(),p.own()))return terminal(p);
            Position passed=play(p,-1);descend(p,passed,ply);
            return -negamax(passed,depth,-beta,-alpha,ply+1,64);
        }
        if(depth<=0)return eval(p,ply);
        const auto index=key(p);const Entry entry=table[index];int preferred=-1;
        // A bound may cut off a search; it must never be treated as an exact score.
        if(entry.depth>=0&&entry.own==p.own()&&entry.opp==p.opponent()){
            ++ttHits;preferred=entry.move;
            if(entry.depth>=depth&&(entry.flag==0||(entry.flag==1&&entry.score>=beta)||(entry.flag==2&&entry.score<=alpha))){++ttCutoffs;return entry.score;}
        }
        const int originalAlpha=alpha;int best=-inf,bestMove=-1,searched=0;
        std::vector<int> tried;
        for(int m:ordered(p,moves,preferred,previous)){
            Position child=play(p,m);descend(p,child,ply);int value;
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
                for(int earlier:tried){updateHistory(history[side][earlier],-bonus/4);updateHistory(continuation[side][previous][earlier],-bonus/4);}
                break;
            }
            tried.push_back(m);
        }
        table[index]={p.own(),p.opponent(),depth,best,bestMove,best<=originalAlpha?2:best>=beta?1:0};
        return best;
    }
    int rootMove(const Position& p,int move,int depth,int previousScore){
        Position child=play(p,move);descend(p,child,0);
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
    Search(int ms,const nnue::Network* net):deadline(start+std::chrono::milliseconds(ms)),table(1<<18),network(net),useDeadline(ms>0){}
    SearchResult run(const Position& p,SearchOptions opt){
        SearchResult result;Bits moves=legal(p);
        if(!moves){
            if(over(p)){result.score=terminal(p);result.exact=true;return result;}
            result=run(play(p,-1),opt);result.move=-1;result.score=-result.score;result.candidates.clear();return result;
        }
        if(network)accumulators[0]=network->refresh(p);
        auto list=ordered(p,moves,-1,64);
        result.move=list.front();auto fallback=play(p,result.move);descend(p,fallback,0);result.score=-eval(fallback,1);
        if(opt.level==0){
            auto rng=opt.seed?opt.seed:1;rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
            for(int m:list){auto child=play(p,m);descend(p,child,0);result.candidates.push_back({m,-eval(child,1)});}
            std::stable_sort(result.candidates.begin(),result.candidates.end(),[](auto a,auto b){return a.score>b.score;});
            auto picked=result.candidates[rng%std::min<std::size_t>(3,result.candidates.size())];result.move=picked.move;result.score=picked.score;result.depth=1;
        }else{
            std::array<int,64> previousScores{};
            for(int depth=1;depth<=std::min(opt.maxDepth,p.empty());++depth){
                try{
                    std::vector<Candidate> candidates;
                    for(int m:list)candidates.push_back({m,rootMove(p,m,depth,previousScores[m])});
                    // Publish only fully completed iterations. All root candidate scores are exact
                    // at the reported depth, including after aspiration fail-high/fail-low retries.
                    std::stable_sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.score>b.score;});
                    result.move=candidates.front().move;result.score=candidates.front().score;result.depth=depth;result.candidates=candidates;
                    list.clear();for(auto c:candidates){list.push_back(c.move);previousScores[c.move]=c.score;}
                    if(depth>=p.empty()){result.exact=true;break;}
                }catch(const Timeout&){break;}
            }
        }
        result.nodes=nodes;result.ttHits=ttHits;result.ttCutoffs=ttCutoffs;result.betaCutoffs=betaCutoffs;
        result.pvsResearches=pvsResearches;result.aspirationRetries=aspirationRetries;result.evaluations=evaluations;
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
    return std::make_unique<Search>(opt.milliseconds,opt.network)->run(p,opt);
}
}

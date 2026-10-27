#include "../cpp/othello.hpp"
#include "../cpp/api.h"
#include "../cpp/nnue.hpp"
#include "../cpp/cnn.hpp"
#include "../cpp/resnet.hpp"
#include "../cpp/mcts.hpp"
#include "../cpp/search_table.hpp"
#include <iostream>
#include <random>
#include <cstdlib>
using namespace othello;
void check(bool yes,const char* why) {if(!yes){std::cerr<<"FAIL: "<<why<<'\n';std::exit(1);}}
Bits referenceFlips(const Position& p,int m) {
    auto s=board(p);if(m<0||m>63||s[m]!='.')return 0;
    Bits result=0;char own=p.turn==1?'B':'W',opp=p.turn==1?'W':'B';
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        if(!dy&&!dx)continue;
        int y=m/8+dy,x=m%8+dx;Bits f=0;
        while(y>=0&&y<8&&x>=0&&x<8&&s[y*8+x]==opp){f|=1ULL<<(y*8+x);y+=dy;x+=dx;}
        if(y>=0&&y<8&&x>=0&&x<8&&s[y*8+x]==own)result|=f;
    }return result;
}
std::uint64_t perft(Position p,int d) {
    if(!d)return 1;
    auto b=legal(p);if(!b)return over(p)?1:perft(play(p,-1),d);
    std::uint64_t n=0;while(b){int m=first(b);b&=b-1;n+=perft(play(p,m),d-1);}return n;
}
int exactDiff(Position p) {
    auto b=legal(p);if(!b)return over(p)?count(p.own())-count(p.opponent()):-exactDiff(play(p,-1));
    int best=-65;while(b){int m=first(b);b&=b-1;best=std::max(best,-exactDiff(play(p,m)));}return best;
}
int referenceValue(Position p,int depth,const nnue::Network* net=nullptr){
    auto b=legal(p);
    if(!b){if(over(p)){int d=count(p.own())-count(p.opponent());return d>0?100000+d*100:d<0?-100000+d*100:0;}return -referenceValue(play(p,-1),depth,net);}
    if(!depth)return net?net->evaluate(p):evaluate(p);
    int best=-2000000;while(b){int m=first(b);b&=b-1;best=std::max(best,-referenceValue(play(p,m),depth-1,net));}return best;
}
void policyChecks(){
    Position p;SearchResult analysis;analysis.move=19;analysis.score=100;analysis.depth=4;
    analysis.candidates={{19,100},{26,92},{37,68},{44,-20}};
    auto original=analysis.candidates;std::array<int,64> counts{};
    for(unsigned seed=0;seed<4096;++seed){
        auto selected=selectMove(p,analysis,{},seed);
        check(selected.sampled&&selected.eligible==3&&selected.temperatureMilli==32000&&selected.lossLimit==64,"opening schedule and score filter");
        ++counts[selected.move];
        check(selected.bestScore-selected.score<=selected.lossLimit,"opening selection stays within score-loss budget");
        auto repeated=selectMove(p,analysis,{},seed);
        check(repeated.move==selected.move,"same seed and root scores reproduce selection");
        auto reversed=analysis;std::reverse(reversed.candidates.begin(),reversed.candidates.end());
        check(selectMove(p,reversed,{},seed).move==selected.move,"candidate tie order does not change seeded selection");
    }
    check(counts[19]>counts[26]&&counts[26]>counts[37]&&counts[44]==0,"weighted sampling favors better moves and excludes bad moves");
    check(analysis.move==19&&analysis.score==100&&analysis.candidates[1].score==92,"selection does not mutate analysis");
    check(!selectMove(p,analysis,{0,32,64},71).sampled,"opening selection can be disabled");
    auto exact=analysis;exact.exact=true;check(selectMove(p,exact,{},71).move==19,"exact result is never weakened");
    auto terminal=analysis;terminal.score=100100;terminal.candidates[0].score=100100;
    check(!selectMove(p,terminal,{},71).sampled,"known terminal scores disable random selection");
    for(int ply=0;ply<12;++ply){auto moves=legal(p);if(!moves)p=play(p,-1);p=play(p,first(legal(p)));}
    auto end=selectMove(p,analysis,{},71);check(end.move==19&&!end.sampled&&end.temperatureMilli==0,"policy ends after twelve placements");
    p=Position{};
    for(int ply=0;ply<11;++ply){if(!legal(p))p=play(p,-1);p=play(p,first(legal(p)));}
    auto b=legal(p);SearchResult late;late.depth=4;late.score=100;late.move=first(b);
    while(b){int m=first(b);b&=b-1;late.candidates.push_back({m,m==late.move?100:92});}
    auto selected=selectMove(p,late,{},71);
    check(selected.temperatureMilli==2666&&selected.lossLimit==5&&!selected.sampled,"temperature and near-score window decay together");
    bool rejected=false;try{selectMove(Position{},analysis,{-1,16,32},1);}catch(...){rejected=true;}
    check(rejected,"invalid opening policy rejected");
}
void checkSearch(const Position& p,int depth,const nnue::Network* net=nullptr){
    auto r=search(p,{0,depth,2,1,net});
    check(r.score==referenceValue(p,depth,net),"persistent TT preserves fixed-depth minimax");
    for(auto c:r.candidates)check(c.score==-referenceValue(play(p,c.move),depth-1,net),"persistent TT preserves every candidate score");
}
void mctsChecks(){
    Position p;clearMctsCache();
    auto firstRun=mcts(p,{0,128,19});
    check(firstRun.analysis.nodes==128&&flips(p,firstRun.analysis.move),"MCTS fixed simulation budget and legal move");
    auto rootVisits=firstRun.visits;auto rootCandidates=firstRun.analysis.candidates;
    check(firstRun.analysis.candidates.size()==4&&firstRun.visits.size()==4,"MCTS exposes all root visits");
    clearMctsCache();auto repeated=mcts(p,{0,128,19});
    check(repeated.analysis.move==firstRun.analysis.move&&repeated.visits==rootVisits,"MCTS seeded simulation runs reproduce");
    for(std::size_t i=0;i<rootCandidates.size();++i)check(rootCandidates[i].score==repeated.analysis.candidates[i].score,"MCTS reproducible candidate values");
    auto child=play(p,repeated.analysis.move);auto warm=mcts(child,{0,128,19});
    check(warm.reusedNodes>0&&flips(child,warm.analysis.move),"MCTS retains the reached subtree");
    nnue::Network net;auto withNet=mcts(p,{0,64,19,&net});
    check(withNet.reusedNodes==0,"MCTS clears tree on evaluator change");
    net.outputBias=100;auto changed=mcts(p,{0,64,19,&net});
    check(changed.reusedNodes==0,"MCTS invalidates modified evaluator contents");
    clearMctsCache();auto bounded=mcts(p,{0,10000,19,nullptr,1.5,65});
    check(bounded.memoryLimited&&bounded.treeNodes<=65,"MCTS bounds node storage");
    bool rejected=false;try{mcts(p,{0,0});}catch(...){rejected=true;}check(rejected,"MCTS requires a budget");
    std::mt19937 rng(1606);int passes=0;
    for(int game=0;game<12;++game){
        p=Position{};
        while(!over(p)&&p.empty()>4){auto moves=legal(p);
            if(!moves){p=play(p,-1);continue;}
            int nth=rng()%count(moves);while(nth--)moves&=moves-1;p=play(p,first(moves));}
        int expected=exactDiff(p);clearMctsCache();auto exact=mcts(p,{0,200000,23});
        check(exact.analysis.exact,"MCTS proves small endgames");
        int target=expected>0?100000+expected*100:expected<0?-100000+expected*100:0;
        check(exact.analysis.score==target,"MCTS terminal backups agree with independent minimax");
        if(legal(p))check(-exactDiff(play(p,exact.analysis.move))==expected,"MCTS chooses minimax endgame move");
        else if(!over(p)){++passes;check(exact.analysis.move==-1,"MCTS returns forced pass");}
    }
    p=parse(std::string(64,'B'),-1);clearMctsCache();auto terminal=mcts(p,{0,1});
    check(terminal.analysis.exact&&terminal.analysis.move==-1&&terminal.analysis.score==-106400,"MCTS terminal value uses current player's view");
    p=parse("WWBBBWBBBBBBBWBBBBWWWWWBBBBWBWBBBBWBWWBBBBWWWBWBBWWWWWBBBW.BBBBB",-1);
    check(!legal(p)&&legal(p.opponent(),p.own()),"MCTS forced-pass fixture");
    clearMctsCache();auto passed=mcts(p,{0,1000});int passValue=exactDiff(p);
    check(passed.analysis.move==-1&&passed.analysis.exact&&passed.analysis.score==(passValue>0?100000+passValue*100:passValue<0?-100000+passValue*100:0),"MCTS backs up forced passes correctly");
    clearMctsCache();auto timed=mcts(Position{},{5});check(flips(Position{},timed.analysis.move),"timed MCTS returns a legal move");
    int cells[64]{},output[136]{},policy[6]{},visits[64]{};double stats[72]{};
    check(ot_nnue_mcts_play(nullptr,cells,1,1,0,1,12,32,64,output,policy,visits,stats)==-1,"MCTS C ABI rejects null model");
    clearMctsCache();
}
void cnnChecks(){
    auto weights=std::make_shared<nnue::Cnn>();std::mt19937 rng(941);
    for(auto& w:weights->weights)w=float(int(rng()%401)-200)/1000;
    nnue::Network model;model.architecture=nnue::Architecture::Cnn;model.hidden=0;model.cnn=weights;
    Position p;
    for(int ply=0;ply<18;++ply){
        if(!legal(p))p=play(p,-1);
        auto next=play(p,first(legal(p)));
        auto acc=model.update(model.refresh(p),p,next);
        check(acc.blackBoard==next.black&&acc.whiteBoard==next.white,"CNN board state updates correctly");
        check(model.evaluate(acc,next.turn)==model.evaluate(next),"CNN direct and board-state evaluation agree");
        check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"CNN SIMD and independent scalar inference agree");
        auto swapped=next;std::swap(swapped.black,swapped.white);swapped.turn=-swapped.turn;
        check(model.evaluate(next)==model.evaluate(swapped),"CNN preserves color relabeling");p=next;
    }
    checkSearch(p,2,&model);
    weights->weights.back()+=0.5f; // Same pointer, changed weights: cache snapshot must be independent.
    checkSearch(p,2,&model);
    nnue::Network legacy;checkSearch(p,2,&legacy);checkSearch(p,2,&model);
    clearSearchCache();
}
void residualChecks(){
    std::mt19937 rng(1532);
    for(int blocks:{2,4}){
        auto weights=std::make_shared<nnue::ResNet>();weights->blocks=blocks;
        weights->weights.resize(nnue::ResNet::parameters(blocks));
        for(auto& w:weights->weights)w=float(int(rng()%201)-100)/2000;
        nnue::Network model;model.architecture=nnue::Architecture::Residual;model.hidden=0;model.residual=weights;
        Position p;
        for(int ply=0;ply<8;++ply){
            if(!legal(p))p=play(p,-1);
            auto next=play(p,first(legal(p)));
            auto acc=model.update(model.refresh(p),p,next);
            check(model.evaluate(acc,next.turn)==model.evaluate(next),"residual board state updates agree");
            check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"residual SIMD and independent scalar inference agree");
            auto renamed=next;std::swap(renamed.black,renamed.white);renamed.turn=-renamed.turn;
            check(model.evaluate(next)==model.evaluate(renamed),"residual preserves color relabeling");p=next;
        }
        checkSearch(p,2,&model);
        auto snapshot=model.snapshot();weights->weights.back()+=0.5f;
        check(!model.sameParameters(*snapshot),"residual cache snapshot owns independent weights");
        checkSearch(p,2,&model);
        nnue::Network old;checkSearch(p,2,&old);checkSearch(p,2,&model);
    }
    clearSearchCache();
}
void cacheChecks(nnue::Network& model){
    const auto originalModel=model;
    // Force all positions into one bucket to exercise replacement independently
    // of board hash distribution.
    detail::TranspositionTable table(1);table.newSearch();
    std::array<Position,5> positions;
    int i=0;Bits moves=legal(Position{});
    while(moves){int m=first(moves);moves&=moves-1;positions[i++]=play(Position{},m);}
    positions[4]=play(positions[0],first(legal(positions[0])));
    for(i=0;i<4;++i)table.store(positions[i],i+1,100+i,0,detail::Bound::Exact);
    for(i=0;i<4;++i)check(table.probe(positions[i])!=nullptr,"TT bucket retains colliding boards");
    table.store(positions[3],1,0,1,detail::Bound::Upper);
    check(table.probe(positions[3])->depth==4&&table.probe(positions[3])->score==103,"TT retains deeper result");
    table.store(positions[3],4,0,1,detail::Bound::Lower);
    check(table.probe(positions[3])->bound==detail::Bound::Exact,"TT retains exact score over same-depth bound");
    table.store(positions[4],2,0,1,detail::Bound::Lower);
    check(!table.probe(positions[0])&&table.probe(positions[3]),"TT replaces weakest collision");
    for(i=0;i<4;++i)table.newSearch();
    table.store(positions[4],2,0,1,detail::Bound::Lower);
    table.store(positions[0],2,0,1,detail::Bound::Exact);
    check(table.probe(positions[4])&&!table.probe(positions[1]),"TT ages old entries before recent ones");
    table.clear();for(auto& p:positions)check(!table.probe(p),"TT explicit clear");

    Position p;std::mt19937 rng(91);
    for(i=0;i<20;++i){Bits b=legal(p);if(!b){p=play(p,-1);b=legal(p);}int n=rng()%count(b);while(n--)b&=b-1;p=play(p,first(b));}
    clearSearchCache();auto cold=search(p,{0,4,2,1,&model});auto warm=search(p,{0,4,2,1,&model});
    check(cold.score==warm.score&&warm.ttCutoffs>0&&warm.nodes<cold.nodes,"same-model searches reuse TT across calls");
    check(warm.accumulatorUpdates<cold.accumulatorUpdates,"TT cutoffs avoid lazy NNUE updates");
    checkSearch(p,2,&model); // Deeper cached scores must not leak into shallow requests.
    checkSearch(play(p,cold.move),3,&model); // Reuse across actual moves.
    auto replacement=model;
    for(auto& x:replacement.output)x=-x;
    checkSearch(p,3,&replacement);
    model=replacement; // Same address, different parameters.
    checkSearch(p,3,&model);
    model.hidden=256;checkSearch(p,3,&model);
    model.architecture=nnue::Architecture::Turn;checkSearch(p,3,&model);
    model.opponentOutput[0]=192;checkSearch(p,3,&model);
    model.outputBias=12000;checkSearch(p,3,&model);
    checkSearch(p,3); // Neural -> heuristic -> neural.
    checkSearch(p,3,&model);
    checkSearch(p,3);
    ot_search_clear_cache();model=originalModel;
}
int main() {
    cnnChecks();residualChecks();mctsChecks();
    policyChecks();
    Position p;
    check(legal(p)==((1ULL<<19)|(1ULL<<26)|(1ULL<<37)|(1ULL<<44)),"initial legal squares");
    const std::uint64_t expected[]={1,4,12,56,244,1396,8200,55092,390216};
    for(int i=0;i<=8;++i)check(perft(p,i)==expected[i],"opening perft");
    check(board(parse(board(p),1))==board(p),"board round trip");
    check(count(play(p,19).black)==4&&count(play(p,19).white)==1,"D3 flips D4");
    bool rejected=false;try{play(p,-1);}catch(...){rejected=true;}check(rejected,"reject voluntary pass");
    check(square("h8")==63&&square("A1")==0,"coordinates");
    std::mt19937 rng(2026);int passes=0,exactChecks=0;
    nnue::Network model;
    for(auto& row:model.weights)for(auto& value:row)value=int(rng()%161)-80;
    for(auto& value:model.bias)value=int(rng()%201)-30;
    for(auto& value:model.output)value=int(rng()%401)-200;
    for(int hidden:{32,256}){
        model.hidden=hidden;auto acc=model.refresh(p);auto next=play(p,19);
        auto incremental=model.update(acc,p,next),full=model.refresh(next);
        check(incremental.black==full.black&&incremental.white==full.white,"both NNUE widths update correctly");
        check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"both NNUE widths match tiny_nn");
    }
    model.hidden=32;
    cacheChecks(model);
    auto turnModel=model;turnModel.architecture=nnue::Architecture::Turn;turnModel.outputBias=2000;
    for(int i=0;i<nnue::MaxHidden;++i)turnModel.opponentOutput[i]=-model.output[i]/2+5;
    nnue::Network initiative;initiative.architecture=nnue::Architecture::Turn;
    initiative.outputBias=16384;
    check(initiative.evaluate(p)>0&&initiative.evaluate(initiative.refresh(p),-1)>0,"turn-aware model can represent initiative for either color");
    int searchChecks=0;
    for(int game=0;game<120;++game) {
        p=Position{};int steps=0;
        while(!over(p)) {
            check(++steps<=120,"games terminate");Bits expectedLegal=0;
            check(p.hash==hashBoard(p.black,p.white,p.turn),"incremental Zobrist matches full recomputation");
            for(int m=0;m<64;++m){auto f=referenceFlips(p,m);check(f==flips(p,m),"ray flips agree");if(f)expectedLegal|=1ULL<<m;}
            check(legal(p)==expectedLegal,"bitboard moves agree with array oracle");
            int cells[64],moves[64],out[136];auto s=board(p);
            for(int i=0;i<64;++i)cells[i]=s[i]=='B'?1:s[i]=='W'?-1:0;
            check(ot_legal(cells,p.turn,moves)==count(expectedLegal),"C API legal");
            auto b=legal(p);if(!b){
                ++passes;check(ot_play(cells,p.turn,-1,out)==1,"C API pass");
                if(p.empty()<=6){auto r=search(p,{0,64,2,1});int diff=exactDiff(p);
                    int expectedScore=diff>0?100000+diff*100:diff<0?-100000+diff*100:0;
                    check(r.move==-1&&r.exact&&r.score==expectedScore,"search through forced pass");}
                if(p.empty()<=6&&game<12){
                    auto r=search(p,{0,64,2,1,&model});
                    check(r.exact&&r.score==referenceValue(p,64,&model),"lazy NNUE search through forced pass");
                    auto t=search(p,{0,64,2,1,&turnModel});
                    check(t.exact&&t.score==referenceValue(p,64,&turnModel),"turn-aware search through forced pass");
                }
                p=play(p,-1);continue;
            }
            if(p.empty()<=6 && exactChecks<12) {
                auto r=search(p,{0,64,2,1});check(r.exact,"endgame completed");
                check(-exactDiff(play(p,r.move))==exactDiff(p),"search matches exhaustive endgame");++exactChecks;
            }
            std::vector<int> list;while(b){list.push_back(first(b));b&=b-1;}int m=list[rng()%list.size()];
            auto next=play(p,m);check(ot_play(cells,p.turn,m,out)==1,"C API play");
            auto updated=model.update(model.refresh(p),p,next),refreshed=model.refresh(next);
            check(updated.black==refreshed.black&&updated.white==refreshed.white,"quantized NNUE incremental update equals full refresh");
            check(model.evaluate(updated,next.turn)==-model.evaluate(updated,-next.turn),"NNUE turn antisymmetry");
            if(steps%9==0){
                check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"tiny_nn SIMD FP32 reference matches integer NNUE");
                auto t=turnModel.update(turnModel.refresh(p),p,next),full=turnModel.refresh(next);
                check(t.black==full.black&&t.white==full.white,"turn-aware accumulators update incrementally");
                check(std::abs(turnModel.evaluate(next)-turnModel.evaluateReference(next))<=1,"turn-aware integer and FP32 reference agree");
                auto swapped=next;std::swap(swapped.black,swapped.white);swapped.turn=-swapped.turn;
                check(turnModel.evaluate(next)==turnModel.evaluate(swapped),"turn-aware model preserves color relabeling");
            }
            if(game<3&&steps%9==0&&searchChecks<20){
                for(int width:{0,32,256}){
                    model.hidden=width?width:32;const nnue::Network* net=width?&model:nullptr;
                    auto r=search(p,{0,3,2,1,net});check(r.depth==std::min(3,p.empty()),"fixed depth completed");
                    check(r.score==referenceValue(p,3,net),"PVS/TT/aspiration match exhaustive minimax");
                    for(auto c:r.candidates)check(c.score==-referenceValue(play(p,c.move),2,net),"root candidates retain exact depth scores");
                    if(width){turnModel.hidden=width;checkSearch(p,3,&turnModel);}
                }model.hidden=32;++searchChecks;
            }
            for(int i=0;i<64;++i)check(out[i]==((next.black>>i&1)?1:(next.white>>i&1)?-1:0),"C API board");
            check(count(next.black|next.white)==count(p.black|p.white)+1,"one new disc per move");p=next;
        }
        check(!legal(p.black,p.white)&&!legal(p.white,p.black),"both sides stuck at end");
    }
    check(passes>0,"forced passes covered");
    p=parse(std::string(64,'B'),1);check(over(p),"full board over");
    p=parse("B..............................................................W",-1);check(over(p),"early finish with empty squares");
    auto timed=search(Position{},{40,64,2,1});check(flips(Position{},timed.move)!=0&&timed.depth>0,"timed search returns completed legal move");
    check(timed.ttHits>0,"Zobrist transposition reuse");
    clearSearchCache();auto optimized=search(Position{},{0,6,2,1});
    check(optimized.pvsResearches>0&&optimized.betaCutoffs>0&&optimized.aspirationRetries>0,"PVS, beta cutoffs and aspiration retries exercised");
    std::cout<<"PASS: perft depth 8; 120 games vs independent oracle; "<<passes<<" forced passes; exhaustive endgames; C ABI; timed AI\n";
}

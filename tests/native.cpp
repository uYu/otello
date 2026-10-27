#include "../cpp/othello.hpp"
#include "../cpp/api.h"
#include "../cpp/nnue.hpp"
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
int main() {
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
            if(steps%9==0)check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"tiny_nn SIMD FP32 reference matches integer NNUE");
            if(game<3&&steps%9==0&&searchChecks<20){
                for(int width:{0,32,256}){
                    model.hidden=width?width:32;const nnue::Network* net=width?&model:nullptr;
                    auto r=search(p,{0,3,2,1,net});check(r.depth==std::min(3,p.empty()),"fixed depth completed");
                    check(r.score==referenceValue(p,3,net),"PVS/TT/aspiration match exhaustive minimax");
                    for(auto c:r.candidates)check(c.score==-referenceValue(play(p,c.move),2,net),"root candidates retain exact depth scores");
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
    auto optimized=search(Position{},{0,6,2,1});
    check(optimized.pvsResearches>0&&optimized.betaCutoffs>0&&optimized.aspirationRetries>0,"PVS, beta cutoffs and aspiration retries exercised");
    std::cout<<"PASS: perft depth 8; 120 games vs independent oracle; "<<passes<<" forced passes; exhaustive endgames; C ABI; timed AI\n";
}

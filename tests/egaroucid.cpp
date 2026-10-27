#include "../cpp/egaroucid.hpp"
#include <egaroucid/egaroucid.h>
#include <cstdlib>
#include <iostream>
#include <random>
using namespace othello;
void check(bool ok,const char* message){if(!ok){std::cerr<<"FAIL: "<<message<<'\n';std::exit(1);}}
std::array<int,64> encode(const Position& p){
    std::array<int,64> cells{};
    for(int i=0;i<64;++i)cells[i]=p.black>>i&1?EGAROUCID_BLACK:p.white>>i&1?EGAROUCID_WHITE:EGAROUCID_EMPTY;
    return cells;
}
int main(){
    check(ot_egaroucid_create("/this/egaroucid/resource/directory/does/not/exist")==nullptr,"missing resources fail explicitly");
    check(std::string(ot_egaroucid_last_error()).size()>0,"native errors have a message");
    Egaroucid engine;
    auto book=engine.analyze(Position{},1,true);
    check(book.bookHit&&book.depth==-1&&book.nodes==0,"official book is actually used at opening");
    check(flips(Position{},book.move)!=0,"book move is legal");
    EgaroucidResult lookup,repeat;
    check(engine.probeBook(Position{},lookup,1,20261027)&&lookup.nodes==0&&lookup.depth==-1,"book-only lookup hits without search");
    check(engine.probeBook(Position{},repeat,1,20261027)&&lookup.move==repeat.move&&lookup.value==repeat.value,"seeded book-only draw reproduces");
    auto searched=engine.analyze(Position{},1,false);
    check(!searched.bookHit&&searched.depth>=1&&searched.nodes>0,"disabling use_book executes search");
    // D3 E3 F4: the pinned official book offers -6 and -8 responses.
    // This verifies variety beyond merely randomizing symmetry/tied best moves.
    Position opening=parse("...................BW......BBB.....BW...........................",-1);
    auto best=engine.analyze(opening,1,true,0);
    check(best.bookHit&&best.value==-6,"known strict book baseline");
    for(int i=0;i<64;++i)check(engine.analyze(opening,1,true,0).value==best.value,"strict option remains best-score only");
    int nearBest=0,bestCount=0;Bits observedMoves=0;
    for(int i=0;i<4096;++i){
        auto r=engine.analyze(opening,1); // Default variety = 1, not an opt-in test-only mode.
        check(r.bookHit&&flips(opening,r.move)!=0,"randomized book returns legal moves");
        check(r.value<=best.value&&r.value>=best.value-2,"default variety respects the 2-unit bound");
        observedMoves|=1ULL<<r.move;
        if(r.value<best.value)++nearBest;else ++bestCount;
    }
    check(nearBest>0&&count(observedMoves)>1,"same position admits near-best alternatives");
    check(bestCount>2048,"weighted selection favors best book scores");
    for(int i=0;i<256;++i){auto r=engine.analyze(opening,1,true,2);check(r.bookHit&&r.value>=best.value-4,"beginner variety respects the 4-unit bound");}
    bool rejected=false;try{engine.analyze(opening,1,true,4);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"out-of-range variety rejected");
    std::mt19937 rng(731);int passes=0,bookMisses=0;
    for(int game=0;game<8;++game){
        Position p;
        while(!over(p)){
            bool hit=engine.probeBook(p,lookup,1,731);
            check(lookup.nodes==0,"book-only probe never searches");
            if(hit)check(flips(p,lookup.move)!=0,"book-only hit is legal for either color");else ++bookMisses;
            auto cells=encode(p);int moves[64],n=0;
            check(egaroucid_get_legal_moves(cells.data(),p.turn==1?0:1,moves,&n,nullptr)==EGAROUCID_OK,"upstream legal API");
            Bits mask=0;for(int i=0;i<n;++i)mask|=1ULL<<moves[i];check(mask==legal(p),"both colors and all coordinates match");
            if(!mask){++passes;auto r=engine.analyze(p,1);check(r.move==-1,"forced pass remains explicit");p=play(p,-1);continue;}
            int m=moves[rng()%n],nFlipped=0,flipped[64];
            check(egaroucid_get_flipped_discs(cells.data(),p.turn==1?0:1,m,flipped,&nFlipped,nullptr)==EGAROUCID_OK,"upstream flips API");
            Bits f=0;for(int i=0;i<nFlipped;++i)f|=1ULL<<flipped[i];check(f==flips(p,m),"flip coordinates match");
            p=play(p,m);
        }
        auto r=engine.analyze(p,1);check(r.move==-1&&r.exact,"terminal result has no move");
    }
    check(passes>0,"forced pass tested");
    check(bookMisses>0,"book-only misses return without an upstream fallback");
    for(int game=0;game<2;++game){
        Position p;int plies=0,hits=0;
        while(!over(p)){
            check(++plies<=120,"AI games terminate");
            auto r=engine.analyze(p,1);if(r.bookHit)++hits;p=play(p,r.move);
        }
        check(hits>0,"complete game uses opening book");
    }
    void* handle=ot_egaroucid_create(nullptr);check(handle!=nullptr,"C ABI initializes");
    int cells[64]{},result[8];cells[27]=cells[36]=-1;cells[28]=cells[35]=1;
    check(ot_egaroucid_search(handle,cells,1,1,1,result)==1&&result[5]==1,"C ABI book hit");
    check(ot_egaroucid_book(handle,cells,1,1,42,result)==1&&result[4]==0&&result[5]==1,"C ABI book-only hit");
    check(ot_egaroucid_book(handle,cells,1,9,42,result)==-1,"C ABI book-only variety validation");
    check(ot_egaroucid_search(handle,cells,1,1,0,result)==1&&result[5]==0,"C ABI no-book search");
    cells[0]=7;check(ot_egaroucid_search(handle,cells,1,1,1,result)==-1,"C ABI invalid cells rejected");
    check(ot_egaroucid_search(nullptr,cells,1,1,1,result)==-1,"C ABI invalid handles rejected");
    ot_egaroucid_destroy(handle);
    std::cout<<"PASS: Egaroucid "<<engine.version()<<"; 4096 weighted book samples ("<<nearBest<<" near-best); strict/variety bounds; book on/off; 8 games legal/flips parity; forced passes; 2 complete AI games; C ABI\n";
}

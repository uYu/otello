#include "api.h"
#include "othello.hpp"
#include "nnue.hpp"
#include <climits>
struct ot_nnue {othello::nnue::Network network;};
namespace {
thread_local std::string nnueError;
othello::Position decode(const int* cells, int turn) {
    if(!cells || (turn!=1 && turn!=-1)) throw std::invalid_argument("Invalid position");
    othello::Position p; p.black=p.white=0; p.turn=turn;
    for(int i=0;i<64;++i) {
        if(cells[i]==1) p.black|=1ULL<<i;
        else if(cells[i]==-1) p.white|=1ULL<<i;
        else if(cells[i]!=0) throw std::invalid_argument("Invalid cell");
    }
    p.hash=othello::hashBoard(p.black,p.white,p.turn);return p;
}
void encode(const othello::SearchResult& r,int* output){
    output[0]=r.move;output[1]=r.score;output[2]=r.depth;output[3]=r.elapsed;
    output[4]=int(std::min<std::uint64_t>(INT_MAX,r.nodes));output[5]=r.exact;output[6]=int(r.candidates.size());
    output[135]=int(std::min<std::uint64_t>(INT_MAX,r.ttHits));
    for(std::size_t i=0;i<r.candidates.size();++i){output[7+i*2]=r.candidates[i].move;output[8+i*2]=r.candidates[i].score;}
}
}
extern "C" {
int ot_legal(const int* cells,int turn,int* moves) {
    try { if(!moves) return -1; auto b=othello::legal(decode(cells,turn));int n=0;
        while(b){moves[n++]=othello::first(b);b&=b-1;} return n;
    } catch(...) {return -1;}
}
int ot_play(const int* cells,int turn,int move,int* output) {
    try { if(!output) return 0; auto p=decode(cells,turn);auto next=othello::play(p,move);
        for(int i=0;i<64;++i) output[i]=(next.black>>i&1)?1:(next.white>>i&1)?-1:0;
        output[64]=next.turn;output[65]=othello::over(next);output[66]=othello::count(othello::flips(p,move));return 1;
    } catch(...) {return 0;}
}
int ot_search(const int* cells,int turn,int ms,int depth,int level,unsigned seed,int* output) {
    try {if(!output)return -1;auto r=othello::search(decode(cells,turn),{ms,depth,level,seed});
        encode(r,output);return 1;
    } catch(...) {return -1;}
}
ot_nnue* ot_nnue_load(const char* path){
    try{nnueError.clear();if(!path)throw std::invalid_argument("Model path is null");
        return new ot_nnue{othello::nnue::Network::load(path)};
    }catch(const std::exception& e){nnueError=e.what();return nullptr;}catch(...){nnueError="Cannot load NNUE model";return nullptr;}
}
void ot_nnue_destroy(ot_nnue* model){delete model;}
int ot_nnue_evaluate(const ot_nnue* model,const int* cells,int turn,int* score){
    try{nnueError.clear();if(!model||!score)throw std::invalid_argument("Null NNUE argument");
        *score=model->network.evaluate(decode(cells,turn));return 1;
    }catch(const std::exception& e){nnueError=e.what();return -1;}catch(...){nnueError="NNUE evaluation failed";return -1;}
}
int ot_nnue_search(const ot_nnue* model,const int* cells,int turn,int ms,int depth,int level,unsigned seed,int* output){
    try{nnueError.clear();if(!model||!output)throw std::invalid_argument("Null NNUE argument");
        encode(othello::search(decode(cells,turn),{ms,depth,level,seed,&model->network}),output);return 1;
    }catch(const std::exception& e){nnueError=e.what();return -1;}catch(...){nnueError="NNUE search failed";return -1;}
}
const char* ot_nnue_error(){return nnueError.c_str();}
// Single-threaded shared scratch space only for the browser adapter. Native callers use caller-owned buffers above.
int* ot_input() {static int cells[64];return cells;}
int* ot_output() {static int output[136];return output;}
}

#include "api.h"
#include "othello.hpp"
#include "nnue.hpp"
#include "mcts.hpp"
#include "default_engine.hpp"
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
void ot_search_clear_cache(void){othello::clearSearchCache();}
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
const char* ot_default_model_sha256(){return othello::defaultNetworkSha256();}
int ot_default_evaluate(const int* cells,int turn,int* score){
    try{nnueError.clear();if(!score)throw std::invalid_argument("Null default score");
        *score=othello::defaultNetwork().evaluate(decode(cells,turn));return 1;
    }catch(const std::exception& e){nnueError=e.what();return -1;}catch(...){nnueError="Default evaluation failed";return -1;}
}
int ot_default_search(const int* cells,int turn,int ms,unsigned seed,int* output){
    try{nnueError.clear();if(!output)throw std::invalid_argument("Null default output");
        auto p=decode(cells,turn);auto r=othello::mcts(p,{ms,0,seed,&othello::defaultNetwork()});
        auto selected=othello::selectMove(p,r.analysis,{},seed);
        r.analysis.move=selected.move;r.analysis.score=selected.score;encode(r.analysis,output);return 1;
    }catch(const std::exception& e){nnueError=e.what();return -1;}catch(...){nnueError="Default search failed";return -1;}
}
void ot_mcts_clear_cache(void){othello::clearMctsCache();}
int ot_nnue_mcts_play(const ot_nnue* model,const int* cells,int turn,int ms,unsigned simulations,unsigned seed,
                      int plies,int temperature,int maxLoss,int* output,int* policyOutput,int* visits,double* stats){
    try{
        nnueError.clear();if(!model||!output||!policyOutput||!visits||!stats)throw std::invalid_argument("Null MCTS argument");
        auto p=decode(cells,turn);othello::selectMove(p,{}, {plies,temperature,maxLoss},seed);
        auto r=othello::mcts(p,{ms,simulations,seed,&model->network});
        auto selected=othello::selectMove(p,r.analysis,{plies,temperature,maxLoss},seed);
        r.analysis.move=selected.move;r.analysis.score=selected.score;encode(r.analysis,output);
        for(std::size_t i=0;i<r.visits.size();++i){visits[i]=int(std::min<std::uint64_t>(INT_MAX,r.visits[i]));stats[8+i]=r.values[i];}
        policyOutput[0]=selected.eligible;policyOutput[1]=selected.temperatureMilli;policyOutput[2]=selected.lossLimit;
        policyOutput[3]=selected.bestScore;policyOutput[4]=selected.bestScore-selected.score;policyOutput[5]=selected.sampled;
        stats[0]=double(r.analysis.nodes);stats[1]=double(r.analysis.evaluations);stats[2]=r.meanDepth*1000;
        stats[3]=double(r.reusedNodes);stats[4]=double(r.treeNodes);stats[5]=r.analysis.elapsed;
        stats[6]=r.analysis.exact;stats[7]=r.memoryLimited;return 1;
    }catch(const std::exception& e){nnueError=e.what();return -1;}catch(...){nnueError="MCTS failed";return -1;}
}
int ot_nnue_play(const ot_nnue* model,const int* cells,int turn,int ms,int depth,int level,unsigned seed,
                  int plies,int temperature,int maxLoss,int* output,int* policyOutput){
    try{
        nnueError.clear();if(!model||!output||!policyOutput)throw std::invalid_argument("Null NNUE play argument");
        auto p=decode(cells,turn);
        // Validate before spending the search budget.
        othello::selectMove(p,{}, {plies,temperature,maxLoss},seed);
        auto r=othello::search(p,{ms,depth,level,seed,&model->network});
        auto selected=othello::selectMove(p,r,{plies,temperature,maxLoss},seed);
        r.move=selected.move;r.score=selected.score;encode(r,output);
        policyOutput[0]=selected.eligible;policyOutput[1]=selected.temperatureMilli;policyOutput[2]=selected.lossLimit;
        policyOutput[3]=selected.bestScore;policyOutput[4]=selected.bestScore-selected.score;policyOutput[5]=selected.sampled;
        return 1;
    }catch(const std::exception& e){nnueError=e.what();return -1;}catch(...){nnueError="NNUE play failed";return -1;}
}
// Single-threaded shared scratch space only for the browser adapter. Native callers use caller-owned buffers above.
int* ot_input() {static int cells[64];return cells;}
int* ot_output() {static int output[136];return output;}
}

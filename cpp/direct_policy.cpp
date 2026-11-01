// Policy-only browser adapter. Uses the exact frozen v2 Pattern implementation
// and embedded model; no search implementation is linked into this module.
#include "pattern.hpp"
#include <sstream>
namespace {
#include "model/default_nnue.inc"
int cells[64];
double logits[64];
const othello::nnue::Pattern& network(){
    static const auto model=[](){
        std::string bytes(reinterpret_cast<const char*>(defaultModelBytes),sizeof(defaultModelBytes));
        std::istringstream stream(bytes,std::ios::in|std::ios::binary);
        char magic[8];stream.read(magic,8);
        if(std::string(magic,8)!=std::string("OTPAT9\0\0",8))throw std::runtime_error("Expected approved v2 model");
        return othello::nnue::Pattern::load(stream,false,true,false,false,false,true);
    }();
    return *model;
}
}
extern "C" {
int* ot_policy_input(){return cells;}
double* ot_policy_output(){return logits;}
const char* ot_policy_model_sha256(){return defaultModelSha256;}
int ot_policy_infer(int turn){
    try{
        if(turn!=1&&turn!=-1)return -1;
        othello::Position p;p.black=p.white=0;p.turn=turn;
        for(int i=0;i<64;++i){
            if(cells[i]==1)p.black|=1ULL<<i;
            else if(cells[i]==-1)p.white|=1ULL<<i;
            else if(cells[i]!=0)return -1;
        }
        const auto prediction=network().predict(p);
        std::copy(prediction.logits.begin(),prediction.logits.end(),logits);
        return 1;
    }catch(...){return -1;}
}
}

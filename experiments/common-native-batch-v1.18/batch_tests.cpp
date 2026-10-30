#include "nnue.hpp"
#include "cnn.hpp"
#include "mcts.hpp"
#include <cstdlib>
#include <iostream>
#include <random>

using namespace othello;
using Callback=int(*)(int,const int*,const int*,const std::uint64_t*,int*,double*);
extern "C" int ot_cnn_set_batch_evaluator(Callback,int);
extern "C" int ot_cnn_set_native_batch_evaluator(int);
extern "C" void ot_cnn_native_batch_validate(int);
extern "C" void ot_cnn_native_batch_validation_errors(double*,double*);
void check(bool value,const char* why){if(!value){std::cerr<<why<<'\n';std::exit(1);}}
int bad(int,const int*,const int*,const std::uint64_t*,int*,double*){return -1;}
int exact(Position p){Bits moves=legal(p);if(!moves){if(legal(p.opponent(),p.own()))return -exact(play(p,-1));
    int diff=count(p.own())-count(p.opponent());return (diff>0)-(diff<0);}
    int best=-2;while(moves){int m=first(moves);moves&=moves-1;best=std::max(best,-exact(play(p,m)));}return best;}
int main(int argc,char** argv){
    std::mt19937 rng(718);
    for(int c:{8,16,32}){
        auto weights=std::make_shared<nnue::CnnPolicy>();weights->channels=c;
        weights->weights.resize(c*18+c+c*c*9+c+64*c*64+64+65+65*65);
        for(auto& w:weights->weights)w=float(int(rng()%101)-50)/1000;
        nnue::Network model;model.architecture=nnue::Architecture::CnnPolicy;model.valueTarget=nnue::ValueTarget::Outcome;
        model.hidden=0;model.cnnPolicy=weights;
        ot_cnn_native_batch_validate(1);
        for(int size:{1,8,32,128}){
            check(ot_cnn_set_native_batch_evaluator(size)==1,"native batch setup");clearMctsCache();
            auto result=mcts(Position{},{0,257,19,&model});
            check(result.analysis.nodes==257&&flips(Position{},result.analysis.move),"partial final batch honors simulation budget");
        }
        double value=0,policy=0;ot_cnn_native_batch_validation_errors(&value,&policy);
        check(value<=1&&policy<1e-5,"batched inference agrees with scalar model");
        ot_cnn_native_batch_validate(0);
        ot_cnn_set_batch_evaluator(bad,16);clearMctsCache();bool failed=false;
        try{mcts(Position{},{0,63,1,&model});}catch(...){failed=true;}check(failed,"failed callback rejected");
        ot_cnn_set_native_batch_evaluator(16);
        auto recovered=mcts(Position{},{0,63,1,&model});check(recovered.analysis.nodes==63,"virtual reservations recover after callback failure");
        Position late;
        while(count(late.black|late.white)<58&&!over(late)){
            auto moves=legal(late);if(!moves){late=play(late,-1);continue;}
            int skip=int(rng()%count(moves));while(skip--)moves&=moves-1;late=play(late,first(moves));
        }
        clearMctsCache();auto solved=mcts(late,{0,200000,1,&model});
        check(solved.analysis.exact&&solved.analysis.score==exact(late)*100000,"batch preserves exact outcome proof");
        auto terminal=parse(std::string(64,'B'),-1);clearMctsCache();
        check(mcts(terminal,{0,1,1,&model}).analysis.score==-100000,"batch terminal outcome");
    }
    check(argc==2,"approved NNUE model argument required");
    auto baseline=nnue::Network::load(argv[1]);
    ot_cnn_native_batch_validate(1);
    for(int size:{1,8,16,32,64,128}){
        ot_cnn_set_native_batch_evaluator(size);clearMctsCache();
        check(mcts(Position{},{0,257,31,&baseline}).analysis.nodes==257,"NNUE batched budget and exact inference");
    }
    double value=0,policy=0;ot_cnn_native_batch_validation_errors(&value,&policy);
    check(value==0&&policy<1e-12,"NNUE batch preserves integer semantics");ot_cnn_native_batch_validate(0);
    ot_cnn_set_batch_evaluator(nullptr,1);clearMctsCache();auto original=mcts(Position{},{0,512,31,&baseline});
    ot_cnn_set_native_batch_evaluator(1);clearMctsCache();auto batched=mcts(Position{},{0,512,31,&baseline});
    check(original.analysis.move==batched.analysis.move&&original.analysis.score==batched.analysis.score&&original.visits==batched.visits,
          "NNUE batch 1 reproduces original fixed-work search");
    check(ot_cnn_set_native_batch_evaluator(0)==-1&&ot_cnn_set_native_batch_evaluator(129)==-1,"batch size bounds");
    std::cout<<"PASS: batch budgets, all widths, scalar parity, failed-evaluator cleanup and exact endgames\n";
}

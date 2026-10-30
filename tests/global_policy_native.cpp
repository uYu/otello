#include "pattern.hpp"
#include <cmath>
#include <iostream>
#include <random>
using namespace othello;
static void equal(const nnue::Pattern::Prediction& a,const nnue::Pattern::Prediction& b,Bits mask=~Bits(0)){
    if(a.values!=b.values)throw std::runtime_error("value mismatch");
    while(mask){int s=first(mask);mask&=mask-1;if(a.logits[s]!=b.logits[s])throw std::runtime_error("policy mismatch");}
}
int main(int argc,char** argv){
    if(argc!=2)return 2;auto network=nnue::Network::load(argv[1]);const auto& model=*network.pattern;
    if(model.variant!=6)return 3;
    std::mt19937 rng(49149);Position p;int passes=0;
    for(int i=0;i<512;++i){
        if(over(p))p=Position{};auto moves=legal(p);auto full=model.predict(p),masked=model.predictLegal(p,moves);
        equal(full,masked,moves);
        auto known=model.policy(p,full.values);for(int s=0;s<64;++s)if(known[s]!=full.logits[s])return 4;
        if(model.outputs(model.refresh(p),p.turn)!=full.values)return 5;
        auto state=model.refreshState(p);equal(model.predict(state,p.turn),full);equal(model.predictLegal(state,p.turn,moves),masked,moves);
        int move=-1;if(moves){int skip=rng()%count(moves);while(skip--)moves&=moves-1;move=first(moves);}else ++passes;
        auto next=play(p,move);auto updated=model.updateState(state,next);equal(model.predict(updated,next.turn),model.predict(next));
        Position swapped=p;std::swap(swapped.black,swapped.white);swapped.turn=-swapped.turn;equal(full,model.predict(swapped));
        for(int mirror=0;mirror<2;++mirror)for(int rotation=0;rotation<4;++rotation){
            Position transformed=p;transformed.black=transformed.white=0;int mapping[64];
            for(int s=0;s<64;++s){int x=s%8,y=s/8;if(mirror)x=7-x;for(int r=0;r<rotation;++r){int old=x;x=7-y;y=old;}mapping[s]=y*8+x;
                if(p.black>>s&1)transformed.black|=Bits(1)<<mapping[s];if(p.white>>s&1)transformed.white|=Bits(1)<<mapping[s];}
            auto prediction=model.predict(transformed);if(prediction.values!=full.values)return 6;
            for(int s=0;s<64;++s)if(std::abs(prediction.logits[mapping[s]]-full.logits[s])>1e-12)return 7;
        }
        auto changed=model;changed.globalUpBias[0]++;
        if(changed==model)return 8;
        p=next;
    }
    std::cout<<"global policy native: 512 positions, legal/known/state/update/swap/D4/parameter equality passed; passes="<<passes<<'\n';
}

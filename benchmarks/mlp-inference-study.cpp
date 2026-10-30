// Isolated research harness; never linked into the default engine.
#include "pattern.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#if defined(STUDY_PROFILE)
extern unsigned long long study_ns[5];
#endif
using namespace othello;
int main(int argc,char** argv){
    if(argc!=5)return 2;
    auto network=nnue::Network::load(argv[1]);const auto& model=*network.pattern;
    std::ifstream input(argv[2]);std::vector<Position> boards;std::string cells;int turn;
    while(input>>cells>>turn)boards.push_back(parse(cells,turn));
    if(boards.empty())return 3;
    std::vector<Bits> masks;for(const auto& p:boards)masks.push_back(legal(p));
    int iterations=std::stoi(argv[3]),mode=std::stoi(argv[4]);
    std::array<std::vector<int>,3> groups;
    for(int i=0;i<int(boards.size());++i){int placed=60-boards[i].empty();groups[placed<20?0:placed<44?1:2].push_back(i);}
    std::int64_t checksum=0;double logitChecksum=0;
    auto predict=[&](int i){return model.predictLegal(boards[i],mode==0?0:mode==1?masks[i]:~Bits(0));};
    for(int i=0;i<2048;++i)checksum+=predict(i%boards.size()).values[0];
    std::cout<<std::setprecision(12)<<"{\"groups\":[";
    for(int g=0;g<3;++g){
        if(groups[g].empty())return 4;
#if defined(STUDY_PROFILE)
        for(auto& ns:study_ns)ns=0;
#endif
        double meanLegal=0;for(int i:groups[g])meanLegal+=count(masks[i]);
        auto tick=std::chrono::steady_clock::now();
        for(int j=0;j<iterations;++j){int i=groups[g][j%groups[g].size()];auto prediction=predict(i);
            checksum+=prediction.values[0];Bits selected=mode==0?0:mode==1?masks[i]:~Bits(0);
            while(selected){int s=first(selected);selected&=selected-1;logitChecksum+=prediction.logits[s];}}
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tick).count();
        if(g)std::cout<<',';
        std::cout<<"{\"phase\":"<<g<<",\"positions\":"<<groups[g].size()<<",\"meanLegalMoves\":"<<meanLegal/groups[g].size()<<",\"nanoseconds\":"<<double(ns)/iterations;
#if defined(STUDY_PROFILE)
        std::cout<<",\"stageNanoseconds\":[";for(int k=0;k<5;++k){if(k)std::cout<<',';std::cout<<double(study_ns[k])/iterations;}std::cout<<']';
#endif
        std::cout<<'}';
    }
    std::cout<<"],\"checksum\":"<<checksum<<",\"logitChecksum\":"<<logitChecksum<<"}\n";
}

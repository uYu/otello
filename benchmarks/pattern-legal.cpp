#include "pattern.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace othello;
int main(int argc,char** argv){
    if(argc!=3)return 2;auto network=nnue::Network::load(argv[1]);if(!network.pattern)return 3;
    const auto& model=*network.pattern;int iterations=std::stoi(argv[2]);if(iterations<1)return 4;
    std::mt19937 rng(3412);std::vector<Position> boards;std::vector<Bits> masks;Position p;
    for(int i=0;i<512;++i){if(over(p))p=Position{};boards.push_back(p);Bits moves=legal(p);masks.push_back(moves);int move=-1;
        if(moves){int skip=rng()%count(moves);while(skip--)moves&=moves-1;move=first(moves);}p=play(p,move);}
    std::array<std::vector<int>,3> groups;std::array<double,3> active{},legalCounts{};std::int64_t checksum=0;
    auto predict=[&](int i){
#if defined(OTELLO_PATTERN_LEGAL_POLICY)
        return model.predictLegal(boards[i],masks[i]);
#else
        return model.predict(boards[i]);
#endif
    };
    for(int i=0;i<int(boards.size());++i){int occupied=count(boards[i].black|boards[i].white),group=occupied<24?0:occupied<48?1:2;groups[group].push_back(i);
        legalCounts[group]+=count(masks[i]);for(const auto& shape:nnue::Pattern::shapes()){Bits mask=0;for(int j=0;j<shape.length;++j)mask|=Bits(1)<<shape.squares[j];active[group]+=bool(mask&masks[i]);}
        auto full=model.predict(boards[i]),masked=model.predictLegal(boards[i],masks[i]);if(full.values!=masked.values)return 5;
        Bits mask=masks[i];double maximum=-1e30;while(mask){int square=first(mask);mask&=mask-1;if(full.logits[square]!=masked.logits[square])return 6;maximum=std::max(maximum,full.logits[square]);}
        mask=masks[i];while(mask){int square=first(mask);mask&=mask-1;checksum+=std::llround(full.logits[square]*1000000);}}
    for(int i=0;i<2048;++i)checksum+=predict(i%512).values[0];
    std::cout<<"{\"groups\":[";
    for(int group=0;group<3;++group){const auto& indices=groups[group];double logitChecksum=0;
        auto tick=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i){int index=indices[i%indices.size()];auto prediction=predict(index);checksum+=prediction.values[0];Bits mask=masks[index];
            while(mask){int square=first(mask);mask&=mask-1;logitChecksum+=prediction.logits[square];}}
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tick).count();
        if(group)std::cout<<',';
        std::cout<<"{\"stage\":"<<group<<",\"nanoseconds\":"<<double(ns)/iterations<<",\"meanActivePatterns\":"<<active[group]/indices.size()<<",\"meanLegalMoves\":"<<legalCounts[group]/indices.size()<<",\"logitChecksum\":"<<logitChecksum<<'}';}
    std::cout<<"],\"checksum\":"<<checksum<<"}\n";
}

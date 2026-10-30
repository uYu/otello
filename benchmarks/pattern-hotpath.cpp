#include "pattern.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
using namespace othello;
int main(int argc,char** argv){
    if(argc!=3)return 2;
    auto net=nnue::Network::load(argv[1]);if(!net.pattern)return 3;
    const auto& model=*net.pattern;int iterations=std::stoi(argv[2]);if(iterations<1)return 4;
    std::mt19937 rng(3218);std::vector<Position> positions;Position p;
    for(int i=0;i<128;++i){if(over(p))p=Position{};positions.push_back(p);Bits moves=legal(p);int move=-1;
        if(moves){int skip=rng()%count(moves);while(skip--)moves&=moves-1;move=first(moves);}p=play(p,move);}
    std::vector<nnue::Accumulator> sums;for(const auto& board:positions)sums.push_back(model.refresh(board));
    std::int64_t checksum=0;
    for(int i=0;i<2048;++i)checksum+=model.predict(positions[i%128]).values[0];
    auto measure=[&](auto call){auto start=std::chrono::steady_clock::now();
        for(int i=0;i<iterations;++i)checksum+=call(i%128);
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count())/iterations;};
    double encoding=measure([&](int i){const auto& board=positions[i];std::int64_t sum=0;
#if defined(OTELLO_PATTERN_FAST_CODES)
        for(auto code:nnue::Pattern::encode(board.own(),board.opponent()))sum+=code;
#else
        for(const auto& shape:nnue::Pattern::shapes())sum+=nnue::Pattern::code(shape,board.own(),board.opponent());
#endif
        return sum;});
    double dense=measure([&](int i){return model.outputs(sums[i],positions[i].turn)[0];});
    double joint=measure([&](int i){return model.predict(positions[i]).values[0];});
    std::cout<<"{\"encodingNs\":"<<encoding<<",\"denseNs\":"<<dense<<",\"jointNs\":"<<joint<<",\"checksum\":"<<checksum<<"}\n";
}

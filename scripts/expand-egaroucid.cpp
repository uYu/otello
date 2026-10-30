// Exhaustive position expansion from verified compact transcripts, without sampling.
#include "othello.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <filesystem>
#include <memory>
using namespace othello;
namespace {
using Key=std::pair<Bits,Bits>;
struct Hash {std::size_t operator()(const Key& k)const{return k.first^(k.second*0x9e3779b97f4a7c15ULL);}};
Bits mirror(Bits x){
    x=((x>>1)&0x5555555555555555ULL)|((x&0x5555555555555555ULL)<<1);
    x=((x>>2)&0x3333333333333333ULL)|((x&0x3333333333333333ULL)<<2);
    return ((x>>4)&0x0f0f0f0f0f0f0f0fULL)|((x&0x0f0f0f0f0f0f0f0fULL)<<4);
}
Bits transpose(Bits x){
    Bits t=(x^(x>>7))&0x00aa00aa00aa00aaULL;x^=t^(t<<7);
    t=(x^(x>>14))&0x0000cccc0000ccccULL;x^=t^(t<<14);
    t=(x^(x>>28))&0x00000000f0f0f0f0ULL;return x^t^(t<<28);
}
Bits rotate(Bits x){return mirror(transpose(x));}
int mapMove(int m,int s){int x=m%8,y=m/8;if(s&4)x=7-x;
    for(int r=0;r<(s&3);++r){int old=x;x=7-y;y=old;}return 8*y+x;}
struct Canonical {Bits own,opp;int move;};
Canonical canonical(const Position& p,int move){
    Key best{~0ULL,~0ULL};int bestMove=64;
    for(int m=0;m<2;++m){Bits own=m?mirror(p.own()):p.own(),opp=m?mirror(p.opponent()):p.opponent();
        for(int r=0;r<4;++r){Key key{own,opp};int candidate=mapMove(move,m*4+r);
            if(key<best||(key==best&&candidate<bestMove)){best=key;bestMove=candidate;}
            own=rotate(own);opp=rotate(opp);}}
    return {best.first,best.second,bestMove};
}
Bits read64(const unsigned char* p){Bits v=0;for(int i=0;i<8;++i)v|=Bits(p[i])<<(8*i);return v;}
void write64(unsigned char* p,Bits v){for(int i=0;i<8;++i)p[i]=v>>(8*i);}
}
int main(int argc,char** argv){
 std::ios::sync_with_stdio(false);
 try{
    if(argc!=5)throw std::runtime_error("Usage: expand-egaroucid GAMES.bin OUTPUT_DIRECTORY HOLDOUT.bin BUCKETS");
    int buckets=std::stoi(argv[4]);if(buckets<1||buckets>128)throw std::runtime_error("Invalid bucket count");
    std::unordered_set<Key,Hash> frozen;
    std::ifstream holdout(argv[3],std::ios::binary);if(!holdout)throw std::runtime_error("Missing holdout file");
    unsigned char keyBytes[16];while(holdout.read(reinterpret_cast<char*>(keyBytes),16))frozen.insert({read64(keyBytes),read64(keyBytes+8)});
    if(holdout.gcount())throw std::runtime_error("Truncated holdout file");
    std::filesystem::create_directories(argv[2]);std::vector<std::unique_ptr<std::ofstream>> streams;
    for(int i=0;i<buckets;++i){auto path=std::filesystem::path(argv[2])/(std::to_string(i)+".bin");
        auto s=std::make_unique<std::ofstream>(path,std::ios::binary);if(!*s)throw std::runtime_error("Cannot create bucket");streams.push_back(std::move(s));}
    std::ifstream input(argv[1],std::ios::binary);if(!input)throw std::runtime_error("Missing games file");
    std::uint64_t games=0,positions=0,blocked=0,bySplit[3]={},byPly[60]={};unsigned char record[64];
    while(input.read(reinterpret_cast<char*>(record),64)){
        int random=record[60],n=record[61],expected=static_cast<signed char>(record[62]),split=record[63];
        if(n>60||random>n||split>2)throw std::runtime_error("Invalid compact record metadata");
        for(int i=n;i<60;++i)if(record[i]!=255)throw std::runtime_error("Invalid compact padding");
        Position p;
        for(int ply=0;ply<n;++ply){
            int move=record[ply];if(move>63)throw std::runtime_error("Invalid square");Bits f=flips(p,move);
            if(!f){if(legal(p))throw std::runtime_error("Illegal move");p.turn=-p.turn;f=flips(p,move);if(!f)throw std::runtime_error("Illegal move after pass");}
            if(ply>=random){
                auto c=canonical(p,move);bool excluded=frozen.count({c.own,c.opp});
                unsigned char row[32]={};write64(row,c.own);write64(row+8,c.opp);write64(row+16,legal(c.own,c.opp));
                row[24]=c.move;row[25]=((expected>0)-(expected<0))*p.turn;row[26]=expected*p.turn;
                row[27]=ply;row[28]=split;row[29]=excluded;
                Bits mix=c.own^(c.opp*0x9e3779b97f4a7c15ULL);auto bucket=splitmix(mix)%buckets;
                streams[bucket]->write(reinterpret_cast<char*>(row),32);
                ++positions;++bySplit[split];++byPly[ply];blocked+=excluded;
            }
            if(p.turn==1){p.black|=f|(1ULL<<move);p.white&=~f;}else{p.white|=f|(1ULL<<move);p.black&=~f;}
            p.turn=-p.turn;
        }
        if(!over(p)||count(p.black)-count(p.white)!=expected)throw std::runtime_error("Terminal result differs from compact record");
        ++games;
    }
    if(input.gcount()||input.bad())throw std::runtime_error("Truncated compact input");
    for(auto& stream:streams){stream->flush();if(!*stream)throw std::runtime_error("Bucket write failed");}
    std::cout<<"{\"games\":"<<games<<",\"positions\":"<<positions<<",\"frozenObservations\":"<<blocked<<",\"splitOccurrences\":[";
    for(int i=0;i<3;++i){if(i)std::cout<<',';std::cout<<bySplit[i];}std::cout<<"],\"placementOccurrences\":[";
    for(int i=0;i<60;++i){if(i)std::cout<<',';std::cout<<byPly[i];}std::cout<<"]}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

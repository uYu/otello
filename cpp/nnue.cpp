#include "nnue.hpp"
#include <cmath>
#include <fstream>
#include "model/tiny_nn.h"

namespace othello::nnue {
namespace {
std::uint32_t read(std::istream& in,int bytes){
    std::uint32_t value=0;
    for(int i=0;i<bytes;++i){int c=in.get();if(c==EOF)throw std::runtime_error("Truncated Othello NNUE model");value|=std::uint32_t(c)<<(i*8);}return value;
}
void change(Accumulator& acc,const Network& net,int square,bool black,int direction){
    int b=(black?0:64)+square,w=(black?64:0)+square;
    for(int j=0;j<net.hidden;++j){acc.black[j]+=direction*net.weights[b][j];acc.white[j]+=direction*net.weights[w][j];}
}
}
Network Network::load(const std::string& path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open Othello NNUE model: "+path);
    char magic[8]{};in.read(magic,8);
    if(!in||std::string(magic,8)!=std::string("OTNNUE1\0",8))throw std::runtime_error("Not an Othello NNUE v1 model");
    Network net;
    if(read(in,4)!=Features)throw std::runtime_error("Unsupported Othello NNUE features");
    net.hidden=int(read(in,4));
    if((net.hidden!=32&&net.hidden!=64&&net.hidden!=128&&net.hidden!=256)||read(in,4)!=Scale||read(in,4)!=Scale)throw std::runtime_error("Unsupported Othello NNUE dimensions/scales");
    for(auto& row:net.weights)for(int j=0;j<net.hidden;++j){auto u=read(in,2);row[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
    for(int j=0;j<net.hidden;++j){auto u=read(in,4);std::int64_t v=u>=0x80000000U?std::int64_t(u)-0x100000000LL:u;
        if(v < -10000000||v > 10000000)throw std::runtime_error("NNUE bias outside safe accumulator range");net.bias[j]=std::int32_t(v);}
    for(int j=0;j<net.hidden;++j){auto u=read(in,2);net.output[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in Othello NNUE model");
    return net;
}
Accumulator Network::refresh(const Position& p) const{
    Accumulator acc{bias,bias};Bits b=p.black,w=p.white;
    while(b){int i=first(b);b&=b-1;change(acc,*this,i,true,1);}
    while(w){int i=first(w);w&=w-1;change(acc,*this,i,false,1);}return acc;
}
Accumulator Network::update(const Accumulator& parent,const Position& before,const Position& after) const{
    Accumulator acc=parent;
    Bits changed=(before.black^after.black)|(before.white^after.white);
    while(changed){int i=first(changed);Bits bit=1ULL<<i;changed&=changed-1;
        if(before.black&bit)change(acc,*this,i,true,-1);else if(before.white&bit)change(acc,*this,i,false,-1);
        if(after.black&bit)change(acc,*this,i,true,1);else if(after.white&bit)change(acc,*this,i,false,1);
    }return acc;
}
int Network::evaluate(const Accumulator& acc,int turn) const{
    std::int64_t total=0;
    for(int j=0;j<hidden;++j)total+=std::int64_t(std::clamp(acc.black[j],0,Scale)-std::clamp(acc.white[j],0,Scale))*output[j];
    // The shared two-perspective head makes color/turn antisymmetry exact.
    return int(std::lround(1000.0*std::tanh(double(total)*turn/(2.0*Scale*Scale))));
}
int Network::evaluateReference(const Position& p) const{
    std::vector<float> matrix(Features*hidden);
    std::array<float,Features> black{},white{};
    std::vector<float> offsets(hidden),head(hidden),b(hidden),w(hidden);
    for(int i=0;i<Features;++i)for(int j=0;j<hidden;++j)matrix[j*Features+i]=float(weights[i][j])/Scale;
    for(int j=0;j<hidden;++j){offsets[j]=float(bias[j])/Scale;head[j]=float(output[j])/Scale;}
    for(int i=0;i<64;++i){
        if(p.black>>i&1){black[i]=1;white[64+i]=1;}
        if(p.white>>i&1){black[64+i]=1;white[i]=1;}
    }
    tiny_nn::linear(matrix.data(),offsets.data(),black.data(),b.data(),Features,hidden);
    tiny_nn::linear(matrix.data(),offsets.data(),white.data(),w.data(),Features,hidden);
    for(int j=0;j<hidden;++j){b[j]=std::clamp(b[j],0.0f,1.0f);w[j]=std::clamp(w[j],0.0f,1.0f);}
    double raw=(tiny_nn::dot(b.data(),head.data(),hidden)-tiny_nn::dot(w.data(),head.data(),hidden))*0.5;
    return int(std::lround(1000.0*std::tanh(raw*p.turn)));
}
}

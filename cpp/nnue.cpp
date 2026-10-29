#include "nnue.hpp"
#include "cnn.hpp"
#include "resnet.hpp"
#include "pattern.hpp"
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
    return load(in);
}
Network Network::load(std::istream& in){
    char magic[8]{};in.read(magic,8);
    const auto format=std::string(magic,8);
    if(in&&format==std::string("OTPAT1\0\0",8)){
        Network net;net.architecture=Architecture::Pattern;net.hidden=0;net.pattern=Pattern::load(in);return net;
    }
    if(in&&format==std::string("OTRES1\0\0",8)){
        Network net;net.architecture=Architecture::Residual;net.hidden=0;net.residual=ResNet::load(in);return net;
    }
    if(in&&format==std::string("OTCNN1\0\0",8)){
        Network net;net.architecture=Architecture::Cnn;net.hidden=0;net.cnn=Cnn::load(in);return net;
    }
    if(!in||(format!=std::string("OTNNUE1\0",8)&&format!=std::string("OTNNUE2\0",8)))throw std::runtime_error("Not an Othello NNUE model");
    Network net;
    net.architecture=format==std::string("OTNNUE2\0",8)?Architecture::Turn:Architecture::Legacy;
    if(read(in,4)!=Features)throw std::runtime_error("Unsupported Othello NNUE features");
    net.hidden=int(read(in,4));
    if((net.hidden!=32&&net.hidden!=64&&net.hidden!=128&&net.hidden!=256)||read(in,4)!=Scale||read(in,4)!=Scale)throw std::runtime_error("Unsupported Othello NNUE dimensions/scales");
    for(auto& row:net.weights)for(int j=0;j<net.hidden;++j){auto u=read(in,2);row[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
    for(int j=0;j<net.hidden;++j){auto u=read(in,4);std::int64_t v=u>=0x80000000U?std::int64_t(u)-0x100000000LL:u;
        if(v < -10000000||v > 10000000)throw std::runtime_error("NNUE bias outside safe accumulator range");net.bias[j]=std::int32_t(v);}
    for(int j=0;j<net.hidden;++j){auto u=read(in,2);net.output[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
    if(net.architecture==Architecture::Turn){
        for(int j=0;j<net.hidden;++j){auto u=read(in,2);net.opponentOutput[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
        auto u=read(in,4);net.outputBias=std::int32_t(u>=0x80000000U?std::int64_t(u)-0x100000000LL:u);
    }
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in Othello NNUE model");
    return net;
}
Accumulator Network::refresh(const Position& p) const{
    if(pattern)return pattern->refresh(p);
    if(spatial()){Accumulator acc;acc.blackBoard=p.black;acc.whiteBoard=p.white;return acc;}
    Accumulator acc{bias,bias};Bits b=p.black,w=p.white;
    while(b){int i=first(b);b&=b-1;change(acc,*this,i,true,1);}
    while(w){int i=first(w);w&=w-1;change(acc,*this,i,false,1);}return acc;
}
Accumulator Network::update(const Accumulator& parent,const Position& before,const Position& after) const{
    if(pattern)return pattern->update(parent,before,after);
    Accumulator acc=parent;
    if(spatial()){acc.blackBoard=after.black;acc.whiteBoard=after.white;return acc;}
    Bits changed=(before.black^after.black)|(before.white^after.white);
    while(changed){int i=first(changed);Bits bit=1ULL<<i;changed&=changed-1;
        if(before.black&bit)change(acc,*this,i,true,-1);else if(before.white&bit)change(acc,*this,i,false,-1);
        if(after.black&bit)change(acc,*this,i,true,1);else if(after.white&bit)change(acc,*this,i,false,1);
    }return acc;
}
int Network::evaluate(const Accumulator& acc,int turn) const{
    if(pattern)return pattern->evaluate(acc,turn);
    if(architecture==Architecture::Residual)return residual->evaluate(acc.blackBoard,acc.whiteBoard,turn);
    if(architecture==Architecture::Cnn)return cnn->evaluate(acc.blackBoard,acc.whiteBoard,turn);
    if(architecture==Architecture::Turn){
        const auto& own=turn==1?acc.black:acc.white;
        const auto& opp=turn==1?acc.white:acc.black;
        std::int64_t total=outputBias;
        for(int j=0;j<hidden;++j)total+=std::int64_t(std::clamp(own[j],0,Scale))*output[j]
            +std::int64_t(std::clamp(opp[j],0,Scale))*opponentOutput[j];
        return int(std::lround(1000.0*std::tanh(double(total)/(Scale*Scale))));
    }
    std::int64_t total=0;
    for(int j=0;j<hidden;++j)total+=std::int64_t(std::clamp(acc.black[j],0,Scale)-std::clamp(acc.white[j],0,Scale))*output[j];
    // The shared two-perspective head makes color/turn antisymmetry exact.
    return int(std::lround(1000.0*std::tanh(double(total)*turn/(2.0*Scale*Scale))));
}
int Network::evaluateReference(const Position& p) const{
    if(pattern)return pattern->evaluate(pattern->refresh(p),p.turn);
    if(architecture==Architecture::Residual)return residual->evaluateReference(p.black,p.white,p.turn);
    if(architecture==Architecture::Cnn)return cnn->evaluateReference(p.black,p.white,p.turn);
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
    if(architecture==Architecture::Turn){
        std::vector<float> other(hidden);for(int j=0;j<hidden;++j)other[j]=float(opponentOutput[j])/Scale;
        const auto& own=p.turn==1?b:w;const auto& opp=p.turn==1?w:b;
        double raw=tiny_nn::dot(own.data(),head.data(),hidden)+tiny_nn::dot(opp.data(),other.data(),hidden)
            +double(outputBias)/(Scale*Scale);
        return int(std::lround(1000.0*std::tanh(raw)));
    }
    double raw=(tiny_nn::dot(b.data(),head.data(),hidden)-tiny_nn::dot(w.data(),head.data(),hidden))*0.5;
    return int(std::lround(1000.0*std::tanh(raw*p.turn)));
}
int Network::evaluate(const Position& p) const{
    if(architecture==Architecture::Residual)return residual->evaluate(p.black,p.white,p.turn);
    return architecture==Architecture::Cnn?cnn->evaluate(p.black,p.white,p.turn):evaluate(refresh(p),p.turn);
}
bool Network::sameParameters(const Network& other) const{
    return architecture==other.architecture&&hidden==other.hidden&&weights==other.weights&&bias==other.bias
        &&output==other.output&&opponentOutput==other.opponentOutput&&outputBias==other.outputBias
        &&(cnn==other.cnn||(cnn&&other.cnn&&cnn->weights==other.cnn->weights))
        &&(pattern==other.pattern||(pattern&&other.pattern&&*pattern==*other.pattern))
        &&(residual==other.residual||(residual&&other.residual&&residual->blocks==other.residual->blocks&&residual->weights==other.residual->weights));
}
std::unique_ptr<Network> Network::snapshot() const{
    auto copy=std::make_unique<Network>(*this);if(cnn)copy->cnn=std::make_shared<Cnn>(*cnn);
    if(pattern)copy->pattern=std::make_shared<Pattern>(*pattern);
    if(residual)copy->residual=std::make_shared<ResNet>(*residual);return copy;
}
}

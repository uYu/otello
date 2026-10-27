#include "cnn.hpp"
#include "model/tiny_nn.h"
#include <cmath>
#include <cstring>
#include <istream>

namespace othello::nnue {
namespace {
constexpr int C1Bias=144,C2=152,C2Bias=728,Fc=736,FcBias=17120,Out=17152,OutBias=17184;
std::uint32_t read32(std::istream& in){
    std::uint32_t value=0;
    for(int i=0;i<4;++i){int c=in.get();if(c==EOF)throw std::runtime_error("Truncated Othello CNN");value|=std::uint32_t(c)<<(8*i);}
    return value;
}
template<int In> void conv(const float* input,const float* weights,const float* bias,float* output){
    std::array<float,In*100> padded{};
    for(int c=0;c<In;++c)for(int y=0;y<8;++y)std::copy_n(input+c*64+y*8,8,padded.data()+c*100+(y+1)*10+1);
    std::array<float,In*9> patch{};
    for(int y=0;y<8;++y)for(int x=0;x<8;++x){
        for(int c=0;c<In;++c)for(int dy=0;dy<3;++dy)
            std::copy_n(padded.data()+c*100+(y+dy)*10+x,3,patch.data()+c*9+dy*3);
        for(int o=0;o<Cnn::Channels;++o)output[o*64+y*8+x]=std::max(0.0f,bias[o]+tiny_nn::dot(weights+o*In*9,patch.data(),In*9));
    }
}
template<int In> void referenceConv(const float* input,const float* weights,const float* bias,float* output){
    for(int o=0;o<Cnn::Channels;++o)for(int y=0;y<8;++y)for(int x=0;x<8;++x){
        double value=bias[o];
        for(int c=0;c<In;++c)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            int yy=y+dy,xx=x+dx;
            if(yy>=0&&yy<8&&xx>=0&&xx<8)value+=double(input[c*64+yy*8+xx])*weights[o*In*9+c*9+(dy+1)*3+dx+1];
        }
        output[o*64+y*8+x]=float(std::max(0.0,value));
    }
}
std::array<float,128> inputs(Bits black,Bits white,int turn){
    std::array<float,128> result{};Bits own=turn==1?black:white,opp=turn==1?white:black;
    while(own){int i=first(own);own&=own-1;result[i]=1;}
    while(opp){int i=first(opp);opp&=opp-1;result[64+i]=1;}
    return result;
}
}
std::shared_ptr<const Cnn> Cnn::load(std::istream& in){
    if(read32(in)!=8||read32(in)!=2||read32(in)!=Channels||read32(in)!=Head)throw std::runtime_error("Unsupported CNN dimensions");
    auto result=std::make_shared<Cnn>();
    for(auto& w:result->weights){auto bits=read32(in);std::memcpy(&w,&bits,sizeof(w));
        if(!std::isfinite(w)||std::abs(w)>1000)throw std::runtime_error("Invalid CNN weight");}
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in Othello CNN");
    return result;
}
int Cnn::evaluate(Bits black,Bits white,int turn) const{
    auto input=inputs(black,white,turn);
    std::array<float,Channels*64> firstLayer{},secondLayer{};
    std::array<float,Head> hidden{};
    conv<2>(input.data(),weights.data(),weights.data()+C1Bias,firstLayer.data());
    conv<Channels>(firstLayer.data(),weights.data()+C2,weights.data()+C2Bias,secondLayer.data());
    tiny_nn::linear_relu(weights.data()+Fc,weights.data()+FcBias,secondLayer.data(),hidden.data(),Channels*64,Head);
    const float raw=tiny_nn::dot(weights.data()+Out,hidden.data(),Head)+weights[OutBias];
    return int(std::lround(1000.0*std::tanh(double(raw))));
}
int Cnn::evaluateReference(Bits black,Bits white,int turn) const{
    auto input=inputs(black,white,turn);
    std::array<float,Channels*64> firstLayer{},secondLayer{};
    std::array<float,Head> hidden{};
    referenceConv<2>(input.data(),weights.data(),weights.data()+C1Bias,firstLayer.data());
    referenceConv<Channels>(firstLayer.data(),weights.data()+C2,weights.data()+C2Bias,secondLayer.data());
    for(int o=0;o<Head;++o){double value=weights[FcBias+o];
        for(int i=0;i<Channels*64;++i)value+=double(secondLayer[i])*weights[Fc+o*Channels*64+i];
        hidden[o]=float(std::max(0.0,value));}
    double raw=weights[OutBias];for(int i=0;i<Head;++i)raw+=double(hidden[i])*weights[Out+i];
    return int(std::lround(1000.0*std::tanh(raw)));
}
}

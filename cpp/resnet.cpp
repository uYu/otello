#include "resnet.hpp"
#include "model/tiny_nn.h"
#include <cmath>
#include <cstring>
#include <istream>

namespace othello::nnue {
namespace {
std::uint32_t read32(std::istream& in){
    std::uint32_t value=0;
    for(int i=0;i<4;++i){int c=in.get();if(c==EOF)throw std::runtime_error("Truncated residual model");value|=std::uint32_t(c)<<(8*i);}
    return value;
}
template<int In> void conv3(const float* input,const float* weights,float* output,bool relu,bool reference){
    const float* bias=weights+ResNet::Channels*In*9;
    if(reference){
        for(int o=0;o<ResNet::Channels;++o)for(int y=0;y<8;++y)for(int x=0;x<8;++x){
            double sum=bias[o];
            for(int c=0;c<In;++c)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                int yy=y+dy,xx=x+dx;
                if(yy>=0&&yy<8&&xx>=0&&xx<8)sum+=double(input[c*64+yy*8+xx])*weights[o*In*9+c*9+(dy+1)*3+dx+1];
            }
            output[o*64+y*8+x]=float(relu?std::max(0.0,sum):sum);
        }
        return;
    }
    std::array<float,In*100> padded{};
    for(int c=0;c<In;++c)for(int y=0;y<8;++y)std::copy_n(input+c*64+y*8,8,padded.data()+c*100+(y+1)*10+1);
    std::array<float,In*9> patch{};
    for(int y=0;y<8;++y)for(int x=0;x<8;++x){
        for(int c=0;c<In;++c)for(int dy=0;dy<3;++dy)
            std::copy_n(padded.data()+c*100+(y+dy)*10+x,3,patch.data()+c*9+dy*3);
        for(int o=0;o<ResNet::Channels;++o){
            float sum=bias[o]+tiny_nn::dot(weights+o*In*9,patch.data(),In*9);
            output[o*64+y*8+x]=relu?std::max(0.0f,sum):sum;
        }
    }
}
int forward(const ResNet& net,Bits black,Bits white,int turn,bool reference){
    std::array<float,128> input{};Bits own=turn==1?black:white,opp=turn==1?white:black;
    while(own){int i=first(own);own&=own-1;input[i]=1;}
    while(opp){int i=first(opp);opp&=opp-1;input[64+i]=1;}
    std::array<float,ResNet::Channels*64> state{},middle{},last{};
    const float* w=net.weights.data();
    conv3<2>(input.data(),w,state.data(),true,reference);w+=32*2*9+32;
    for(int block=0;block<net.blocks;++block){
        conv3<32>(state.data(),w,middle.data(),true,reference);w+=32*32*9+32;
        conv3<32>(middle.data(),w,last.data(),false,reference);w+=32*32*9+32;
        for(int i=0;i<32*64;++i)state[i]=std::max(0.0f,state[i]+last[i]);
    }
    std::array<float,64> projected{},hidden{};
    for(int i=0;i<64;++i){
        double sum=w[32];for(int c=0;c<32;++c)sum+=double(state[c*64+i])*w[c];
        projected[i]=float(std::max(0.0,sum));
    }
    w+=33;
    if(reference){
        for(int o=0;o<64;++o){double sum=w[64*64+o];
            for(int i=0;i<64;++i)sum+=double(projected[i])*w[o*64+i];hidden[o]=float(std::max(0.0,sum));}
    }else tiny_nn::linear_relu(w,w+64*64,projected.data(),hidden.data(),64,64);
    w+=64*64+64;
    double raw=w[64];for(int i=0;i<64;++i)raw+=double(hidden[i])*w[i];
    return int(std::lround(1000.0*std::tanh(raw)));
}
}
std::shared_ptr<const ResNet> ResNet::load(std::istream& in){
    if(read32(in)!=8||read32(in)!=2||read32(in)!=Channels)throw std::runtime_error("Unsupported residual dimensions");
    int blocks=int(read32(in));
    if((blocks!=2&&blocks!=4)||read32(in)!=Head)throw std::runtime_error("Unsupported residual blocks/head");
    auto net=std::make_shared<ResNet>();net->blocks=blocks;net->weights.resize(parameters(blocks));
    for(auto& w:net->weights){auto bits=read32(in);std::memcpy(&w,&bits,sizeof(w));
        if(!std::isfinite(w)||std::abs(w)>1000)throw std::runtime_error("Invalid residual weight");}
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in residual model");
    return net;
}
int ResNet::evaluate(Bits black,Bits white,int turn) const{return forward(*this,black,white,turn,false);}
int ResNet::evaluateReference(Bits black,Bits white,int turn) const{return forward(*this,black,white,turn,true);}
}

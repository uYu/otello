#include "cnn.hpp"
#include "nnue.hpp"
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
std::shared_ptr<const CnnPolicy> CnnPolicy::load(std::istream& in){
    if(read32(in)!=8||read32(in)!=2)throw std::runtime_error("Unsupported policy CNN inputs");
    int c=int(read32(in));
    if((c!=8&&c!=16&&c!=32)||read32(in)!=64||read32(in)!=65)throw std::runtime_error("Unsupported policy CNN dimensions");
    auto result=std::make_shared<CnnPolicy>();result->channels=c;
    const int parameters=c*18+c+c*c*9+c+64*(c*64)+64+64+1+65*64+65;
    result->weights.resize(parameters);
    for(auto& w:result->weights){auto bits=read32(in);std::memcpy(&w,&bits,sizeof(w));
        if(!std::isfinite(w)||std::abs(w)>1000)throw std::runtime_error("Invalid policy CNN weight");}
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in policy CNN");
    return result;
}
Inference CnnPolicy::infer(Bits black,Bits white,int turn,Bits moves,bool terminal,bool reference) const{
    auto input=inputs(black,white,turn);
    std::array<float,32*64> a{},b{};
    std::array<float,64> hidden{};
    const int c=channels,c1b=c*18,c2=c1b+c,c2b=c2+c*c*9;
    const int fc=c2b+c,fcb=fc+64*c*64,out=fcb+64,outb=out+64,pol=outb+1,polb=pol+65*64;
    auto convolution=[&](const float* src,const float* w,const float* bias,float* dst,int n){
        std::array<float,32*100> padded{};
        for(int k=0;k<n;++k)for(int y=0;y<8;++y)
            std::copy_n(src+k*64+y*8,8,padded.data()+k*100+(y+1)*10+1);
        std::array<float,32*9> patch{};
        for(int y=0;y<8;++y)for(int x=0;x<8;++x){
            if(!reference)for(int k=0;k<n;++k)for(int dy=0;dy<3;++dy)
                std::copy_n(padded.data()+k*100+(y+dy)*10+x,3,patch.data()+k*9+dy*3);
            for(int o=0;o<c;++o){
                double value;
                if(reference){value=bias[o];
                    for(int k=0;k<n;++k)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                        int yy=y+dy,xx=x+dx;
                        if(yy>=0&&yy<8&&xx>=0&&xx<8)value+=double(src[k*64+yy*8+xx])*w[o*n*9+k*9+(dy+1)*3+dx+1];
                    }
                }else value=bias[o]+tiny_nn::dot(w+o*n*9,patch.data(),n*9);
                dst[o*64+y*8+x]=float(std::max(0.0,value));
            }
        }
    };
    const float* w=weights.data();
    convolution(input.data(),w,w+c1b,a.data(),2);
    convolution(a.data(),w+c2,w+c2b,b.data(),c);
    auto linear=[&](const float* row,const float* src,int n,float bias){
        if(!reference)return double(bias+tiny_nn::dot(row,src,n));
        double value=bias;for(int i=0;i<n;++i)value+=double(row[i])*src[i];return value;
    };
    for(int j=0;j<64;++j)hidden[j]=float(std::max(0.0,linear(w+fc+j*c*64,b.data(),c*64,w[fcb+j])));
    Inference result;result.score=int(std::lround(1000*std::tanh(linear(w+out,hidden.data(),64,w[outb]))));
    if(!moves){if(!terminal)result.probabilities[64]=1;return result;}
    double maximum=-1e300;Bits remaining=moves;
    while(remaining){int m=first(remaining);remaining&=remaining-1;
        result.probabilities[m]=linear(w+pol+m*64,hidden.data(),64,w[polb+m]);
        maximum=std::max(maximum,result.probabilities[m]);}
    double sum=0;remaining=moves;
    while(remaining){int m=first(remaining);remaining&=remaining-1;
        result.probabilities[m]=std::exp(result.probabilities[m]-maximum);sum+=result.probabilities[m];}
    while(moves){int m=first(moves);moves&=moves-1;result.probabilities[m]/=sum;}
    return result;
}
}

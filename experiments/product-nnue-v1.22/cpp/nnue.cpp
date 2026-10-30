#include "nnue.hpp"
#include "cnn.hpp"
#include "resnet.hpp"
#include <cmath>
#include <fstream>
#include <limits>
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
std::array<double,256> deeper(const TrunkBlock& block,const std::array<std::int32_t,MaxHidden>& acc){
    std::array<double,256> input{},result{};std::array<double,32> bottleneck{};
    for(int i=0;i<256;++i)input[i]=double(std::clamp(acc[i],0,Scale))/Scale;
    for(int j=0;j<32;++j){
        double total=double(block.downBias[j])/(Scale*Scale);
        for(int i=0;i<256;++i)total+=input[i]*block.down[j][i]/Scale;
        bottleneck[j]=std::clamp(total,0.0,1.0);
    }
    for(int j=0;j<256;++j){
        double total=input[j]+double(block.upBias[j])/(Scale*Scale);
        for(int i=0;i<32;++i)total+=bottleneck[i]*block.up[j][i]/Scale;
        result[j]=std::clamp(total,0.0,1.0);
    }
    return result;
}
Inference productInfer(const ProductHead& head,const Accumulator& acc,int turn,Bits moves,bool terminal){
    std::array<int,256> joined{};
    const auto& own=turn==1?acc.black:acc.white;const auto& opp=turn==1?acc.white:acc.black;
    for(int i=0;i<128;++i){
        joined[i]=(std::clamp(own[i],0,256)*std::clamp(own[i+128],0,256)+128)/256;
        joined[i+128]=(std::clamp(opp[i],0,256)*std::clamp(opp[i+128],0,256)+128)/256;
    }
    Inference result;std::int64_t total=head.valueBias;
    for(int i=0;i<256;++i)total+=std::int64_t(joined[i])*head.value[i];
    result.score=int(std::lround(1000*std::tanh(double(total)/65536)));
    if(!moves){if(!terminal)result.probabilities[64]=1;return result;}
    std::array<double,32> hidden{};
    for(int j=0;j<32;++j){std::int64_t sum=head.bias[j];for(int i=0;i<256;++i)sum+=std::int64_t(joined[i])*head.weights[j][i];hidden[j]=std::clamp(double(sum)/65536,0.0,1.0);}
    double maximum=-1e300;Bits remaining=moves;
    while(remaining){int m=first(remaining);remaining&=remaining-1;double logit=double(head.outputBias[m])/65536;
        for(int j=0;j<32;++j)logit+=hidden[j]*head.output[m][j]/256;result.probabilities[m]=logit;maximum=std::max(maximum,logit);}
    double sum=0;remaining=moves;
    while(remaining){int m=first(remaining);remaining&=remaining-1;result.probabilities[m]=std::exp(result.probabilities[m]-maximum);sum+=result.probabilities[m];}
    while(moves){int m=first(moves);moves&=moves-1;result.probabilities[m]/=sum;}return result;
}

}
Network Network::load(const std::string& path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open Othello NNUE model: "+path);
    return load(in);
}
Network Network::load(std::istream& in){
    char magic[8]{};in.read(magic,8);
    const auto format=std::string(magic,8);
    if(in&&(format==std::string("OTCNN2\0\0",8)||format==std::string("OTCNN3\0\0",8))){
        Network net;net.architecture=Architecture::CnnPolicy;net.valueTarget=ValueTarget::Outcome;
        net.hidden=0;net.cnnPolicy=CnnPolicy::load(in,format==std::string("OTCNN3\0\0",8));return net;
    }
    if(in&&format==std::string("OTRES1\0\0",8)){
        Network net;net.architecture=Architecture::Residual;net.hidden=0;net.residual=ResNet::load(in);return net;
    }
    if(in&&format==std::string("OTCNN1\0\0",8)){
        Network net;net.architecture=Architecture::Cnn;net.hidden=0;net.cnn=Cnn::load(in);return net;
    }
    const bool product=format==std::string("OTNNUE6\0",8);
    const bool deep=format==std::string("OTNNUE5\0",8);
    const bool outcome=format==std::string("OTNNUE4\0",8)||deep||product;
    const bool withPolicy=format==std::string("OTNNUE3\0",8)||outcome;
    if(!in||(format!=std::string("OTNNUE1\0",8)&&format!=std::string("OTNNUE2\0",8)&&!withPolicy))throw std::runtime_error("Not an Othello NNUE model");
    Network net;
    if(outcome)net.valueTarget=ValueTarget::Outcome;
    net.architecture=(format==std::string("OTNNUE2\0",8)||withPolicy)?Architecture::Turn:Architecture::Legacy;
    if(read(in,4)!=Features)throw std::runtime_error("Unsupported Othello NNUE features");
    net.hidden=int(read(in,4));
    if((net.hidden!=32&&net.hidden!=64&&net.hidden!=128&&net.hidden!=256)||read(in,4)!=Scale||read(in,4)!=Scale)throw std::runtime_error("Unsupported Othello NNUE dimensions/scales");
    for(auto& row:net.weights)for(int j=0;j<net.hidden;++j){auto u=read(in,2);row[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
    for(int j=0;j<net.hidden;++j){auto u=read(in,4);std::int64_t v=u>=0x80000000U?std::int64_t(u)-0x100000000LL:u;
        if(v < -10000000||v > 10000000)throw std::runtime_error("NNUE bias outside safe accumulator range");net.bias[j]=std::int32_t(v);}
    if(product){
        if(net.hidden!=256||read(in,4)!=256||read(in,4)!=32||read(in,4)!=65)throw std::runtime_error("Unsupported product NNUE dimensions");
        auto head=std::make_shared<ProductHead>();
        auto i16=[&](){auto u=read(in,2);return std::int16_t(u>=32768?int(u)-65536:int(u));};
        auto i32=[&](){auto u=read(in,4);return std::int32_t(u>=0x80000000U?std::int64_t(u)-0x100000000LL:u);};
        for(auto& v:head->value)v=i16();head->valueBias=i32();
        for(auto& row:head->weights)for(auto& v:row)v=i16();for(auto& v:head->bias)v=i32();
        for(auto& row:head->output)for(auto& v:row)v=i16();for(auto& v:head->outputBias)v=i32();
        if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in product NNUE");net.productHead=head;return net;
    }
    for(int j=0;j<net.hidden;++j){auto u=read(in,2);net.output[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
    if(net.architecture==Architecture::Turn){
        for(int j=0;j<net.hidden;++j){auto u=read(in,2);net.opponentOutput[j]=std::int16_t(u>=32768?int(u)-65536:int(u));}
        auto u=read(in,4);net.outputBias=std::int32_t(u>=0x80000000U?std::int64_t(u)-0x100000000LL:u);
    }
    if(withPolicy){
        if(net.hidden!=256||read(in,4)!=32||read(in,4)!=65)throw std::runtime_error("Unsupported NNUE policy dimensions");
        auto policy=std::make_shared<PolicyHead>();
        auto i16=[&](){auto u=read(in,2);return std::int16_t(u>=32768?int(u)-65536:int(u));};
        auto i32=[&](){auto u=read(in,4);return std::int32_t(u>=0x80000000U?std::int64_t(u)-0x100000000LL:u);};
        for(auto& row:policy->weights)for(auto& v:row)v=i16();
        for(auto& v:policy->bias)v=i32();
        for(auto& row:policy->output)for(auto& v:row)v=i16();
        for(auto& v:policy->outputBias)v=i32();
        net.policyHead=policy;
    }
    if(deep){
        if(read(in,4)!=256||read(in,4)!=32)throw std::runtime_error("Unsupported NNUE trunk dimensions");
        auto block=std::make_shared<TrunkBlock>();
        auto i16=[&](){auto u=read(in,2);return std::int16_t(u>=32768?int(u)-65536:int(u));};
        auto i32=[&](){auto u=read(in,4);return std::int32_t(u>=0x80000000U?std::int64_t(u)-0x100000000LL:u);};
        for(auto& row:block->down)for(auto& v:row)v=i16();
        for(auto& v:block->downBias)v=i32();
        for(auto& row:block->up)for(auto& v:row)v=i16();
        for(auto& v:block->upBias)v=i32();
        net.trunkBlock=block;
    }
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in Othello NNUE model");
    return net;
}
Accumulator Network::refresh(const Position& p) const{
    if(spatial()){Accumulator acc;acc.blackBoard=p.black;acc.whiteBoard=p.white;return acc;}
    Accumulator acc{bias,bias};Bits b=p.black,w=p.white;
    while(b){int i=first(b);b&=b-1;change(acc,*this,i,true,1);}
    while(w){int i=first(w);w&=w-1;change(acc,*this,i,false,1);}return acc;
}
Accumulator Network::update(const Accumulator& parent,const Position& before,const Position& after) const{
    Accumulator acc=parent;
    if(spatial()){acc.blackBoard=after.black;acc.whiteBoard=after.white;return acc;}
    Bits changed=(before.black^after.black)|(before.white^after.white);
    while(changed){int i=first(changed);Bits bit=1ULL<<i;changed&=changed-1;
        if(before.black&bit)change(acc,*this,i,true,-1);else if(before.white&bit)change(acc,*this,i,false,-1);
        if(after.black&bit)change(acc,*this,i,true,1);else if(after.white&bit)change(acc,*this,i,false,1);
    }return acc;
}
int Network::evaluate(const Accumulator& acc,int turn) const{
    if(productHead)return productInfer(*productHead,acc,turn,0,true).score;
    if(cnnPolicy)return cnnPolicy->infer(acc.blackBoard,acc.whiteBoard,turn,0,true).score;
    if(architecture==Architecture::Residual)return residual->evaluate(acc.blackBoard,acc.whiteBoard,turn);
    if(architecture==Architecture::Cnn)return cnn->evaluate(acc.blackBoard,acc.whiteBoard,turn);
    if(architecture==Architecture::Turn){
        const auto& own=turn==1?acc.black:acc.white;
        const auto& opp=turn==1?acc.white:acc.black;
        if(trunkBlock){
            const auto a=deeper(*trunkBlock,own),b=deeper(*trunkBlock,opp);
            double total=double(outputBias)/(Scale*Scale);
            for(int j=0;j<hidden;++j)total+=a[j]*output[j]/Scale+b[j]*opponentOutput[j]/Scale;
            return int(std::lround(1000.0*std::tanh(total)));
        }
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
    if(cnnPolicy)return cnnPolicy->infer(p.black,p.white,p.turn,0,true,true).score;
    if(trunkBlock||productHead)return evaluate(refresh(p),p.turn);
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
    return architecture==other.architecture&&valueTarget==other.valueTarget&&hidden==other.hidden&&weights==other.weights&&bias==other.bias
        &&output==other.output&&opponentOutput==other.opponentOutput&&outputBias==other.outputBias
        &&(productHead==other.productHead||(productHead&&other.productHead&&*productHead==*other.productHead))
        &&(trunkBlock==other.trunkBlock||(trunkBlock&&other.trunkBlock&&*trunkBlock==*other.trunkBlock))
        &&(policyHead==other.policyHead||(policyHead&&other.policyHead&&policyHead->weights==other.policyHead->weights
           &&policyHead->bias==other.policyHead->bias&&policyHead->output==other.policyHead->output&&policyHead->outputBias==other.policyHead->outputBias))
        &&(cnn==other.cnn||(cnn&&other.cnn&&cnn->weights==other.cnn->weights))
        &&(cnnPolicy==other.cnnPolicy||(cnnPolicy&&other.cnnPolicy&&*cnnPolicy==*other.cnnPolicy))
        &&(residual==other.residual||(residual&&other.residual&&residual->blocks==other.residual->blocks&&residual->weights==other.residual->weights));
}
std::unique_ptr<Network> Network::snapshot() const{
    auto copy=std::make_unique<Network>(*this);if(cnn)copy->cnn=std::make_shared<Cnn>(*cnn);
    if(cnnPolicy)copy->cnnPolicy=std::make_shared<CnnPolicy>(*cnnPolicy);
    if(productHead)copy->productHead=std::make_shared<ProductHead>(*productHead);
    if(policyHead)copy->policyHead=std::make_shared<PolicyHead>(*policyHead);
    if(trunkBlock)copy->trunkBlock=std::make_shared<TrunkBlock>(*trunkBlock);
    if(residual)copy->residual=std::make_shared<ResNet>(*residual);return copy;
}
std::array<double,65> Network::policy(const Position& p) const{
    std::array<double,65> result{};Bits moves=legal(p);
    if(!moves){if(legal(p.opponent(),p.own()))result[64]=1;return result;}
    if(cnnPolicy)return cnnPolicy->infer(p.black,p.white,p.turn,moves,false).probabilities;
    if(productHead)return productInfer(*productHead,refresh(p),p.turn,moves,false).probabilities;
    return policyFromFeatures(refresh(p),p.turn,moves);
}
std::array<double,65> Network::policyFromFeatures(const Accumulator& acc,int turn,Bits moves,
        const std::array<double,256>* sharedOwn,const std::array<double,256>* sharedOpp) const{
    std::array<double,65> result{};
    if(!policyHead){double prior=1.0/count(moves);while(moves){int m=first(moves);moves&=moves-1;result[m]=prior;}return result;}
    const auto& own=turn==1?acc.black:acc.white;const auto& opp=turn==1?acc.white:acc.black;
    std::array<double,256> localOwn{},localOpp{};
    if(trunkBlock&&!sharedOwn){localOwn=deeper(*trunkBlock,own);localOpp=deeper(*trunkBlock,opp);sharedOwn=&localOwn;sharedOpp=&localOpp;}
    std::array<double,32> hidden{};
    for(int j=0;j<32;++j){
        if(trunkBlock){
            double total=double(policyHead->bias[j])/(Scale*Scale);
            for(int i=0;i<256;++i)total+=(*sharedOwn)[i]*policyHead->weights[j][i]/Scale
                +(*sharedOpp)[i]*policyHead->weights[j][256+i]/Scale;
            hidden[j]=std::clamp(total,0.0,1.0);continue;
        }
        std::int64_t total=policyHead->bias[j];
        for(int i=0;i<256;++i)total+=std::int64_t(std::clamp(own[i],0,Scale))*policyHead->weights[j][i]
            +std::int64_t(std::clamp(opp[i],0,Scale))*policyHead->weights[j][256+i];
        hidden[j]=std::clamp(double(total)/(Scale*Scale),0.0,1.0);
    }
    double maximum=-std::numeric_limits<double>::infinity();Bits remaining=moves;
    while(remaining){int m=first(remaining);remaining&=remaining-1;double logit=double(policyHead->outputBias[m])/(Scale*Scale);
        for(int j=0;j<32;++j)logit+=hidden[j]*policyHead->output[m][j]/Scale;
        result[m]=logit;maximum=std::max(maximum,logit);}
    double sum=0;remaining=moves;
    while(remaining){int m=first(remaining);remaining&=remaining-1;result[m]=std::exp(result[m]-maximum);sum+=result[m];}
    while(moves){int m=first(moves);moves&=moves-1;result[m]/=sum;}return result;
}
Inference Network::infer(const Position& p) const{
    Bits moves=legal(p);return infer(p,moves,!moves&&!legal(p.opponent(),p.own()));
}
Inference Network::infer(const Position& p,Bits moves,bool terminal) const{
    if(cnnPolicy)return cnnPolicy->infer(p.black,p.white,p.turn,moves,terminal);
    Inference result;auto acc=refresh(p);
    if(productHead)return productInfer(*productHead,acc,p.turn,moves,terminal);
    std::array<double,256> own{},opp{};
    if(trunkBlock){
        own=deeper(*trunkBlock,p.turn==1?acc.black:acc.white);
        opp=deeper(*trunkBlock,p.turn==1?acc.white:acc.black);
        double total=double(outputBias)/(Scale*Scale);
        for(int j=0;j<hidden;++j)total+=own[j]*output[j]/Scale+opp[j]*opponentOutput[j]/Scale;
        result.score=int(std::lround(1000.0*std::tanh(total)));
    }else result.score=evaluate(acc,p.turn);
    if(moves)result.probabilities=policyFromFeatures(acc,p.turn,moves,trunkBlock?&own:nullptr,trunkBlock?&opp:nullptr);
    else if(!terminal)result.probabilities[64]=1;
    return result;
}
}

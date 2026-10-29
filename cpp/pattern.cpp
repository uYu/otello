#include "pattern.hpp"
#include <cmath>
#include <istream>

namespace othello::nnue {
namespace {
std::uint32_t read(std::istream& in,int bytes){
    std::uint32_t x=0;for(int i=0;i<bytes;++i){int c=in.get();if(c==EOF)throw std::runtime_error("Truncated pattern model");x|=std::uint32_t(c)<<(8*i);}return x;
}
int rounded(double x){return int(std::round(x));}
constexpr int groupCounts[3]={4,30,4};
int bucket(int square){int x=std::min(square%8,7-square%8),y=std::min(square/8,7-square/8);if(x>y)std::swap(x,y);return x*4-x*(x-1)/2+y-x;}
}
const std::array<Pattern::Shape,Pattern::Count>& Pattern::shapes(){
    static const auto result=[](){
        std::array<Shape,Count> list{};int n=0;
        for(int axis=0;axis<2;++axis)for(int i=0;i<8;++i){
            auto& s=list[n++];s.length=8;int distance=std::min(i,7-i);s.offset=distance*6561;s.group=distance?1:0;
            for(int j=0;j<8;++j)s.squares[j]=axis?j*8+i:i*8+j;
        }
        for(int direction=0;direction<2;++direction)for(int delta=-4;delta<=4;++delta){
            auto& s=list[n++];s.group=1;s.offset=26244;
            for(int length=4;length<8-std::abs(delta);++length){int size=1;for(int k=0;k<length;++k)size*=3;s.offset+=size;}
            for(int y=0;y<8;++y){int x=y+delta;if(x>=0&&x<8)s.squares[s.length++]=y*8+(direction?7-x:x);}
        }
        for(int corner=0;corner<4;++corner){auto& s=list[n++];s.offset=36045;s.group=2;
            for(int y=0;y<3;++y)for(int x=0;x<3;++x)s.squares[s.length++]=(corner&2?7-y:y)*8+(corner&1?7-x:x);
        }
        return list;
    }();return result;
}
int Pattern::code(const Shape& shape,Bits own,Bits opponent){int value=shape.offset,power=1;for(int j=0;j<shape.length;++j){Bits bit=Bits(1)<<shape.squares[j];value+=power*((own&bit)?1:(opponent&bit)?2:0);power*=3;}return value;}
std::shared_ptr<Pattern> Pattern::load(std::istream& in){
    if(read(in,4)!=Rows||read(in,4)!=Dim||read(in,4)!=Hidden||read(in,4)!=Inputs||read(in,4)!=Outputs||read(in,4)!=256)
        throw std::runtime_error("Unsupported pattern model dimensions/scales");
    auto model=std::make_shared<Pattern>();model->embedding.resize(Rows*Dim);
    auto i16=[&](auto& values){for(auto& x:values){auto u=read(in,2);x=std::int16_t(u>=32768?int(u)-65536:int(u));}};
    auto i32=[&](auto& values){for(auto& x:values){auto u=read(in,4);auto v=u>=0x80000000U?std::int64_t(u)-0x100000000LL:std::int64_t(u);
        if(std::abs(v)>10000000)throw std::runtime_error("Pattern bias outside safe range");x=std::int32_t(v);}};
    i16(model->embedding);i16(model->trunk);i32(model->trunkBias);i16(model->head);i32(model->headBias);
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in pattern model");return model;
}
Accumulator Pattern::refresh(const Position& p) const{
    Accumulator a;a.blackBoard=p.black;a.whiteBoard=p.white;
    for(const auto& s:shapes()){int b=code(s,p.black,p.white),w=code(s,p.white,p.black);
        for(int j=0;j<Dim;++j){a.black[s.group*Dim+j]+=embedding[b*Dim+j];a.white[s.group*Dim+j]+=embedding[w*Dim+j];}}
    return a;
}
Accumulator Pattern::update(const Accumulator& parent,const Position& before,const Position& after) const{
    auto a=parent;a.blackBoard=after.black;a.whiteBoard=after.white;Bits changes=(before.black^after.black)|(before.white^after.white);
    for(const auto& s:shapes()){
        bool affected=false;for(int k=0;k<s.length;++k)affected|=bool(changes&(Bits(1)<<s.squares[k]));if(!affected)continue;
        int oldB=code(s,before.black,before.white),oldW=code(s,before.white,before.black),newB=code(s,after.black,after.white),newW=code(s,after.white,after.black);
        for(int j=0;j<Dim;++j){a.black[s.group*Dim+j]+=int(embedding[newB*Dim+j])-embedding[oldB*Dim+j];a.white[s.group*Dim+j]+=int(embedding[newW*Dim+j])-embedding[oldW*Dim+j];}
    }return a;
}
std::array<int,Pattern::Outputs> Pattern::outputs(const Accumulator& a,int turn,int outputCount) const{
    if(outputCount<1||outputCount>Outputs)throw std::invalid_argument("Invalid pattern output count");
    std::array<int,Inputs> x{};const auto& sums=turn==1?a.black:a.white;
    for(int i=0;i<96;++i)x[i]=std::clamp(rounded(double(sums[i])/groupCounts[i/Dim]),-1024,1024);
    x[96]=rounded(double(64-count(a.blackBoard|a.whiteBoard))*256/60);
    std::array<int,Hidden> hidden{};
    for(int i=0;i<Hidden;++i){std::int64_t sum=trunkBias[i];for(int j=0;j<Inputs;++j)sum+=std::int64_t(trunk[i*Inputs+j])*x[j];hidden[i]=std::clamp(rounded(double(sum)/256),0,256);}
    std::array<int,Outputs> result{};
    for(int i=0;i<outputCount;++i){std::int64_t sum=headBias[i];for(int j=0;j<Hidden;++j)sum+=std::int64_t(head[i*Hidden+j])*hidden[j];result[i]=std::clamp(rounded(double(sum)/256),-1024,1024);}
    return result;
}
int Pattern::evaluate(const Accumulator& acc,int turn) const{return rounded(1000*std::tanh(double(outputs(acc,turn,1)[0])/256));}
std::array<double,64> Pattern::policy(const Position& p) const{
    return policy(p,outputs(refresh(p),p.turn));
}
std::array<double,64> Pattern::policy(const Position& p,const std::array<int,Outputs>& out) const{
    std::array<double,64> logits{};std::array<int,64> counts{};
    for(const auto& s:shapes()){
        int index=code(s,p.own(),p.opponent());std::int64_t sum=0;
        for(int j=0;j<Dim;++j)sum+=std::int64_t(embedding[index*Dim+j])*out[4+j];
        double projection=double(sum)/(256*256*std::sqrt(double(Dim)));
        for(int j=0;j<s.length;++j){int square=s.squares[j];logits[square]+=projection;++counts[square];}
    }
    for(int square=0;square<64;++square)logits[square]=logits[square]/counts[square]+double(out[4+Dim+bucket(square)])/256;
    return logits;
}
bool Pattern::operator==(const Pattern& b) const{return embedding==b.embedding&&trunk==b.trunk&&trunkBias==b.trunkBias&&head==b.head&&headBias==b.headBias;}
}

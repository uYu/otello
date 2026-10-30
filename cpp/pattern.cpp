#include "pattern.hpp"
#include "pattern_kernels.hpp"
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
struct PolicyGeometry {
    std::array<Bits,Pattern::Count> masks{};
    std::array<int,64> counts{},buckets{};
};
const PolicyGeometry& policyGeometry(){
    static const auto geometry=[](){PolicyGeometry g;const auto& shapes=Pattern::shapes();
        for(int i=0;i<Pattern::Count;++i)for(int j=0;j<shapes[i].length;++j){int square=shapes[i].squares[j];g.masks[i]|=Bits(1)<<square;++g.counts[square];}
        for(int square=0;square<64;++square)g.buckets[square]=bucket(square);return g;}();
    return geometry;
}
// Rows are packed in original shape order. Keep each square's FP addition order.
void maskedPolicy(Pattern::Prediction& result,const std::int16_t* entries,const Bits* destinations,int active,Bits mask){
    std::array<std::int16_t,Pattern::Dim> gates{};
    for(int j=0;j<Pattern::Dim;++j)gates[j]=std::int16_t(result.values[4+j]);
    std::array<std::int64_t,Pattern::Count> projections;int i=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;i+4<=active;i+=4)pattern_kernels::dot4(entries+i*Pattern::Dim,gates.data(),Pattern::Dim,Pattern::Dim,projections.data()+i);
#endif
    for(;i<active;++i)projections[i]=pattern_kernels::dot(entries+i*Pattern::Dim,gates.data(),Pattern::Dim);
    for(i=0;i<active;++i){double projection=double(projections[i])/(256*256*std::sqrt(double(Pattern::Dim)));Bits targets=destinations[i];
        while(targets){int square=first(targets);targets&=targets-1;result.logits[square]+=projection;}}
    const auto& geometry=policyGeometry();
    while(mask){int square=first(mask);mask&=mask-1;
        result.logits[square]=result.logits[square]/geometry.counts[square]+double(result.values[4+Pattern::Dim+geometry.buckets[square]])/256;}
}
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
std::array<int,Pattern::Count> Pattern::encode(Bits own,Bits opponent){
    // Convert binary occupancy masks to ternary weights without per-cell branches.
    static const auto ternary=[](){std::array<int,512> table{};
        for(int i=0;i<512;++i){int power=1;for(int j=0;j<9;++j){table[i]+=((i>>j)&1)*power;power*=3;}}return table;}();
    auto mirror=[](Bits x){x=((x>>1)&0x5555555555555555ULL)|((x&0x5555555555555555ULL)<<1);
        x=((x>>2)&0x3333333333333333ULL)|((x&0x3333333333333333ULL)<<2);
        return ((x>>4)&0x0f0f0f0f0f0f0f0fULL)|((x&0x0f0f0f0f0f0f0f0fULL)<<4);};
    opponent&=~own; // Preserve code()'s own-first semantics even for overlapping masks.
    Bits boards[2][2]={{own,opponent},{mirror(own),mirror(opponent)}};
    std::array<int,Count> result{};const auto& list=shapes();
    for(int row=0;row<8;++row)result[row]=list[row].offset+ternary[(own>>(8*row))&255]+2*ternary[(opponent>>(8*row))&255];
    for(int column=0;column<8;++column){
        auto pack=[&](Bits board){return (((board>>column)&0x0101010101010101ULL)*0x0102040810204080ULL)>>56;};
        result[8+column]=list[8+column].offset+ternary[pack(own)]+2*ternary[pack(opponent)];
    }
    struct Diagonal {Bits mask;int shift;};
    static const auto diagonals=[](){std::array<Diagonal,18> plans{};const auto& shapes=Pattern::shapes();
        for(int i=0;i<18;++i){const auto& shape=shapes[16+i];auto& plan=plans[i];plan.shift=shape.squares[0]^(i>=9?7:0);
            for(int j=0;j<shape.length;++j)plan.mask|=Bits(1)<<(9*j);}return plans;}();
    for(int i=0;i<18;++i){const auto& plan=diagonals[i];auto pack=[&](Bits board){return (((board>>plan.shift)&plan.mask)*0x0101010101010101ULL)>>56;};
        result[16+i]=list[16+i].offset+ternary[pack(boards[i/9][0])]+2*ternary[pack(boards[i/9][1])];}
    for(int corner=0;corner<4;++corner){
        auto pack=[&](Bits board){if(corner&2)board=__builtin_bswap64(board);return (board&7)|((board>>5)&56)|((board>>10)&448);};
        result[34+corner]=list[34+corner].offset+ternary[pack(boards[corner&1][0])]+2*ternary[pack(boards[corner&1][1])];
    }
    return result;
}
std::shared_ptr<Pattern> Pattern::load(std::istream& in,bool int8){
    const auto rows=read(in,4),dim=read(in,4),width=read(in,4);
    if(rows!=Rows||dim!=Dim||(width!=32&&width!=64)||read(in,4)!=Inputs||read(in,4)!=Outputs||read(in,4)!=256)
        throw std::runtime_error("Unsupported pattern model dimensions/scales");
    auto model=std::make_shared<Pattern>();model->hidden=int(width);
    if(int8){
        model->embeddingStep=int(read(in,4));
        if(model->embeddingStep!=1&&model->embeddingStep!=2&&model->embeddingStep!=4&&model->embeddingStep!=8)
            throw std::runtime_error("Unsupported INT8 pattern step");
        model->embedding8.resize(Rows*Dim);
        for(auto& v:model->embedding8){auto u=read(in,1);v=std::int8_t(u>=128?int(u)-256:int(u));}
    }else model->embedding.resize(Rows*Dim);
    auto i16=[&](auto& values,int n){for(int i=0;i<n;++i){auto& x=values[i];auto u=read(in,2);x=std::int16_t(u>=32768?int(u)-65536:int(u));}};
    auto i32=[&](auto& values,int n){for(int i=0;i<n;++i){auto& x=values[i];auto u=read(in,4);auto v=u>=0x80000000U?std::int64_t(u)-0x100000000LL:std::int64_t(u);
        if(std::abs(v)>10000000)throw std::runtime_error("Pattern bias outside safe range");x=std::int32_t(v);}};
    if(!int8)i16(model->embedding,Rows*Dim);
    i16(model->trunk,model->hidden*Inputs);i32(model->trunkBias,model->hidden);i16(model->head,Outputs*model->hidden);i32(model->headBias,Outputs);
    if(in.get()!=EOF)throw std::runtime_error("Trailing bytes in pattern model");return model;
}
Accumulator Pattern::refresh(const Position& p) const{
    Accumulator a;a.blackBoard=p.black;a.whiteBoard=p.white;
    for(const auto& s:shapes()){int b=code(s,p.black,p.white),w=code(s,p.white,p.black);
        for(int j=0;j<Dim;++j){a.black[s.group*Dim+j]+=embeddingValue(b*Dim+j);a.white[s.group*Dim+j]+=embeddingValue(w*Dim+j);}}
    return a;
}
Accumulator Pattern::update(const Accumulator& parent,const Position& before,const Position& after) const{
    auto a=parent;a.blackBoard=after.black;a.whiteBoard=after.white;Bits changes=(before.black^after.black)|(before.white^after.white);
    for(const auto& s:shapes()){
        bool affected=false;for(int k=0;k<s.length;++k)affected|=bool(changes&(Bits(1)<<s.squares[k]));if(!affected)continue;
        int oldB=code(s,before.black,before.white),oldW=code(s,before.white,before.black),newB=code(s,after.black,after.white),newW=code(s,after.white,after.black);
        for(int j=0;j<Dim;++j){a.black[s.group*Dim+j]+=embeddingValue(newB*Dim+j)-embeddingValue(oldB*Dim+j);a.white[s.group*Dim+j]+=embeddingValue(newW*Dim+j)-embeddingValue(oldW*Dim+j);}
    }return a;
}
std::array<int,Pattern::Outputs> Pattern::outputs(const Accumulator& a,int turn,int outputCount) const{
    return outputs((turn==1?a.black:a.white).data(),a.blackBoard|a.whiteBoard,outputCount);
}
std::array<int,Pattern::Outputs> Pattern::outputs(const std::int32_t* sums,Bits occupied,int outputCount) const{
    if(outputCount<1||outputCount>Outputs)throw std::invalid_argument("Invalid pattern output count");
    alignas(64) std::array<std::int16_t,Inputs> x{};
    for(int group=0;group<3;++group)for(int j=0;j<Dim;++j){int i=group*Dim+j;
#if defined(OTELLO_PATTERN_INTEGER_ROUNDING)
        auto value=group==1?pattern_kernels::roundedDivide<30>(sums[i]):pattern_kernels::roundedDivide<4>(sums[i]);
        x[i]=std::int16_t(std::clamp<std::int64_t>(value,-1024,1024));
#else
        x[i]=std::clamp(rounded(double(sums[i])/groupCounts[group]),-1024,1024);
#endif
    }
#if defined(OTELLO_PATTERN_INTEGER_ROUNDING)
    x[96]=std::int16_t(pattern_kernels::roundedDivide<60>((64-count(occupied))*256));
#else
    x[96]=rounded(double(64-count(occupied))*256/60);
#endif
    auto scaled=[](std::int64_t sum,int low,int high){
#if defined(OTELLO_PATTERN_INTEGER_ROUNDING)
        return int(std::clamp<std::int64_t>(pattern_kernels::roundedDivide<256>(sum),low,high));
#else
        return std::clamp(rounded(double(sum)/256),low,high);
#endif
    };
    alignas(64) std::array<std::int16_t,MaxHidden> hiddenValues{};
    int row=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;row+4<=hidden;row+=4){std::int64_t totals[4];pattern_kernels::dot4(trunk.data()+row*Inputs,x.data(),Inputs,Inputs,totals);
        for(int k=0;k<4;++k)hiddenValues[row+k]=std::int16_t(scaled(trunkBias[row+k]+totals[k],0,256));}
#endif
    for(;row<hidden;++row)hiddenValues[row]=std::int16_t(scaled(trunkBias[row]+pattern_kernels::dot(trunk.data()+row*Inputs,x.data(),Inputs),0,256));
    std::array<int,Outputs> result{};row=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;row+4<=outputCount;row+=4){std::int64_t totals[4];pattern_kernels::dot4(head.data()+row*hidden,hiddenValues.data(),hidden,hidden,totals);
        for(int k=0;k<4;++k)result[row+k]=scaled(headBias[row+k]+totals[k],-1024,1024);}
#endif
    for(;row<outputCount;++row)result[row]=scaled(headBias[row]+pattern_kernels::dot(head.data()+row*hidden,hiddenValues.data(),hidden),-1024,1024);
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
        for(int j=0;j<Dim;++j)sum+=std::int64_t(embeddingValue(index*Dim+j))*out[4+j];
        double projection=double(sum)/(256*256*std::sqrt(double(Dim)));
        for(int j=0;j<s.length;++j){int square=s.squares[j];logits[square]+=projection;++counts[square];}
    }
    for(int square=0;square<64;++square)logits[square]=logits[square]/counts[square]+double(out[4+Dim+bucket(square)])/256;
    return logits;
}
Pattern::Prediction Pattern::predict(const Position& p) const{
    Accumulator acc;acc.blackBoard=p.black;acc.whiteBoard=p.white;
    auto& sums=p.turn==1?acc.black:acc.white;
    alignas(64) std::array<std::int16_t,Count*Dim> entries{};
    const auto& list=shapes();
#if defined(OTELLO_PATTERN_FAST_CODES)
    const auto indices=encode(p.own(),p.opponent());
#endif
    for(int i=0;i<Count;++i){
        const auto& shape=list[i];
#if defined(OTELLO_PATTERN_FAST_CODES)
        int offset=indices[i]*Dim;
#else
        int offset=code(shape,p.own(),p.opponent())*Dim;
#endif
        if(embedding8.empty())pattern_kernels::gather(embedding.data()+offset,sums.data()+shape.group*Dim,entries.data()+i*Dim);
        else pattern_kernels::gather(embedding8.data()+offset,embeddingStep,sums.data()+shape.group*Dim,entries.data()+i*Dim);
    }
    Prediction result;result.values=outputs(acc,p.turn);
    std::array<std::int16_t,Dim> gates{};for(int j=0;j<Dim;++j)gates[j]=std::int16_t(result.values[4+j]);
    static const auto counts=[](){std::array<int,64> n{};for(const auto& s:shapes())for(int j=0;j<s.length;++j)++n[s.squares[j]];return n;}();
    static const auto buckets=[](){std::array<int,64> b{};for(int i=0;i<64;++i)b[i]=bucket(i);return b;}();
    std::array<std::int64_t,Count> projections{};int shapeIndex=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;shapeIndex+4<=Count;shapeIndex+=4)pattern_kernels::dot4(entries.data()+shapeIndex*Dim,gates.data(),Dim,Dim,projections.data()+shapeIndex);
#endif
    for(;shapeIndex<Count;++shapeIndex)projections[shapeIndex]=pattern_kernels::dot(entries.data()+shapeIndex*Dim,gates.data(),Dim);
    for(int i=0;i<Count;++i){
        std::int64_t total=projections[i];
        double projection=double(total)/(256*256*std::sqrt(double(Dim)));
        const auto& shape=list[i];for(int j=0;j<shape.length;++j)result.logits[shape.squares[j]]+=projection;
    }
    for(int i=0;i<64;++i)result.logits[i]=result.logits[i]/counts[i]+double(result.values[4+Dim+buckets[i]])/256;
    return result;
}
Pattern::Prediction Pattern::predictLegal(const Position& p,Bits mask) const{
    Accumulator acc;acc.blackBoard=p.black;acc.whiteBoard=p.white;auto& sums=p.turn==1?acc.black:acc.white;
    alignas(64) std::array<std::int16_t,Count*Dim> entries;
    std::array<Bits,Count> destinations;int active=0;
    const auto& list=shapes();const auto& geometry=policyGeometry();
#if defined(OTELLO_PATTERN_FAST_CODES)
    const auto indices=encode(p.own(),p.opponent());
#endif
    for(int i=0;i<Count;++i){const auto& shape=list[i];
#if defined(OTELLO_PATTERN_FAST_CODES)
        int offset=indices[i]*Dim;
#else
        int offset=code(shape,p.own(),p.opponent())*Dim;
#endif
        Bits targets=geometry.masks[i]&mask;std::int16_t* saved=nullptr;
        if(targets){saved=entries.data()+active*Dim;destinations[active++]=targets;}
        if(embedding8.empty())pattern_kernels::gather(embedding.data()+offset,sums.data()+shape.group*Dim,saved);
        else pattern_kernels::gather(embedding8.data()+offset,embeddingStep,sums.data()+shape.group*Dim,saved);
    }
    Prediction result;result.values=outputs(acc,p.turn);
    maskedPolicy(result,entries.data(),destinations.data(),active,mask);return result;
}
Pattern::Prediction Pattern::predictLegal(const State& state,int turn,Bits mask) const{
    Prediction result;result.values=outputs(state.sums[turn==1?0:1].data(),state.blackBoard|state.whiteBoard);
    alignas(64) std::array<std::int16_t,Count*Dim> entries;
    std::array<Bits,Count> destinations;int active=0;const auto& geometry=policyGeometry();
    for(int i=0;i<Count;++i){Bits targets=geometry.masks[i]&mask;if(!targets)continue;
        destinations[active]=targets;int offset=state.codes[turn==1?0:1][i]*Dim;
        for(int j=0;j<Dim;++j)entries[active*Dim+j]=std::int16_t(embeddingValue(offset+j));++active;}
    maskedPolicy(result,entries.data(),destinations.data(),active,mask);return result;
}
Pattern::State Pattern::refreshState(const Position& p) const{
    State state;state.blackBoard=p.black;state.whiteBoard=p.white;
    for(int i=0;i<Count;++i){const auto& shape=shapes()[i];
        state.codes[0][i]=std::uint16_t(code(shape,p.black,p.white));
        state.codes[1][i]=std::uint16_t(code(shape,p.white,p.black));
        for(int side=0;side<2;++side){int offset=state.codes[side][i]*Dim;
            if(embedding8.empty())pattern_kernels::gather(embedding.data()+offset,state.sums[side].data()+shape.group*Dim,nullptr);
            else pattern_kernels::gather(embedding8.data()+offset,embeddingStep,state.sums[side].data()+shape.group*Dim,nullptr);}
    }return state;
}
Pattern::State Pattern::updateState(const State& before,const Position& p,unsigned* changedPatterns) const{
    struct Occurrence {int shape,power;};
    static const auto occurrences=[](){std::array<std::vector<Occurrence>,64> result;
        for(int i=0;i<Count;++i){const auto& shape=shapes()[i];int power=1;
            for(int j=0;j<shape.length;++j){result[shape.squares[j]].push_back({i,power});power*=3;}}
        return result;}();
    State state=before;state.blackBoard=p.black;state.whiteBoard=p.white;
    Bits changed=(before.blackBoard^p.black)|(before.whiteBoard^p.white),affected=0;
    while(changed){int square=first(changed);changed&=changed-1;Bits bit=Bits(1)<<square;
        int oldB=(before.blackBoard&bit)?1:(before.whiteBoard&bit)?2:0,newB=(p.black&bit)?1:(p.white&bit)?2:0;
        int oldW=(before.whiteBoard&bit)?1:(before.blackBoard&bit)?2:0,newW=(p.white&bit)?1:(p.black&bit)?2:0;
        for(const auto& occurrence:occurrences[square]){
            int i=occurrence.shape;affected|=Bits(1)<<i;
            state.codes[0][i]=std::uint16_t(int(state.codes[0][i])+(newB-oldB)*occurrence.power);
            state.codes[1][i]=std::uint16_t(int(state.codes[1][i])+(newW-oldW)*occurrence.power);
        }
    }
    if(changedPatterns)*changedPatterns=unsigned(count(affected));
    while(affected){int i=first(affected);affected&=affected-1;int group=shapes()[i].group;
        for(int side=0;side<2;++side){int oldOffset=before.codes[side][i]*Dim,newOffset=state.codes[side][i]*Dim;
            for(int j=0;j<Dim;++j)state.sums[side][group*Dim+j]+=embeddingValue(newOffset+j)-embeddingValue(oldOffset+j);}
    }return state;
}
Pattern::Prediction Pattern::predict(const State& state,int turn) const{
    int side=turn==1?0:1;Prediction result;result.values=outputs(state.sums[side].data(),state.blackBoard|state.whiteBoard);
    static const auto counts=[](){std::array<int,64> n{};for(const auto& s:shapes())for(int j=0;j<s.length;++j)++n[s.squares[j]];return n;}();
    static const auto buckets=[](){std::array<int,64> b{};for(int i=0;i<64;++i)b[i]=bucket(i);return b;}();
    for(int i=0;i<Count;++i){int offset=state.codes[side][i]*Dim;std::int64_t total=0;
        for(int j=0;j<Dim;++j)total+=std::int64_t(embeddingValue(offset+j))*result.values[4+j];
        double projection=double(total)/(256*256*std::sqrt(double(Dim)));const auto& shape=shapes()[i];
        for(int j=0;j<shape.length;++j)result.logits[shape.squares[j]]+=projection;
    }
    for(int i=0;i<64;++i)result.logits[i]=result.logits[i]/counts[i]+double(result.values[4+Dim+buckets[i]])/256;
    return result;
}
bool Pattern::operator==(const Pattern& b) const{return hidden==b.hidden&&embedding==b.embedding&&embedding8==b.embedding8&&embeddingStep==b.embeddingStep&&trunk==b.trunk&&trunkBias==b.trunkBias&&head==b.head&&headBias==b.headBias;}
}

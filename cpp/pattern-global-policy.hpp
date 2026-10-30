// Native candidate kernel: ordered38 projections -> one16-hidden -> all64 logits.
#pragma once
#include "pattern.hpp"
#include "pattern_kernels.hpp"
#include <cmath>
namespace othello::nnue {
inline void applyGlobalPolicy(Pattern::Prediction& result,const Pattern& model,
        const std::int16_t* entries,const Bits* destinations,Bits mask,
        const std::array<int,64>& counts,const std::array<int,64>& buckets){
    if(!mask)return;
    alignas(64) std::array<std::int16_t,32> gates{};
    for(int j=0;j<32;++j)gates[j]=std::int16_t(result.values[4+j]);
    std::array<std::int64_t,38> projections{};int row=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;row+4<=38;row+=4)pattern_kernels::dot4(entries+row*32,gates.data(),32,32,projections.data()+row);
#endif
    for(;row<38;++row)projections[row]=pattern_kernels::dot(entries+row*32,gates.data(),32);
    alignas(64) std::array<std::int16_t,38> features{};
    for(int i=0;i<38;++i){
        double projection=double(projections[i])/(65536*std::sqrt(32.0));
        Bits targets=destinations[i];
        while(targets){int square=first(targets);targets&=targets-1;result.logits[square]+=projection;}
        // Match the trained Torch FP32 division before quantizing the new input.
        // The original policy continues to use its unchanged double projection.
        const float feature=(float(projections[i])/65536.0f)/std::sqrt(32.0f);
        features[i]=std::int16_t(std::clamp(int(std::round(double(feature)*256)),-1024,1024));
    }
    auto scaled=[](std::int64_t v,int lo,int hi){return std::clamp(int(std::round(double(v)/256)),lo,hi);};
    alignas(64) std::array<std::int16_t,16> hidden{};row=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;row<16;row+=4){std::int64_t totals[4];pattern_kernels::dot4(model.globalDown.data()+row*38,features.data(),38,38,totals);
        for(int k=0;k<4;++k)hidden[row+k]=std::int16_t(scaled(model.globalDownBias[row+k]+totals[k],0,256));}
#endif
    for(;row<16;++row)hidden[row]=std::int16_t(scaled(model.globalDownBias[row]+pattern_kernels::dot(model.globalDown.data()+row*38,features.data(),38),0,256));
    std::array<int,64> correction{};row=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;row<64;row+=4){std::int64_t totals[4];pattern_kernels::dot4(model.globalUp.data()+row*16,hidden.data(),16,16,totals);
        for(int k=0;k<4;++k)correction[row+k]=scaled(model.globalUpBias[row+k]+totals[k],-1024,1024);}
#endif
    for(;row<64;++row)correction[row]=scaled(model.globalUpBias[row]+pattern_kernels::dot(model.globalUp.data()+row*16,hidden.data(),16),-1024,1024);
    while(mask){int square=first(mask);mask&=mask-1;
        result.logits[square]=result.logits[square]/counts[square]+double(result.values[36+buckets[square]])/256+double(correction[square])/256;}
}
}

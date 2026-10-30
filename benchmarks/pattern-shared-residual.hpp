// Candidate-only shared residual trunk. No per-move nonlinear network.
#pragma once
#include "pattern_kernels.hpp"
#include <algorithm>
#include <array>
#include <cmath>
namespace othello::nnue {
inline void applySharedResidual(const std::int16_t* down,const std::int32_t* downBias,
        const std::int16_t* up,const std::int32_t* upBias,int width,std::int16_t* trunk){
    auto scaled=[](std::int64_t value,int lo,int hi){return std::clamp(int(std::round(double(value)/256)),lo,hi);};
    alignas(64) std::array<std::int16_t,16> middle{};
    int row=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;row+4<=width;row+=4){std::int64_t totals[4];pattern_kernels::dot4(down+row*32,trunk,32,32,totals);
        for(int k=0;k<4;++k)middle[row+k]=std::int16_t(scaled(downBias[row+k]+totals[k],0,256));}
#endif
    for(;row<width;++row)middle[row]=std::int16_t(scaled(downBias[row]+pattern_kernels::dot(down+row*32,trunk,32),0,256));
    row=0;
#if defined(OTELLO_PATTERN_BLOCK4)
    for(;row<32;row+=4){std::int64_t totals[4];pattern_kernels::dot4(up+row*width,middle.data(),width,width,totals);
        for(int k=0;k<4;++k)trunk[row+k]=std::int16_t(std::clamp(int(trunk[row+k])+scaled(upBias[row+k]+totals[k],-256,256),0,256));}
#endif
    for(;row<32;++row)trunk[row]=std::int16_t(std::clamp(int(trunk[row])+scaled(upBias[row]+pattern_kernels::dot(up+row*width,middle.data(),width),-256,256),0,256));
}
}

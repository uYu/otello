#pragma once
#include <cstdint>
#if defined(OTELLO_PATTERN_SIMD) && defined(__aarch64__)
#include <arm_neon.h>
#elif defined(OTELLO_PATTERN_SIMD) && defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace othello::nnue::pattern_kernels {
// Pattern intermediates are bounded well below 2^53; divisor > 0 is even.
// Match std::round(value/divisor), including negative ties, without FP.
template<int Divisor> inline std::int64_t roundedDivide(std::int64_t value){
    static_assert(Divisor>0&&Divisor%2==0);
    return value>=0?(value+Divisor/2)/Divisor:-((-value+Divisor/2)/Divisor);
}
// Contract: n <= 128, abs(input[i]) <= 1024. The individual SIMD lanes
// cannot overflow INT32 even with arbitrary INT16 weights. Reduce into INT64.
inline std::int64_t dot(const std::int16_t* weights,const std::int16_t* input,int n){
    int i=0;std::int64_t total=0;
#if defined(OTELLO_PATTERN_SIMD) && defined(__aarch64__)
    int32x4_t sums=vdupq_n_s32(0);
    for(;i+8<=n;i+=8){
        auto a=vld1q_s16(weights+i),b=vld1q_s16(input+i);
        sums=vmlal_s16(sums,vget_low_s16(a),vget_low_s16(b));
        sums=vmlal_s16(sums,vget_high_s16(a),vget_high_s16(b));
    }
    total=vaddlvq_s32(sums);
#elif defined(OTELLO_PATTERN_SIMD) && defined(__SSE2__)
    __m128i sums=_mm_setzero_si128();
    for(;i+8<=n;i+=8){
        auto a=_mm_loadu_si128(reinterpret_cast<const __m128i*>(weights+i));
        auto b=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i));
        sums=_mm_add_epi32(sums,_mm_madd_epi16(a,b));
    }
    alignas(16) std::int32_t lanes[4];_mm_storeu_si128(reinterpret_cast<__m128i*>(lanes),sums);
    for(auto v:lanes)total+=v;
#endif
    for(;i<n;++i)total+=std::int64_t(weights[i])*input[i];
    return total;
}

// Four output rows share each input load. Same input/range contract as dot().
inline void dot4(const std::int16_t* weights,const std::int16_t* input,int n,int stride,std::int64_t* totals){
    int i=0;
#if defined(OTELLO_PATTERN_SIMD) && defined(__aarch64__)
    int32x4_t lo[4]={},hi[4]={};
    for(;i+8<=n;i+=8){auto x=vld1q_s16(input+i);
        for(int row=0;row<4;++row){auto w=vld1q_s16(weights+row*stride+i);
            lo[row]=vmlal_s16(lo[row],vget_low_s16(w),vget_low_s16(x));
            hi[row]=vmlal_s16(hi[row],vget_high_s16(w),vget_high_s16(x));}}
    for(int row=0;row<4;++row)totals[row]=vaddlvq_s32(vaddq_s32(lo[row],hi[row]));
#elif defined(OTELLO_PATTERN_SIMD) && defined(__SSE2__)
    __m128i sums[4]={_mm_setzero_si128(),_mm_setzero_si128(),_mm_setzero_si128(),_mm_setzero_si128()};
    for(;i+8<=n;i+=8){auto x=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i));
        for(int row=0;row<4;++row){auto w=_mm_loadu_si128(reinterpret_cast<const __m128i*>(weights+row*stride+i));sums[row]=_mm_add_epi32(sums[row],_mm_madd_epi16(w,x));}}
    for(int row=0;row<4;++row){alignas(16) std::int32_t lanes[4];_mm_storeu_si128(reinterpret_cast<__m128i*>(lanes),sums[row]);totals[row]=0;for(auto v:lanes)totals[row]+=v;}
#else
    for(int row=0;row<4;++row)totals[row]=0;
#endif
    for(;i<n;++i)for(int row=0;row<4;++row)totals[row]+=std::int64_t(weights[row*stride+i])*input[i];
}

// Decode exactly 32 table entries, optionally retaining the decoded row for policy.
inline void gather(const std::int16_t* row,std::int32_t* sums,std::int16_t* saved){
    int i=0;
#if defined(OTELLO_PATTERN_SIMD) && defined(__aarch64__)
    for(;i<32;i+=8){auto v=vld1q_s16(row+i);if(saved)vst1q_s16(saved+i,v);
        vst1q_s32(sums+i,vaddq_s32(vld1q_s32(sums+i),vmovl_s16(vget_low_s16(v))));
        vst1q_s32(sums+i+4,vaddq_s32(vld1q_s32(sums+i+4),vmovl_s16(vget_high_s16(v))));}
#elif defined(OTELLO_PATTERN_SIMD) && defined(__SSE2__)
    for(;i<32;i+=8){auto v=_mm_loadu_si128(reinterpret_cast<const __m128i*>(row+i));
        if(saved)_mm_storeu_si128(reinterpret_cast<__m128i*>(saved+i),v);
        auto sign=_mm_cmpgt_epi16(_mm_setzero_si128(),v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sums+i),_mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sums+i)),_mm_unpacklo_epi16(v,sign)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sums+i+4),_mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sums+i+4)),_mm_unpackhi_epi16(v,sign)));}
#endif
    for(;i<32;++i){sums[i]+=row[i];if(saved)saved[i]=row[i];}
}
inline void gather(const std::int8_t* row,int step,std::int32_t* sums,std::int16_t* saved){
    int i=0;
#if defined(OTELLO_PATTERN_SIMD) && defined(__aarch64__)
    for(;i<32;i+=8){auto v=vmulq_n_s16(vmovl_s8(vld1_s8(row+i)),std::int16_t(step));if(saved)vst1q_s16(saved+i,v);
        vst1q_s32(sums+i,vaddq_s32(vld1q_s32(sums+i),vmovl_s16(vget_low_s16(v))));
        vst1q_s32(sums+i+4,vaddq_s32(vld1q_s32(sums+i+4),vmovl_s16(vget_high_s16(v))));}
#elif defined(OTELLO_PATTERN_SIMD) && defined(__SSE2__)
    for(;i<32;i+=8){auto bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row+i));
        auto v=_mm_mullo_epi16(_mm_unpacklo_epi8(bytes,_mm_cmpgt_epi8(_mm_setzero_si128(),bytes)),_mm_set1_epi16(std::int16_t(step)));
        if(saved)_mm_storeu_si128(reinterpret_cast<__m128i*>(saved+i),v);
        auto sign=_mm_cmpgt_epi16(_mm_setzero_si128(),v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sums+i),_mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sums+i)),_mm_unpacklo_epi16(v,sign)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sums+i+4),_mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sums+i+4)),_mm_unpackhi_epi16(v,sign)));}
#endif
    for(;i<32;++i){int v=int(row[i])*step;sums[i]+=v;if(saved)saved[i]=std::int16_t(v);}
}
}

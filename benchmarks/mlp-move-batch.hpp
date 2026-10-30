// Research-only kernels. Included by isolated copies of pattern.cpp.
#pragma once
#include "pattern.hpp"
#include "pattern_kernels.hpp"
#include <cmath>
#include <optional>
#if defined(STUDY_GEMM) && defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace othello::nnue::study {
struct Prepared {
    const Pattern* model;
    std::array<float,32*32> weights{};
    std::array<float,32> output{};
    bool exact=true,fusedExact=true;
    explicit Prepared(const Pattern& m):model(&m){
        for(int k=0;k<m.policyWidth;++k){std::int64_t localBound=0,globalBound=std::abs(std::int64_t(m.policyHiddenBias[k])),positionBound=0;
            for(int j=0;j<32;++j){int w=m.policyHidden[k*74+j];weights[k*32+j]=float(w);localBound+=std::abs(w)*1024;
                globalBound+=std::abs(int(m.policyHidden[k*74+32+j]))*256;}
            for(int j=0;j<10;++j)positionBound=std::max(positionBound,std::int64_t(std::abs(int(m.policyHidden[k*74+64+j]))*256));
            exact&=localBound<=(1<<24);fusedExact&=localBound+globalBound+positionBound<=(1<<24);
            output[k]=float(m.policyOutput[k]);
        }
        std::int64_t outputBound=std::abs(std::int64_t(m.policyOutputBias[0]));
        for(int k=0;k<m.policyWidth;++k)outputBound+=std::abs(int(m.policyOutput[k]))*256;
        fusedExact&=outputBound<=(1<<24);
    }
};
inline thread_local const Prepared* activePrepared=nullptr;
struct Scope {
    const Prepared* previous;
    explicit Scope(const Prepared& prepared):previous(activePrepared){activePrepared=&prepared;}
    ~Scope(){activePrepared=previous;}
};
inline int scale(std::int64_t value,int low,int high){
    return std::clamp(int(std::round(double(value)/256)),low,high);
}
inline void normalize(const std::int32_t* local,int divisor,std::int16_t* output){
    int j=0;
#if defined(STUDY_VECTOR_NORMALIZE) && defined(OTELLO_PATTERN_SIMD) && defined(__aarch64__)
    // Local sums have magnitude <= 6*32768 and divisors are 2..6. FP32
    // division cannot cross a half boundary in this range; half ties for even
    // divisors are exact. FCVTAS has std::round's away-from-zero tie rule.
    // The runner exhaustively checks every sum/divisor against double rounding.
    for(;j<32;j+=4){
        auto v=vcvtq_f32_s32(vld1q_s32(local+j));
        auto r=vcvtaq_s32_f32(vdivq_f32(v,vdupq_n_f32(float(divisor))));
        r=vmaxq_s32(vdupq_n_s32(-1024),vminq_s32(vdupq_n_s32(1024),r));
        vst1_s16(output+j,vmovn_s32(r));
    }
#endif
    for(;j<32;++j)output[j]=std::int16_t(std::clamp(int(std::round(double(local[j])/divisor)),-1024,1024));
}

inline bool policyGemm(const Pattern& model,const std::array<std::array<std::int32_t,32>,64>& local,
        const std::array<int,64>& counts,const std::array<int,64>& buckets,
        const std::int16_t* trunk,Bits mask,Pattern::Prediction& result){
#if defined(STUDY_GEMM) && defined(__APPLE__)
    alignas(64) float localWeights[32*32],features[64*32],dots[64*32];
    const float* weights=localWeights;
#if defined(STUDY_PREPACK)
    std::optional<Prepared> owned;
    const Prepared* prepared=activePrepared&&activePrepared->model==&model?activePrepared:nullptr;
    if(!prepared){owned.emplace(model);prepared=&*owned;}
    if(!prepared->exact)return false;
    weights=prepared->weights.data();
#endif
    std::array<std::int64_t,32> global{};
    for(int k=0;k<model.policyWidth;++k){
#if !defined(STUDY_PREPACK)
        std::int64_t bound=0;
        for(int j=0;j<32;++j){int w=model.policyHidden[k*74+j];localWeights[k*32+j]=float(w);bound+=std::abs(w)*1024;}
        // Every product and every possible intermediate integer sum fits
        // exactly into FP32. Unsafe models use the integer kernel below.
        if(bound>(1<<24))return false;
#endif
        global[k]=model.policyHiddenBias[k]+pattern_kernels::dot(model.policyHidden.data()+k*74+32,trunk,32);
    }
    std::array<int,64> squares{};int moves=0;
    while(mask){int square=first(mask);mask&=mask-1;squares[moves]=square;
        std::int16_t normalized[32];normalize(local[square].data(),counts[square],normalized);
        for(int j=0;j<32;++j)features[moves*32+j]=float(normalized[j]);++moves;}
    if(!moves)return true;
    // One complete N x 32 times 32 x H matrix multiply for this board.
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,moves,model.policyWidth,32,
        1.0f,features,32,weights,32,0.0f,dots,model.policyWidth);
#if defined(STUDY_FUSED) && defined(__aarch64__)
    if(prepared->fusedExact){
        alignas(64) float addition[10][32];
        for(int bucket=0;bucket<10;++bucket)for(int k=0;k<model.policyWidth;++k)
            addition[bucket][k]=float(global[k]+std::int64_t(model.policyHidden[k*74+64+bucket])*256);
        for(int move=0;move<moves;++move){int square=squares[move];auto sum=vdupq_n_f32(0);
            for(int k=0;k<model.policyWidth;k+=4){
                auto x=vaddq_f32(vld1q_f32(dots+move*model.policyWidth+k),vld1q_f32(addition[buckets[square]]+k));
                auto rounded=vcvtaq_s32_f32(vmulq_n_f32(x,1.0f/256));
                rounded=vmaxq_s32(vdupq_n_s32(0),vminq_s32(vdupq_n_s32(256),rounded));
                sum=vfmaq_f32(sum,vcvtq_f32_s32(rounded),vld1q_f32(prepared->output.data()+k));
            }
            auto total=std::int64_t(vaddvq_f32(sum))+model.policyOutputBias[0];
            result.logits[square]=double(scale(total,-1024,1024))/256;
        }
        return true;
    }
#endif
    for(int move=0;move<moves;++move){int square=squares[move];std::int16_t hidden[32];
        for(int k=0;k<model.policyWidth;++k){auto total=global[k]+std::int64_t(dots[move*model.policyWidth+k])+
                std::int64_t(model.policyHidden[k*74+64+buckets[square]])*256;
            hidden[k]=std::int16_t(scale(total,0,256));}
        auto total=model.policyOutputBias[0]+pattern_kernels::dot(model.policyOutput.data(),hidden,model.policyWidth);
        result.logits[square]=double(scale(total,-1024,1024))/256;}
    return true;
#else
    return false;
#endif
}
inline void policyBatch(const Pattern& model,const std::array<std::array<std::int32_t,32>,64>& local,
        const std::array<int,64>& counts,const std::array<int,64>& buckets,
        const std::int16_t* trunk,Bits mask,Pattern::Prediction& result){
    if(policyGemm(model,local,counts,buckets,trunk,mask,result))return;
    std::array<std::int64_t,32> global{};
    for(int k=0;k<model.policyWidth;++k)global[k]=model.policyHiddenBias[k]+
        pattern_kernels::dot(model.policyHidden.data()+k*74+32,trunk,32);
    // SIMD lanes represent moves, rather than channels of one move.
    alignas(64) std::int16_t features[32][64]{};
    std::array<int,64> squares{};int moves=0;
    while(mask){int square=first(mask);mask&=mask-1;squares[moves]=square;
        std::int16_t normalized[32];normalize(local[square].data(),counts[square],normalized);
        for(int j=0;j<32;++j)features[j][moves]=normalized[j];++moves;
    }
    for(int begin=0;begin<moves;begin+=4){
        const int lanes=std::min(4,moves-begin);
        alignas(64) std::int16_t hidden[4][32]{};
        for(int k=0;k<model.policyWidth;++k){
            std::int64_t totals[4]{};
#if defined(OTELLO_PATTERN_SIMD) && defined(__aarch64__)
            auto total=vdupq_n_s32(0);
            for(int j=0;j<32;++j)total=vmlal_n_s16(total,vld1_s16(features[j]+begin),model.policyHidden[k*74+j]);
            std::int32_t sums[4];vst1q_s32(sums,total);
            for(int lane=0;lane<4;++lane)totals[lane]=sums[lane];
#elif defined(OTELLO_PATTERN_SIMD) && defined(__SSE2__)
            auto total=_mm_setzero_si128();
            for(int j=0;j<32;++j){auto x=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(features[j]+begin));
                auto w=_mm_set1_epi32(std::uint16_t(model.policyHidden[k*74+j]));
                total=_mm_add_epi32(total,_mm_madd_epi16(_mm_unpacklo_epi16(x,_mm_setzero_si128()),w));}
            std::int32_t sums[4];_mm_storeu_si128(reinterpret_cast<__m128i*>(sums),total);
            for(int lane=0;lane<4;++lane)totals[lane]=sums[lane];
#else
            for(int j=0;j<32;++j)for(int lane=0;lane<lanes;++lane)
                totals[lane]+=std::int64_t(features[j][begin+lane])*model.policyHidden[k*74+j];
#endif
            for(int lane=0;lane<lanes;++lane){int square=squares[begin+lane];
                hidden[lane][k]=std::int16_t(scale(global[k]+totals[lane]+
                    std::int64_t(model.policyHidden[k*74+64+buckets[square]])*256,0,256));}
        }
        for(int lane=0;lane<lanes;++lane){auto total=model.policyOutputBias[0]+
                pattern_kernels::dot(model.policyOutput.data(),hidden[lane],model.policyWidth);
            result.logits[squares[begin+lane]]=double(scale(total,-1024,1024))/256;}
    }
}
}

#if defined(STUDY_DEFINE_CHECK)
extern "C" int ot_study_check_normalize(){
    for(int divisor=2;divisor<=6;++divisor)for(int first=-6*32768;first<=6*32768;first+=32){
        std::int32_t input[32];std::int16_t output[32];
        for(int j=0;j<32;++j)input[j]=std::min(first+j,6*32768);
        othello::nnue::study::normalize(input,divisor,output);
        for(int j=0;j<32;++j)if(output[j]!=std::clamp(int(std::round(double(input[j])/divisor)),-1024,1024))return 0;
    }
    return 1;
}
#endif

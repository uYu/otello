#include "native_batch.hpp"
#include "cnn.hpp"
#include "nnue.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

thread_local const othello::nnue::CnnPolicy* activeBatchCnn=nullptr;
thread_local const othello::nnue::Network* activeBatchNetwork=nullptr;
namespace {thread_local bool validateBatch=false;thread_local double maxValueError=0,maxPolicyError=0;}
extern "C" void ot_cnn_native_batch_validate(int enabled){validateBatch=enabled;if(enabled)maxValueError=maxPolicyError=0;}
extern "C" void ot_cnn_native_batch_validation_errors(double* value,double* policy){if(value)*value=maxValueError;if(policy)*policy=maxPolicyError;}
namespace {
struct Scratch {std::vector<float> input,first,second,patch,matrix,hidden,output;};
thread_local Scratch scratch;
struct PackedNnue {
    std::unique_ptr<othello::nnue::Network> snapshot;
    std::vector<float> transform,policy,features,activation,joined,policyHidden;
    std::array<bool,32> exactFloatPolicy{};
};
thread_local PackedNnue packed;
void convolution(const std::vector<float>& input,int batch,int in,int out,const float* weights,const float* bias,std::vector<float>& result){
    const int rows=batch*64,columns=in*9;
    scratch.patch.resize(rows*columns);scratch.matrix.resize(rows*out);result.resize(batch*out*64);
    for(int b=0;b<batch;++b)for(int y=0;y<8;++y)for(int x=0;x<8;++x){
        float* row=scratch.patch.data()+((b*8+y)*8+x)*columns;
        for(int c=0;c<in;++c)for(int dy=0;dy<3;++dy)for(int dx=0;dx<3;++dx){
            int yy=y+dy-1,xx=x+dx-1;
            row[c*9+dy*3+dx]=(yy>=0&&yy<8&&xx>=0&&xx<8)?input[(b*in+c)*64+yy*8+xx]:0;
        }
    }
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,out,columns,1,scratch.patch.data(),columns,
                weights,columns,0,scratch.matrix.data(),out);
    for(int b=0;b<batch;++b)for(int p=0;p<64;++p)for(int c=0;c<out;++c)
        result[(b*out+c)*64+p]=std::max(0.0f,scratch.matrix[(b*64+p)*out+c]+bias[c]);
}
}
void prepareNativeModel(const othello::nnue::Network& model){
    if(model.cnnPolicy)return;
    if(packed.snapshot&&packed.snapshot->sameParameters(model))return;
    packed.snapshot=model.snapshot();packed.transform.resize(128*256);packed.policy.resize(32*512);
    if(model.hidden!=256||!model.policyHead||model.trunkBlock)return;
    for(int i=0;i<128;++i)for(int j=0;j<256;++j)packed.transform[i*256+j]=float(model.weights[i][j]);
    for(int j=0;j<32;++j){
        std::int64_t bound=std::abs(std::int64_t(model.policyHead->bias[j]));
        for(int i=0;i<512;++i){int w=model.policyHead->weights[j][i];packed.policy[j*512+i]=float(w);bound+=std::abs(w)*256;}
        // Under this bound every integer product and partial sum is exactly FP32.
        packed.exactFloatPolicy[j]=bound<(1LL<<24);
    }
}
namespace {
int nativeNnueBatch(int n,const int* boards,const int* turns,const std::uint64_t* legal,int* scores,double* probabilities){
    const auto& model=*activeBatchNetwork;
    if(model.hidden!=256||!model.policyHead||model.trunkBlock){
        for(int b=0;b<n;++b){othello::Position p;p.black=p.white=0;p.turn=turns[b];
            for(int i=0;i<64;++i){if(boards[b*64+i]==1)p.black|=1ULL<<i;else if(boards[b*64+i]==-1)p.white|=1ULL<<i;}
            auto result=model.infer(p,legal[b],false);scores[b]=result.score;std::copy(result.probabilities.begin(),result.probabilities.end(),probabilities+b*65);}
        return 1;
    }
    packed.features.resize(n*2*128);packed.activation.resize(n*2*256);packed.joined.resize(n*512);packed.policyHidden.resize(n*32);
    for(int b=0;b<n;++b)for(int i=0;i<64;++i){
        float own=boards[b*64+i]==turns[b]?1.0f:0.0f,opp=boards[b*64+i]==-turns[b]?1.0f:0.0f;
        packed.features[(b*2)*128+i]=own;packed.features[(b*2)*128+64+i]=opp;
        packed.features[(b*2+1)*128+i]=opp;packed.features[(b*2+1)*128+64+i]=own;
    }
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n*2,256,128,1,packed.features.data(),128,
                packed.transform.data(),256,0,packed.activation.data(),256);
    for(int b=0;b<n;++b)for(int j=0;j<256;++j){
        packed.joined[b*512+j]=std::clamp(packed.activation[(b*2)*256+j]+float(model.bias[j]),0.0f,256.0f);
        packed.joined[b*512+256+j]=std::clamp(packed.activation[(b*2+1)*256+j]+float(model.bias[j]),0.0f,256.0f);
    }
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,32,512,1,packed.joined.data(),512,
                packed.policy.data(),512,0,packed.policyHidden.data(),32);
    for(int b=0;b<n;++b){
        std::int64_t total=model.outputBias;
        for(int j=0;j<256;++j)total+=std::int64_t(packed.joined[b*512+j])*model.output[j]
            +std::int64_t(packed.joined[b*512+256+j])*model.opponentOutput[j];
        scores[b]=int(std::lround(1000*std::tanh(double(total)/65536)));
        std::array<double,32> hidden{};
        for(int j=0;j<32;++j){std::int64_t sum;
            if(packed.exactFloatPolicy[j])sum=std::llround(packed.policyHidden[b*32+j])+model.policyHead->bias[j];
            else{sum=model.policyHead->bias[j];for(int i=0;i<512;++i)sum+=std::int64_t(packed.joined[b*512+i])*model.policyHead->weights[j][i];}
            hidden[j]=std::clamp(double(sum)/65536,0.0,1.0);
        }
        double* p=probabilities+b*65;std::fill_n(p,65,0.0);if(!legal[b]){p[64]=1;continue;}
        double maximum=-1e300;
        for(int m=0;m<64;++m)if(legal[b]&(1ULL<<m)){
            double logit=double(model.policyHead->outputBias[m])/65536;
            for(int j=0;j<32;++j)logit+=hidden[j]*model.policyHead->output[m][j]/256;
            p[m]=logit;maximum=std::max(maximum,logit);
        }
        double sum=0;for(int m=0;m<64;++m)if(legal[b]&(1ULL<<m)){p[m]=std::exp(p[m]-maximum);sum+=p[m];}
        for(int m=0;m<64;++m)if(legal[b]&(1ULL<<m))p[m]/=sum;
    }
    return 1;
}
}
int nativeCnnBatch(int n,const int* boards,const int* turns,const std::uint64_t* legal,int* scores,double* probabilities){
    if(!activeBatchNetwork||n<1||n>128)return -1;
    if(!activeBatchCnn){int success=nativeNnueBatch(n,boards,turns,legal,scores,probabilities);
        if(success!=1)return success;
        if(validateBatch)for(int b=0;b<n;++b){othello::Position p;p.black=p.white=0;p.turn=turns[b];
            for(int i=0;i<64;++i){if(boards[b*64+i]==1)p.black|=1ULL<<i;else if(boards[b*64+i]==-1)p.white|=1ULL<<i;}
            auto reference=activeBatchNetwork->infer(p,legal[b],false);
            maxValueError=std::max(maxValueError,double(std::abs(reference.score-scores[b])));
            for(int m=0;m<65;++m)maxPolicyError=std::max(maxPolicyError,std::abs(reference.probabilities[m]-probabilities[b*65+m]));
            if(maxValueError>0||maxPolicyError>1e-12)return -1;
        }
        return 1;
    }
    const int c=activeBatchCnn->channels,h=activeBatchCnn->headWidth;
    const float* w=activeBatchCnn->weights.data();
    const int c1b=c*18,c2=c1b+c,c2b=c2+c*c*9,fc=c2b+c,fcb=fc+h*c*64;
    const int out=fcb+h,outb=out+h,pol=outb+1,polb=pol+65*h;
    scratch.input.resize(n*128);
    for(int b=0;b<n;++b)for(int p=0;p<64;++p){
        scratch.input[b*128+p]=boards[b*64+p]==turns[b]?1.0f:0.0f;
        scratch.input[b*128+64+p]=boards[b*64+p]==-turns[b]?1.0f:0.0f;
    }
    convolution(scratch.input,n,2,c,w,w+c1b,scratch.first);
    convolution(scratch.first,n,c,c,w+c2,w+c2b,scratch.second);
    scratch.hidden.resize(n*h);scratch.output.resize(n*66);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,h,c*64,1,scratch.second.data(),c*64,
                w+fc,c*64,0,scratch.hidden.data(),h);
    for(int b=0;b<n;++b)for(int j=0;j<h;++j)
        scratch.hidden[b*h+j]=std::max(0.0f,scratch.hidden[b*h+j]+w[fcb+j]);
    // Small output heads share the batched trunk; no Python or device transfer.
    for(int b=0;b<n;++b){
        double value=w[outb];for(int j=0;j<h;++j)value+=double(scratch.hidden[b*h+j])*w[out+j];
        scores[b]=int(std::lround(1000*std::tanh(value)));
        double* p=probabilities+b*65;std::fill_n(p,65,0.0);
        if(!legal[b]){p[64]=1;continue;}
        double maximum=-1e300;
        for(int m=0;m<64;++m)if(legal[b]&(1ULL<<m)){
            double logit=w[polb+m];for(int j=0;j<h;++j)logit+=double(scratch.hidden[b*h+j])*w[pol+m*h+j];
            p[m]=logit;maximum=std::max(maximum,logit);
        }
        double sum=0;for(int m=0;m<64;++m)if(legal[b]&(1ULL<<m)){p[m]=std::exp(p[m]-maximum);sum+=p[m];}
        for(int m=0;m<64;++m)if(legal[b]&(1ULL<<m))p[m]/=sum;
    }
    if(validateBatch)for(int b=0;b<n;++b){
        std::uint64_t black=0,white=0;
        for(int i=0;i<64;++i){if(boards[b*64+i]==1)black|=1ULL<<i;else if(boards[b*64+i]==-1)white|=1ULL<<i;}
        auto reference=activeBatchCnn->infer(black,white,turns[b],legal[b],false);
        maxValueError=std::max(maxValueError,double(std::abs(reference.score-scores[b])));
        for(int m=0;m<65;++m)maxPolicyError=std::max(maxPolicyError,std::abs(reference.probabilities[m]-probabilities[b*65+m]));
        if(maxValueError>1||maxPolicyError>1e-5)return -1;
    }
    return 1;
}

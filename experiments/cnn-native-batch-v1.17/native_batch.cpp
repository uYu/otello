#include "native_batch.hpp"
#include "cnn.hpp"
#include "nnue.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

thread_local const othello::nnue::CnnPolicy* activeBatchCnn=nullptr;
namespace {thread_local bool validateBatch=false;thread_local double maxValueError=0,maxPolicyError=0;}
extern "C" void ot_cnn_native_batch_validate(int enabled){validateBatch=enabled;if(enabled)maxValueError=maxPolicyError=0;}
extern "C" void ot_cnn_native_batch_validation_errors(double* value,double* policy){if(value)*value=maxValueError;if(policy)*policy=maxPolicyError;}
namespace {
struct Scratch {std::vector<float> input,first,second,patch,matrix,hidden,output;};
thread_local Scratch scratch;
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
int nativeCnnBatch(int n,const int* boards,const int* turns,const std::uint64_t* legal,int* scores,double* probabilities){
    if(!activeBatchCnn||n<1||n>128)return -1;
    const int c=activeBatchCnn->channels;
    const float* w=activeBatchCnn->weights.data();
    const int c1b=c*18,c2=c1b+c,c2b=c2+c*c*9,fc=c2b+c,fcb=fc+64*c*64;
    const int out=fcb+64,outb=out+64,pol=outb+1,polb=pol+65*64;
    scratch.input.resize(n*128);
    for(int b=0;b<n;++b)for(int p=0;p<64;++p){
        scratch.input[b*128+p]=boards[b*64+p]==turns[b]?1.0f:0.0f;
        scratch.input[b*128+64+p]=boards[b*64+p]==-turns[b]?1.0f:0.0f;
    }
    convolution(scratch.input,n,2,c,w,w+c1b,scratch.first);
    convolution(scratch.first,n,c,c,w+c2,w+c2b,scratch.second);
    scratch.hidden.resize(n*64);scratch.output.resize(n*66);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,64,c*64,1,scratch.second.data(),c*64,
                w+fc,c*64,0,scratch.hidden.data(),64);
    for(int b=0;b<n;++b)for(int j=0;j<64;++j)
        scratch.hidden[b*64+j]=std::max(0.0f,scratch.hidden[b*64+j]+w[fcb+j]);
    // Small output heads share the batched trunk; no Python or device transfer.
    for(int b=0;b<n;++b){
        double value=w[outb];for(int j=0;j<64;++j)value+=double(scratch.hidden[b*64+j])*w[out+j];
        scores[b]=int(std::lround(1000*std::tanh(value)));
        double* p=probabilities+b*65;std::fill_n(p,65,0.0);
        if(!legal[b]){p[64]=1;continue;}
        double maximum=-1e300;
        for(int m=0;m<64;++m)if(legal[b]&(1ULL<<m)){
            double logit=w[polb+m];for(int j=0;j<64;++j)logit+=double(scratch.hidden[b*64+j])*w[pol+m*64+j];
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

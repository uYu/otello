#include "../cpp/othello.hpp"
#include "../cpp/api.h"
#include "../cpp/nnue.hpp"
#include "../cpp/pattern.hpp"
#include "../cpp/pattern_kernels.hpp"
#include "../cpp/cnn.hpp"
#include "../cpp/resnet.hpp"
#include "../cpp/mcts.hpp"
#include "../cpp/search_table.hpp"
#include <iostream>
#include <random>
#include <cstdlib>
#include <sstream>
using namespace othello;
void check(bool yes,const char* why) {if(!yes){std::cerr<<"FAIL: "<<why<<'\n';std::exit(1);}}
Bits referenceFlips(const Position& p,int m) {
    auto s=board(p);if(m<0||m>63||s[m]!='.')return 0;
    Bits result=0;char own=p.turn==1?'B':'W',opp=p.turn==1?'W':'B';
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        if(!dy&&!dx)continue;
        int y=m/8+dy,x=m%8+dx;Bits f=0;
        while(y>=0&&y<8&&x>=0&&x<8&&s[y*8+x]==opp){f|=1ULL<<(y*8+x);y+=dy;x+=dx;}
        if(y>=0&&y<8&&x>=0&&x<8&&s[y*8+x]==own)result|=f;
    }return result;
}
std::uint64_t perft(Position p,int d) {
    if(!d)return 1;
    auto b=legal(p);if(!b)return over(p)?1:perft(play(p,-1),d);
    std::uint64_t n=0;while(b){int m=first(b);b&=b-1;n+=perft(play(p,m),d-1);}return n;
}
int exactDiff(Position p) {
    auto b=legal(p);if(!b)return over(p)?count(p.own())-count(p.opponent()):-exactDiff(play(p,-1));
    int best=-65;while(b){int m=first(b);b&=b-1;best=std::max(best,-exactDiff(play(p,m)));}return best;
}
int referenceValue(Position p,int depth,const nnue::Network* net=nullptr){
    auto b=legal(p);
    if(!b){if(over(p)){int d=count(p.own())-count(p.opponent());return d>0?100000+d*100:d<0?-100000+d*100:0;}return -referenceValue(play(p,-1),depth,net);}
    if(!depth)return net?net->evaluate(p):evaluate(p);
    int best=-2000000;while(b){int m=first(b);b&=b-1;best=std::max(best,-referenceValue(play(p,m),depth-1,net));}return best;
}
void policyChecks(){
    Position p;SearchResult analysis;analysis.move=19;analysis.score=100;analysis.depth=4;
    analysis.candidates={{19,100},{26,92},{37,68},{44,-20}};
    auto original=analysis.candidates;std::array<int,64> counts{};
    for(unsigned seed=0;seed<4096;++seed){
        auto selected=selectMove(p,analysis,{},seed);
        check(selected.sampled&&selected.eligible==3&&selected.temperatureMilli==32000&&selected.lossLimit==64,"opening schedule and score filter");
        ++counts[selected.move];
        check(selected.bestScore-selected.score<=selected.lossLimit,"opening selection stays within score-loss budget");
        auto repeated=selectMove(p,analysis,{},seed);
        check(repeated.move==selected.move,"same seed and root scores reproduce selection");
        auto reversed=analysis;std::reverse(reversed.candidates.begin(),reversed.candidates.end());
        check(selectMove(p,reversed,{},seed).move==selected.move,"candidate tie order does not change seeded selection");
    }
    check(counts[19]>counts[26]&&counts[26]>counts[37]&&counts[44]==0,"weighted sampling favors better moves and excludes bad moves");
    check(analysis.move==19&&analysis.score==100&&analysis.candidates[1].score==92,"selection does not mutate analysis");
    check(!selectMove(p,analysis,{0,32,64},71).sampled,"opening selection can be disabled");
    auto exact=analysis;exact.exact=true;check(selectMove(p,exact,{},71).move==19,"exact result is never weakened");
    auto terminal=analysis;terminal.score=100100;terminal.candidates[0].score=100100;
    check(!selectMove(p,terminal,{},71).sampled,"known terminal scores disable random selection");
    for(int ply=0;ply<12;++ply){auto moves=legal(p);if(!moves)p=play(p,-1);p=play(p,first(legal(p)));}
    auto end=selectMove(p,analysis,{},71);check(end.move==19&&!end.sampled&&end.temperatureMilli==0,"policy ends after twelve placements");
    p=Position{};
    for(int ply=0;ply<11;++ply){if(!legal(p))p=play(p,-1);p=play(p,first(legal(p)));}
    auto b=legal(p);SearchResult late;late.depth=4;late.score=100;late.move=first(b);
    while(b){int m=first(b);b&=b-1;late.candidates.push_back({m,m==late.move?100:92});}
    auto selected=selectMove(p,late,{},71);
    check(selected.temperatureMilli==2666&&selected.lossLimit==5&&!selected.sampled,"temperature and near-score window decay together");
    bool rejected=false;try{selectMove(Position{},analysis,{-1,16,32},1);}catch(...){rejected=true;}
    check(rejected,"invalid opening policy rejected");
}
void checkSearch(const Position& p,int depth,const nnue::Network* net=nullptr){
    auto r=search(p,{0,depth,2,1,net});
    check(r.score==referenceValue(p,depth,net),"persistent TT preserves fixed-depth minimax");
    for(auto c:r.candidates)check(c.score==-referenceValue(play(p,c.move),depth-1,net),"persistent TT preserves every candidate score");
}
void patternInt8Checks(){
    std::mt19937 rng(3218);
    for(int trial=0;trial<20000;++trial){
        Bits own=(Bits(rng())<<32)|rng(),opponent=(Bits(rng())<<32)|rng();
        auto codes=nnue::Pattern::encode(own,opponent);
        for(int i=0;i<nnue::Pattern::Count;++i)
            check(codes[i]==nnue::Pattern::code(nnue::Pattern::shapes()[i],own,opponent),"packed Pattern codes equal generic ternary encoding including overlapping masks");
    }
    for(int value=-983040;value<=983040;++value){
        check(nnue::pattern_kernels::roundedDivide<4>(value)==std::round(double(value)/4),"integer pool rounding matches FP including negative ties");
        check(nnue::pattern_kernels::roundedDivide<30>(value)==std::round(double(value)/30),"integer interior rounding matches FP");
    }
    for(int empty=0;empty<=64;++empty)
        check(nnue::pattern_kernels::roundedDivide<60>(empty*256)==std::round(double(empty)*256/60),"integer phase rounding matches FP");
    for(std::int64_t quotient:{-17000000LL,-1LL,0LL,1LL,17000000LL})for(int remainder=-257;remainder<=257;++remainder){
        auto value=quotient*256+remainder;
        check(nnue::pattern_kernels::roundedDivide<256>(value)==std::round(double(value)/256),"integer dense rounding matches FP at extreme legal dot magnitudes");
    }
    std::array<std::int16_t,129> a{},b{};
    std::array<std::int16_t,4*129> matrix{};
    for(int trial=0;trial<100;++trial){
        for(int i=0;i<129;++i){a[i]=std::int16_t(rng());b[i]=std::int16_t(int(rng()%2049)-1024);}
        if(trial==0){a.fill(-32768);b.fill(-1024);}
        for(int n:{1,7,8,31,32,64,97,128}){
            std::int64_t expected=0;for(int i=0;i<n;++i)expected+=std::int64_t(a[i])*b[i];
            check(nnue::pattern_kernels::dot(a.data(),b.data(),n)==expected,"integer SIMD dot equals INT64 reference including tails and extreme weights");
            for(int row=0;row<4;++row)for(int i=0;i<129;++i)matrix[row*129+i]=trial==0?-32768:std::int16_t(rng());
            std::int64_t actual[4];nnue::pattern_kernels::dot4(matrix.data(),b.data(),n,129,actual);
            for(int row=0;row<4;++row){std::int64_t reference=0;for(int i=0;i<n;++i)reference+=std::int64_t(matrix[row*129+i])*b[i];
                check(actual[row]==reference,"four-row SIMD layout equals INT64 reference including odd strides and tails");}
        }
    }
    for(int width:{32,64})for(int step:{1,2,4,8}){
        nnue::Pattern full;full.hidden=width;full.embedding.resize(full.Rows*full.Dim);
        std::string binary("OTPAT3\0\0",8);
        auto append=[&](std::uint32_t v,int bytes){for(int i=0;i<bytes;++i)binary.push_back(char((v>>(8*i))&255));};
        for(auto v:{full.Rows,full.Dim,width,full.Inputs,full.Outputs,256,step})append(v,4);
        for(auto& v:full.embedding){int raw=int(rng()%256)-128;v=std::int16_t(raw*step);append(std::uint32_t(raw),1);}
        for(int i=0;i<width*full.Inputs;++i){full.trunk[i]=std::int16_t(int(rng()%513)-256);append(std::uint32_t(full.trunk[i]),2);}
        for(int i=0;i<width;++i){full.trunkBias[i]=int(rng()%10000)-5000;append(std::uint32_t(full.trunkBias[i]),4);}
        for(int i=0;i<width*full.Outputs;++i){full.head[i]=std::int16_t(int(rng()%513)-256);append(std::uint32_t(full.head[i]),2);}
        for(auto& v:full.headBias){v=int(rng()%10000)-5000;append(std::uint32_t(v),4);}
        std::istringstream stream(binary);auto net=nnue::Network::load(stream);auto q=std::const_pointer_cast<nnue::Pattern>(net.pattern);
        check(q&&q->embedding.empty()&&q->embedding8.size()==full.embedding.size(),"INT8 loader keeps compressed table in memory");
        check(net.valueTarget==nnue::ValueTarget::Outcome&&q->hidden==width,"OTPAT3 preserves outcome semantics and width");
        Position p;auto acc=q->refresh(p);auto state=q->refreshState(p);
        for(int ply=0;ply<60&&!over(p);++ply){
            auto expected=full.predict(p),actual=q->predict(p),cached=q->predict(state,p.turn);
            check(actual.values==expected.values&&actual.logits==expected.logits,"INT8 decode and SIMD predict equal expanded INT16 reference exactly");
            check(actual.values==cached.values&&actual.logits==cached.logits,"INT8 cache prediction equals full prediction");
            Bits randomMask=(Bits(rng())<<32)|rng();
            for(Bits mask:{legal(p),Bits(0),~Bits(0),Bits(1)<<63,randomMask}){
                auto masked=q->predictLegal(p,mask),cachedMasked=q->predictLegal(state,p.turn,mask);
                check(masked.values==actual.values&&cachedMasked.values==actual.values,"legal-only policy preserves all 46 value and gate outputs");
                for(int square=0;square<64;++square){double wanted=mask&(Bits(1)<<square)?actual.logits[square]:0;
                    check(masked.logits[square]==wanted&&cachedMasked.logits[square]==wanted,"masked full and incremental policy logits exactly match full inference at every selected square");}
            }
            auto ref=full.refresh(p);check(acc.black==ref.black&&acc.white==ref.white,"INT8 incremental accumulator equals expanded table");
            Bits moves=legal(p);int move=-1;if(moves){int skip=rng()%count(moves);while(skip--)moves&=moves-1;move=first(moves);}
            auto next=play(p,move);acc=q->update(acc,p,next);state=q->updateState(state,next);p=next;
        }
        auto copy=net.snapshot();++q->embedding8[0];check(!net.sameParameters(*copy),"INT8 mutation invalidates model snapshot");
        auto rejects=[&](const std::string& bytes){bool rejected=false;try{std::istringstream bad(bytes);nnue::Network::load(bad);}catch(...){rejected=true;}check(rejected,"OTPAT3 rejects invalid scale, truncation and trailing bytes");};
        rejects(binary.substr(0,binary.size()-1));rejects(binary+"x");binary[32]=3;rejects(binary);
    }
}
// Independent scalar reference for the new file geometry and long dense rows.
void patternStructureChecks(){
    std::mt19937 rng(37137);
    for(int scenario:{1,2,3,4,5,6,7}){
        int variant=scenario==7?5:scenario==6?4:std::min(scenario,3),policyWidth=scenario==4?16:scenario==5?32:8;
        int rows=variant==2?55728:28503,inputs=variant==2?321:97,headCount=variant==3?4:46;
        std::string bytes(variant==5?"OTPAT7\0\0":variant==4?"OTPAT6\0\0":variant==3?"OTPAT5\0\0":"OTPAT4\0\0",8);
        auto append=[&](std::uint32_t v,int n){for(int i=0;i<n;++i)bytes.push_back(char(v>>(8*i)));};
        for(int v:{rows,32,32,inputs,headCount,256})append(v,4);
        if(variant==3){append(74,4);append(policyWidth,4);}else append(variant>=4?1:variant,4);
        if(variant>=4){append(36,4);append(48,4);}
        for(int i=0;i<rows*32+32*inputs;++i)append(std::uint32_t(int(rng()%513)-256),2);
        for(int i=0;i<32;++i)append(std::uint32_t(int(rng()%10001)-5000),4);
        for(int i=0;i<headCount*32;++i)append(std::uint32_t(int(rng()%513)-256),2);
        for(int i=0;i<headCount;++i)append(std::uint32_t(int(rng()%10001)-5000),4);
        if(variant==3){
            for(int i=0;i<policyWidth*74;++i)append(std::uint32_t(int(rng()%513)-256),2);
            for(int i=0;i<policyWidth;++i)append(std::uint32_t(int(rng()%10001)-5000),4);
            for(int i=0;i<policyWidth;++i)append(std::uint32_t(int(rng()%513)-256),2);
            append(std::uint32_t(int(rng()%10001)-5000),4);
        }
        if(variant==4){for(int i=0;i<4*32;++i)append(std::uint32_t(int(rng()%513)-256),2);for(int i=0;i<4;++i)append(std::uint32_t(int(rng()%10001)-5000),4);}
        if(variant==5){for(int i=0;i<42*32;++i)append(std::uint32_t(int(rng()%513)-256),2);for(int i=0;i<42;++i)append(std::uint32_t(int(rng()%10001)-5000),4);}
        std::istringstream stream(bytes);auto net=nnue::Network::load(stream);const auto& q=*net.pattern;
        check(net.valueTarget==nnue::ValueTarget::Outcome&&q.variant==variant&&(variant!=3||q.policyWidth==policyWidth),"OTPAT4 retains outcome and independent variant");
        check(q.embedding.size()==std::size_t(rows*32),"OTPAT4 retains compact table in memory");
        auto reference=[&](const Position& p){
            nnue::Pattern::Prediction result;std::array<std::int64_t,320> sums{};
            std::array<std::array<int,32>,38> entries{};std::array<int,10> sizes{};
            for(int i=0;i<38;++i){const auto& shape=q.shapes()[i];int group=variant==2?(i<16?shape.offset/6561:i<34?shape.length:9):shape.group;
                ++sizes[group];int raw=q.code(shape,p.own(),p.opponent());
                // Derive symmetry representatives separately from the loader's map.
                if(variant!=2){int value=raw-shape.offset,digits[9]{},other=0,power=1;
                    for(int k=0;k<shape.length;++k){digits[k]=value%3;value/=3;}
                    for(int k=0;k<shape.length;++k){int j=shape.length==9?(k%3)*3+k/3:shape.length-1-k;other+=digits[j]*power;power*=3;}
                    check(q.canonical[raw]==q.canonical[shape.offset+other],"D4 orbit rows are tied");}
                int offset=q.embeddingOffset(raw);
                for(int j=0;j<32;++j){entries[i][j]=q.embedding[offset+j];sums[group*32+j]+=entries[i][j];}
            }
            std::array<int,321> x{};int groups=variant==2?10:3;
            for(int g=0;g<groups;++g)for(int j=0;j<32;++j)x[g*32+j]=std::clamp(int(std::round(double(sums[g*32+j])/sizes[g])),-1024,1024);
            x[inputs-1]=int(std::round(double(p.empty())*256/60));std::array<int,32> hidden{};
            for(int j=0;j<32;++j){std::int64_t total=q.trunkBias[j];for(int k=0;k<inputs;++k)total+=std::int64_t(q.trunk[j*inputs+k])*x[k];hidden[j]=std::clamp(int(std::round(double(total)/256)),0,256);}
            for(int j=0;j<46;++j){std::int64_t total=q.headBias[j];for(int k=0;k<32;++k)total+=std::int64_t(q.head[j*32+k])*hidden[k];result.values[j]=std::clamp(int(std::round(double(total)/256)),-1024,1024);}
            if(variant==4){
                int numerator=std::clamp(p.empty()-36,0,12);
                for(int j=0;j<4;++j){std::int64_t total=q.earlyValueBias[j];for(int k=0;k<32;++k)total+=std::int64_t(q.earlyValue[j*32+k])*hidden[k];
                    int early=std::clamp(int(std::round(double(total)/256)),-1024,1024);
                    result.values[j]=int(std::round(double(numerator*early+(12-numerator)*result.values[j])/12));}
            }
            if(variant==5){
                int numerator=std::clamp(p.empty()-36,0,12);
                for(int j=0;j<42;++j){std::int64_t total=q.earlyPolicyBias[j];for(int k=0;k<32;++k)total+=std::int64_t(q.earlyPolicy[j*32+k])*hidden[k];
                    int early=std::clamp(int(std::round(double(total)/256)),-1024,1024);
                    result.values[4+j]=int(std::round(double(numerator*early+(12-numerator)*result.values[4+j])/12));}
            }
            if(variant==3){
                for(int square=0;square<64;++square){std::array<int,74> input{};int n=0;
                    for(int i=0;i<38;++i){const auto& shape=q.shapes()[i];bool contains=false;for(int k=0;k<shape.length;++k)contains|=shape.squares[k]==square;
                        if(contains){++n;for(int j=0;j<32;++j)input[j]+=entries[i][j];}}
                    for(int j=0;j<32;++j){input[j]=std::clamp(int(std::round(double(input[j])/n)),-1024,1024);input[32+j]=hidden[j];}
                    int x=std::min(square%8,7-square%8),y=std::min(square/8,7-square/8);if(x>y)std::swap(x,y);int bucket=x*4-x*(x-1)/2+y-x;input[64+bucket]=256;
                    std::array<int,32> ph{};
                    for(int k=0;k<policyWidth;++k){std::int64_t total=q.policyHiddenBias[k];for(int j=0;j<74;++j)total+=std::int64_t(q.policyHidden[k*74+j])*input[j];ph[k]=std::clamp(int(std::round(double(total)/256)),0,256);}
                    std::int64_t total=q.policyOutputBias[0];for(int k=0;k<policyWidth;++k)total+=std::int64_t(q.policyOutput[k])*ph[k];
                    result.logits[square]=double(std::clamp(int(std::round(double(total)/256)),-1024,1024))/256;
                }
                return result;
            }
            std::array<int,64> counts{};
            for(int i=0;i<38;++i){std::int64_t total=0;for(int j=0;j<32;++j)total+=std::int64_t(entries[i][j])*result.values[4+j];
                double projected=double(total)/(65536*std::sqrt(32.0));const auto& shape=q.shapes()[i];for(int k=0;k<shape.length;++k){int square=shape.squares[k];++counts[square];result.logits[square]+=projected;}}
            for(int square=0;square<64;++square){int x=std::min(square%8,7-square%8),y=std::min(square/8,7-square/8);if(x>y)std::swap(x,y);int bucket=x*4-x*(x-1)/2+y-x;
                result.logits[square]=result.logits[square]/counts[square]+double(result.values[36+bucket])/256;}
            return result;
        };
        Position p;auto state=q.refreshState(p);auto acc=q.refresh(p);
        for(int ply=0;ply<62&&!over(p);++ply){auto wanted=reference(p),actual=q.predict(p),cached=q.predict(state,p.turn);
            check(actual.values==wanted.values&&actual.logits==wanted.logits,"structural SIMD inference equals independent scalar arithmetic");
            check(cached.values==actual.values&&cached.logits==actual.logits,"structural cached state refresh fallback is exact");
            check(q.outputs(acc,p.turn)==actual.values&&net.evaluate(acc,p.turn)==net.evaluate(p),"structural generic accumulator evaluation is exact");
            check(q.policy(p,actual.values)==actual.logits,"structural separate policy preserves supplied gates");
            for(Bits mask:{legal(p),Bits(0),~Bits(0),Bits(1)<<63}){auto masked=q.predictLegal(p,mask),fromState=q.predictLegal(state,p.turn,mask);
                check(masked.values==actual.values&&fromState.values==actual.values,"structural legal policy preserves value");
                for(int k=0;k<64;++k){double expected=mask&(Bits(1)<<k)?actual.logits[k]:0;check(masked.logits[k]==expected&&fromState.logits[k]==expected,"structural legal and state projections match full prediction");}}
            if(variant!=2)for(int symmetry=0;symmetry<8;++symmetry){Position transformed{0,0,p.turn};std::array<int,64> to{};
                for(int square=0;square<64;++square){int x=square%8,y=square/8;if(symmetry&4)x=7-x;for(int j=0;j<(symmetry&3);++j){int old=x;x=7-y;y=old;}to[square]=y*8+x;
                    if(p.black&(Bits(1)<<square))transformed.black|=Bits(1)<<to[square];if(p.white&(Bits(1)<<square))transformed.white|=Bits(1)<<to[square];}
                auto other=q.predict(transformed);check(other.values==actual.values,"tied value and global gates are exactly D4 invariant");
                for(int k=0;k<64;++k)check(std::abs(other.logits[to[k]]-actual.logits[k])<2e-12,"tied policy is D4 equivariant");}
            Bits moves=legal(p);int move=-1;if(moves){int skip=rng()%count(moves);while(skip--)moves&=moves-1;move=first(moves);}auto next=play(p,move);
            state=q.updateState(state,next);acc=q.update(acc,p,next);p=next;
        }
        if(variant==2){auto mutableModel=std::const_pointer_cast<nnue::Pattern>(net.pattern);
            std::fill(mutableModel->embedding.begin(),mutableModel->embedding.end(),32767);
            std::fill(mutableModel->trunk.begin(),mutableModel->trunk.end(),-32768);
            auto wanted=reference(Position{}),actual=q.predict(Position{});
            check(actual.values==wanted.values&&actual.logits==wanted.logits,"321-wide extreme INT16 rows reduce bounded chunks to INT64 without SIMD overflow");}
        if(variant==3){auto snapshot=net.snapshot();auto mutableModel=std::const_pointer_cast<nnue::Pattern>(net.pattern);++mutableModel->policyOutput[0];
            check(!net.sameParameters(*snapshot),"direct policy parameter changes invalidate the cached model snapshot");}
        if(variant==4){auto snapshot=net.snapshot();auto mutableModel=std::const_pointer_cast<nnue::Pattern>(net.pattern);++mutableModel->earlyValue[0];
            check(!net.sameParameters(*snapshot),"phase value changes invalidate the cached model snapshot");}
        if(variant==5){auto snapshot=net.snapshot();auto mutableModel=std::const_pointer_cast<nnue::Pattern>(net.pattern);++mutableModel->earlyPolicy[0];
            check(!net.sameParameters(*snapshot),"phase policy changes invalidate the cached model snapshot");}
        auto rejects=[&](std::string bad){bool rejected=false;try{std::istringstream input(bad);nnue::Network::load(input);}catch(...){rejected=true;}check(rejected,"OTPAT4 rejects malformed geometry and data");};
        rejects(bytes.substr(0,bytes.size()-1));rejects(bytes+"x");
        if(variant==3){auto bad=bytes;bad[36]=9;rejects(bad);}
        if(variant>=4){for(int offset:{36,40}){auto bad=bytes;bad[offset]=char(127);rejects(bad);}auto bad=bytes;bad[40]=36;rejects(bad);}
        for(int offset:{8,12,16,20,24,28,32}){auto bad=bytes;bad[offset]=char(127);rejects(bad);}
    }
}
void patternChecks(){
    auto weights=std::make_shared<nnue::Pattern>();weights->embedding.resize(nnue::Pattern::Rows*nnue::Pattern::Dim);
    std::mt19937 rng(1832);
    for(auto& x:weights->embedding)x=int(rng()%257)-128;
    for(auto& x:weights->trunk)x=int(rng()%129)-64;
    for(auto& x:weights->head)x=int(rng()%129)-64;
    nnue::Network net;net.architecture=nnue::Architecture::Pattern;net.hidden=0;net.pattern=weights;
    for(int game=0;game<8;++game){
        Position p;auto acc=net.refresh(p);auto state=weights->refreshState(p);const auto rootState=state;
        while(!over(p)){
            Bits moves=legal(p);int m=-1;
            if(moves){int skip=rng()%count(moves);while(skip--)moves&=moves-1;m=first(moves);}
            auto after=play(p,m);auto updated=net.update(acc,p,after),fresh=net.refresh(after);
            auto incremental=weights->updateState(state,after),stateFresh=weights->refreshState(after);
            check(incremental.codes==stateFresh.codes&&incremental.sums==stateFresh.sums,"compact Pattern incremental codes and sums match full refresh");
            auto skipped=weights->updateState(rootState,after);
            check(skipped.codes==stateFresh.codes&&skipped.sums==stateFresh.sums,"Pattern direct update from a distant ancestor is exact");
            auto cachedPrediction=weights->predict(incremental,after.turn),fullPrediction=weights->predict(after);
            check(cachedPrediction.values==fullPrediction.values&&cachedPrediction.logits==fullPrediction.logits,"incremental Pattern joint prediction is exactly equal");
            state=incremental;
            check(updated.black==fresh.black&&updated.white==fresh.white,"pattern incremental sums match refresh after every flip/pass");
            check(net.evaluate(updated,after.turn)==net.evaluate(after),"pattern incremental score matches refresh");
            auto swapped=after;std::swap(swapped.black,swapped.white);swapped.turn=-after.turn;
            check(net.evaluate(after)==net.evaluate(swapped),"pattern color-relative value invariance");
            auto policy=weights->policy(after),other=weights->policy(swapped);
            auto fused=weights->predict(after);
            check(fused.values==weights->outputs(weights->refresh(after),after.turn),"fused Pattern value/WDL/gates equal full refresh");
            check(fused.logits==policy,"fused Pattern policy equals separate inference exactly");
            for(int i=0;i<64;++i)check(std::isfinite(policy[i])&&policy[i]==other[i],"pattern policy is finite and color-relative");
            acc=updated;p=after;
        }
    }
    auto compareResults=[](const MctsResult& a,const MctsResult& b){
        check(a.analysis.move==b.analysis.move&&a.analysis.score==b.analysis.score&&a.analysis.exact==b.analysis.exact&&a.analysis.nodes==b.analysis.nodes,"incremental MCTS preserves move, value, proof and simulation count");
        check(a.visits==b.visits&&a.values==b.values,"incremental MCTS preserves all root visits and means");
    };
    for(std::size_t entries:{64u,4096u}){
        Position root;clearMctsCache();auto cold=mcts(root,{0,512,1832,&net});auto next=play(root,cold.analysis.move);
        auto fullWarm=mcts(next,{0,512,1832,&net});
        clearMctsCache();auto incremental=mcts(root,{0,512,1832,&net,1.5,2097152,true,entries});
        compareResults(cold,incremental);
        check(incremental.patternUpdates>0&&incremental.patternRefreshes>0,"MCTS exercises incremental updates and refresh fallback");
        check(incremental.patternCacheBytes<=entries*1024,"incremental MCTS cache remains bounded below 1KiB per entry");
        auto cachedWarm=mcts(next,{0,512,1832,&net,1.5,2097152,true,entries});compareResults(fullWarm,cachedWarm);
        check(cachedWarm.reusedNodes>0,"incremental cache invalidation preserves reached-subtree reuse");
    }
    clearMctsCache();mcts(Position{},{0,128,1832,&net,1.5,2097152,true,64});
    ++weights->embedding[0];auto mutated=mcts(Position{},{0,128,1832,&net,1.5,2097152,true,64});
    check(mutated.reusedNodes==0&&mutated.patternRefreshes>0,"model mutation invalidates incremental state cache");
    clearMctsCache();auto freshMutated=mcts(Position{},{0,128,1832,&net});compareResults(mutated,freshMutated);
    --weights->embedding[0];weights->hidden=64;
    clearMctsCache();auto wideFull=mcts(Position{},{0,128,1832,&net});
    clearMctsCache();auto wideCached=mcts(Position{},{0,128,1832,&net,1.5,2097152,true,64});compareResults(wideFull,wideCached);
    weights->hidden=32;
    auto disabled=mcts(Position{},{0,128,1832,&net,1.5,2097152,false,64});
    check(disabled.patternUpdates==0&&disabled.patternCacheBytes==0,"uniform-policy ablation disables incremental Pattern cache");
    Position forced=parse("WWBBBWBBBBBBBWBBBBWWWWWBBBBWBWBBBBWBWWBBBBWWWBWBBWWWWWBBBW.BBBBB",-1);
    auto passPrediction=weights->predictLegal(forced,legal(forced));
    check(passPrediction.values==weights->predict(forced).values&&passPrediction.logits==std::array<double,64>{},"forced pass skips all policy logits and preserves value outputs");
    unsigned affected=99;auto beforePass=weights->refreshState(forced),afterPass=weights->updateState(beforePass,play(forced,-1),&affected);
    check(affected==0&&beforePass.codes==afterPass.codes&&beforePass.sums==afterPass.sums,"forced pass reuses unchanged Pattern accumulators");
    net.valueTarget=nnue::ValueTarget::Outcome;
    clearMctsCache();auto fullPass=mcts(forced,{0,1024,1832,&net});
    clearMctsCache();auto cachedPass=mcts(forced,{0,1024,1832,&net,1.5,2097152,true,64});compareResults(fullPass,cachedPass);
    bool rejectedCache=false;try{mcts(Position{},{0,1,1832,&net,1.5,2097152,true,63});}catch(...){rejectedCache=true;}
    check(rejectedCache,"incremental MCTS rejects unsupported cache sizes");
    net.valueTarget=nnue::ValueTarget::DiscDifference;
    Position p;clearSearchCache();checkSearch(p,2,&net);
    clearMctsCache();auto firstRun=mcts(p,{0,128,1832,&net});
    clearMctsCache();auto second=mcts(p,{0,128,1832,&net});
    check(firstRun.visits==second.visits&&firstRun.analysis.move==second.analysis.move,"pattern policy MCTS reproducible");
    auto warm=mcts(play(p,second.analysis.move),{0,128,1832,&net});
    check(warm.reusedNodes>0&&flips(play(p,second.analysis.move),warm.analysis.move),"pattern policy priors survive subtree compaction");
    auto uniform=mcts(p,{0,128,1832,&net,1.5,2097152,false});
    check(uniform.reusedNodes==0&&flips(p,uniform.analysis.move),"pattern policy ablation invalidates MCTS tree");
    auto copy=net.snapshot();weights->embedding[0]++;
    check(!net.sameParameters(*copy),"pattern cache snapshot detects mutated embedding");
    weights->head.fill(0);weights->headBias.fill(0);weights->headBias[36]=4*65536;
    p=Position{};
    while(!over(p)&&(!(legal(p)&corners)||!(legal(p)&~corners))){Bits moves=legal(p);p=play(p,moves?first(moves):-1);}
    check(!over(p),"policy-prior fixture reaches a corner choice");
    clearMctsCache();auto prioritized=mcts(p,{0,2,1832,&net});
    check((Bits(1)<<prioritized.analysis.move)&corners,"learned policy actually prioritizes high-prior legal moves");
}
void mctsChecks(){
    {
        nnue::Network policyNet;policyNet.architecture=nnue::Architecture::Turn;policyNet.hidden=256;
        auto head=std::make_shared<nnue::PolicyHead>();head->outputBias[19]=4*256*256;
        policyNet.policyHead=head;Position start;
        auto probs=policyNet.policy(start);double sum=0;
        for(int i=0;i<65;++i){sum+=probs[i];check(probs[i]==0||((legal(start)>>i)&1),"policy masks illegal moves");}
        check(std::abs(sum-1)<1e-12&&probs[19]>.9,"policy legal softmax normalization");
        auto snapshot=policyNet.snapshot();head->outputBias[19]=3*256*256;
        check(!policyNet.sameParameters(*snapshot),"policy snapshot is independent and affects cache identity");
        clearMctsCache();auto guided=mcts(start,{0,2,19,&policyNet});
        check(guided.analysis.move==19,"learned priors order first unvisited expansion");
        auto warm=mcts(play(start,guided.analysis.move),{0,16,19,&policyNet});
        check(warm.reusedNodes>0&&flips(play(start,guided.analysis.move),warm.analysis.move),"NNUE policy priors survive subtree compaction");
        auto uniform=mcts(play(start,guided.analysis.move),{0,16,19,&policyNet,1.5,2097152,false});
        check(uniform.reusedNodes==0,"NNUE policy ablation invalidates MCTS tree");
        head->outputBias[19]=0;head->outputBias[44]=4*256*256;
        auto changed=mcts(start,{0,2,19,&policyNet});
        check(changed.reusedNodes==0&&changed.analysis.move==44,"policy-only changes invalidate MCTS tree");
        auto terminal=policyNet.policy(parse(std::string(64,'B'),-1));
        check(std::all_of(terminal.begin(),terminal.end(),[](double x){return x==0;}),"terminal policy is zero");
        auto forced=parse("WWBBBWBBBBBBBWBBBBWWWWWBBBBWBWBBBBWBWWBBBBWWWBWBBWWWWWBBBW.BBBBB",-1);
        auto forcedPolicy=policyNet.policy(forced);
        check(forcedPolicy[64]==1&&std::count(forcedPolicy.begin(),forcedPolicy.end(),0.0)==64,"policy gives forced pass probability one");
        clearMctsCache();auto solved=mcts(forced,{0,1000,19,&policyNet});
        int diff=exactDiff(forced);int exact=diff>0?100000+diff*100:diff<0?-100000+diff*100:0;
        check(solved.analysis.exact&&solved.analysis.move==-1&&solved.analysis.score==exact,"learned policy preserves exact pass/endgame backups");
        auto flipped=start;std::swap(flipped.black,flipped.white);flipped.turn=-start.turn;
        flipped.hash=hashBoard(flipped.black,flipped.white,flipped.turn);
        check(policyNet.policy(start)==policyNet.policy(flipped),"policy preserves color relabeling");
        auto block=std::make_shared<nnue::TrunkBlock>();policyNet.trunkBlock=block;
        auto deepSnapshot=policyNet.snapshot();clearMctsCache();mcts(start,{0,8,19,&policyNet});
        block->upBias[0]=256;
        check(!policyNet.sameParameters(*deepSnapshot),"deep trunk snapshot is independent and part of cache identity");
        auto deepChanged=mcts(start,{0,8,19,&policyNet});
        check(deepChanged.reusedNodes==0,"trunk-only changes invalidate MCTS tree");
        clearMctsCache();
    }
    Position p;clearMctsCache();
    auto firstRun=mcts(p,{0,128,19});
    check(firstRun.analysis.nodes==128&&flips(p,firstRun.analysis.move),"MCTS fixed simulation budget and legal move");
    auto rootVisits=firstRun.visits;auto rootCandidates=firstRun.analysis.candidates;
    clearMctsCache();auto bigger=mcts(p,{0,128,19,nullptr,1.5,8388608});
    check(bigger.visits==rootVisits&&bigger.analysis.move==firstRun.analysis.move,"larger capacity preserves fixed-budget search when neither capacity binds");
    clearMctsCache();auto capped=mcts(p,{0,1000,19,nullptr,1.5,65});
    check(capped.memoryLimited&&capped.treeNodes==65,"explicit small node budget stops at its exact bound");
    bool rejectedCapacity=false;try{mcts(p,{0,1,19,nullptr,1.5,16777217});}catch(...){rejectedCapacity=true;}
    check(rejectedCapacity,"node budget above experimental safety bound rejected");
    check(firstRun.analysis.candidates.size()==4&&firstRun.visits.size()==4,"MCTS exposes all root visits");
    clearMctsCache();auto repeated=mcts(p,{0,128,19});
    check(repeated.analysis.move==firstRun.analysis.move&&repeated.visits==rootVisits,"MCTS seeded simulation runs reproduce");
    for(std::size_t i=0;i<rootCandidates.size();++i)check(rootCandidates[i].score==repeated.analysis.candidates[i].score,"MCTS reproducible candidate values");
    auto child=play(p,repeated.analysis.move);auto warm=mcts(child,{0,128,19});
    check(warm.reusedNodes>0&&flips(child,warm.analysis.move),"MCTS retains the reached subtree");
    nnue::Network net;auto withNet=mcts(p,{0,64,19,&net});
    check(withNet.reusedNodes==0,"MCTS clears tree on evaluator change");
    net.outputBias=100;auto changed=mcts(p,{0,64,19,&net});
    check(changed.reusedNodes==0,"MCTS invalidates modified evaluator contents");
    clearMctsCache();auto bounded=mcts(p,{0,10000,19,nullptr,1.5,65});
    check(bounded.memoryLimited&&bounded.treeNodes<=65,"MCTS bounds node storage");
    bool rejected=false;try{mcts(p,{0,0});}catch(...){rejected=true;}check(rejected,"MCTS requires a budget");
    std::mt19937 rng(1606);int passes=0;
    nnue::Network outcomeNet;outcomeNet.architecture=nnue::Architecture::Turn;outcomeNet.hidden=256;
    outcomeNet.valueTarget=nnue::ValueTarget::Outcome;outcomeNet.policyHead=std::make_shared<nnue::PolicyHead>();
    for(int game=0;game<12;++game){
        p=Position{};
        while(!over(p)&&p.empty()>4){auto moves=legal(p);
            if(!moves){p=play(p,-1);continue;}
            int nth=rng()%count(moves);while(nth--)moves&=moves-1;p=play(p,first(moves));}
        int expected=exactDiff(p);clearMctsCache();auto exact=mcts(p,{0,200000,23});
        check(exact.analysis.exact,"MCTS proves small endgames");
        int target=expected>0?100000+expected*100:expected<0?-100000+expected*100:0;
        check(exact.analysis.score==target,"MCTS terminal backups agree with independent minimax");
        if(legal(p))check(-exactDiff(play(p,exact.analysis.move))==expected,"MCTS chooses minimax endgame move");
        else if(!over(p)){++passes;check(exact.analysis.move==-1,"MCTS returns forced pass");}
        clearMctsCache();auto outcome=mcts(p,{0,200000,23,&outcomeNet});int sign=(expected>0)-(expected<0);
        check(outcome.analysis.exact&&outcome.analysis.score==sign*100000,"outcome MCTS exact backups match independent minimax winner");
        if(legal(p)){int child=-exactDiff(play(p,outcome.analysis.move));check((child>0)-(child<0)==sign,"outcome MCTS picks a minimax outcome");}
    }
    p=parse(std::string(64,'B'),-1);clearMctsCache();auto terminal=mcts(p,{0,1});
    check(terminal.analysis.exact&&terminal.analysis.move==-1&&terminal.analysis.score==-106400,"MCTS terminal value uses current player's view");
    clearMctsCache();auto outcomeLoss=mcts(p,{0,1,1,&outcomeNet});
    check(outcomeLoss.analysis.exact&&outcomeLoss.analysis.score==-100000,"outcome terminal loss is not a fictitious 64-disc margin");
    auto drawn=parse(std::string(32,'B')+std::string(32,'W'),1);clearMctsCache();auto outcomeDraw=mcts(drawn,{0,1,1,&outcomeNet});
    check(outcomeDraw.analysis.exact&&outcomeDraw.analysis.score==0,"outcome terminal draw is zero");
    auto identity=outcomeNet.snapshot();outcomeNet.valueTarget=nnue::ValueTarget::DiscDifference;
    check(!outcomeNet.sameParameters(*identity),"value target participates in model/cache identity");
    auto targetChanged=mcts(drawn,{0,1,1,&outcomeNet});check(targetChanged.reusedNodes==0,"changing only value semantics invalidates tree");
    outcomeNet.valueTarget=nnue::ValueTarget::Outcome;
    p=parse("WWBBBWBBBBBBBWBBBBWWWWWBBBBWBWBBBBWBWWBBBBWWWBWBBWWWWWBBBW.BBBBB",-1);
    check(!legal(p)&&legal(p.opponent(),p.own()),"MCTS forced-pass fixture");
    clearMctsCache();auto passed=mcts(p,{0,1000});int passValue=exactDiff(p);
    check(passed.analysis.move==-1&&passed.analysis.exact&&passed.analysis.score==(passValue>0?100000+passValue*100:passValue<0?-100000+passValue*100:0),"MCTS backs up forced passes correctly");
    clearMctsCache();auto outcomePass=mcts(p,{0,1000,1,&outcomeNet});
    check(outcomePass.analysis.move==-1&&outcomePass.analysis.exact&&outcomePass.analysis.score==((passValue>0)-(passValue<0))*100000,"outcome forced pass reverses winner perspective correctly");
    clearMctsCache();auto timed=mcts(Position{},{5});check(flips(Position{},timed.analysis.move),"timed MCTS returns a legal move");
    int cells[64]{},output[136]{},policy[6]{},visits[64]{};double stats[72]{};
    check(ot_nnue_mcts_play(nullptr,cells,1,1,0,1,12,32,64,output,policy,visits,stats)==-1,"MCTS C ABI rejects null model");
    clearMctsCache();
}
void cnnPolicyChecks(){
    for(int c:{8,16,32}){
        auto weights=std::make_shared<nnue::CnnPolicy>();weights->channels=c;
        weights->weights.resize(c*18+c+c*c*9+c+64*c*64+64+65+65*65);
        std::mt19937 rng(725+c);for(auto& w:weights->weights)w=float(int(rng()%101)-50)/1000;
        nnue::Network model;model.architecture=nnue::Architecture::CnnPolicy;
        model.valueTarget=nnue::ValueTarget::Outcome;model.hidden=0;model.cnnPolicy=weights;
        check(model.hasPolicy()&&model.spatial(),"CNN uses learned policy and spatial state");
        Position p;
        for(int ply=0;ply<8;++ply){
            if(!legal(p))p=play(p,-1);
            auto fast=model.infer(p);auto slow=weights->infer(p.black,p.white,p.turn,legal(p),false,true);
            check(std::abs(fast.score-slow.score)<=1,"policy CNN SIMD and scalar value agree");
            double sum=0;for(int i=0;i<65;++i){sum+=fast.probabilities[i];
                check(std::abs(fast.probabilities[i]-slow.probabilities[i])<1e-6,"policy CNN SIMD and scalar policy agree");
                if(i<64&&!(legal(p)&(1ULL<<i)))check(fast.probabilities[i]==0,"CNN masks illegal moves");}
            check(std::abs(sum-1)<1e-12,"CNN legal probabilities sum to one");
            auto swapped=p;std::swap(swapped.black,swapped.white);swapped.turn=-swapped.turn;
            auto renamed=model.infer(swapped);check(renamed.score==fast.score&&renamed.probabilities==fast.probabilities,"CNN color relabeling");
            auto next=play(p,first(legal(p)));
            check(model.evaluate(model.update(model.refresh(p),p,next),next.turn)==model.evaluate(next),"policy CNN accumulator update");p=next;
        }
        auto snapshot=model.snapshot();check(snapshot->sameParameters(model),"CNN snapshot matches");
        weights->weights.back()+=.1f;check(!snapshot->sameParameters(model),"CNN snapshot owns independent weights");
        clearMctsCache();auto searched=mcts(Position{},{0,32,17,&model});
        check(flips(Position{},searched.analysis.move)!=0&&searched.analysis.evaluations>0,"CNN executes learned-prior MCTS");
        auto terminal=parse(std::string(64,'B'),-1);clearMctsCache();
        check(mcts(terminal,{0,1,1,&model}).analysis.score==-100000,"CNN MCTS terminal backup uses outcome");
    }
    clearMctsCache();
}
void cnnChecks(){
    auto weights=std::make_shared<nnue::Cnn>();std::mt19937 rng(941);
    for(auto& w:weights->weights)w=float(int(rng()%401)-200)/1000;
    nnue::Network model;model.architecture=nnue::Architecture::Cnn;model.hidden=0;model.cnn=weights;
    Position p;
    for(int ply=0;ply<18;++ply){
        if(!legal(p))p=play(p,-1);
        auto next=play(p,first(legal(p)));
        auto acc=model.update(model.refresh(p),p,next);
        check(acc.blackBoard==next.black&&acc.whiteBoard==next.white,"CNN board state updates correctly");
        check(model.evaluate(acc,next.turn)==model.evaluate(next),"CNN direct and board-state evaluation agree");
        check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"CNN SIMD and independent scalar inference agree");
        auto swapped=next;std::swap(swapped.black,swapped.white);swapped.turn=-swapped.turn;
        check(model.evaluate(next)==model.evaluate(swapped),"CNN preserves color relabeling");p=next;
    }
    checkSearch(p,2,&model);
    weights->weights.back()+=0.5f; // Same pointer, changed weights: cache snapshot must be independent.
    checkSearch(p,2,&model);
    nnue::Network legacy;checkSearch(p,2,&legacy);checkSearch(p,2,&model);
    clearSearchCache();
}
void residualChecks(){
    std::mt19937 rng(1532);
    for(int blocks:{2,4}){
        auto weights=std::make_shared<nnue::ResNet>();weights->blocks=blocks;
        weights->weights.resize(nnue::ResNet::parameters(blocks));
        for(auto& w:weights->weights)w=float(int(rng()%201)-100)/2000;
        nnue::Network model;model.architecture=nnue::Architecture::Residual;model.hidden=0;model.residual=weights;
        Position p;
        for(int ply=0;ply<8;++ply){
            if(!legal(p))p=play(p,-1);
            auto next=play(p,first(legal(p)));
            auto acc=model.update(model.refresh(p),p,next);
            check(model.evaluate(acc,next.turn)==model.evaluate(next),"residual board state updates agree");
            check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"residual SIMD and independent scalar inference agree");
            auto renamed=next;std::swap(renamed.black,renamed.white);renamed.turn=-renamed.turn;
            check(model.evaluate(next)==model.evaluate(renamed),"residual preserves color relabeling");p=next;
        }
        checkSearch(p,2,&model);
        auto snapshot=model.snapshot();weights->weights.back()+=0.5f;
        check(!model.sameParameters(*snapshot),"residual cache snapshot owns independent weights");
        checkSearch(p,2,&model);
        nnue::Network old;checkSearch(p,2,&old);checkSearch(p,2,&model);
    }
    clearSearchCache();
}
void cacheChecks(nnue::Network& model){
    const auto originalModel=model;
    // Force all positions into one bucket to exercise replacement independently
    // of board hash distribution.
    detail::TranspositionTable table(1);table.newSearch();
    std::array<Position,5> positions;
    int i=0;Bits moves=legal(Position{});
    while(moves){int m=first(moves);moves&=moves-1;positions[i++]=play(Position{},m);}
    positions[4]=play(positions[0],first(legal(positions[0])));
    for(i=0;i<4;++i)table.store(positions[i],i+1,100+i,0,detail::Bound::Exact);
    for(i=0;i<4;++i)check(table.probe(positions[i])!=nullptr,"TT bucket retains colliding boards");
    table.store(positions[3],1,0,1,detail::Bound::Upper);
    check(table.probe(positions[3])->depth==4&&table.probe(positions[3])->score==103,"TT retains deeper result");
    table.store(positions[3],4,0,1,detail::Bound::Lower);
    check(table.probe(positions[3])->bound==detail::Bound::Exact,"TT retains exact score over same-depth bound");
    table.store(positions[4],2,0,1,detail::Bound::Lower);
    check(!table.probe(positions[0])&&table.probe(positions[3]),"TT replaces weakest collision");
    for(i=0;i<4;++i)table.newSearch();
    table.store(positions[4],2,0,1,detail::Bound::Lower);
    table.store(positions[0],2,0,1,detail::Bound::Exact);
    check(table.probe(positions[4])&&!table.probe(positions[1]),"TT ages old entries before recent ones");
    table.clear();for(auto& p:positions)check(!table.probe(p),"TT explicit clear");

    Position p;std::mt19937 rng(91);
    for(i=0;i<20;++i){Bits b=legal(p);if(!b){p=play(p,-1);b=legal(p);}int n=rng()%count(b);while(n--)b&=b-1;p=play(p,first(b));}
    clearSearchCache();auto cold=search(p,{0,4,2,1,&model});auto warm=search(p,{0,4,2,1,&model});
    check(cold.score==warm.score&&warm.ttCutoffs>0&&warm.nodes<cold.nodes,"same-model searches reuse TT across calls");
    check(warm.accumulatorUpdates<cold.accumulatorUpdates,"TT cutoffs avoid lazy NNUE updates");
    checkSearch(p,2,&model); // Deeper cached scores must not leak into shallow requests.
    checkSearch(play(p,cold.move),3,&model); // Reuse across actual moves.
    auto replacement=model;
    for(auto& x:replacement.output)x=-x;
    checkSearch(p,3,&replacement);
    model=replacement; // Same address, different parameters.
    checkSearch(p,3,&model);
    model.hidden=256;checkSearch(p,3,&model);
    model.architecture=nnue::Architecture::Turn;checkSearch(p,3,&model);
    model.opponentOutput[0]=192;checkSearch(p,3,&model);
    model.outputBias=12000;checkSearch(p,3,&model);
    checkSearch(p,3); // Neural -> heuristic -> neural.
    checkSearch(p,3,&model);
    checkSearch(p,3);
    ot_search_clear_cache();model=originalModel;
}
int main() {
    cnnChecks();cnnPolicyChecks();residualChecks();mctsChecks();patternChecks();patternInt8Checks();patternStructureChecks();
    policyChecks();
    Position p;
    check(legal(p)==((1ULL<<19)|(1ULL<<26)|(1ULL<<37)|(1ULL<<44)),"initial legal squares");
    const std::uint64_t expected[]={1,4,12,56,244,1396,8200,55092,390216};
    for(int i=0;i<=8;++i)check(perft(p,i)==expected[i],"opening perft");
    check(board(parse(board(p),1))==board(p),"board round trip");
    check(count(play(p,19).black)==4&&count(play(p,19).white)==1,"D3 flips D4");
    bool rejected=false;try{play(p,-1);}catch(...){rejected=true;}check(rejected,"reject voluntary pass");
    check(square("h8")==63&&square("A1")==0,"coordinates");
    std::mt19937 rng(2026);int passes=0,exactChecks=0;
    nnue::Network model;
    for(auto& row:model.weights)for(auto& value:row)value=int(rng()%161)-80;
    for(auto& value:model.bias)value=int(rng()%201)-30;
    for(auto& value:model.output)value=int(rng()%401)-200;
    for(int hidden:{32,256}){
        model.hidden=hidden;auto acc=model.refresh(p);auto next=play(p,19);
        auto incremental=model.update(acc,p,next),full=model.refresh(next);
        check(incremental.black==full.black&&incremental.white==full.white,"both NNUE widths update correctly");
        check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"both NNUE widths match tiny_nn");
    }
    model.hidden=32;
    cacheChecks(model);
    auto turnModel=model;turnModel.architecture=nnue::Architecture::Turn;turnModel.outputBias=2000;
    for(int i=0;i<nnue::MaxHidden;++i)turnModel.opponentOutput[i]=-model.output[i]/2+5;
    nnue::Network initiative;initiative.architecture=nnue::Architecture::Turn;
    initiative.outputBias=16384;
    check(initiative.evaluate(p)>0&&initiative.evaluate(initiative.refresh(p),-1)>0,"turn-aware model can represent initiative for either color");
    int searchChecks=0;
    for(int game=0;game<120;++game) {
        p=Position{};int steps=0;
        while(!over(p)) {
            check(++steps<=120,"games terminate");Bits expectedLegal=0;
            check(p.hash==hashBoard(p.black,p.white,p.turn),"incremental Zobrist matches full recomputation");
            for(int m=0;m<64;++m){auto f=referenceFlips(p,m);check(f==flips(p,m),"ray flips agree");if(f)expectedLegal|=1ULL<<m;}
            check(legal(p)==expectedLegal,"bitboard moves agree with array oracle");
            int cells[64],moves[64],out[136];auto s=board(p);
            for(int i=0;i<64;++i)cells[i]=s[i]=='B'?1:s[i]=='W'?-1:0;
            check(ot_legal(cells,p.turn,moves)==count(expectedLegal),"C API legal");
            auto b=legal(p);if(!b){
                ++passes;check(ot_play(cells,p.turn,-1,out)==1,"C API pass");
                if(p.empty()<=6){auto r=search(p,{0,64,2,1});int diff=exactDiff(p);
                    int expectedScore=diff>0?100000+diff*100:diff<0?-100000+diff*100:0;
                    check(r.move==-1&&r.exact&&r.score==expectedScore,"search through forced pass");}
                if(p.empty()<=6&&game<12){
                    auto r=search(p,{0,64,2,1,&model});
                    check(r.exact&&r.score==referenceValue(p,64,&model),"lazy NNUE search through forced pass");
                    auto t=search(p,{0,64,2,1,&turnModel});
                    check(t.exact&&t.score==referenceValue(p,64,&turnModel),"turn-aware search through forced pass");
                }
                p=play(p,-1);continue;
            }
            if(p.empty()<=6 && exactChecks<12) {
                auto r=search(p,{0,64,2,1});check(r.exact,"endgame completed");
                check(-exactDiff(play(p,r.move))==exactDiff(p),"search matches exhaustive endgame");++exactChecks;
            }
            std::vector<int> list;while(b){list.push_back(first(b));b&=b-1;}int m=list[rng()%list.size()];
            auto next=play(p,m);check(ot_play(cells,p.turn,m,out)==1,"C API play");
            int flippedSquares[64];std::fill(std::begin(flippedSquares),std::end(flippedSquares),-2);
            check(ot_play_with_flips(cells,p.turn,m,out,flippedSquares)==1,"C API play returns its computed flip list");
            Bits flippedMask=0;for(int i=0;i<out[66];++i)flippedMask|=1ULL<<flippedSquares[i];
            check(flippedMask==flips(p,m)&&out[66]==count(flippedMask),"flip list and count match the validated transition");
            if(out[66]<64)check(flippedSquares[out[66]]==-2,"C API flip list writes only its documented count");
            auto updated=model.update(model.refresh(p),p,next),refreshed=model.refresh(next);
            check(updated.black==refreshed.black&&updated.white==refreshed.white,"quantized NNUE incremental update equals full refresh");
            check(model.evaluate(updated,next.turn)==-model.evaluate(updated,-next.turn),"NNUE turn antisymmetry");
            if(steps%9==0){
                check(std::abs(model.evaluate(next)-model.evaluateReference(next))<=1,"tiny_nn SIMD FP32 reference matches integer NNUE");
                auto t=turnModel.update(turnModel.refresh(p),p,next),full=turnModel.refresh(next);
                check(t.black==full.black&&t.white==full.white,"turn-aware accumulators update incrementally");
                check(std::abs(turnModel.evaluate(next)-turnModel.evaluateReference(next))<=1,"turn-aware integer and FP32 reference agree");
                auto swapped=next;std::swap(swapped.black,swapped.white);swapped.turn=-swapped.turn;
                check(turnModel.evaluate(next)==turnModel.evaluate(swapped),"turn-aware model preserves color relabeling");
            }
            if(game<3&&steps%9==0&&searchChecks<20){
                for(int width:{0,32,256}){
                    model.hidden=width?width:32;const nnue::Network* net=width?&model:nullptr;
                    auto r=search(p,{0,3,2,1,net});check(r.depth==std::min(3,p.empty()),"fixed depth completed");
                    check(r.score==referenceValue(p,3,net),"PVS/TT/aspiration match exhaustive minimax");
                    for(auto c:r.candidates)check(c.score==-referenceValue(play(p,c.move),2,net),"root candidates retain exact depth scores");
                    if(width){turnModel.hidden=width;checkSearch(p,3,&turnModel);}
                }model.hidden=32;++searchChecks;
            }
            for(int i=0;i<64;++i)check(out[i]==((next.black>>i&1)?1:(next.white>>i&1)?-1:0),"C API board");
            check(count(next.black|next.white)==count(p.black|p.white)+1,"one new disc per move");p=next;
        }
        check(!legal(p.black,p.white)&&!legal(p.white,p.black),"both sides stuck at end");
    }
    check(passes>0,"forced passes covered");
    p=parse(std::string(64,'B'),1);check(over(p),"full board over");
    p=parse("B..............................................................W",-1);check(over(p),"early finish with empty squares");
    auto timed=search(Position{},{40,64,2,1});check(flips(Position{},timed.move)!=0&&timed.depth>0,"timed search returns completed legal move");
    check(timed.ttHits>0,"Zobrist transposition reuse");
    clearSearchCache();auto optimized=search(Position{},{0,6,2,1});
    check(optimized.pvsResearches>0&&optimized.betaCutoffs>0&&optimized.aspirationRetries>0,"PVS, beta cutoffs and aspiration retries exercised");
    std::cout<<"PASS: perft depth 8; 120 games vs independent oracle; "<<passes<<" forced passes; exhaustive endgames; C ABI; timed AI\n";
}

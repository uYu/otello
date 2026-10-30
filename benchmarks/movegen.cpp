#include "othello.hpp"
#include <iostream>
#include <random>
#include <iomanip>
#include <atomic>
using namespace othello;
template<int D,int N=1> inline Bits step(Bits b){if constexpr(N==1)return shift(b,D);else return step<D,N/2>(step<D,N/2>(b));}
template<int D> inline Bits prefix(Bits own,Bits opp){
    Bits g=own,p=opp;g|=p&step<D>(g);p&=step<D>(p);g|=p&step<D,2>(g);p&=step<D,2>(p);g|=p&step<D,4>(g);return g&opp;
}
Bits prefixLegal(Bits own,Bits opp){
    return (step<0>(prefix<0>(own,opp))|step<1>(prefix<1>(own,opp))|step<2>(prefix<2>(own,opp))|step<3>(prefix<3>(own,opp))|step<4>(prefix<4>(own,opp))|step<5>(prefix<5>(own,opp))|step<6>(prefix<6>(own,opp))|step<7>(prefix<7>(own,opp)))&~(own|opp);
}
template<int D> inline Bits rayPrefix(Bits move,Bits own,Bits opp){Bits c=prefix<D>(move,opp);return step<D>(c)&own?c:0;}
Bits prefixFlips(const Position& p,int move){
    if(move<0||move>63||((p.black|p.white)&(1ULL<<move)))return 0;Bits m=1ULL<<move,a=p.own(),b=p.opponent();
    return rayPrefix<0>(m,a,b)|rayPrefix<1>(m,a,b)|rayPrefix<2>(m,a,b)|rayPrefix<3>(m,a,b)|rayPrefix<4>(m,a,b)|rayPrefix<5>(m,a,b)|rayPrefix<6>(m,a,b)|rayPrefix<7>(m,a,b);
}
std::array<std::array<Bits,8>,64> makeRays(){std::array<std::array<Bits,8>,64> rays{};for(int i=0;i<64;++i)for(int d=0;d<8;++d){Bits b=shift(1ULL<<i,d);while(b){rays[i][d]|=b;b=shift(b,d);}}return rays;}
const auto rays=makeRays();
Bits tableFlips(const Position& p,int move){
    if(move<0||move>63||((p.black|p.white)&(1ULL<<move)))return 0;Bits result=0,own=p.own(),opp=p.opponent();
    for(int d=0;d<8;++d){Bits ray=rays[move][d],blockers=ray&~opp;if(!blockers)continue;bool ascending=d==0||d==2||d==4||d==5;
        Bits bound=ascending?blockers&-blockers:1ULL<<(63-__builtin_clzll(blockers));
        if(bound&own)result|=ray&(ascending?bound-1:~(bound|(bound-1)));
    }return result;
}
struct Sample{Position p;int move;};
volatile Bits sink=0;
template<class F> double bench(const std::vector<Sample>& data,F f){
    auto start=std::chrono::steady_clock::now();Bits total=0;
    for(int repeat=0;repeat<1200;++repeat){std::atomic_signal_fence(std::memory_order_seq_cst);for(const auto& x:data)total+=f(x);}
    sink=total;return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/(1200*data.size());
}
int main(){std::mt19937 rng(10919);std::vector<Sample> data;data.reserve(8192);Position p;
    while(data.size()<8192){auto moves=legal(p);if(!moves){if(over(p)){p=Position{};continue;}p=play(p,-1);continue;}
        int n=rng()%count(moves);while(n--)moves&=moves-1;int m=first(moves);data.push_back({p,m});p=play(p,m);}
    for(const auto& s:data){if(prefixLegal(s.p.own(),s.p.opponent())!=legal(s.p)){std::cerr<<"legal mismatch";return 1;}
        for(int m=0;m<64;++m)if(prefixFlips(s.p,m)!=flips(s.p,m)||tableFlips(s.p,m)!=flips(s.p,m)){std::cerr<<"flip mismatch";return 1;}}
    for(int i=0;i<100000;++i){Position q;q.black=(Bits(rng())<<32)|rng();q.white=((Bits(rng())<<32)|rng())&~q.black;q.turn=rng()%2?1:-1;
        if(prefixLegal(q.own(),q.opponent())!=legal(q)){std::cerr<<"random legal mismatch";return 1;}
        int m=rng()%64;if(prefixFlips(q,m)!=flips(q,m)||tableFlips(q,m)!=flips(q,m)){std::cerr<<"random flip mismatch";return 1;}}
    std::cout<<std::fixed<<std::setprecision(3)<<"{\"positions\":"<<data.size()<<",\"randomBoards\":100000,\"rounds\":[";
    for(int round=0;round<5;++round){if(round)std::cout<<',';std::cout<<"{\"legalCurrentNs\":"<<bench(data,[](auto& x){return legal(x.p);})
      <<",\"legalPrefixNs\":"<<bench(data,[](auto& x){return prefixLegal(x.p.own(),x.p.opponent());})
      <<",\"flipsCurrentNs\":"<<bench(data,[](auto& x){return flips(x.p,x.move);})
      <<",\"flipsPrefixNs\":"<<bench(data,[](auto& x){return prefixFlips(x.p,x.move);})
      <<",\"flipsRayTableNs\":"<<bench(data,[](auto& x){return tableFlips(x.p,x.move);})<<'}';}
    std::cout<<"]}\n";
}

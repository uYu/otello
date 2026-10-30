#include "othello.hpp"
#include <cmath>

namespace othello {
MoveSelection selectMove(const Position& p,const SearchResult& analysis,OpeningPolicy policy,std::uint32_t seed){
    if(policy.plies<0||policy.plies>60||policy.temperature<0||policy.temperature>1000||policy.maxLoss<0||policy.maxLoss>1000)
        throw std::invalid_argument("Invalid opening policy");
    MoveSelection result;
    result.move=analysis.move;result.score=result.bestScore=analysis.score;result.eligible=analysis.move>=0?1:0;
    const int placed=60-p.empty();
    if(policy.plies==0||policy.temperature==0||placed<0||placed>=policy.plies||analysis.exact||analysis.depth<1||analysis.candidates.empty())return result;
    Bits seen=0,moves=legal(p);
    std::array<Candidate,64> candidates{};int n=0;
    bool terminalScore=false;
    for(auto c:analysis.candidates){
        if(c.move<0||c.move>=64||!(moves&(1ULL<<c.move))||(seen&(1ULL<<c.move)))throw std::invalid_argument("Invalid root candidates");
        seen|=1ULL<<c.move;candidates[n++]=c;
        if(c.score>result.bestScore){result.bestScore=c.score;result.move=c.move;result.score=c.score;}
        terminalScore|=std::abs(c.score)>=100000;
    }
    if(seen!=moves)throw std::invalid_argument("Opening selection needs all root scores");
    if(terminalScore)return result;
    const int remaining=policy.plies-placed;
    result.temperatureMilli=policy.temperature*1000*remaining/policy.plies;
    result.lossLimit=policy.maxLoss*remaining/policy.plies;
    // Canonical move order makes seeded selection independent of root tie order.
    std::sort(candidates.begin(),candidates.begin()+n,[](auto a,auto b){return a.move<b.move;});
    std::array<double,64> weights{};double total=0;result.eligible=0;
    for(int i=0;i<n;++i){
        const auto loss=std::int64_t(result.bestScore)-candidates[i].score;
        if(loss<=result.lossLimit){
            weights[i]=std::exp(-double(loss)*1000/result.temperatureMilli);
            total+=weights[i];++result.eligible;
        }
    }
    if(result.eligible<=1)return result;
    Bits state=(Bits(seed)<<32)^p.hash^0x4f54454c4c4f504cULL;
    const double target=double(splitmix(state)>>11)*0x1.0p-53*total;
    double cumulative=0;
    for(int i=0;i<n;++i){
        cumulative+=weights[i];
        if(weights[i]>0&&target<cumulative){result.move=candidates[i].move;result.score=candidates[i].score;break;}
    }
    result.sampled=true;return result;
}
}

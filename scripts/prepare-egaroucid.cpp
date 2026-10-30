// Streaming local corpus validator. No model/search code or default is changed.
#include "othello.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <map>
#include <cstring>
using namespace othello;
namespace {
using Key=std::pair<Bits,Bits>;
struct KeyHash {std::size_t operator()(const Key& k)const{return k.first^(k.second*0x9e3779b97f4a7c15ULL);}};
int maps[8][64];
Bits mapped(Bits b,int s){Bits out=0;while(b){int m=first(b);b&=b-1;out|=1ULL<<maps[s][m];}return out;}
struct Canonical {Bits own,opp,legal;int move;};
Canonical canonical(const Position& p,int move){
    Key best{~0ULL,~0ULL};int chosen=0,bestMove=64;
    for(int s=0;s<8;++s){Key k{mapped(p.own(),s),mapped(p.opponent(),s)};
        if(k<best||(k==best&&maps[s][move]<bestMove)){best=k;chosen=s;bestMove=maps[s][move];}}
    return {best.first,best.second,mapped(legal(p),chosen),bestMove};
}
Bits fnv(const unsigned char* data,int n){Bits h=14695981039346656037ULL;for(int i=0;i<n;++i){h^=data[i];h*=1099511628211ULL;}return h;}
int split(const std::vector<int>& actions){
    std::vector<unsigned char> best;
    for(int s=0;s<8;++s){std::vector<unsigned char> v;int placements=0;
        for(int m:actions){v.push_back(m<0?64:maps[s][m]);if(m>=0&&++placements==8)break;}
        if(s==0||v<best)best=v;}
    auto bucket=fnv(best.data(),int(best.size()))%100;
    return bucket<80?0:bucket<90?1:2;
}
void little(std::ostream& stream,Bits v,int n){for(int i=0;i<n;++i)stream.put(char(v>>(8*i)));}
Bits readLittle(std::istream& stream){Bits v=0;for(int i=0;i<8;++i){int c=stream.get();if(c<0)throw std::runtime_error("Truncated holdout key");v|=Bits(c)<<(8*i);}return v;}
}
int main(int argc,char** argv){
 std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
 try{
    if(argc!=7)throw std::runtime_error("Usage: prepare-egaroucid RANDOM_PLIES CLEAN.bin SAMPLES.bin HOLDOUT.bin SAMPLE_PER_STAGE SEED");
    int randomPlies=std::stoi(argv[1]),budget=std::stoi(argv[5]);Bits seed=std::stoull(argv[6]);
    if(randomPlies<0||randomPlies>59||budget<0)throw std::runtime_error("Invalid sampling options");
    for(int s=0;s<8;++s)for(int m=0;m<64;++m){int x=m%8,y=m/8;if(s&4)x=7-x;
        for(int r=0;r<(s&3);++r){int old=x;x=7-y;y=old;}maps[s][m]=8*y+x;}
    std::unordered_set<Key,KeyHash> holdout;
    std::ifstream frozen(argv[4],std::ios::binary);if(!frozen)throw std::runtime_error("Cannot read holdout keys");
    while(frozen.peek()!=EOF){Bits own=readLittle(frozen),opp=readLittle(frozen);holdout.insert({own,opp});}
    std::ofstream clean(argv[2],std::ios::binary),samples(argv[3],std::ios::binary);
    if(!clean||!samples)throw std::runtime_error("Cannot create output");
    std::map<std::string,std::uint64_t> rejects;
    std::uint64_t records=0,accepted=0,sampled=0,excludedHoldout=0,passes=0,early=0,splitCounts[3]={},eligible=0;
    std::string line;
    while(std::getline(std::cin,line)){
        ++records;
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        if(line.empty()||line.size()%2||line.size()>120){++rejects["format"];continue;}
        unsigned char moves[60];bool valid=true;int n=int(line.size()/2);
        for(int i=0;i<n;++i){char x=line[2*i],y=line[2*i+1];if(x>='A'&&x<='H')x+=32;
            if(x<'a'||x>'h'||y<'1'||y>'8'){valid=false;break;}moves[i]=(y-'1')*8+x-'a';}
        if(!valid){++rejects["coordinate"];continue;}
        Bits gameHash=fnv(moves,n);Position p;std::vector<int> actions;
        struct Row{Canonical c;int turn,ply;};std::vector<Row> rows;
        int group=-1,gamePasses=0;
        for(int ply=0;ply<n;++ply){
            int move=moves[ply];Bits f=flips(p,move);
            if(!f){
                if(legal(p)){valid=false;break;}
                p.turn=-p.turn;actions.push_back(-1);++gamePasses;f=flips(p,move);
                if(!f){valid=false;break;}
            }
            if(ply>=8&&group<0)group=split(actions);
            if(ply>=randomPlies){
                ++eligible;
                // Equal expected observations per placement stage before opening-group filtering.
                Bits state=gameHash^seed^(Bits(ply)*0x9e3779b97f4a7c15ULL);Bits h=splitmix(state);
                long double probability=std::min(1.L,static_cast<long double>(budget)/(1000000.L*std::max(1,ply-7)));
                if(group==0&&static_cast<long double>(h)/18446744073709551616.L<probability){
                    auto c=canonical(p,move);
                    if(holdout.count({c.own,c.opp}))++excludedHoldout;
                    else rows.push_back({c,p.turn,ply});
                }
            }
            // Reuse repository rules; hashes are unused by this offline converter.
            if(p.turn==1){p.black|=f|(1ULL<<move);p.white&=~f;}
            else {p.white|=f|(1ULL<<move);p.black&=~f;}
            p.turn=-p.turn;actions.push_back(move);
        }
        if(!valid){++rejects["illegalMove"];continue;}
        if(!over(p)){++rejects["incomplete"];continue;}
        if(n<randomPlies){++rejects["shorterThanRandomPrefix"];continue;}
        if(group<0)group=split(actions);
        int diff=count(p.black)-count(p.white),outcome=(diff>0)-(diff<0);
        for(int i=0;i<60;++i)clean.put(char(i<n?moves[i]:255));
        clean.put(char(randomPlies));clean.put(char(n));clean.put(char(diff));clean.put(char(group));
        for(const auto& row:rows){
            little(samples,row.c.own,8);little(samples,row.c.opp,8);little(samples,row.c.legal,8);little(samples,gameHash,8);
            samples.put(char(row.c.move));samples.put(char(outcome*row.turn));samples.put(char(row.ply));samples.put(char(randomPlies));
            samples.put(char(diff*row.turn));little(samples,0,3);++sampled;
        }
        ++accepted;++splitCounts[group];passes+=gamePasses;early+=n<60;
    }
    clean.flush();samples.flush();if(!clean||!samples||std::cin.bad())throw std::runtime_error("I/O failure");
    std::cout<<"{\"records\":"<<records<<",\"accepted\":"<<accepted<<",\"sampled\":"<<sampled
        <<",\"excludedHoldout\":"<<excludedHoldout<<",\"forcedPasses\":"<<passes<<",\"earlyTerminal\":"<<early
        <<",\"eligibleOccurrencesBeforeRejection\":"<<eligible<<",\"splits\":["<<splitCounts[0]<<','<<splitCounts[1]<<','<<splitCounts[2]<<"],\"rejected\":{";
    bool comma=false;for(auto [reason,n]:rejects){if(comma)std::cout<<',';std::cout<<'"'<<reason<<"\":"<<n;comma=true;}
    std::cout<<"}}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

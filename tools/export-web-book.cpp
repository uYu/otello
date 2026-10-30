// SPDX-License-Identifier: GPL-3.0-or-later
// Independent converter for the pinned Egaroucid egbk3 data format.
#include "othello.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
using namespace othello;
Bits horizontal(Bits x){x=((x>>1)&0x5555555555555555ULL)|((x&0x5555555555555555ULL)<<1);x=((x>>2)&0x3333333333333333ULL)|((x&0x3333333333333333ULL)<<2);return ((x>>4)&0x0f0f0f0f0f0f0f0fULL)|((x&0x0f0f0f0f0f0f0f0fULL)<<4);}
Bits transpose(Bits x){Bits t=(x^(x<<28))&0x0f0f0f0f00000000ULL;x^=t^(t>>28);t=(x^(x<<14))&0x3333000033330000ULL;x^=t^(t>>14);t=(x^(x<<7))&0x5500550055005500ULL;return x^t^(t>>7);}
Bits symmetry(Bits x,int s){if(s&4)x=transpose(x);if(s&1)x=horizontal(x);if(s&2)x=__builtin_bswap64(x);return x;}
struct Key{Bits own,opp;bool operator==(const Key& b)const{return own==b.own&&opp==b.opp;}};
struct Hash{std::size_t operator()(const Key& k)const{Bits a=k.own^(k.opp*0x9e3779b97f4a7c15ULL);return splitmix(a);}};
Key canonical(Key key){Key best=key;for(int s=1;s<8;++s){Key k{symmetry(key.own,s),symmetry(key.opp,s)};if(k.own<best.own||(k.own==best.own&&k.opp<best.opp))best=k;}return best;}
struct Value{int score,level;};
struct Entry{std::string key;std::vector<int> moves;};
std::string text(Key k){std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<k.own<<std::setw(16)<<k.opp;return out.str();}
unsigned shard(const std::string& key){std::uint32_t h=2166136261U;for(unsigned char c:key)h=(h^c)*16777619U;return h&4095;}
std::uint64_t little(const unsigned char* p,int n){std::uint64_t v=0;for(int i=0;i<n;++i)v|=std::uint64_t(p[i])<<(i*8);return v;}
int main(int argc,char** argv){
 if(argc!=3){std::cerr<<"Usage: export-web-book INPUT.egbk3 OUTPUT_DIRECTORY\n";return 1;}
 for(int s=0;s<8;++s)for(int i=0;i<64;++i){int x=i%8,y=i/8;if(s&4)std::swap(x,y);if(s&1)x=7-x;if(s&2)y=7-y;if(symmetry(1ULL<<i,s)!=(1ULL<<(y*8+x)))throw std::runtime_error("Bad symmetry");}
 std::ifstream input(argv[1],std::ios::binary);unsigned char header[14]{};input.read(reinterpret_cast<char*>(header),14);
 if(!input||std::string(reinterpret_cast<char*>(header),9)!="DICUORAGE"||header[9]!=3)throw std::runtime_error("Bad book header");
 auto records=little(header+10,4);std::unordered_map<Key,Value,Hash> values;values.reserve(records*2);
 for(std::uint64_t i=0;i<records;++i){unsigned char row[25];input.read(reinterpret_cast<char*>(row),25);if(!input)throw std::runtime_error("Truncated book");Key k{little(row,8),little(row+8,8)};int v=static_cast<std::int8_t>(row[16]),level=row[17];if((k.own&k.opp)||v< -64||v>64)continue;k=canonical(k);auto found=values.find(k);if(found==values.end()||level>=found->second.level)values[k]={v,level};}
 if(input.get()!=EOF)throw std::runtime_error("Trailing book data");
 std::array<std::vector<Entry>,4096> output;std::uint64_t entries=0,edges=0;
 for(const auto& item:values){Key key=item.first;Position p;p.black=key.own;p.white=key.opp;p.turn=1;Bits moves=legal(p);std::vector<std::pair<int,int>> links;int best=-1000;
  while(moves){int m=first(moves);moves&=moves-1;Bits f=flips(p,m);Key child{p.white&~f,p.black|f|(1ULL<<m)};int sign=-1;if(!legal(child.own,child.opp)){std::swap(child.own,child.opp);sign=1;}auto found=values.find(canonical(child));if(found!=values.end()){int v=sign*found->second.score;best=std::max(best,v);links.emplace_back(m,v);}}
  // Same book-loss cutoff as pinned get_random; fixed variety 1 keeps a 2-disc window.
  if(links.empty()||best<item.second.score-8)continue;
  Entry e;e.key=text(key);e.moves.push_back(best);for(auto [move,value]:links)if(value>=best-2){e.moves.push_back(move);e.moves.push_back(value);++edges;}
  output[shard(e.key)].push_back(std::move(e));++entries;
 }
 std::filesystem::create_directories(argv[2]);
 for(unsigned bucket=0;bucket<4096;++bucket){auto& rows=output[bucket];std::sort(rows.begin(),rows.end(),[](const auto&a,const auto&b){return a.key<b.key;});std::ostringstream name;name<<std::hex<<std::setfill('0')<<std::setw(3)<<bucket<<".json";std::ofstream out(std::filesystem::path(argv[2])/name.str());out<<'{';bool first=true;for(const auto&e:rows){if(!first)out<<',';first=false;out<<'"'<<e.key<<"\":[";for(std::size_t j=0;j<e.moves.size();++j){if(j)out<<',';out<<e.moves[j];}out<<']';}out<<"}\n";}
 std::cout<<"{\"sourceRecords\":"<<records<<",\"validPositions\":"<<values.size()<<",\"entries\":"<<entries<<",\"moves\":"<<edges<<",\"shards\":4096}\n";
}

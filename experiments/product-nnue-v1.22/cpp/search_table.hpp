#pragma once
#include "othello.hpp"
#include <limits>

namespace othello::detail {
enum class Bound : std::uint8_t { Exact, Lower, Upper };
struct Entry {
    Bits own=0, opp=0;
    int score=0;
    std::uint32_t generation=0;
    std::int8_t depth=-1, move=-1;
    Bound bound=Bound::Exact;
    bool matches(const Position& p) const {return depth>=0&&own==p.own()&&opp==p.opponent();}
};

// Four entries per bucket, with the same 8 MiB entry budget as v1.
class TranspositionTable {
    using Bucket=std::array<Entry,4>;
    std::vector<Bucket> buckets;
    std::uint32_t generation=0;
    Bucket& bucket(const Position& p){return buckets[std::size_t(p.hash)&(buckets.size()-1)];}
public:
    explicit TranspositionTable(std::size_t bucketCount=1<<16):buckets(bucketCount){
        if(!bucketCount||(bucketCount&(bucketCount-1)))throw std::invalid_argument("TT bucket count must be a power of two");
    }
    void clear(){for(auto& b:buckets)b=Bucket{};generation=0;}
    void newSearch(){if(++generation==0){clear();generation=1;}}
    const Entry* probe(const Position& p){
        for(auto& entry:bucket(p))if(entry.matches(p))return &entry;
        return nullptr;
    }
    void store(const Position& p,int depth,int score,int move,Bound bound){
        auto& b=bucket(p);Entry* victim=nullptr;
        for(auto& entry:b){
            if(entry.matches(p)){
                // Keep deeper results and exact values when a narrow re-search
                // only produces a bound at the same horizon.
                entry.generation=generation;
                if(entry.depth>depth||(entry.depth==depth&&entry.bound==Bound::Exact&&bound!=Bound::Exact))return;
                victim=&entry;break;
            }
        }
        if(!victim){
            int lowest=std::numeric_limits<int>::max();
            for(auto& entry:b){
                if(entry.depth<0){victim=&entry;break;}
                const auto age=std::min<std::uint32_t>(generation-entry.generation,64);
                const int quality=entry.depth+(entry.bound==Bound::Exact?2:0)-int(age)*8;
                if(quality<lowest){lowest=quality;victim=&entry;}
            }
        }
        *victim={p.own(),p.opponent(),score,generation,std::int8_t(depth),std::int8_t(move),bound};
    }
};
static_assert(sizeof(Entry)==32,"Keep the TT entry budget stable");
}

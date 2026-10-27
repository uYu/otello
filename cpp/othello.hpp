#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace othello {
namespace nnue {struct Network;}
using Bits = std::uint64_t;
constexpr Bits notA = 0xfefefefefefefefeULL, notH = 0x7f7f7f7f7f7f7f7fULL;
constexpr Bits corners = 0x8100000000000081ULL;
// Reproducible Zobrist keys, generated with SplitMix64. Hashes include side to move.
constexpr Bits splitmix(Bits& seed) {
    Bits z = (seed += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
constexpr std::array<Bits,129> makeZobrist() {
    std::array<Bits,129> keys{};Bits seed=0x4f5448454c4c4f31ULL;
    for(auto& key:keys)key=splitmix(seed);return keys;
}
inline constexpr auto zobristKeys=makeZobrist();
inline Bits hashBoard(Bits black, Bits white, int turn) {
    Bits h=turn==-1?zobristKeys[128]:0;
    for(int i=0;i<64;++i){if(black>>i&1)h^=zobristKeys[i];if(white>>i&1)h^=zobristKeys[64+i];}return h;
}
inline int count(Bits b) { return __builtin_popcountll(b); }
inline int first(Bits b) { return __builtin_ctzll(b); }
inline Bits shift(Bits b, int d) {
    switch(d) {
        case 0: return (b & notH) << 1;
        case 1: return (b & notA) >> 1;
        case 2: return b << 8;
        case 3: return b >> 8;
        case 4: return (b & notH) << 9;
        case 5: return (b & notA) << 7;
        case 6: return (b & notH) >> 7;
        default: return (b & notA) >> 9;
    }
}
struct Position {
    Bits black = (1ULL << 28) | (1ULL << 35);
    Bits white = (1ULL << 27) | (1ULL << 36);
    int turn = 1; // 1 = black, -1 = white. A1 is bit zero.
    Bits hash = hashBoard(black,white,turn);
    Bits own() const { return turn == 1 ? black : white; }
    Bits opponent() const { return turn == 1 ? white : black; }
    int empty() const { return 64 - count(black | white); }
};
inline Bits legal(Bits own, Bits opp) {
    Bits result = 0, empty = ~(own | opp);
    for(int d = 0; d < 8; ++d) {
        Bits captured = shift(own, d) & opp;
        for(int n = 0; n < 5; ++n) captured |= shift(captured, d) & opp;
        result |= shift(captured, d) & empty;
    }
    return result;
}
inline Bits legal(const Position& p) { return legal(p.own(), p.opponent()); }
inline Bits flips(const Position& p, int move) {
    if(move < 0 || move > 63 || ((p.black | p.white) & (1ULL << move))) return 0;
    Bits result = 0, own = p.own(), opp = p.opponent();
    for(int d = 0; d < 8; ++d) {
        Bits ray = shift(1ULL << move, d), captured = 0;
        while(ray & opp) { captured |= ray; ray = shift(ray, d); }
        if(ray & own) result |= captured;
    }
    return result;
}
inline bool over(const Position& p) { return !legal(p.black, p.white) && !legal(p.white, p.black); }
// A pass is explicit in the API and accepted only when the current side cannot move.
inline Position play(const Position& p, int move) {
    Position next = p;
    if(move == -1) {
        if(legal(p) || over(p)) throw std::invalid_argument("Pass is not legal");
    } else {
        Bits f = flips(p, move);
        if(!f) throw std::invalid_argument("Illegal move");
        if(p.turn == 1) { next.black |= f | (1ULL << move); next.white &= ~f; }
        else { next.white |= f | (1ULL << move); next.black &= ~f; }
        next.hash ^= zobristKeys[(p.turn==1?0:64)+move];
        Bits changed=f;
        while(changed){int i=first(changed);changed&=changed-1;next.hash^=zobristKeys[i]^zobristKeys[64+i];}
    }
    next.turn = -p.turn;
    next.hash ^= zobristKeys[128];
    return next;
}
inline std::string board(const Position& p) {
    std::string s(64, '.');
    for(int i = 0; i < 64; ++i) {
        if(p.black & (1ULL << i)) s[i] = 'B';
        else if(p.white & (1ULL << i)) s[i] = 'W';
    }
    return s;
}
inline Position parse(const std::string& s, int turn) {
    if(s.size() != 64 || (turn != 1 && turn != -1)) throw std::invalid_argument("Expected 64 board cells and turn 1 or -1");
    Position p; p.black = p.white = 0; p.turn = turn;
    for(int i = 0; i < 64; ++i) {
        if(s[i] == 'B' || s[i] == 'b' || s[i] == 'X') p.black |= 1ULL << i;
        else if(s[i] == 'W' || s[i] == 'w' || s[i] == 'O') p.white |= 1ULL << i;
        else if(s[i] != '.' && s[i] != '-') throw std::invalid_argument("Invalid board character");
    }
    p.hash=hashBoard(p.black,p.white,p.turn);return p;
}
inline std::string coordinate(int i) { return i == -1 ? "pass" : std::string{char('A' + i % 8), char('1' + i / 8)}; }
inline int square(std::string s) {
    if(s == "pass" || s == "-1") return -1;
    if(s.size() == 2 && s[0] >= 'a' && s[0] <= 'h') s[0] -= 'a' - 'A';
    if(s.size() != 2 || s[0] < 'A' || s[0] > 'H' || s[1] < '1' || s[1] > '8') throw std::invalid_argument("Expected A1-H8 or pass");
    return (s[1] - '1') * 8 + s[0] - 'A';
}
struct Candidate { int move, score; };
struct SearchResult {
    int move = -1, score = 0, depth = 0, elapsed = 0;
    std::uint64_t nodes = 0;
    std::uint64_t ttHits = 0;
    std::uint64_t ttCutoffs = 0, betaCutoffs = 0, pvsResearches = 0, aspirationRetries = 0, evaluations = 0;
    std::uint64_t accumulatorUpdates = 0;
    bool exact = false;
    std::vector<Candidate> candidates;
};
struct SearchOptions { int milliseconds = 800, maxDepth = 64, level = 2; std::uint32_t seed = 1; const nnue::Network* network = nullptr; };
struct OpeningPolicy { int plies=12, temperature=32, maxLoss=64; };
struct MoveSelection {
    int move=-1, score=0, bestScore=0, eligible=0, temperatureMilli=0, lossLimit=0;
    bool sampled=false;
};
// Select from a fully completed root iteration. Search scores/TT stay untouched.
// Plies count placed discs since the standard four-disc start; passes do not count.
MoveSelection selectMove(const Position& position,const SearchResult& analysis,
                         OpeningPolicy policy,std::uint32_t seed);
SearchResult search(const Position& position, SearchOptions options = {});
// Searches on one thread reuse an 8 MiB TT. Changing evaluator contents clears
// it automatically. Scores are reused only at the requested horizon.
void clearSearchCache(); // clears the calling thread's TT, e.g. for a cold benchmark
int evaluate(const Position& position);
}

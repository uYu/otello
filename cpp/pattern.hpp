#pragma once
#include "nnue.hpp"

namespace othello::nnue {
// Independent Othello experiment: ternary line/corner embeddings, three pooled
// regions, phase-conditioned trunk, disc/WDL and global-conditioned policy heads.
// All parameters use scale 256. This is not a Rapfi weight format.
struct Pattern {
    static constexpr int Dim=32, Hidden=32, Count=38, Rows=55728, Inputs=97, Outputs=46;
    struct Shape {std::array<int,9> squares{};int length=0,offset=0,group=0;};
    std::vector<std::int16_t> embedding;
    std::array<std::int16_t,Hidden*Inputs> trunk{};
    std::array<std::int32_t,Hidden> trunkBias{};
    // disc, win/loss/draw logits, 32 policy gates, 10 positional biases
    std::array<std::int16_t,Outputs*Hidden> head{};
    std::array<std::int32_t,Outputs> headBias{};
    static const std::array<Shape,Count>& shapes();
    static int code(const Shape& shape,Bits own,Bits opponent);
    static std::shared_ptr<Pattern> load(std::istream& input);
    Accumulator refresh(const Position& p) const;
    Accumulator update(const Accumulator& acc,const Position& before,const Position& after) const;
    std::array<int,Outputs> outputs(const Accumulator& acc,int turn,int count=Outputs) const;
    int evaluate(const Accumulator& acc,int turn) const;
    // Policy logits are unnormalized; callers mask illegal moves before softmax.
    std::array<double,64> policy(const Position& p) const;
    std::array<double,64> policy(const Position& p,const std::array<int,Outputs>& outputs) const;
    bool operator==(const Pattern& other) const;
};
}

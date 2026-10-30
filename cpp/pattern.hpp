#pragma once
#include "nnue.hpp"

namespace othello::nnue {
// Independent Othello experiment: ternary line/corner embeddings, three pooled
// regions, phase-conditioned trunk, scalar/WDL and global-conditioned policy heads.
// All parameters use scale 256. This is not a Rapfi weight format.
struct Pattern {
    static constexpr int Dim=32, Hidden=32, MaxHidden=64, Count=38, Rows=55728, Inputs=97, Outputs=46;
    struct State {
        std::array<std::array<std::uint16_t,Count>,2> codes{};
        std::array<std::array<std::int32_t,96>,2> sums{};
        Bits blackBoard=0,whiteBoard=0;
    };
    struct Prediction {std::array<int,Outputs> values{};std::array<double,64> logits{};};
    struct Shape {std::array<int,9> squares{};int length=0,offset=0,group=0;};
    int hidden=Hidden;
    std::vector<std::int16_t> embedding;
    // OTPAT3 stores only the embedding in INT8; reconstructed units remain 1/256.
    std::vector<std::int8_t> embedding8;
    int embeddingStep=1;
    int embeddingValue(int index) const {return embedding8.empty()?embedding[index]:int(embedding8[index])*embeddingStep;}
    std::array<std::int16_t,MaxHidden*Inputs> trunk{};
    std::array<std::int32_t,MaxHidden> trunkBias{};
    // scalar value (target in Network), win/loss/draw logits, 32 policy gates, 10 positional biases
    std::array<std::int16_t,Outputs*MaxHidden> head{};
    std::array<std::int32_t,Outputs> headBias{};
    static const std::array<Shape,Count>& shapes();
    static int code(const Shape& shape,Bits own,Bits opponent);
    static std::array<int,Count> encode(Bits own,Bits opponent);
    static std::shared_ptr<Pattern> load(std::istream& input,bool int8=false);
    Accumulator refresh(const Position& p) const;
    Accumulator update(const Accumulator& acc,const Position& before,const Position& after) const;
    std::array<int,Outputs> outputs(const Accumulator& acc,int turn,int count=Outputs) const;
    int evaluate(const Accumulator& acc,int turn) const;
    Prediction predict(const Position& position) const;
    // Only masked policy logits are computed; other logits are zero. Values unchanged.
    Prediction predictLegal(const Position& position,Bits policyMask) const;
    State refreshState(const Position& position) const;
    State updateState(const State& before,const Position& after,unsigned* changedPatterns=nullptr) const;
    Prediction predict(const State& state,int turn) const;
    Prediction predictLegal(const State& state,int turn,Bits policyMask) const;
    // Policy logits are unnormalized; callers mask illegal moves before softmax.
    std::array<double,64> policy(const Position& p) const;
    std::array<double,64> policy(const Position& p,const std::array<int,Outputs>& outputs) const;
    bool operator==(const Pattern& other) const;
private:
    std::array<int,Outputs> outputs(const std::int32_t* sums,Bits occupied,int count=Outputs) const;
};
}

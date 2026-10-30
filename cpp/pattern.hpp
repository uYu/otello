#pragma once
#include "nnue.hpp"

namespace othello::nnue {
// Independent Othello experiment: ternary line/corner embeddings, three pooled
// regions, phase-conditioned trunk, scalar/WDL and global-conditioned policy heads.
// All parameters use scale 256. This is not a Rapfi weight format.
struct Pattern {
    static constexpr int Dim=32, Hidden=32, MaxHidden=64, Count=38, Rows=55728, Inputs=97, MaxInputs=321, Outputs=46;
    struct State {
        std::array<std::array<std::uint16_t,Count>,2> codes{};
        std::array<std::array<std::int32_t,96>,2> sums{};
        Bits blackBoard=0,whiteBoard=0;
    };
    struct Prediction {std::array<int,Outputs> values{};std::array<double,64> logits{};};
    struct Shape {std::array<int,9> squares{};int length=0,offset=0,group=0;};
    int hidden=Hidden;
    // OTPAT4 independent structural ablations: 1=tied D4 table, 2=ten pools.
    int variant=0,inputs=Inputs,rows=Rows,policyWidth=8;
    std::vector<std::uint16_t> canonical;
    std::vector<std::int16_t> embedding;
    // OTPAT3 stores only the embedding in INT8; reconstructed units remain 1/256.
    std::vector<std::int8_t> embedding8;
    int embeddingStep=1;
    int embeddingOffset(int code) const {return (canonical.empty()?code:canonical[code])*Dim;}
    int embeddingValue(int index) const {if(!canonical.empty())index=embeddingOffset(index/Dim)+index%Dim;return embedding8.empty()?embedding[index]:int(embedding8[index])*embeddingStep;}
    std::array<std::int16_t,MaxHidden*MaxInputs> trunk{};
    std::array<std::int32_t,MaxHidden> trunkBias{};
    // scalar value (target in Network), win/loss/draw logits, 32 policy gates, 10 positional biases
    std::array<std::int16_t,Outputs*MaxHidden> head{};
    std::array<std::int32_t,Outputs> headBias{};
    // OTPAT5: ordinary shared [local32, global32, position10] -> 8 -> 1 MLP.
    std::array<std::int16_t,32*74> policyHidden{};
    std::array<std::int32_t,32> policyHiddenBias{};
    std::array<std::int16_t,32> policyOutput{};
    std::array<std::int32_t,1> policyOutputBias{};
    // OTPAT6: opening value/WLD readout, blended with head[0:4] by empties.
    int phaseLow=36,phaseHigh=48;
    std::array<std::int16_t,4*Hidden> earlyValue{};
    std::array<std::int32_t,4> earlyValueBias{};
    // OTPAT7: opening gates/biases, blended into the original policy readout.
    std::array<std::int16_t,42*Hidden> earlyPolicy{};
    std::array<std::int32_t,42> earlyPolicyBias{};
    // OTPAT9 global policy: fixed38->16->64; expanded D4-tied weights.
    std::array<std::int16_t,16*38> globalDown{};
    std::array<std::int32_t,16> globalDownBias{};
    std::array<std::int16_t,64*16> globalUp{};
    std::array<std::int32_t,64> globalUpBias{};
    static const std::array<Shape,Count>& shapes();
    static int code(const Shape& shape,Bits own,Bits opponent);
    static std::array<int,Count> encode(Bits own,Bits opponent);
    static std::shared_ptr<Pattern> load(std::istream& input,bool int8=false,bool structural=false,bool directPolicy=false,bool phaseValue=false,bool phasePolicy=false,bool globalPolicy=false);
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
    Prediction predictStructural(const Position& position,Bits mask,const std::array<int,Outputs>* known=nullptr) const;
};
}

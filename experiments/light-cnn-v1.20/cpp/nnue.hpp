#pragma once
#include "othello.hpp"
#include <memory>
#include <iosfwd>

namespace othello::nnue {
constexpr int Features=128, Hidden=32, MaxHidden=256, Scale=256;
enum class Architecture { Legacy=1, Turn=2, Cnn=3, Residual=4, CnnPolicy=5 };
enum class ValueTarget { DiscDifference, Outcome };
struct Cnn;
struct CnnPolicy;
struct ResNet;
struct TrunkBlock {
    static constexpr int Width=32;
    std::array<std::array<std::int16_t,256>,Width> down{};
    std::array<std::int32_t,Width> downBias{};
    std::array<std::array<std::int16_t,Width>,256> up{};
    std::array<std::int32_t,256> upBias{};
    bool operator==(const TrunkBlock& b) const{return down==b.down&&downBias==b.downBias&&up==b.up&&upBias==b.upBias;}
};
struct PolicyHead {
    static constexpr int Width=32, Moves=65;
    std::array<std::array<std::int16_t,512>,Width> weights{};
    std::array<std::int32_t,Width> bias{}; // scale^2
    std::array<std::array<std::int16_t,Width>,Moves> output{};
    std::array<std::int32_t,Moves> outputBias{}; // scale^2
};
struct Accumulator { std::array<std::int32_t,MaxHidden> black{},white{}; Bits blackBoard=0,whiteBoard=0; };
struct Inference {int score=0;std::array<double,65> probabilities{};};
// Othello-specific quantized network, not a Stockfish .nnue file.
// Color perspectives share a feature transformer. V1 subtracts perspectives;
// v2 orders them by side to move and uses independent output weights.
struct Network {
    Architecture architecture=Architecture::Legacy;
    ValueTarget valueTarget=ValueTarget::DiscDifference;
    int hidden=Hidden;
    std::array<std::array<std::int16_t,MaxHidden>,Features> weights{};
    std::array<std::int32_t,MaxHidden> bias{};
    std::array<std::int16_t,MaxHidden> output{};
    std::array<std::int16_t,MaxHidden> opponentOutput{};
    std::int32_t outputBias=0; // v2 output bias, scaled by Scale * Scale
    std::shared_ptr<const Cnn> cnn;
    std::shared_ptr<const CnnPolicy> cnnPolicy;
    std::shared_ptr<const ResNet> residual;
    std::shared_ptr<const PolicyHead> policyHead;
    std::shared_ptr<const TrunkBlock> trunkBlock;
    bool spatial() const{return architecture==Architecture::Cnn||architecture==Architecture::Residual||architecture==Architecture::CnnPolicy;}
    bool hasPolicy() const{return bool(policyHead)||bool(cnnPolicy);}
    bool sameParameters(const Network& other) const;
    std::unique_ptr<Network> snapshot() const;
    static Network load(const std::string& path);
    static Network load(std::istream& input);
    Accumulator refresh(const Position& position) const;
    Accumulator update(const Accumulator& parent,const Position& before,const Position& after) const;
    int evaluate(const Accumulator& accumulator,int turn) const;
    int evaluate(const Position& position) const;
    int evaluateReference(const Position& position) const;
    // Masked legal probabilities; index 64 is forced pass, terminal is all zero.
    std::array<double,65> policy(const Position& position) const;
    Inference infer(const Position& position) const;
    // Internal search entry: reuse already computed legality/terminal status.
    Inference infer(const Position& position,Bits legalMoves,bool terminal) const;
private:
    std::array<double,65> policyFromFeatures(const Accumulator& accumulator,int turn,Bits moves,
        const std::array<double,256>* deepOwn=nullptr,const std::array<double,256>* deepOpp=nullptr) const;
};
}

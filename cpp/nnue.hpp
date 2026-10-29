#pragma once
#include "othello.hpp"
#include <memory>
#include <iosfwd>

namespace othello::nnue {
constexpr int Features=128, Hidden=32, MaxHidden=256, Scale=256;
enum class Architecture { Legacy=1, Turn=2, Cnn=3, Residual=4, Pattern=5 };
struct Cnn;
struct ResNet;
struct Pattern;
struct Accumulator { std::array<std::int32_t,MaxHidden> black{},white{}; Bits blackBoard=0,whiteBoard=0; };
// Othello-specific quantized network, not a Stockfish .nnue file.
// Color perspectives share a feature transformer. V1 subtracts perspectives;
// v2 orders them by side to move and uses independent output weights.
struct Network {
    Architecture architecture=Architecture::Legacy;
    int hidden=Hidden;
    std::array<std::array<std::int16_t,MaxHidden>,Features> weights{};
    std::array<std::int32_t,MaxHidden> bias{};
    std::array<std::int16_t,MaxHidden> output{};
    std::array<std::int16_t,MaxHidden> opponentOutput{};
    std::int32_t outputBias=0; // v2 output bias, scaled by Scale * Scale
    std::shared_ptr<const Cnn> cnn;
    std::shared_ptr<const ResNet> residual;
    std::shared_ptr<const Pattern> pattern;
    bool spatial() const{return architecture==Architecture::Cnn||architecture==Architecture::Residual;}
    bool sameParameters(const Network& other) const;
    std::unique_ptr<Network> snapshot() const;
    static Network load(const std::string& path);
    static Network load(std::istream& input);
    Accumulator refresh(const Position& position) const;
    Accumulator update(const Accumulator& parent,const Position& before,const Position& after) const;
    int evaluate(const Accumulator& accumulator,int turn) const;
    int evaluate(const Position& position) const;
    int evaluateReference(const Position& position) const;
};
}

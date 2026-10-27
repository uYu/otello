#pragma once
#include "othello.hpp"

namespace othello::nnue {
constexpr int Features=128, Hidden=32, MaxHidden=256, Scale=256;
struct Accumulator { std::array<std::int32_t,MaxHidden> black{},white{}; };
// Othello-specific quantized network, not a Stockfish .nnue file.
// Two color perspectives share a feature transformer and an output head.
struct Network {
    int hidden=Hidden;
    std::array<std::array<std::int16_t,MaxHidden>,Features> weights{};
    std::array<std::int32_t,MaxHidden> bias{};
    std::array<std::int16_t,MaxHidden> output{};
    static Network load(const std::string& path);
    Accumulator refresh(const Position& position) const;
    Accumulator update(const Accumulator& parent,const Position& before,const Position& after) const;
    int evaluate(const Accumulator& accumulator,int turn) const;
    int evaluate(const Position& position) const {return evaluate(refresh(position),position.turn);}
    int evaluateReference(const Position& position) const;
};
}

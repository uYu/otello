#pragma once
#include "othello.hpp"
#include <iosfwd>
#include <memory>

namespace othello::nnue {
// Experimental 8x8 value CNN. NCHW layout, 3x3 convolutions, zero padding,
// ReLU, flatten -> 32 -> 1 -> tanh. Inputs are own/opponent occupancy.
struct Cnn {
    static constexpr int Channels=8, Head=32, Parameters=17185;
    std::array<float,Parameters> weights{};
    static std::shared_ptr<const Cnn> load(std::istream& input);
    int evaluate(Bits black,Bits white,int turn) const;
    int evaluateReference(Bits black,Bits white,int turn) const;
};
}

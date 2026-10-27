#pragma once
#include "othello.hpp"
#include <iosfwd>
#include <memory>

namespace othello::nnue {
// Side-to-move 8x8 residual value network. Batch normalization is folded into
// convolution weights/biases at export. NCHW, zero padding, post-add ReLU.
struct ResNet {
    static constexpr int Channels=32, Head=64;
    static constexpr int parameters(int blocks){return 4866+18496*blocks;}
    int blocks=2;
    std::vector<float> weights=std::vector<float>(parameters(2));
    static std::shared_ptr<const ResNet> load(std::istream& input);
    int evaluate(Bits black,Bits white,int turn) const;
    int evaluateReference(Bits black,Bits white,int turn) const;
};
}

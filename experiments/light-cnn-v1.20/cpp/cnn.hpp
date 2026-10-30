#pragma once
#include "othello.hpp"
#include <iosfwd>
#include <memory>

namespace othello::nnue {
struct Inference;
// Experimental 8x8 value CNN. NCHW layout, 3x3 convolutions, zero padding,
// ReLU, flatten -> 32 -> 1 -> tanh. Inputs are own/opponent occupancy.
struct Cnn {
    static constexpr int Channels=8, Head=32, Parameters=17185;
    std::array<float,Parameters> weights{};
    static std::shared_ptr<const Cnn> load(std::istream& input);
    int evaluate(Bits black,Bits white,int turn) const;
    int evaluateReference(Bits black,Bits white,int turn) const;
};
// OTCNN2: two 3x3 convolutions, shared dense64, outcome and learned policy.
struct CnnPolicy {
    int channels=32;
    int headWidth=64;
    std::vector<float> weights;
    static std::shared_ptr<const CnnPolicy> load(std::istream& input,bool compact=false);
    Inference infer(Bits black,Bits white,int turn,Bits moves,bool terminal,bool reference=false) const;
    bool operator==(const CnnPolicy& b) const{return channels==b.channels&&headWidth==b.headWidth&&weights==b.weights;}
};
}

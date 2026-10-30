#pragma once
#include "othello.hpp"

namespace othello {
struct MctsOptions {
    int milliseconds=1000;
    std::uint64_t simulations=0; // zero: time only; positive: also cap new simulations
    std::uint32_t seed=1;
    const nnue::Network* network=nullptr;
    double exploration=1.5;
    std::size_t maxNodes=2097152;
};
struct MctsResult {
    SearchResult analysis;
    std::vector<std::uint64_t> visits; // same order as analysis.candidates
    std::vector<double> values; // unrounded root-player means, for decision audit
    std::uint64_t reusedNodes=0,treeNodes=0;
    double meanDepth=0;
    bool memoryLimited=false;
};
// PUCT uses learned legal priors when a policy head is present, uniform otherwise.
// Every unvisited child is sampled once (learned-prior order, seeded ties).
// Values follow the model target: disc difference / 64 or win/draw/loss expectation.
// No rollouts or alpha-beta fallback.
MctsResult mcts(const Position& position,MctsOptions options={});
void clearMctsCache();
}

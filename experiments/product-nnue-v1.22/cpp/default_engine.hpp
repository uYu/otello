#pragma once
#include "nnue.hpp"
namespace othello {
// Exact approved policy/value NNUE weights embedded for offline native/Wasm use.
const nnue::Network& defaultNetwork();
const char* defaultNetworkSha256();
}

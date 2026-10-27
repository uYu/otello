#pragma once
#include "nnue.hpp"
namespace othello {
// Exact frozen v1.3 NNUE weights embedded for offline native/Wasm use.
const nnue::Network& defaultNetwork();
const char* defaultNetworkSha256();
}

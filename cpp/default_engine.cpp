#include "default_engine.hpp"
#include <sstream>
namespace othello {
namespace {
#include "model/default_nnue.inc"
}
const nnue::Network& defaultNetwork(){
    static const auto network=[](){
        std::string bytes(reinterpret_cast<const char*>(defaultModelBytes),sizeof(defaultModelBytes));
        std::istringstream input(bytes,std::ios::in|std::ios::binary);
        return std::make_unique<nnue::Network>(nnue::Network::load(input));
    }();
    return *network;
}
const char* defaultNetworkSha256(){return defaultModelSha256;}
}

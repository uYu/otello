#pragma once
#include <cstdint>
namespace othello::nnue {struct CnnPolicy;struct Network;}
extern thread_local const othello::nnue::CnnPolicy* activeBatchCnn;
extern thread_local const othello::nnue::Network* activeBatchNetwork;
void prepareNativeModel(const othello::nnue::Network& model);
int nativeCnnBatch(int n,const int* boards,const int* turns,const std::uint64_t* legal,int* scores,double* probabilities);

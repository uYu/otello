#pragma once
#include <cstdint>
namespace othello::nnue {struct CnnPolicy;}
extern thread_local const othello::nnue::CnnPolicy* activeBatchCnn;
int nativeCnnBatch(int n,const int* boards,const int* turns,const std::uint64_t* legal,int* scores,double* probabilities);

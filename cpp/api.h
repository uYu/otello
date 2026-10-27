#pragma once
// Portable C ABI. All buffers are owned by the caller. No WebAssembly dependency.
#ifdef __cplusplus
extern "C" {
#endif
// cells[64]: 0 empty, 1 black, -1 white. turn: 1 black or -1 white.
// Return -1 for invalid arguments, otherwise the number of legal moves written.
int ot_legal(const int* cells, int turn, int* moves);
// move: 0..63, or -1 for a forced pass. Returns 1 if accepted, 0 if illegal.
// output[67]: cells[0..63], next turn, game over, number of flipped discs.
int ot_play(const int* cells, int turn, int move, int* output);
// output[136]: move, score, completed depth, elapsed ms, nodes (capped at INT_MAX),
// exact (0/1), candidate count, then (move, score) pairs; output[135] = TT hits.
// Return 1 or -1 on error.
int ot_search(const int* cells, int turn, int milliseconds, int max_depth, int level, unsigned seed, int* output);
// Optional native NNUE evaluator. Models are immutable after loading and may be
// shared across independent searches. Keep the handle alive until calls finish.
typedef struct ot_nnue ot_nnue;
ot_nnue* ot_nnue_load(const char* path); // null on error
void ot_nnue_destroy(ot_nnue* model);
int ot_nnue_evaluate(const ot_nnue* model, const int* cells, int turn, int* score);
int ot_nnue_search(const ot_nnue* model, const int* cells, int turn, int milliseconds,
                   int max_depth, int level, unsigned seed, int* output);
const char* ot_nnue_error(void); // per-thread message, valid until the next NNUE call
#ifdef __cplusplus
}
#endif

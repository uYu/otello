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
// Searches cache transpositions per calling thread. Clear for a new session or
// a cold benchmark; evaluator changes invalidate cached scores automatically.
void ot_search_clear_cache(void);
// Optional native NNUE evaluator. Models are immutable after loading and may be
// shared across independent searches. Keep the handle alive until calls finish.
typedef struct ot_nnue ot_nnue;
ot_nnue* ot_nnue_load(const char* path); // null on error
void ot_nnue_destroy(ot_nnue* model);
int ot_nnue_evaluate(const ot_nnue* model, const int* cells, int turn, int* score);
// Pattern experiment diagnostics: output[68] = normalized disc estimate,
// win/loss/draw probabilities, then 64 raw policy logits (illegal moves unmasked).
int ot_pattern_predict(const ot_nnue* model,const int* cells,int turn,double* output);
int ot_nnue_search(const ot_nnue* model, const int* cells, int turn, int milliseconds,
                   int max_depth, int level, unsigned seed, int* output);
// NNUE gameplay with a decaying opening policy. Analysis API above remains best-move search.
// output[136] contains the selected move/score and the unchanged ranked candidates.
// policy_output[6]: eligible moves, temperature * 1000, score-loss limit,
// best score, selected score loss, sampled (0/1). Use opening_plies=0 to disable.
int ot_nnue_play(const ot_nnue* model, const int* cells, int turn, int milliseconds,
                 int max_depth, int level, unsigned seed, int opening_plies,
                 int temperature, int max_loss, int* output, int* policy_output);
const char* ot_nnue_error(void); // per-thread message, valid until the next NNUE call
// PUCT with uniform priors for value-only models and learned priors for Pattern.
// Same output[136] layout; depth is maximum
// visited path length, not a completed alpha-beta horizon. visits[64] align
// with output candidates. stats[72]: first 8 are new simulations, evaluations, mean depth
// *1000, retained nodes, tree nodes, elapsed ms, exact, memory limit hit;
// stats[8..] are unrounded root-player candidate means in the same order.
int ot_nnue_mcts_play(const ot_nnue* model, const int* cells, int turn,
                     int milliseconds, unsigned simulations, unsigned seed,
                     int opening_plies, int temperature, int max_loss,
                     int* output, int* policy_output, int* visits, double* stats);
void ot_mcts_clear_cache(void);
// Same as ot_nnue_mcts_play, with explicit learned-policy ablation (0/1).
int ot_nnue_mcts_play_ex(const ot_nnue* model,const int* cells,int turn,
                        int milliseconds,unsigned simulations,unsigned seed,
                        int opening_plies,int temperature,int max_loss,int use_policy,
                        int* output,int* policy_output,int* visits,double* stats);
// Offline default: embedded v1.3 NNUE + MCTS + the v1.3 opening policy.
// Same output[136] layout; depth is maximum visited path, output[135]=0.
int ot_default_search(const int* cells,int turn,int milliseconds,unsigned seed,int* output);
int ot_default_evaluate(const int* cells,int turn,int* score);
const char* ot_default_model_sha256(void);
#ifdef __cplusplus
}
#endif

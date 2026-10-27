#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// Native-only C ABI. nullptr on initialization failure; last_error explains why.
void* ot_egaroucid_create(const char* resources);
void ot_egaroucid_destroy(void* engine);
// cells: 0 empty, 1 black, -1 white; turn: 1 black, -1 white.
// output[8]: move, disc-value, depth, elapsed ms, nodes (capped), book-hit, exact, use-book.
// Return 1 on success, -1 on error. Pass use_book=1 to enable the opening book.
// Book selection defaults to weighted variety within 2 book-score units of the best.
int ot_egaroucid_search(void* engine, const int* cells, int turn, int level, int use_book, int* output);
// Book only: 1 hit, 0 miss (including pass/terminal), -1 error. output[8] uses
// the same layout; hit has depth=-1 and nodes=0. Does not perform AI search.
// variety 0..3; seed=0 keeps upstream randomness, nonzero fixes the book draw.
int ot_egaroucid_book(void* engine,const int* cells,int turn,int variety,unsigned seed,int* output);
// Thread-local error string; valid until this thread's next adapter call.
const char* ot_egaroucid_last_error(void);
#ifdef __cplusplus
}
#endif

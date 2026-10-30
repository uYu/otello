// SPDX-License-Identifier: GPL-3.0-or-later
// Compile the unmodified pinned upstream C API in this translation unit so a
// book-only extension can reuse its initialized Book and coordinate helpers.
#include "lib/egaroucid_c_api.cpp"

extern "C" int otello_egaroucid_book_probe(const int* cells,int player,int variety,unsigned seed,int* move,int* value){
    if(!cells||!move||!value||(player!=EGAROUCID_BLACK&&player!=EGAROUCID_WHITE)||variety<0||variety>3)
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    *move=-1;*value=0;
    std::lock_guard<std::mutex> guard(g_init_mutex);
    if(!g_initialized||!g_book_available)return EGAROUCID_ERROR_NOT_INITIALIZED;
    Board board;
    if(!board_from_array(cells,player,&board))return EGAROUCID_ERROR_INVALID_ARGUMENT;
    if(!board.get_legal())return 0;
    // Seed zero preserves normal application randomness; experiments supply a
    // per-position seed, making book choice reproducible for a fixed position.
    if(seed)raw_myrandom.seed(seed);
    auto result=book.get_random(&board,variety);
    if(!is_valid_policy(result.policy))return 0;
    *move=to_public_move(result.policy);*value=result.value;
    return 1;
}

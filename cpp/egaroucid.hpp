#pragma once
#include "othello.hpp"
#include "egaroucid_api.h"
#include <memory>

namespace othello {
struct EgaroucidResult {
    int move = -1, value = 0, depth = 0, elapsed = 0;
    std::uint64_t nodes = 0;
    bool bookHit = false, exact = false;
};
// Native-only adapter. Upstream global state is serialized across all instances.
// Initialization requires a usable book and verifies an opening-book hit.
class Egaroucid {
public:
    explicit Egaroucid(const std::string& resources = "");
    ~Egaroucid();
    Egaroucid(const Egaroucid&) = delete;
    Egaroucid& operator=(const Egaroucid&) = delete;
    // bookVariety: 0 = best-score only; 1..3 = weighted variety within 2..6 book-score units.
    EgaroucidResult analyze(const Position& position, int level = 5, bool useBook = true, int bookVariety = 1);
    // Book lookup only: false is a miss, with zero search nodes. Never runs the
    // Egaroucid evaluator/search as a fallback. Nonzero seed fixes this draw.
    bool probeBook(const Position& position,EgaroucidResult& result,int bookVariety=1,unsigned seed=0);
    std::string version() const;
    static std::string defaultResources();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}


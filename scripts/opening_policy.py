"""Independent replay/audit helpers for the native decaying opening selector."""
import math

MASK = (1 << 64) - 1


def splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, z ^ (z >> 31)


state = 0x4F5448454C4C4F31
KEYS = []
for _ in range(129):
    state, key = splitmix(state)
    KEYS.append(key)


def board_hash(board, turn):
    value = KEYS[128] if turn == -1 else 0
    for i, color in enumerate(board):
        if color:
            value ^= KEYS[i if color == 1 else 64 + i]
    return value


def choose(board, turn, candidates, seed, plies=12, temperature=32, max_loss=64, exact=False):
    """Reproduce a choice from recorded root scores, independent of native code."""
    best = max(c['score'] for c in candidates)
    baseline = next(c for c in candidates if c['score'] == best)
    meta = dict(seed=seed, eligible=1, temperatureMilli=0, lossLimit=0,
                bestScore=best, scoreLoss=0, sampled=False)
    placed = sum(bool(c) for c in board) - 4
    if not plies or not temperature or placed >= plies or exact or any(abs(c['score']) >= 100000 for c in candidates):
        return baseline['move'], baseline['score'], meta
    remaining = plies - placed
    meta['temperatureMilli'] = temperature * 1000 * remaining // plies
    meta['lossLimit'] = max_loss * remaining // plies
    candidates = sorted(candidates, key=lambda c: c['move'])
    weights = [math.exp(-(best - c['score']) * 1000 / meta['temperatureMilli'])
               if best - c['score'] <= meta['lossLimit'] else 0.0 for c in candidates]
    meta['eligible'] = sum(best - c['score'] <= meta['lossLimit'] for c in candidates)
    if meta['eligible'] <= 1:
        return baseline['move'], baseline['score'], meta
    _, draw = splitmix(((seed << 32) ^ board_hash(board, turn) ^ 0x4F54454C4C4F504C) & MASK)
    target = (draw >> 11) * 2.0 ** -53 * sum(weights)
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if weight > 0 and target < cumulative:
            meta['scoreLoss'] = best - candidate['score']
            meta['sampled'] = True
            return candidate['move'], candidate['score'], meta
    raise AssertionError('Invalid weighted draw')


def transform(move, symmetry):
    if move == -1:
        return -1
    x, y = move % 8, move // 8
    if symmetry & 4:
        x = 7 - x
    for _ in range(symmetry & 3):
        x, y = 7 - y, x
    return y * 8 + x


# Only the four transformations that preserve the initial black/white layout.
# A 90-degree rotation alone swaps the initial colors and is not a valid
# relabeling of a complete game with Black moving first.
START = {28: 1, 35: 1, 27: -1, 36: -1}
SYMMETRIES = [s for s in range(8) if {transform(m, s): c for m, c in START.items()} == START]


def canonical_moves(moves):
    return min(tuple(transform(m, s) for m in moves) for s in SYMMETRIES)

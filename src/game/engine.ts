import createModule from './wasm/othello.mjs';
import type { Analysis, Difficulty, Position } from './types.ts';
let module: Awaited<ReturnType<typeof createModule>>;
let pending: Promise<void> | undefined;
export function initEngine() { return pending ??= createModule().then(m => { module = m; }); }
function input(p: Position) {
  const ptr = module._ot_input();
  module.HEAP32.set(p.cells, ptr / 4);
  return ptr;
}
export function initial(): Position {
  const cells = Array<number>(64).fill(0);
  cells[27] = cells[36] = -1; cells[28] = cells[35] = 1;
  return { cells, turn: 1, over: false, passed: null, lastMove: null };
}
export function legal(p: Position): number[] {
  const n = module._ot_legal(input(p), p.turn, module._ot_output());
  if (n < 0) throw new Error('无效局面');
  const offset = module._ot_output() / 4;
  return Array.from(module.HEAP32.subarray(offset, offset + n));
}
export function play(p: Position, move: number): Position {
  if (p.over || module._ot_play(input(p), p.turn, move, module._ot_output()) !== 1) throw new Error('这个位置不能落子');
  const out = Array.from(module.HEAP32.subarray(module._ot_output() / 4, module._ot_output() / 4 + 67));
  const next: Position = { cells: out.slice(0, 64), turn: out[64] as 1 | -1, over: !!out[65], passed: null, lastMove: move < 0 ? p.lastMove : move };
  if (!next.over && !legal(next).length) {
    const passed = next.turn;
    return { ...play(next, -1), passed };
  }
  return next;
}
export function flips(p: Position, move: number): number[] {
  if (module._ot_play_with_flips) {
    if (p.over) return [];
    const out=module._ot_output();
    if (module._ot_play_with_flips(input(p),p.turn,move,out,out+67*4)!==1) return [];
    const start=out/4;
    return Array.from(module.HEAP32.subarray(start+67,start+67+module.HEAP32[start+66]));
  }
  if (!legal(p).includes(move)) return [];
  const next = play(p, move);
  return p.cells.flatMap((c, i) => c !== 0 && c !== next.cells[i] ? [i] : []);
}
export function score(p: Position) { return { black: p.cells.filter(c => c === 1).length, white: p.cells.filter(c => c === -1).length }; }
export function analyze(p: Position, difficulty: Difficulty, seed = 1): Analysis {
  const mcts = difficulty !== 'easy';
  const ok = mcts
    ? module._ot_default_search(input(p), p.turn, difficulty === 'hard' ? 1000 : 100, seed, module._ot_output())
    : module._ot_search(input(p), p.turn, 80, 1, 0, seed, module._ot_output());
  if (ok !== 1) throw new Error('AI 分析失败');
  const out = Array.from(module.HEAP32.subarray(module._ot_output() / 4, module._ot_output() / 4 + 136));
  return { move: out[0], score: out[1], depth: out[2], elapsed: out[3], nodes: out[4], ttHits: mcts ? null : out[135], exact: !!out[5], search: mcts ? 'mcts' : 'ab', evaluator: mcts ? 'nnue' : 'heuristic', candidates: Array.from({ length: out[6] }, (_, i) => ({ move: out[7 + i * 2], score: out[8 + i * 2] })) };
}

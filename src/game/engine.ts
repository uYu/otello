import { evaluateSmallModel } from './beginner.ts';
import type { SmallValueModel } from './beginner.ts';
import beginnerModel from '../../models/beginner-error-policy-80-e2-lr0003-v1.json' with { type: 'json' };
import beginnerDefault from '../../models/beginner-default.json' with { type: 'json' };
import { sampleBeginnerPolicy } from './beginner-policy.ts';
import { analyzeDirectPolicy, initDirectPolicy } from './direct-policy.ts';
import createModule from './wasm/othello.mjs';
import type { Analysis, Difficulty, Position } from './types.ts';
import { SEARCH_MILLISECONDS } from './types.ts';
let module: Awaited<ReturnType<typeof createModule>>;
let pending: Promise<void> | undefined;
export function initEngine() { return pending ??= Promise.all([createModule().then(m => { module = m; }),initDirectPolicy()]).then(()=>{}); }
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
export function analyzeSmallModel(p: Position, model: SmallValueModel): Analysis {
    const started = performance.now();
    const candidates = legal(p).map(move => {
      const next = play(p, move);
      // play() automatically passes. The child may still have the same actor.
      const value = evaluateSmallModel(model, next.cells, next.turn);
      return { move, score: next.turn === p.turn ? value : -value };
    }).sort((a, b) => b.score - a.score || a.move - b.move);
    return { move: candidates[0]?.move ?? -1, score: candidates[0]?.score ?? evaluateSmallModel(model, p.cells, p.turn),
      depth: candidates.length ? 1 : 0, elapsed: performance.now() - started, nodes: candidates.length,
      ttHits: null, exact: false, search: 'one-ply', evaluator: 'beginner', engine: 'builtin',
      engineVersion: model.version, useBook: false, bookHit: false, candidates };
}
export function analyze(p: Position, difficulty: Difficulty, seed = 1): Analysis {
  if (difficulty === 'easy') return sampleBeginnerPolicy(beginnerModel, p, seed, beginnerDefault.temperature);
  if (difficulty === 'normal') return analyzeDirectPolicy(p,seed);
  const ok = module._ot_default_search(input(p), p.turn, SEARCH_MILLISECONDS[difficulty], seed, module._ot_output());
  if (ok !== 1) throw new Error('AI 分析失败');
  const out = Array.from(module.HEAP32.subarray(module._ot_output() / 4, module._ot_output() / 4 + 136));
  return { move: out[0], score: out[1], depth: out[2], elapsed: out[3], nodes: out[4], ttHits: null,
    exact: !!out[5], search: 'mcts', evaluator: 'nnue',
    candidates: Array.from({ length: out[6] }, (_, i) => ({ move: out[7 + i * 2], score: out[8 + i * 2] })) };
}

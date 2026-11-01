// Experimental direct-board heads; these are intentionally outside the default worker import tree.
import model from '../../models/beginner-board-heads-v1.json' with { type: 'json' };
import { legal, play } from './engine.ts';
import type { Analysis, Color, Position } from './types.ts';

const transforms = Array.from({ length: 8 }, (_, symmetry) => Array.from({ length: 64 }, (_, i) => {
  let x = i % 8, y = Math.floor(i / 8);
  if (symmetry & 4) [x, y] = [y, x];
  if (symmetry & 1) x = 7-x;
  if (symmetry & 2) y = 7-y;
  return y*8+x;
}));

function encode(cells: readonly number[], side: Color, mapping: number[]) {
  return mapping.map(i => cells[i] * side === 1 ? 0 : cells[i] * side === -1 ? 1 : 2);
}

export function evaluateBoardValue(cells: readonly number[], side: Color) {
  let total = 0;
  for (const mapping of transforms) {
    const channels = encode(cells, side, mapping);
    let output = model.outputBias;
    for (let h = 0; h < 32; h++) {
      let hidden = model.hiddenBias[h];
      for (let i = 0; i < 64; i++) hidden += model.hiddenWeights[h][i*3+channels[i]];
      output += Math.tanh(hidden) * model.outputWeights[h];
    }
    total += Math.tanh(output);
  }
  return total / 8;
}

export function evaluateBoardPolicy(cells: readonly number[], side: Color) {
  const logits = Array<number>(64).fill(0);
  for (const mapping of transforms) {
    const channels = encode(cells, side, mapping);
    for (let square = 0; square < 64; square++) {
      let output = model.policyBias[square];
      for (let i = 0; i < 64; i++) output += model.policyWeights[square][i*3+channels[i]];
      logits[mapping[square]] += output / 8;
    }
  }
  return logits;
}

export function analyzeBoardCandidate(position: Position, mode: 'policy' | 'value'): Analysis {
  const started = performance.now(), moves = legal(position);
  const logits = mode === 'policy' ? evaluateBoardPolicy(position.cells, position.turn) : null;
  const candidates = moves.map(move => {
    if (logits) return { move, score: logits[move] };
    const next = play(position, move);
    return { move, score: evaluateBoardValue(next.cells, next.turn) * (next.turn === position.turn ? 1 : -1) };
  }).sort((a,b) => b.score-a.score || a.move-b.move);
  return { move: candidates[0]?.move ?? -1, score: candidates[0]?.score ?? 0, candidates,
    depth: mode === 'policy' ? 0 : 1, nodes: mode === 'policy' ? 1 : moves.length,
    elapsed: performance.now()-started, ttHits: null, exact: false, useBook: false, bookHit: false,
    search: mode === 'policy' ? 'direct-policy' : 'one-ply', evaluator: 'beginner',
    engine: 'builtin', engineVersion: `${model.version}-${mode}` };
}

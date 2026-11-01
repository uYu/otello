import model from '../../models/beginner8-v1.json' with { type: 'json' };
import type { Color } from './types.ts';

export const BEGINNER_VERSION = model.version;
export const BEGINNER_MODEL = model;
export interface SmallValueModel {
  version: string;
  hidden: number;
  hiddenWeights: number[][];
  hiddenBias: number[];
  outputWeights: number[];
  outputBias: number;
  inferenceViews?: number;
}
const transforms = Array.from({ length: 8 }, (_, symmetry) => Array.from({ length: 64 }, (_, i) => {
  let x = i % 8, y = Math.floor(i / 8);
  if (symmetry & 4) [x, y] = [y, x];
  if (symmetry & 1) x = 7 - x;
  if (symmetry & 2) y = 7 - y;
  return y * 8 + x;
}));

// This is the complete value network. No positional heuristics or v2 weights.
export function evaluateBeginner(cells: readonly number[], side: Color): number {
  return evaluateSmallModel(model, cells, side);
}

export function evaluateSmallModel(network: SmallValueModel, cells: readonly number[], side: Color): number {
  let total = 0;
  const views = network.inferenceViews === 1 ? transforms.slice(0, 1) : transforms;
  for (const mapping of views) {
    let output = network.outputBias;
    for (let h = 0; h < network.hidden; h++) {
      let hidden = network.hiddenBias[h];
      const weights = network.hiddenWeights[h];
      for (let i = 0; i < 64; i++) {
        const cell = cells[mapping[i]] * side;
        hidden += weights.length === 64 ? cell * weights[i] : weights[i*3+(cell===1?0:cell===-1?1:2)];
      }
      output += Math.tanh(hidden) * network.outputWeights[h];
    }
    total += Math.tanh(output);
  }
  return total / views.length;
}

import { analyze, initEngine } from './engine.ts';
import type { Difficulty, Position } from './types.ts';
self.onmessage = async (event: MessageEvent<{ position: Position; difficulty: Difficulty; seed: number }>) => {
  try { await initEngine(); self.postMessage({ result: analyze(event.data.position, event.data.difficulty, event.data.seed) }); }
  catch (error) { self.postMessage({ error: error instanceof Error ? error.message : 'AI 引擎不可用' }); }
};

import { initEngine } from './engine.ts';
import { analyzeGame } from './ai.ts';
import type { Difficulty, Position } from './types.ts';
self.onmessage = async (event: MessageEvent<{ position: Position; difficulty: Difficulty; seed: number }>) => {
  try {
    await initEngine();
    const result=await analyzeGame(event.data.position,event.data.difficulty,event.data.seed);
    self.postMessage({result});
  }
  catch (error) { self.postMessage({ error: error instanceof Error ? error.message : 'AI 引擎不可用' }); }
};

import { analyze, initEngine } from './engine.ts';
import type { Difficulty, Position } from './types.ts';
import { createOpeningBookClient } from './opening-book.ts';
const book=createOpeningBookClient();
self.onmessage = async (event: MessageEvent<{ position: Position; difficulty: Difficulty; seed: number }>) => {
  try {
    await initEngine();
    const opening=await book.lookup(event.data.position,event.data.seed);
    const result=opening??{...analyze(event.data.position,event.data.difficulty,event.data.seed),useBook:true,bookHit:false};
    self.postMessage({result});
  }
  catch (error) { self.postMessage({ error: error instanceof Error ? error.message : 'AI 引擎不可用' }); }
};

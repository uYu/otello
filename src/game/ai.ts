import { analyze } from './engine.ts';
import { createOpeningBookClient } from './opening-book.ts';
import type { Difficulty, Position } from './types.ts';
const defaultBook = createOpeningBookClient();
export async function analyzeGame(position: Position, difficulty: Difficulty, seed: number, book = defaultBook) {
  const useBook = difficulty === 'medium' || difficulty === 'hard';
  const opening = useBook ? await book.lookup(position, seed) : null;
  return opening ?? { ...analyze(position, difficulty, seed), useBook, bookHit: false };
}

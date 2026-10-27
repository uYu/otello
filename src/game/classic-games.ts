import { initial, play } from './engine.ts';

// WTHOR 1977, record 0: the 34–30 final, distinct from their 45–19 round-robin game.
// Complete transcript is replayed through the C++ rules, never hand-placed discs.
export const INAUGURAL_FINAL = {
  title: '1977 · 首届世锦赛决赛',
  black: '井上博',
  white: 'Thomas Heiberg',
  date: '1977-10-29',
  result: { black: 34, white: 30 },
  displayPly: 27,
  source: 'https://www.worldothello.org/about/tournaments/world-othello-championship/woc/1977/story',
  transcriptSource: 'https://www.ffothello.org/informatique/la-base-wthor/',
  transcript: 'f5d6c3f3f4d3c4g6f6e6c5c6d7d8e7g5e3d2g4h3f7b5h5g3b4b6c8b8c7e8f8g8h4h6e2d1c1a3a6a5e1f2g7c2a4h8a2b7f1b3b1b2h7a7a8a1g1g2h2h1',
} as const;

export function classicPosition(ply: number = INAUGURAL_FINAL.displayPly) {
  const coordinates = INAUGURAL_FINAL.transcript.match(/[a-h][1-8]/g)!;
  if (!Number.isInteger(ply) || ply < 0 || ply > coordinates.length) throw new Error('无效的名局手数');
  let position = initial();
  for (const coordinate of coordinates.slice(0, ply)) {
    const square = (Number(coordinate[1]) - 1) * 8 + coordinate.charCodeAt(0) - 97;
    position = play(position, square);
  }
  return { ...position, lastMove: null };
}

export type Color = 1 | -1;
export type Mode = 'ai' | 'local' | 'watch';
export type Difficulty = 'easy' | 'normal' | 'hard';
export type Engine = 'builtin' | 'egaroucid';
export interface Position { cells: number[]; turn: Color; over: boolean; passed: Color | null; lastMove: number | null; }
export interface Save { version: 1; mode: Mode; difficulty: Difficulty; human: Color; actions: number[]; createdAt: string; engine?: Engine; }
export interface Entry { before: Position; after: Position; move: number; color: Color; flipped: number[]; }
export interface Analysis { move: number; score: number; depth: number; elapsed: number; nodes: number; ttHits: number | null; exact: boolean; candidates: { move: number; score: number }[]; engine?: Engine; engineVersion?: string; bookHit?: boolean; useBook?: boolean; }
export const LEVELS: Record<Difficulty, string> = { easy: '入门', normal: '熟练', hard: '高级' };
export const MODES: Record<Mode, string> = { ai: '人机对战', local: '双人同屏', watch: 'AI 观战' };
export const colorName = (c: Color) => c === 1 ? '黑棋' : '白棋';
export const coordinate = (i: number) => i < 0 ? '停着' : `${'ABCDEFGH'[i % 8]}${Math.floor(i / 8) + 1}`;

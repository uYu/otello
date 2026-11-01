import { initial, play } from './engine.ts';
import type { Entry, Save } from './types.ts';
export const SAVE_KEY = 'otello.atelier.v1';
export const MAX_SAVE_BYTES = 16000;
export function history(save: Save): Entry[] {
  let position = initial();
  return save.actions.map(move => {
    const before = position;
    position = play(before, move);
    return { before, after: position, move, color: before.turn, flipped: before.cells.flatMap((c, i) => c && c !== position.cells[i] ? [i] : []) };
  });
}
export function parseSave(text: string): Save {
  if (text.length > MAX_SAVE_BYTES) throw new Error('棋谱文件过大');
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error('不是有效的 JSON 棋谱'); }
  if (!value || typeof value !== 'object') throw new Error('棋谱格式无效');
  const s = value as Partial<Save>;
  if (s.engine !== undefined && s.engine !== 'builtin' && s.engine !== 'egaroucid') throw new Error('棋谱引擎字段无效');
  if (s.version !== 1 || !['ai', 'local', 'watch'].includes(s.mode ?? '') || !['easy', 'normal', 'medium', 'hard'].includes(s.difficulty ?? '') || (s.human !== 1 && s.human !== -1) || typeof s.createdAt !== 'string' || !Number.isFinite(Date.parse(s.createdAt)) || !Array.isArray(s.actions) || s.actions.length > 60 || s.actions.some(m => !Number.isInteger(m) || m < 0 || m > 63)) throw new Error('棋谱版本或字段无效');
  const save: Save = { version: 1, mode: s.mode!, difficulty: s.difficulty!, human: s.human, createdAt: s.createdAt, actions: [...s.actions] };
  if (s.engine !== undefined) save.engine = s.engine;
  try { history(save); } catch { throw new Error('棋谱含有非法落子，无法导入'); }
  return save;
}
export function downloadSave(save: Save) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(save, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = `otello-${save.createdAt.slice(0, 10)}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

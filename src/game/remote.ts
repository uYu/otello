import { legal } from './engine.ts';
import type { Analysis, Difficulty, Position } from './types.ts';
export async function nativeEngineReady(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch('/api/native/health', { signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(22000)]) : AbortSignal.timeout(22000), credentials: 'omit' });
    if (!response.ok) return false;
    const data = await response.json();
    return data.available === true && data.engine === 'egaroucid' && data.useBook === true && data.bookReady === true;
  } catch { return false; }
}
export async function requestNativeAnalysis(position: Position, difficulty: Exclude<Difficulty,'medium'>, signal: AbortSignal): Promise<Analysis> {
  let response: Response;
  try {
    response = await fetch('/api/native/move', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'omit',
      body: JSON.stringify({ version: 1, board: position.cells.map(c => c === 1 ? 'B' : c === -1 ? 'W' : '.').join(''), turn: position.turn, difficulty }),
      signal: AbortSignal.any([signal, AbortSignal.timeout(22000)]),
    });
  } catch { throw new Error('本机 Egaroucid 未响应，请检查引擎服务后重试。'); }
  if (!response.ok) throw new Error('本机 Egaroucid 暂时不可用，请检查引擎服务后重试。');
  const data = await response.json() as Analysis;
  if (!data || data.engine !== 'egaroucid' || data.useBook !== (difficulty !== 'easy') || (difficulty === 'easy' && data.bookHit !== false) || typeof data.bookHit !== 'boolean' || !Number.isInteger(data.move) || !legal(position).includes(data.move) || !Number.isFinite(data.score) || !Number.isInteger(data.depth) || !Number.isFinite(data.elapsed) || !Number.isFinite(data.nodes)) throw new Error('本机 Egaroucid 返回了无效结果。');
  return data;
}
// Adapter contract for a future self-hosted Edax / Egaroucid HTTP bridge.
// Coordinates are zero-based A1=0, B1=1, ..., H8=63; the response must be { move: number }.
export async function requestRemoteMove(endpoint: string, position: Position, signal?: AbortSignal): Promise<number> {
  const url = new URL(endpoint);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('接口地址必须为 HTTP 或 HTTPS');
  const response = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version: 1, board: position.cells.map(c => c === 1 ? 'B' : c === -1 ? 'W' : '.').join(''), turn: position.turn, legalMoves: legal(position) }),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000),
    credentials: 'omit',
  });
  if (!response.ok) throw new Error(`接口返回 HTTP ${response.status}`);
  const data: unknown = await response.json();
  const move = data && typeof data === 'object' && 'move' in data ? data.move : null;
  const moves = legal(position);
  if (typeof move !== 'number' || !Number.isInteger(move) || (move === -1 ? position.over || moves.length > 0 : !moves.includes(move))) throw new Error('接口返回了非法落子');
  return move;
}

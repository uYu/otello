import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { initEngine, initial, legal, play, score, analyze, flips } from '../src/game/engine.ts';
import { classicPosition, INAUGURAL_FINAL } from '../src/game/classic-games.ts';
import { history, parseSave, MAX_SAVE_BYTES } from '../src/game/storage.ts';
import { requestRemoteMove, requestNativeAnalysis, nativeEngineReady } from '../src/game/remote.ts';
import type { Save } from '../src/game/types.ts';
import createModule from '../src/game/wasm/othello.mjs';
await initEngine();
test('hero uses the legal 1977 world final transcript and reproduces the historical result', () => {
  const shown = classicPosition();
  assert.equal(shown.cells.filter(Boolean).length, INAUGURAL_FINAL.displayPly + 4);
  assert.equal(shown.over, false);
  const final = classicPosition(60);
  assert.equal(final.over, true);
  assert.deepEqual(score(final), INAUGURAL_FINAL.result);
});
const fresh = (): Save => ({ version: 1, mode: 'ai', human: 1, difficulty: 'normal', createdAt: '2026-10-27T00:00:00Z', actions: [] });
test('C++ Wasm opening and immutable play', () => {
  const p = initial(); assert.deepEqual(legal(p), [19, 26, 37, 44]);
  assert.deepEqual(flips(p, 19), [27]);
  const next = play(p, 19); assert.deepEqual(score(next), { black: 4, white: 1 });
  assert.deepEqual(score(p), { black: 2, white: 2 }); assert.equal(next.turn, -1);
  assert.throws(() => play(p, 0)); assert.throws(() => play(p, -1));
});
test('complete games, automatic passes and replay validated from actions', () => {
  let passes = 0;
  for (let seed = 1; seed <= 20; seed++) {
    let p = initial(), rng = seed; const save = fresh();
    while (!p.over) {
      const moves = legal(p); assert.ok(moves.length);
      rng = (rng * 1664525 + 1013904223) >>> 0;
      const move = moves[rng % moves.length]; save.actions.push(move); p = play(p, move);
      if (p.passed) passes++;
      assert.ok(save.actions.length <= 60);
    }
    const imported = parseSave(JSON.stringify(save));
    assert.deepEqual(history(imported).at(-1)?.after, p);
    assert.equal(legal(p).length, 0);
  }
  assert.ok(passes > 0);
});
test('corrupt, oversized, unsupported and illegal saves are rejected', () => {
  const s = fresh();
  assert.deepEqual(parseSave(JSON.stringify(s)), s);
  const native = { ...s, engine: 'egaroucid' };
  assert.deepEqual(parseSave(JSON.stringify(native)), native);
  assert.throws(() => parseSave(JSON.stringify({ ...s, engine: 'unknown' })));
  for (const bad of ['not json', 'null', '{}', 'x'.repeat(MAX_SAVE_BYTES + 1), JSON.stringify({ ...s, version: 2 }), JSON.stringify({ ...s, actions: [0] }), JSON.stringify({ ...s, actions: [19, 19] }), JSON.stringify({ ...s, human: 0 }), JSON.stringify({ ...s, createdAt: 'bad' }), JSON.stringify({ ...s, actions: [1.5] })]) assert.throws(() => parseSave(bad));
});
test('four levels use independent beginner, v2 policy, 200ms search and 2s search', () => {
  for (const difficulty of ['easy', 'normal', 'medium', 'hard'] as const) {
    const result = analyze(initial(), difficulty, 123);
    assert.ok(legal(initial()).includes(result.move)); assert.ok(result.depth >= (difficulty === 'hard' || difficulty === 'medium' ? 1 : 0));
    if (difficulty === 'hard' || difficulty === 'medium') {
      assert.equal(result.search, 'mcts'); assert.equal(result.evaluator, 'nnue'); assert.equal(result.ttHits, null);
    } else { assert.equal(result.search, 'direct-policy'); assert.equal(result.evaluator, difficulty==='easy'?'beginner':'nnue'); }
    assert.ok(result.candidates.every(c => legal(initial()).includes(c.move)));
  }
});
test('new middle difficulty and previous saves import without changing their moves',()=>{
  for(const difficulty of ['easy','normal','medium','hard'] as const){
    const save={...fresh(),difficulty,actions:[19]};
    assert.deepEqual(parseSave(JSON.stringify(save)),save);
  }
});
test('native client requires a real book-enabled engine and rejects illegal moves', async () => {
  const original = globalThis.fetch;
  try {
    const valid = { engine: 'egaroucid', useBook: true, bookHit: true, move: 19, score: 0, depth: -1, nodes: 0, elapsed: 0, ttHits: null, candidates: [], exact: false };
    globalThis.fetch = async () => new Response(JSON.stringify(valid));
    assert.equal((await requestNativeAnalysis(initial(), 'normal', new AbortController().signal)).bookHit, true);
    globalThis.fetch = async () => new Response(JSON.stringify({ ...valid, useBook: false }));
    await assert.rejects(requestNativeAnalysis(initial(), 'normal', new AbortController().signal), /无效/);
    globalThis.fetch = async () => new Response(JSON.stringify({ ...valid, useBook: false, bookHit: false }));
    assert.equal((await requestNativeAnalysis(initial(), 'easy', new AbortController().signal)).useBook, false);
    globalThis.fetch = async () => new Response(JSON.stringify(valid));
    await assert.rejects(requestNativeAnalysis(initial(), 'easy', new AbortController().signal), /无效/);
    globalThis.fetch = async () => new Response(JSON.stringify({ ...valid, move: 0 }));
    await assert.rejects(requestNativeAnalysis(initial(), 'normal', new AbortController().signal), /无效/);
    globalThis.fetch = async () => new Response(JSON.stringify({ available: true, engine: 'egaroucid', useBook: true, bookReady: true }));
    assert.equal(await nativeEngineReady(), true);
    globalThis.fetch = async () => new Response('', { status: 503 });
    assert.equal(await nativeEngineReady(), false);
  } finally { globalThis.fetch = original; }
});
test('Wasm and native ABI agree on legal moves, board transition and legacy fixed-depth AI', { skip: !existsSync('.build/native/otello') }, async () => {
  let p = initial();
  const native = (command: string, extra: string[] = []) => JSON.parse(execFileSync('.build/native/otello', [command, '--board', p.cells.map(c => c === 1 ? 'B' : c === -1 ? 'W' : '.').join(''), '--turn', String(p.turn), ...extra], { encoding: 'utf8' }));
  const coord = (m: number) => `${'ABCDEFGH'[m % 8]}${Math.floor(m / 8) + 1}`;
  for (let n = 0; n < 25; n++) {
    const moves = legal(p); assert.deepEqual(native('legal').legalMoves, moves.map(coord));
    const move = moves[(n * 7) % moves.length];
    const result = native('play', ['--move', coord(move)]), next = play(p, move);
    assert.equal(result.board, next.cells.map(c => c === 1 ? 'B' : c === -1 ? 'W' : '.').join(''));
    p = next;
  }
  const result = native('analyze', ['--time', '10000', '--depth', '5', '--network', 'heuristic', '--search', 'ab']);
  const module = await createModule(), input = module._ot_input(), output = module._ot_output();
  module.HEAP32.set(p.cells, input / 4);
  assert.equal(module._ot_search(input,p.turn,0,5,2,1,output),1);
  const values=module.HEAP32.slice(output/4,output/4+136);
  assert.equal(values[2],5); assert.equal(coord(values[0]),result.move); assert.equal(values[1],result.score);
});
test('HTTP adapter validates legal responses, request encoding and errors', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body)); assert.equal(request.board.length, 64); assert.equal(request.turn, 1); assert.deepEqual(request.legalMoves, [19,26,37,44]);
      return new Response(JSON.stringify({ move: 19 }));
    };
    assert.equal(await requestRemoteMove('https://example.com/move', initial()), 19);
    globalThis.fetch = async () => new Response('{"move":0}');
    await assert.rejects(requestRemoteMove('https://example.com/move', initial()), /非法/);
    globalThis.fetch = async () => new Response('{"move":-1}');
    await assert.rejects(requestRemoteMove('https://example.com/move', initial()), /非法/);
    globalThis.fetch = async () => new Response('', { status: 503 });
    await assert.rejects(requestRemoteMove('https://example.com/move', initial()), /503/);
    await assert.rejects(requestRemoteMove('file:///tmp/foo', initial()), /HTTP/);
  } finally { globalThis.fetch = original; }
});

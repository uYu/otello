import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { initEngine, initial, legal, play } from '../../src/game/engine.ts';

const binary = '.build/native-egaroucid/otello';
test('real native CLI, persistent protocol and local HTTP bridge', { timeout: 120000 }, async () => {
  await initEngine();
  const cli = (args: string[]) => JSON.parse(execFileSync(binary, args, { encoding: 'utf8' }));
  const opened = cli(['analyze', '--engine', 'egaroucid', '--eg-level', '1']);
  assert.equal(opened.useBook, true); assert.equal(opened.bookHit, true); assert.equal(opened.depth, -1); assert.equal(opened.nodes, 0);
  assert.equal(opened.bookVariety, 1);
  assert.throws(() => cli(['analyze', '--engine', 'egaroucid', '--book-variety', '4']), /Command failed/);
  assert.ok(legal(initial()).includes(opened.moveIndex));
  const noBook = cli(['analyze', '--engine', 'egaroucid', '--eg-level', '1', '--book', 'off']);
  assert.equal(noBook.bookHit, false); assert.ok(noBook.nodes > 0);
  assert.throws(() => cli(['analyze', '--engine', 'egaroucid', '--time', '100']), /Command failed/);
  const protocol = execFileSync(binary, ['protocol'], {
    input: 'analyze --engine egaroucid --eg-level 1 --book-variety 0\nanalyze --engine egaroucid --eg-level 1 --book off\nquit\n', encoding: 'utf8',
  }).trim().split('\n').map(line => JSON.parse(line));
  assert.equal(protocol.length, 2); assert.equal(protocol[0].bookHit, true); assert.equal(protocol[1].bookHit, false);
  assert.equal(protocol[0].bookVariety, 0);

  const port = 18207, base = `http://127.0.0.1:${port}`;
  const service = spawn(process.execPath, ['scripts/native-api.mjs'], { env: { ...process.env, OTELLO_API_PORT: String(port) }, stdio: ['ignore', 'pipe', 'pipe'] });
  const exited = once(service, 'exit');
  try {
    let startup = '';
    await Promise.race([
      new Promise<void>(resolve => { service.stdout.on('data', chunk => { startup += chunk; if (startup.includes('native service:')) resolve(); }); }),
      exited.then(() => { throw new Error('Native HTTP service stopped before listening'); }),
      new Promise((_, reject) => { const timer = setTimeout(() => reject(new Error('Native HTTP service startup timed out')), 10000); timer.unref(); }),
    ]);
    const health = await fetch(base + '/api/native/health'); assert.equal(health.status, 200);
    const metadata = await health.json(); assert.equal(metadata.bookReady, true); assert.equal(metadata.useBook, true);
    const payload = (board = initial()) => ({ version: 1, board: board.cells.map(c => c === 1 ? 'B' : c === -1 ? 'W' : '.').join(''), turn: board.turn, difficulty: 'easy' });
    const request = (data: unknown, headers: Record<string,string> = {}) => fetch(base + '/api/native/move', { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(data) });
    const first = await request(payload()); const result = await first.json();
    assert.equal(first.status, 200); assert.equal(result.bookHit, false); assert.equal(result.useBook, false); assert.ok(legal(initial()).includes(result.move));
    assert.equal(result.bookVariety, 1);
    for (const difficulty of ['normal', 'hard']) {
      const r = await (await request({ ...payload(), difficulty })).json();
      assert.equal(r.bookVariety, 1); assert.equal(r.bookHit, true);
    }
    const next = play(initial(), result.move), reply = await (await request(payload(next))).json();
    assert.ok(legal(next).includes(reply.move)); // Opposite color and coordinate translation.
    const concurrent = await Promise.all(Array.from({ length: 3 }, () => request(payload()).then(r => r.json())));
    assert.ok(concurrent.every(r => !r.bookHit && !r.useBook && legal(initial()).includes(r.move)));
    assert.equal((await request({ ...payload(), board: 'bad' })).status, 400);
    assert.equal((await request({ ...payload(), difficulty: 'unbounded' })).status, 400);
    assert.equal((await request(payload(), { Origin: 'https://example.com' })).status, 403);
    assert.equal((await fetch(base + '/api/native/move', { method: 'POST', body: 'bad' })).status, 415);
    assert.equal((await request({ ...payload(), padding: 'x'.repeat(5000) })).status, 413);
    const cancellation = new AbortController();
    const cancelled = fetch(base + '/api/native/move', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload()), signal: cancellation.signal });
    cancellation.abort(); await assert.rejects(cancelled);
    assert.equal((await request(payload())).status, 200);
  } finally {
    service.kill('SIGTERM');
    await Promise.race([exited, new Promise<void>(resolve => { const timer = setTimeout(() => { service.kill('SIGKILL'); resolve(); }, 3000); timer.unref(); })]);
  }
});

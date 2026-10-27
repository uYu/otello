import test from 'node:test';
import assert from 'node:assert/strict';
import { initial, initEngine, legal, play } from '../src/game/engine.ts';
import { boardTransition, presentedPosition } from '../src/game/presentation.ts';

await initEngine();

test('animate only the placed disc and the discs flipped by a forward move', () => {
  const before = initial();
  const after = play(before, 19);
  assert.deepEqual(boardTransition(before, after), { placed: 19, flipped: [27] });
  assert.deepEqual(boardTransition(after, before), { placed: null, flipped: [] });
  assert.deepEqual(boardTransition(after, after), { placed: null, flipped: [] });
  const later = play(after, legal(after)[0]);
  assert.deepEqual(boardTransition(before, later), { placed: null, flipped: [] });
});

test('a move that forces a pass still animates only its real flips', () => {
  let position = initial();
  let rng = 5;
  while (!position.over) {
    const before = position;
    const moves = legal(before);
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const move = moves[rng % moves.length];
    position = play(before, move);
    if (!position.passed) continue;
    assert.equal(position.turn, before.turn);
    const transition = boardTransition(before, position);
    assert.equal(transition.placed, move);
    assert.ok(transition.flipped.length > 0);
    for (const i of transition.flipped) {
      assert.equal(before.cells[i], -before.turn);
      assert.equal(position.cells[i], before.turn);
    }
    return;
  }
  assert.fail('The fixture must contain an automatic pass');
});

test('pending passes show the side with no legal move until confirmed, for both colors', () => {
  const checked = new Set<number>();
  for (let seed = 1; seed <= 30 && checked.size < 2; seed++) {
    let position = initial(), rng = seed;
    while (!position.over) {
      const moves = legal(position);
      rng = (rng * 1664525 + 1013904223) >>> 0;
      position = play(position, moves[rng % moves.length]);
      if (!position.passed) continue;
      const nextTurn = position.turn;
      const waiting = presentedPosition(position, true);
      assert.equal(waiting.turn, position.passed);
      assert.deepEqual(legal(waiting), []);
      assert.strictEqual(waiting.cells, position.cells);
      assert.equal(position.turn, nextTurn);
      const confirmed = presentedPosition(position, false);
      assert.strictEqual(confirmed, position);
      assert.ok(legal(confirmed).length > 0);
      checked.add(position.passed);
    }
  }
  assert.deepEqual([...checked].sort(), [-1, 1]);
});

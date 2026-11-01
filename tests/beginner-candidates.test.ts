import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { analyzeBoardCandidate, evaluateBoardPolicy, evaluateBoardValue } from '../src/game/beginner-candidates.ts';
import { initEngine, initial, legal, play } from '../src/game/engine.ts';

const report = JSON.parse(readFileSync('models/beginner-board-heads-v1-training.json', 'utf8'));
await initEngine();

test('direct-board heads match exported Python predictions and the identical tiny-model sample', () => {
  assert.equal(createHash('sha256').update(readFileSync('models/beginner-board-heads-v1.json')).digest('hex'), report.modelSha256);
  const tiny = JSON.parse(readFileSync('models/beginner8-v1-training.json', 'utf8'));
  assert.equal(report.selectedIndicesSha256, tiny.selectedIndicesSha256);
  assert.equal(report.epochs, tiny.epochs);
  assert.equal(report.cacheSha256, tiny.cacheSha256);
  for (const fixture of report.fixtures) {
    const logits = evaluateBoardPolicy(fixture.board, fixture.turn);
    assert.ok(Math.abs(evaluateBoardValue(fixture.board, fixture.turn)-fixture.value) < 1e-6);
    assert.ok(logits.every((v,i) => Math.abs(v-fixture.logits[i]) < 1e-5));
    const rotated = Array.from({length:64},(_,i) => fixture.board[(7-i%8)*8+Math.floor(i/8)]);
    const rotatedLogits = evaluateBoardPolicy(rotated, fixture.turn);
    assert.ok(rotatedLogits.every((v,i) => Math.abs(v-logits[(7-i%8)*8+Math.floor(i/8)]) < 1e-10));
    assert.ok(Math.abs(evaluateBoardValue(rotated, fixture.turn)-fixture.value) < 1e-6);
  }
});

test('policy masks illegal squares; one-ply value preserves the actor on forced passes', () => {
  let passes = 0;
  for (let game=1; game<=5; game++) {
    let p=initial(), ply=0;
    while (!p.over) {
      const moves=legal(p), logits=evaluateBoardPolicy(p.cells,p.turn);
      for (const mode of ['policy','value'] as const) {
        const result=analyzeBoardCandidate(p,mode);
        assert.equal(result.move,result.candidates[0].move);
        assert.equal(result.useBook,false);
        assert.equal(result.nodes,mode === 'policy' ? 1 : moves.length);
        assert.deepEqual(result.candidates.map(c=>c.move).sort((a,b)=>a-b),moves);
        for (const c of result.candidates) {
          const next=play(p,c.move);
          if (next.passed) passes++;
          const expected=mode === 'policy' ? logits[c.move] : evaluateBoardValue(next.cells,next.turn)*(next.turn===p.turn?1:-1);
          assert.equal(c.score,expected);
          assert.ok(c.score<=result.score);
        }
      }
      p=play(p,moves[(ply++*7+game)%moves.length]);
      assert.ok(ply<=60);
    }
  }
  assert.ok(passes>0);
});

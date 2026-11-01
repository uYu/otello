import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { evaluateBeginner } from '../src/game/beginner.ts';
import { analyzeSmallModel, initial, initEngine, legal, play } from '../src/game/engine.ts';

const tiny = JSON.parse(readFileSync('models/beginner8-v1.json', 'utf8'));
const analyze = (p: Parameters<typeof analyzeSmallModel>[0], _difficulty: string, _seed?: number) => analyzeSmallModel(p, tiny);
const report = JSON.parse(readFileSync('models/beginner8-v1-training.json', 'utf8'));
await initEngine();

test('embedded beginner parameters match the trained artifact and Python inference', () => {
  const bytes = readFileSync('models/beginner8-v1.json');
  assert.equal(createHash('sha256').update(bytes).digest('hex'), report.modelSha256);
  const model = JSON.parse(bytes.toString());
  assert.equal(model.parameters, 529);
  assert.equal(model.hiddenWeights.flat().length + model.hiddenBias.length + model.outputWeights.length + 1, 529);
  assert.match(report.initialization, /random/);
  assert.ok(report.test.outcomeMse < report.test.zeroPredictorMse);
  for (const fixture of report.fixtures) {
    assert.ok(Math.abs(evaluateBeginner(fixture.board, fixture.turn) - fixture.value) < 1e-6);
    const rotated = Array.from({ length: 64 }, (_, i) => fixture.board[(7-i%8)*8+Math.floor(i/8)]);
    assert.ok(Math.abs(evaluateBeginner(rotated, fixture.turn) - fixture.value) < 1e-6);
  }
});

test('beginner selects only its maximum successor value, with correct perspective after forced passes', () => {
  let passes = 0, terminalChildren = 0;
  for (let game = 1; game <= 8; game++) {
    let p = initial(), ply = 0;
    while (!p.over) {
      const moves = legal(p), result = analyze(p, 'easy', game);
      assert.equal(result.evaluator, 'beginner');
      assert.equal(result.search, 'one-ply');
      assert.equal(result.depth, 1);
      assert.equal(result.nodes, moves.length);
      assert.equal(result.useBook, false);
      assert.equal(result.exact, false);
      assert.equal(result.move, analyze(p, 'easy', game+111).move);
      assert.deepEqual(result.candidates.map(c => c.move).sort((a,b) => a-b), moves);
      for (const candidate of result.candidates) {
        const next = play(p, candidate.move);
        if (next.passed) passes++;
        if (next.over) terminalChildren++;
        const expected = evaluateBeginner(next.cells, next.turn) * (next.turn === p.turn ? 1 : -1);
        assert.equal(candidate.score, expected);
        assert.ok(candidate.score <= result.score);
      }
      assert.equal(result.score, result.candidates[0].score);
      assert.equal(result.move, result.candidates[0].move);
      p = play(p, moves[(ply++ * 7 + game) % moves.length]);
      assert.ok(ply <= 60);
    }
    assert.equal(analyze(p, 'easy').move, -1);
  }
  assert.ok(passes > 0);
  assert.ok(terminalChildren > 0);
});

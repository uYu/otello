import test from 'node:test';
import assert from 'node:assert/strict';
import createModule from '../src/game/wasm/othello.mjs';

test('Wasm reuses its search cache and preserves shallow candidate scores after a deeper search', async () => {
  const module = await createModule();
  const input = module._ot_input();
  const output = module._ot_output();
  const board = new Int32Array(64);
  board[28] = board[35] = 1;
  board[27] = board[36] = -1;
  const search = (depth: number) => {
    module.HEAP32.set(board, input / 4);
    assert.equal(module._ot_search(input, 1, 0, depth, 2, 1, output), 1);
    // Search may grow Wasm memory; read the current heap after the call.
    const result = module.HEAP32.slice(output / 4, output / 4 + 136);
    const candidates = Array.from({ length: result[6] }, (_, i) => [result[7 + i * 2], result[8 + i * 2]])
      .sort((a, b) => a[0] - b[0]);
    assert.equal(result[2], depth);
    return { score: result[1], nodes: result[4], hits: result[135], candidates };
  };
  module._ot_search_clear_cache();
  const cold = search(6);
  const warm = search(6);
  assert.equal(warm.score, cold.score);
  assert.deepEqual(warm.candidates, cold.candidates);
  assert.ok(warm.nodes < cold.nodes && warm.hits > 0);
  const shallowWarm = search(3);
  module._ot_search_clear_cache();
  const shallowCold = search(3);
  assert.equal(shallowWarm.score, shallowCold.score);
  assert.deepEqual(shallowWarm.candidates, shallowCold.candidates);
});

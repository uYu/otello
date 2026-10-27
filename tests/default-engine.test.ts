import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import createModule from '../src/game/wasm/othello.mjs';
import { initial, initEngine, legal, play } from '../src/game/engine.ts';

test('default NNUE is embedded, matches the measured model, and works offline', async () => {
  const original=globalThis.fetch;
  let module: Awaited<ReturnType<typeof createModule>>;
  try {
    globalThis.fetch=async () => { throw new Error('Network is disabled'); };
    module=await createModule();
  } finally {globalThis.fetch=original;}
  const sha=new TextDecoder().decode(module.HEAPU8.slice(module._ot_default_model_sha256(),module._ot_default_model_sha256()+64));
  assert.equal(sha,JSON.parse(readFileSync('models/nnue256-v1.2.json','utf8')).sha256);
  await initEngine();
  let p=initial();
  for(let i=0;i<10;i++){
    const input=module._ot_input(),out=module._ot_output();module.HEAP32.set(p.cells,input/4);
    assert.equal(module._ot_default_evaluate(input,p.turn,out),1);
    const value=module.HEAP32[out/4];
    const renamed=p.cells.map(x=>-x);module.HEAP32.set(renamed,input/4);
    assert.equal(module._ot_default_evaluate(input,-p.turn,out),1);
    assert.equal(module.HEAP32[out/4],value);
    if(existsSync('.build/native/otello')){
      const native=JSON.parse(execFileSync(resolve('.build/native/otello'),['evaluate','--network',resolve('models/nnue256-v1.2.otnnue'),
        '--board',p.cells.map(x=>x===1?'B':x===-1?'W':'.').join(''),'--turn',String(p.turn)],{encoding:'utf8'}));
      assert.equal(value,native.score);
    }
    const moves=legal(p);p=play(p,moves[(i*7)%moves.length]);
  }
  const input=module._ot_input(),out=module._ot_output();module.HEAP32.set(initial().cells,input/4);
  assert.equal(module._ot_default_search(input,1,5,1701,out),1);
  assert.ok(legal(initial()).includes(module.HEAP32[out/4]));
  module.HEAP32.fill(1,input/4,input/4+64);
  assert.equal(module._ot_default_search(input,-1,5,1701,out),1);
  assert.equal(module.HEAP32[out/4],-1);assert.equal(module.HEAP32[out/4+1],-106400);assert.equal(module.HEAP32[out/4+5],1);
});

test('native CLI defaults to NNUE/MCTS without a model file in its working directory', {skip:!existsSync('.build/native/otello')}, () => {
  const directory=mkdtempSync(join(tmpdir(),'otello-default-'));
  try {
    const result=JSON.parse(execFileSync(resolve('.build/native/otello'),['analyze','--time','10','--seed','1701'],{cwd:directory,encoding:'utf8'}));
    assert.equal(result.search,'mcts');assert.equal(result.evaluator,'othello-nnue-v2');
    assert.equal(result.depthKind,'maximumVisited');
    const weak=JSON.parse(execFileSync(resolve('.build/native/otello'),['analyze','--network','heuristic','--time','10'],{cwd:directory,encoding:'utf8'}));
    assert.equal(weak.evaluator,'heuristic');assert.equal(weak.search,'ab');
  } finally {rmSync(directory,{recursive:true,force:true});}
});

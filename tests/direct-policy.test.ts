import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { analyze, initial, initEngine, legal } from '../src/game/engine.ts';
import { directPolicyLogits } from '../src/game/direct-policy.ts';
import createModule from '../src/game/wasm/direct-policy.mjs';
const reference=JSON.parse(readFileSync('models/second-level-v2-policy-parity.json','utf8'));
await initEngine();
test('second-level Wasm matches explicit native v2 logits for both sides, with no search ABI',async()=>{
  assert.equal(createHash('sha256').update(readFileSync(reference.model)).digest('hex'),reference.modelSha256);
  for(const f of reference.fixtures){
    const p={...initial(),cells:f.board,turn:f.turn};
    const logits=directPolicyLogits(p);
    assert.ok(logits.every((v,i)=>Math.abs(v-f.logits[i])<1e-6));
    assert.deepEqual(directPolicyLogits({...p,cells:p.cells.map((c:number)=>-c),turn:-p.turn as 1|-1}),logits);
  }
  const module=await createModule();
  assert.equal('_ot_default_search' in module,false);
  assert.equal('_ot_search' in module,false);
});
test('second-level selects learned probabilities, masks illegal moves, and safely handles terminals',()=>{
  const f=reference.fixtures[20],p={...initial(),cells:f.board,turn:f.turn};
  const moves=legal(p),logits=directPolicyLogits(p),max=Math.max(...moves.map(m=>logits[m]));
  const weights=moves.map(m=>Math.exp(logits[m]-max)),total=weights.reduce((a,b)=>a+b,0);
  const counts=new Map(moves.map(m=>[m,0]));
  for(let seed=1;seed<=3000;seed++){
    const r=analyze(p,'normal',seed*7919);
    assert.ok(moves.includes(r.move));assert.equal(r.nodes,1);assert.equal(r.depth,0);
    assert.equal(r.search,'direct-policy');assert.equal(r.evaluator,'nnue');assert.equal(r.useBook,false);
    assert.equal(r.move,analyze(p,'normal',seed*7919).move);
    counts.set(r.move,counts.get(r.move)!+1);
  }
  for(let i=0;i<moves.length;i++)assert.ok(Math.abs(counts.get(moves[i])!/3000-weights[i]/total)<.035);
  assert.equal(analyze({...initial(),cells:Array(64).fill(1),over:true},'normal',7).move,-1);
});

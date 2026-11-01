import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { analyze, initial, initEngine, legal, play } from '../src/game/engine.ts';
import { beginnerPolicyLogits } from '../src/game/beginner-policy.ts';
import { beginnerGameSeed } from '../src/game/seeds.ts';
const read=(path:string)=>JSON.parse(readFileSync(path,'utf8'));
const sha=(path:string)=>createHash('sha256').update(readFileSync(path)).digest('hex');
const selected=read('models/beginner-default.json'),model=read(selected.defaultModel),training=read(selected.trainingReport);
await initEngine();

test('active beginner embeds the selected random-initialized 10k model and matches Python logits',()=>{
  assert.equal(sha(selected.defaultModel),selected.modelSha256);
  assert.equal(training.modelSha256,selected.modelSha256);
  assert.equal(model.version,selected.version);
  assert.equal(training.trainPositions,10000);
  assert.equal(training.uniqueCanonicalTrainPositions,10000);
  assert.match(training.initialization,/random/);
  assert.equal(model.parameters,12352);
  for(const fixture of training.fixtures){
    const logits=beginnerPolicyLogits(model,fixture.board,fixture.turn);
    assert.ok(logits.every((value,i)=>Math.abs(value-fixture.logits[i])<1e-5));
  }
});

test('beginner samples only learned legal probabilities with stable game/board seeds',()=>{
  let p=initial();for(let ply=0;ply<15;ply++)p=play(p,legal(p)[(ply*7+3)%legal(p).length]);
  const moves=legal(p),logits=beginnerPolicyLogits(model,p.cells,p.turn);
  const maximum=Math.max(...moves.map(m=>logits[m]));
  const weights=moves.map(m=>Math.exp((logits[m]-maximum)/selected.temperature));
  const total=weights.reduce((a,b)=>a+b,0),counts=new Map(moves.map(m=>[m,0]));
  const samples=3000;
  for(let seed=1;seed<=samples;seed++){
    const result=analyze(p,'easy',seed*7919);
    assert.ok(moves.includes(result.move));
    assert.equal(result.engineVersion,selected.version);
    assert.equal(result.search,'direct-policy');assert.equal(result.depth,0);assert.equal(result.nodes,1);
    assert.equal(result.useBook,false);assert.equal(result.bookHit,false);assert.equal(result.exact,false);
    assert.equal(result.move,analyze(p,'easy',seed*7919).move);
    counts.set(result.move,counts.get(result.move)!+1);
  }
  for(let i=0;i<moves.length;i++)assert.ok(Math.abs(counts.get(moves[i])!/samples-weights[i]/total)<.035);
  const date='2026-11-01T08:00:00.000Z';assert.equal(beginnerGameSeed(date),beginnerGameSeed(date));
  assert.notEqual(beginnerGameSeed(date),beginnerGameSeed('2026-11-01T08:01:00.000Z'));
});

import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {initial,initEngine,legal,play} from '../src/game/engine.ts';
import {analyzeBeginnerPolicy,samplePolicyResult} from '../src/game/beginner-policy.ts';
import {initLegacyBeginner,analyzeLegacyBeginner} from '../benchmarks/beginner-legacy.ts';
const read=(p:string)=>JSON.parse(readFileSync(p,'utf8'));
const sha=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
await initEngine();await initLegacyBeginner();
const paths=['models/second-level-tuning-dense-T1.5-vs-previous-second-100.json','models/second-level-tuning-dense-T1.5-vs-previous-second-confirm-100.json','models/second-level-tuning-dense-T1.5-vs-legacy-confirm-50.json'];
const audits=[];
for(const path of paths){
 const r=read(path),q=r.protocol,model=read(q.model),base=q.baselineModel?read(q.baselineModel):null;
 assert.equal(sha(q.model),q.modelSha256);if(base)assert.equal(sha(q.baselineModel),q.baselineModelSha256);
 let win=0,draw=0,loss=0,actions=0;
 for(const g of r.records){let p=initial();const color=g.black==='candidate'?1:-1;
  for(let i=0;i<g.moves.length;i++){
   const move=g.moves[i];assert.ok(legal(p).includes(move));
   if(i>=g.openingPlies){const seed=q.seed+Math.floor((g.number-1)/2)*104729+(i+1)*7919;
    const expected=p.turn===color?samplePolicyResult(analyzeBeginnerPolicy(model,p),p,seed,q.temperature):base?samplePolicyResult(analyzeBeginnerPolicy(base,p),p,seed,q.baselineTemperature):analyzeLegacyBeginner(p,(i+1)*7919);
    assert.equal(move,expected.move);}
   p=play(p,move);actions++;
  }
  assert.ok(p.over);assert.deepEqual(p.cells,g.finalBoard);const diff=p.cells.reduce((a,b)=>a+b,0)*color;assert.equal(diff||0,g.candidateDiscDifference);if(diff>0)win++;else if(diff<0)loss++;else draw++;
 }
 assert.equal(win,r.results.candidate.win);assert.equal(draw,r.results.candidate.draw);assert.equal(loss,r.results.candidate.loss);
 audits.push({report:path,sha256:sha(path),games:r.records.length,actions,win,draw,loss});
}
writeFileSync('models/second-level-tuning-validation.json',JSON.stringify({audits},null,2)+'\n');console.log(JSON.stringify(audits));

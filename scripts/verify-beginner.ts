import assert from 'node:assert/strict';
import { readFileSync,writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { analyze,initEngine,initial,legal,play } from '../src/game/engine.ts';
import { analyzeLegacyBeginner,initLegacyBeginner } from '../benchmarks/beginner-legacy.ts';
const read=(p:string)=>JSON.parse(readFileSync(p,'utf8'));
const sha=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
const selected=read('models/beginner-default.json');assert.equal(sha(selected.defaultModel),selected.modelSha256);
await initEngine();await initLegacyBeginner();
const total={win:0,draw:0,loss:0};let actions=0,decisions=0;
for(const path of selected.comparisonReports){
  const report=read(path);assert.equal(report.complete,true);assert.equal(report.games,50);
  for(const record of report.records){
    let p=initial();const color=record.black==='candidate'?1:-1;
    for(const [ply,move] of record.moves.entries()){
      assert.ok(legal(p).includes(move));
      if(ply>=8){
        const candidate=p.turn===color;
        const seed=candidate?report.protocol.seed+Math.floor((record.number-1)/2)*104729+(ply+1)*7919:(ply+1)*7919;
        const result=candidate?analyze(p,'easy',seed):analyzeLegacyBeginner(p,seed);
        assert.equal(result.move,move,`${path} game${record.number} ply${ply}`);
        if(candidate)decisions++;
      }
      p=play(p,move);actions++;
    }
    assert.equal(p.over,true);assert.deepEqual(p.cells,record.finalBoard);
    const difference=(p.cells.reduce((a,b)=>a+b,0)*color)||0;
    assert.equal(difference,record.candidateDiscDifference);
    total[difference>0?'win':difference<0?'loss':'draw']++;
  }
}
for(const key of ['win','draw','loss'] as const)assert.equal(total[key],selected.results[key]);
const result={version:selected.version,modelSha256:selected.modelSha256,games:100,actions,decisions,results:total,
  scoreRate:(total.win+total.draw/2)/100,allRecordedActionsExactlyReplayed:true,allFinalBoardsVerified:true,
  scriptSha256:sha('scripts/verify-beginner.ts')};
writeFileSync('models/beginner-default-validation.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));

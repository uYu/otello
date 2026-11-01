import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { analyze, initEngine, initial, legal, play } from '../src/game/engine.ts';
import { analyzeBoardCandidate } from '../src/game/beginner-candidates.ts';
import { initLegacyBeginner, analyzeLegacyBeginner } from '../benchmarks/beginner-legacy.ts';

const read=(path:string)=>JSON.parse(readFileSync(path,'utf8'));
const sha=(path:string)=>createHash('sha256').update(readFileSync(path)).digest('hex');
const corner=(move:number)=>[0,7,56,63].includes(move);
await initEngine();await initLegacyBeginner();
const audits=[];
for(const mode of ['tiny','policy','value'] as const){
  const path=`models/beginner-${mode}-vs-legacy-50.json`,report=read(path);
  assert.equal(report.records.length,50);assert.equal(report.complete,true);
  const counts={win:0,draw:0,loss:0};
  let decisions=0,passes=0,cornersAvailable=0,cornersTaken=0,gifts=0,avoidableGifts=0;
  for(const record of report.records){
    let p=initial();const candidateColor=record.black==='candidate'?1:-1;
    assert.equal(record.openingPlies,8);
    for(const [ply,move] of record.moves.entries()){
      const moves=legal(p);assert.ok(moves.includes(move));
      if(ply>=record.openingPlies){
        if(p.turn===candidateColor){
          const result=mode==='tiny'?analyze(p,'easy'):analyzeBoardCandidate(p,mode);
          assert.equal(result.move,move);assert.equal(result.move,result.candidates[0].move);
          assert.equal(result.candidates.length,moves.length);
          decisions++;
          const corners=moves.filter(corner);
          if(corners.length){cornersAvailable++;if(corner(move))cornersTaken++;}
          const givesCorner=(candidate:number)=>{
            const next=play(p,candidate);
            return !next.over&&next.turn!==p.turn&&legal(next).some(corner);
          };
          if(!legal(p).some(corner)&&givesCorner(move)){
            gifts++;if(moves.some(m=>!givesCorner(m)))avoidableGifts++;
          }
        }else{
          assert.equal(analyzeLegacyBeginner(p,(ply+1)*7919).move,move);
        }
      }
      p=play(p,move);if(p.passed)passes++;
    }
    assert.equal(p.over,true);assert.deepEqual(p.cells,record.finalBoard);
    const diff=p.cells.reduce((a,b)=>a+b,0)*candidateColor;
    assert.equal(diff,record.candidateDiscDifference);
    counts[diff>0?'win':diff<0?'loss':'draw']++;
  }
  for(const key of ['win','draw','loss'] as const)assert.equal(counts[key],report.results.candidate[key]);
  audits.push({mode,report:path,reportSha256:sha(path),games:50,decisions,passes,cornersAvailable,cornersTaken,
    cornerChoiceRate:cornersTaken/cornersAvailable,opponentCornerGiftsWithoutOwnCorner:gifts,avoidableCornerGifts:avoidableGifts,
    limits:'Corner opportunity and one-step gift counts are diagnostics, not a proof of optimal play; passing a corner can be correct.'});
}
const summary=read('models/beginner-legacy50-comparison.json');
summary.audit=audits;
summary.selectionPending=false;
summary.selected='tiny';
summary.selectedModel='models/beginner8-v1.json';
summary.reason='Lowest score among all three candidates against the same old beginner: 58% vs 75% and 92%. The model has held-out learning signal, legal deterministic argmax, D4 symmetry and verified pass handling. Human beginner suitability still needs playtesting; this is not evidence of being weaker than the old beginner.';
summary.auditScriptSha256=sha('scripts/audit-beginner.ts');
writeFileSync('models/beginner-legacy50-comparison.json',JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(audits,null,2));

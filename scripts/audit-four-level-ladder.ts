import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { analyze, initial, initEngine, legal, play } from '../src/game/engine.ts';
import { BOOK_VERSION, createOpeningBookClient } from '../src/game/opening-book.ts';
const read=(p:string)=>JSON.parse(readFileSync(p,'utf8'));
const sha=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
const comparison=read('models/four-level-ladder-v1-comparison.json');assert.equal(comparison.complete,true);
await initEngine();
const book=createOpeningBookClient(async input=>{
  const path=String(input);assert.ok(path.startsWith(`/opening-book/${BOOK_VERSION}/`));
  return new Response(readFileSync(resolve('public',path.slice(1))));
});
const audits=[];
for(const summary of comparison.summaries){
  const report=read(summary.report);assert.equal(sha(summary.report),summary.reportSha256);
  assert.equal(report.complete,true);assert.equal(report.records.length,20);
  for(const [p,digest] of Object.entries(report.protocol.sourceSha256))assert.equal(sha(p),digest);
  let win=0,draw=0,loss=0,actions=0,policies=0,bookMoves=0,searches=0,sumDiscs=0;
  const colors={black:{win:0,draw:0,loss:0},white:{win:0,draw:0,loss:0}};
  for(const game of report.records){
    assert.equal(game.black,game.number%2===1?'candidate':'baseline');
    assert.equal(game.decisions.length,game.moves.length);
    let p=initial();const color=game.black==='candidate'?1:-1;
    for(const [ply,move] of game.moves.entries()){
      const d=game.decisions[ply],role=p.turn===color?'candidate':'baseline',level=report.protocol[role];
      assert.equal(d.role,role);assert.equal(d.level,level);assert.equal(d.move,move);assert.equal(d.ply,ply);
      const seed=report.protocol.seed+game.pair*104729+(ply+1)*7919;assert.equal(d.seed,seed);
      assert.ok(legal(p).includes(move));assert.ok(Number.isFinite(d.wallMs));
      if(level==='easy'||level==='policy'){
        const result=analyze(p,level==='easy'?'easy':'normal',seed);assert.equal(move,result.move);
        assert.equal(d.search,'direct-policy');assert.equal(d.bookHit,false);assert.equal(d.nodes,1);policies++;
      }else{
        const opening=await book.lookup(p,seed);
        if(opening){assert.equal(move,opening.move);assert.equal(d.search,'book');assert.equal(d.bookHit,true);bookMoves++;}
        else{assert.equal(d.search,'mcts');assert.equal(d.bookHit,false);assert.ok(d.nodes>=0);if(d.nodes===0)assert.equal(d.exact,true,'Zero-work search must reuse a proven result');searches++;}
      }
      p=play(p,move);actions++;
    }
    assert.ok(p.over);assert.deepEqual(p.cells,game.finalBoard);
    const diff=p.cells.reduce((a,b)=>a+b,0)*color;assert.equal(diff||0,game.candidateDiscDifference);sumDiscs+=diff;
    const outcome=diff>0?'win':diff<0?'loss':'draw';colors[color===1?'black':'white'][outcome]++;
    if(diff>0)win++;else if(diff<0)loss++;else draw++;
  }
  assert.deepEqual({win,draw,loss},{win:summary.results.win,draw:summary.results.draw,loss:summary.results.loss});
  assert.equal((win+draw/2)/20,summary.results.scoreRate);
  assert.ok(Math.abs(sumDiscs/20-summary.results.meanDiscDifference)<1e-9);
  audits.push({report:summary.report,sha256:sha(summary.report),games:20,actions,policies,bookMoves,searches,colors,
    allPoliciesAndBookMovesExactlyReplayed:true,allMovesLegalAndFinalBoardsVerified:true});
}
writeFileSync('models/four-level-ladder-v1-validation.json',JSON.stringify({complete:true,games:60,audits,
  scriptSha256:sha('scripts/audit-four-level-ladder.ts'),limits:'Time-budgeted MCTS choices cannot be replayed exactly; search moves checked against legality, execution logs and book-miss status.'},null,2)+'\n');
console.log(JSON.stringify(audits));

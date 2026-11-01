// Proposed ladder screening only: no runtime/default changes.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import { resolve } from 'node:path';
import { analyze, initial, initEngine, legal, play } from '../src/game/engine.ts';
import { createOpeningBookClient, BOOK_VERSION } from '../src/game/opening-book.ts';
import createModule from '../src/game/wasm/othello.mjs';
import type { Analysis, Position } from '../src/game/types.ts';

const read=(p:string)=>JSON.parse(readFileSync(p,'utf8'));
const sha=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
const write=(p:string,value:unknown)=>{writeFileSync(p+'.tmp',JSON.stringify(value,null,2)+'\n');renameSync(p+'.tmp',p);};
const version='four-level-ladder-v1',seed=20261201;
const names={easy:'入门',policy:'初级 · v2策略',search200:'中级 · 200ms+开局库',search2000:'高级 · 2s+开局库'};
type Level=keyof typeof names;
const experiments:[Level,Level][]=[['policy','easy'],['search200','policy'],['search2000','search200']];
const items=experiments.map(([candidate,baseline])=>({candidate,baseline,games:20,
  name:`${names[candidate]} vs ${names[baseline]} · 20局`,
  report:`models/${version}-${candidate}-vs-${baseline}-20.json`,focus:'candidate',
  displayNames:{candidate:names[candidate],baseline:names[baseline]},displayOrder:['candidate','baseline']}));
for(const e of items)assert.ok(!existsSync(e.report),`Preserve ${e.report}`);
const preserved=Object.fromEntries(['models/default-model.json','models/v2-default.json','models/beginner-default.json',
  'models/second-level-default.json','src/game/engine.ts','src/game/wasm/othello.mjs','src/game/wasm/direct-policy.mjs'].map(p=>[p,sha(p)]));
const v2=read('models/v2-default.json');
assert.equal(sha(v2.defaultModel),v2.modelSha256);
assert.equal(sha('src/game/wasm/othello.mjs'),v2.artifactSha256['src/game/wasm/othello.mjs']);
const bookBase=`public/opening-book/${BOOK_VERSION}/`;
for(const [p,digest] of Object.entries(read(bookBase+'checksums.json')))assert.equal(sha(bookBase+p),digest);
const book=createOpeningBookClient(async input=>{
  const p=String(input),prefix=`/opening-book/${BOOK_VERSION}/`;
  assert.ok(p.startsWith(prefix));
  // A missing asset throws: a loading failure never becomes a verified miss.
  return new Response(readFileSync(resolve('public',p.slice(1))),{headers:{'Content-Type':'application/json'}});
});
const old=read('models/current-experiment.json');
write('models/current-experiment.json',{version:'拟定四档相邻验证 · 每组20局 · 实际开局库流程',experiments:items,
  archivedExperiments:[...old.experiments,...(old.archivedExperiments??[])]});
await initEngine();
// Independent trees per search tier; caches cleared before every game.
const searchModules={search200:await createModule(),search2000:await createModule()};
for(const m of Object.values(searchModules)){
  const pointer=m._ot_default_model_sha256();
  assert.equal(new TextDecoder().decode(m.HEAPU8.subarray(pointer,pointer+64)),v2.modelSha256);
}
async function choose(level:Level,p:Position,moveSeed:number):Promise<Analysis>{
  if(level==='easy'||level==='policy')return analyze(p,level==='easy'?'easy':'normal',moveSeed);
  const opening=await book.lookup(p,moveSeed);if(opening)return opening;
  const module=searchModules[level],input=module._ot_input(),output=module._ot_output(),ms=level==='search200'?200:2000;
  module.HEAP32.set(p.cells,input/4);
  assert.equal(module._ot_default_search(input,p.turn,ms,moveSeed,output),1);
  const out=Array.from(module.HEAP32.subarray(output/4,output/4+136));
  return {move:out[0],score:out[1],depth:out[2],elapsed:out[3],nodes:out[4],ttHits:null,exact:!!out[5],
    search:'mcts',evaluator:'nnue',useBook:true,bookHit:false,
    candidates:Array.from({length:out[6]},(_,i)=>({move:out[7+i*2],score:out[8+i*2]}))};
}
const summaries=[];
for(const e of items){
  const tally=()=>({win:0,draw:0,loss:0,scoreRate:0,meanDiscDifference:0});
  const results={candidate:tally(),baseline:tally()};
  const counts=()=>({decisions:0,bookHits:0,searchDecisions:0,policyDecisions:0,wallMs:0,searchWallMs:0,nodes:0,exact:0});
  const diagnostics={candidate:counts(),baseline:counts()};
  const records:any[]=[];const started=performance.now();
  const metadata={experiment:'candidate-vs-approved-baseline',version,plannedGames:20,
    configs:[{id:'candidate',name:names[e.candidate]},{id:'baseline',name:names[e.baseline]}],
    protocol:{candidate:e.candidate,baseline:e.baseline,seed,pairedColors:true,pairs:10,
      startingPosition:'Standard four-disc initial board; no forced random opening moves',
      policySeed:'seed+pair*104729+(ply+1)*7919, shared by both roles; policy sampling additionally mixes board',
      bookEnabled:{easy:false,policy:false,search200:true,search2000:true},
      timeBudgetsMs:{easy:0,policy:0,search200:200,search2000:2000},
      bookVersion:BOOK_VERSION,bookChecksumsSha256:sha(bookBase+'checksums.json'),
      models:{easy:read('models/beginner-default.json'),policy:read('models/second-level-default.json'),searchModel:v2.defaultModel,searchModelSha256:v2.modelSha256},
      search:'Exact frozen v2 browser Wasm, default learned-prior PUCT/search selection, independent module per search tier, clear trees before each game, subtree reuse within game',
      execution:'Sequential matches and decisions; no concurrent strength matches competing for CPU',
      sourceSha256:Object.fromEntries(['scripts/verify-four-level-ladder.ts','src/game/engine.ts','src/game/direct-policy.ts',
        'src/game/beginner-policy.ts','src/game/opening-book.ts','src/game/wasm/othello.mjs','src/game/wasm/direct-policy.mjs'].map(p=>[p,sha(p)])),
      limits:'20 games per adjacent pair; screening only, no Elo or human win-rate estimate. Measures deployed book-assisted ladder, not isolated model-only search-time gain.'}};
  function save(complete:boolean,currentGame:unknown=null){
    const state={...metadata,games:records.length,complete,results,diagnostics,seconds:(performance.now()-started)/1000,currentGame};
    write(e.report,{...state,records});write(e.report.replace('.json','.live.json'),state);
  }
  save(false);
  for(let game=0;game<20;game++){
    for(const module of Object.values(searchModules))module._ot_mcts_clear_cache();
    const color=game%2===0?1:-1,black=color===1?'candidate':'baseline',white=color===-1?'candidate':'baseline';
    let p=initial();const moves:number[]=[],decisions:any[]=[];
    const board=()=>({number:game+1,ply:moves.length,black,white,turn:p.turn,board:p.cells,
      blackCount:p.cells.filter(c=>c===1).length,whiteCount:p.cells.filter(c=>c===-1).length});
    save(false,board());
    while(!p.over){
      const role=p.turn===color?'candidate':'baseline',level=e[role];
      const moveSeed=seed+Math.floor(game/2)*104729+(moves.length+1)*7919;
      const tick=performance.now(),result=await choose(level,p,moveSeed),wallMs=performance.now()-tick;
      assert.ok(legal(p).includes(result.move));
      const d=diagnostics[role];d.decisions++;d.wallMs+=wallMs;d.nodes+=result.nodes;
      if(result.bookHit)d.bookHits++;else if(result.search==='mcts'){d.searchDecisions++;d.searchWallMs+=wallMs;}else d.policyDecisions++;
      if(result.exact)d.exact++;
      decisions.push({ply:moves.length,role,level,seed:moveSeed,move:result.move,search:result.search,
        bookHit:!!result.bookHit,nodes:result.nodes,elapsed:result.elapsed,wallMs,exact:result.exact});
      moves.push(result.move);p=play(p,result.move);assert.ok(moves.length<=60);save(false,board());
    }
    const difference=p.cells.reduce((a,b)=>a+b,0)*color;
    results.candidate[difference>0?'win':difference<0?'loss':'draw']++;
    results.baseline[difference<0?'win':difference>0?'loss':'draw']++;
    for(const role of ['candidate','baseline'] as const){const r=results[role];
      r.scoreRate=(r.win+r.draw/2)/(game+1);r.meanDiscDifference=(r.meanDiscDifference*game+difference*(role==='candidate'?1:-1))/(game+1);}
    records.push({number:game+1,pair:Math.floor(game/2),black,white,moves,decisions,finalBoard:p.cells,candidateDiscDifference:difference});
    save(game===19);
    console.log(JSON.stringify({pairing:`${e.candidate}-vs-${e.baseline}`,games:game+1,results:results.candidate,seconds:(performance.now()-started)/1000}));
  }
  for(const record of records){
    let p=initial();for(const move of record.moves){assert.ok(legal(p).includes(move));p=play(p,move);}
    assert.ok(p.over);assert.deepEqual(p.cells,record.finalBoard);
  }
  summaries.push({candidate:e.candidate,baseline:e.baseline,report:e.report,reportSha256:sha(e.report),
    games:20,results:results.candidate,diagnostics,allGamesReplayed:true});
  write(`models/${version}-comparison.json`,{version,complete:summaries.length===items.length,summaries,
    limits:'Adjacent-pair screening; book-assisted proposed ladder; no runtime/default changes or human strength estimates'});
}
for(const [path,digest] of Object.entries(preserved))assert.equal(sha(path),digest);
console.log('Completed 60 games; runtime/default hashes unchanged.');

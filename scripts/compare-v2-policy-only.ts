import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { analyze as builtinAnalyze, initial, initEngine, legal, play } from '../src/game/engine.ts';
import { analyzeBeginnerPolicy, samplePolicyResult } from '../src/game/beginner-policy.ts';
import { analyzeLegacyBeginner, initLegacyBeginner } from '../benchmarks/beginner-legacy.ts';
import type { Analysis, Position } from '../src/game/types.ts';

const read=(p:string)=>JSON.parse(readFileSync(p,'utf8'));
const sha=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
const write=(p:string,value:unknown)=>{writeFileSync(p+'.tmp',JSON.stringify(value,null,2)+'\n');renameSync(p+'.tmp',p);};
const version='v2-policy-only-T1';
const trainingPath='models/second-level-default.json',selected=read(trainingPath);
const training={model:selected.defaultModel,modelSha256:selected.modelSha256,fixtures:read('models/second-level-v2-policy-parity.json').fixtures,preservedSha256:{'models/default-model.json':sha('models/default-model.json'),'src/game/wasm/othello.mjs':sha('src/game/wasm/othello.mjs')}};
const second=read('models/second-level-selection-v2.json');
const beginner=read('models/beginner-default.json');
const experiments=[
  {id:'second',games:100,name:'当前高级权重 · 不搜索 vs 新第二档',baseline:second.model,temperature:second.temperature},
  {id:'legacy',games:50,name:'当前高级权重 · 不搜索 vs 旧入门',baseline:null,temperature:1},
  {id:'beginner',games:50,name:'当前高级权重 · 不搜索 vs 当前入门',baseline:beginner.defaultModel,temperature:beginner.temperature},
].map(e=>({...e,report:`models/${version}-vs-${e.id}-${e.games}.json`,focus:'candidate',
  displayNames:{candidate:'v2策略 · 不搜索',baseline:e.id==='second'?'第二档 · 低Elo T1.5':e.id==='legacy'?'旧入门':'当前入门'},displayOrder:['candidate','baseline']}));
for(const e of experiments)assert.ok(!existsSync(e.report),`Preserve ${e.report}`);
const old=read('models/current-experiment.json');
write('models/current-experiment.json',{version:'第二档：当前高级权重 · 不搜索 · 概率选步',experiments,
  archivedExperiments:[...old.experiments,...(old.archivedExperiments??[])]});
await initEngine();await initLegacyBeginner();
const worker=spawn('.build/venv/bin/python',['scripts/lowelo-global-policy-worker.py',training.model],{stdio:['pipe','pipe','inherit']});
const lines=createInterface({input:worker.stdout})[Symbol.asyncIterator]();
async function predict(p:Position){
  worker.stdin.write(JSON.stringify({board:p.cells,turn:p.turn})+'\n');
  const line=await lines.next();assert.ok(!line.done,'Candidate native worker exited');
  const response=JSON.parse(line.value);assert.equal(response.logits.length,64);
  assert.ok(response.logits.every(Number.isFinite));return response;
}
async function analyze(p:Position,seed:number):Promise<Analysis>{
  const result=builtinAnalyze(p,'normal',seed);
  const {logits}=await predict(p);
  assert.ok(result.candidates.every(c=>Math.abs(c.score-logits[c.move])<1e-6));
  return result;
}
const seed=20261130;
let random=seed;
const draw=()=>{random^=random<<13;random^=random>>>17;random^=random<<5;return random>>>0;};
const openings=Array.from({length:50},()=>{
  let p=initial();const moves:number[]=[];
  for(let i=0;i<8;i++){const choices=legal(p),move=choices[draw()%choices.length];moves.push(move);p=play(p,move);}
  return moves;
});
const summaries=[];
try{
  for(const f of training.fixtures){
    const out=await predict({...initial(),cells:f.board,turn:f.turn});
    for(let i=0;i<64;i++)assert.ok(Math.abs(out.logits[i]-f.logits[i])<2e-5);
  }
  for(const e of experiments){
    const baseline=e.baseline?read(e.baseline):null;
    const expectedHash=e.id==='second'?second.modelSha256:beginner.modelSha256;
    if(e.baseline)assert.equal(sha(e.baseline),expectedHash);
    const tally=()=>({win:0,draw:0,loss:0,scoreRate:0,meanDiscDifference:0});
    const results={candidate:tally(),baseline:tally()},records:any[]=[];
    const metadata={experiment:'candidate-vs-approved-baseline',version,plannedGames:e.games,
      configs:[{id:'candidate',name:e.displayNames.candidate},{id:'baseline',name:e.displayNames.baseline}],
      protocol:{seed,pairedColors:true,openingPlies:8,openings:openings.slice(0,e.games/2),useBook:false,search:false,
        model:training.model,modelSha256:training.modelSha256,temperature:1,
        trainingReport:trainingPath,trainingReportSha256:sha(trainingPath),
        baselineModel:e.baseline,baselineModelSha256:e.baseline?sha(e.baseline):null,baselineTemperature:e.temperature,
        legacySeed:'(ply+1)*7919',policySeed:'20261130+pair*104729+(ply+1)*7919',
        sources:Object.fromEntries(['scripts/compare-v2-policy-only.ts','scripts/lowelo-global-policy-worker.py',
          'src/game/beginner-policy.ts','benchmarks/beginner-legacy.ts','.build/native/libothello.dylib','src/game/wasm/othello.mjs'].map(p=>[p,sha(p)])),
        limits:'Direct trained policy sampling only; eight random legal opening moves, paired colors; not a human win-rate estimate'}};
    const started=performance.now();let actions=0;
    function save(complete:boolean,currentGame:unknown=null){
      const state={...metadata,games:records.length,complete,results,seconds:(performance.now()-started)/1000,currentGame};
      write(e.report,{...state,records});write(e.report.replace('.json','.live.json'),state);
    }
    save(false);
    for(let game=0;game<e.games;game++){
      const color=game%2===0?1:-1,black=color===1?'candidate':'baseline',white=color===-1?'candidate':'baseline';
      let p=initial();const moves=[...openings[Math.floor(game/2)]];
      for(const move of moves)p=play(p,move);
      const board=()=>({number:game+1,ply:moves.length,black,white,turn:p.turn,board:p.cells,
        blackCount:p.cells.filter(c=>c===1).length,whiteCount:p.cells.filter(c=>c===-1).length});
      save(false,board());
      while(!p.over){
        const moveSeed=seed+Math.floor(game/2)*104729+(moves.length+1)*7919;
        const result=p.turn===color?await analyze(p,moveSeed):baseline?
          samplePolicyResult(analyzeBeginnerPolicy(baseline,p),p,moveSeed,e.temperature):analyzeLegacyBeginner(p,(moves.length+1)*7919);
        assert.ok(legal(p).includes(result.move));moves.push(result.move);p=play(p,result.move);actions++;
        assert.ok(moves.length<=60);save(false,board());
      }
      const diff=p.cells.reduce((a,b)=>a+b,0)*color;
      results.candidate[diff>0?'win':diff<0?'loss':'draw']++;
      results.baseline[diff<0?'win':diff>0?'loss':'draw']++;
      for(const role of ['candidate','baseline'] as const){const r=results[role];
        r.scoreRate=(r.win+r.draw/2)/(game+1);r.meanDiscDifference=(r.meanDiscDifference*game+diff*(role==='candidate'?1:-1))/(game+1);}
      records.push({number:game+1,black,white,moves,openingPlies:8,finalBoard:p.cells,candidateDiscDifference:diff});
      save(game===e.games-1,game===e.games-1?null:board());
    }
    // Replay each game with Wasm rules, checking final boards and recorded tallies.
    for(const g of records){let p=initial();for(const move of g.moves){assert.ok(legal(p).includes(move));p=play(p,move);}
      assert.ok(p.over);assert.deepEqual(p.cells,g.finalBoard);}
    const summary={opponent:e.id,report:e.report,reportSha256:sha(e.report),results:results.candidate,
      validatedGames:records.length,candidateInference:'Actual browser normal path; native explicit approved v2 logits checked at every candidate decision',actions};
    summaries.push(summary);console.log(JSON.stringify(summary));
  }
  for(const [path,digest] of Object.entries(training.preservedSha256))assert.equal(sha(path),digest);
  write(`models/${version}-comparison.json`,{version,status:'user-selected-local-playtest',model:training.model,
    modelSha256:training.modelSha256,temperature:1,trainingReport:trainingPath,summaries,
    preservedDefaultsVerified:true,limits:'No MCTS, book, value lookahead or uniform mixture. Second level selected locally; no online deployment.'});
}finally{worker.stdin.end();}

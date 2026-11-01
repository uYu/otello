import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, renameSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { sampleBeginnerPolicy } from '../src/game/beginner-policy.ts';
import { analyzeSmallModel, initial, initEngine, legal, play } from '../src/game/engine.ts';
import { analyzeLegacyBeginner, initLegacyBeginner } from '../benchmarks/beginner-legacy.ts';
const read=(path:string)=>JSON.parse(readFileSync(path,'utf8'));
const sha=(path:string)=>createHash('sha256').update(readFileSync(path)).digest('hex');
const write=(path:string,value:unknown)=>{writeFileSync(path+'.tmp',JSON.stringify(value,null,2)+'\n');renameSync(path+'.tmp',path);};
const confirm=process.argv.includes('--confirm');
const modelIndex=process.argv.indexOf('--model');
const names=modelIndex>=0?[process.argv[modelIndex+1]]:[5,10,100].map(t=>`beginner-low-elo1300-policy-v1-T${t}`);
const seed=confirm?20261104:20261101;
const stage=confirm?'confirm':'screening';
const experiments=names.map(name=>({name:`${name} vs 改动前入门 · ${confirm?'新开局确认':'筛查'}50局`,
  temperature:Number(name.split('-T')[1]), model:`models/${name.split('-T')[0]}.json`,report:`models/${name}-legacy-${stage}-50.json`,games:50,focus:'candidate',
  displayNames:{candidate:name,baseline:'旧入门 · 75%随机+25%一手启发式'},displayOrder:['candidate','baseline']}));
for(const e of experiments){assert.ok(!existsSync(e.report),`Preserve ${e.report}`);assert.ok(existsSync(e.model));}
const old=read('models/current-experiment.json');
write('models/current-experiment.json',{version:'低Elo模仿策略 · 对旧入门各50局',experiments,
  archivedExperiments:[...old.experiments,...(old.archivedExperiments??[])]});
await initEngine();await initLegacyBeginner();
let random=seed;
const draw=()=>{random^=random<<13;random^=random>>>17;random^=random<<5;return random>>>0;};
const openings=Array.from({length:25},()=>{
  let p=initial();const moves:number[]=[];
  for(let i=0;i<8;i++){const choices=legal(p),move=choices[draw()%choices.length];moves.push(move);p=play(p,move);}
  return moves;
});
const summaries=[];
for(const e of experiments){
  const model=read(e.model),training=read(e.model.replace('.json','-training.json'));
  assert.equal(sha(e.model),training.modelSha256);
  const tally=()=>({win:0,draw:0,loss:0,scoreRate:0,meanDiscDifference:0});
  const results={candidate:tally(),baseline:tally()},records:unknown[]=[];
  const diagnostics={decisions:0,cornersAvailable:0,cornersTaken:0,avoidableCornerGifts:0,passes:0,ms:0};
  const started=performance.now();
  const metadata={experiment:'candidate-vs-approved-baseline',version:model.version,plannedGames:50,
    configs:[{id:'candidate',name:model.version},{id:'baseline',name:'旧入门 · 75%随机+25%一手启发式'}],
    protocol:{stage,seed,openings,openingPlies:8,pairedColors:true,useBook:false,model:e.model,modelSha256:sha(e.model),
      trainingReport:e.model.replace('.json','-training.json'),trainingReportSha256:sha(e.model.replace('.json','-training.json')),
      baseline:'Exact pre-model browser 75% legal-random/25% top-three one-ply heuristic; seed=(ply+1)*7919; no book',
      baselineSourceSha256:sha('benchmarks/beginner-legacy.ts'),wasmSha256:sha('src/game/wasm/othello.mjs'),
      temperature:e.temperature,inferenceSourceSha256:{policy:sha('src/game/beginner-policy.ts'),value:sha('src/game/beginner.ts'),selection:sha('src/game/engine.ts')},scriptSha256:sha('scripts/compare-beginner-low-elo-sampling-flat.ts'),
      selection:'EXPERIMENTAL learned softmax policy sampling; no uniform random mixture, heuristic, value, book or search. Pending user choice; no production change',
      limits:'50 paired-color screening/confirmation games; internal Elo is not official; no human win-rate or statistical significance claim'}};
  const snapshot=(complete:boolean,currentGame:unknown=null)=>({...metadata,games:records.length,complete,results,diagnostics,
    seconds:(performance.now()-started)/1000,currentGame,records});
  const live=(complete:boolean,currentGame:unknown=null)=>{
    const {records:_records,...state}=snapshot(complete,currentGame);write(e.report.replace('.json','.live.json'),state);
  };
  write(e.report,snapshot(false));
  for(let game=0;game<50;game++){
    const color=game%2===0?1:-1,black=color===1?'candidate':'baseline',white=color===-1?'candidate':'baseline';
    let p=initial();const moves=[...openings[Math.floor(game/2)]];
    for(const move of moves)p=play(p,move);
    const board=()=>({number:game+1,ply:moves.length,black,white,turn:p.turn,board:p.cells,
      blackCount:p.cells.filter(c=>c===1).length,whiteCount:p.cells.filter(c=>c===-1).length});
    live(false,board());
    while(!p.over){
      const candidate=p.turn===color,startedMove=performance.now();
      const result=candidate?sampleBeginnerPolicy(model,p,(moves.length+1)*7919,e.temperature):analyzeLegacyBeginner(p,(moves.length+1)*7919);
      const elapsed=performance.now()-startedMove;
      assert.ok(legal(p).includes(result.move));
      if(candidate){
        diagnostics.decisions++;diagnostics.ms+=elapsed;
        assert.ok(result.candidates.some(c=>c.move===result.move));
        const corner=(m:number)=>[0,7,56,63].includes(m),choices=legal(p);
        if(choices.some(corner)){diagnostics.cornersAvailable++;if(corner(result.move))diagnostics.cornersTaken++;}
        const gift=(m:number)=>{const next=play(p,m);return !next.over&&next.turn!==p.turn&&legal(next).some(corner);};
        if(!choices.some(corner)&&gift(result.move)&&choices.some(m=>!gift(m)))diagnostics.avoidableCornerGifts++;
      }
      moves.push(result.move);p=play(p,result.move);
      if(p.passed)diagnostics.passes++;
      live(false,board());assert.ok(moves.length<=60);
    }
    const difference=p.cells.reduce((a,b)=>a+b,0)*color;
    results.candidate[difference>0?'win':difference<0?'loss':'draw']++;
    results.baseline[difference<0?'win':difference>0?'loss':'draw']++;
    for(const role of ['candidate','baseline'] as const){
      const r=results[role];r.scoreRate=(r.win+r.draw/2)/(game+1);
      r.meanDiscDifference=(r.meanDiscDifference*game+difference*(role==='candidate'?1:-1))/(game+1);
    }
    records.push({number:game+1,black,white,moves,openingPlies:8,finalBoard:p.cells,candidateDiscDifference:difference});
    write(e.report,snapshot(game===49));live(game===49);
  }
  summaries.push({model:e.model,temperature:e.temperature,report:e.report,results:results.candidate,diagnostics,
    cornerChoiceRate:diagnostics.cornersTaken/diagnostics.cornersAvailable,meanMoveMs:diagnostics.ms/diagnostics.decisions});
  console.log(JSON.stringify(summaries[summaries.length-1]));
}
write(`models/beginner-low-elo1300-sampling-flat-${stage}-summary.json`,{stage,seed,comparisons:summaries,
  limits:'Screening results are used to select low strength; confirmation uses fresh openings. No human win-rate claim.'});

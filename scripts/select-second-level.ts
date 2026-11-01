import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, renameSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { analyzeLocalPolicy } from '../src/game/beginner-local.ts';
import { analyzeBeginnerPolicy, samplePolicyResult } from '../src/game/beginner-policy.ts';
import { analyzeSmallModel, initial, initEngine, legal, play } from '../src/game/engine.ts';
import { analyzeLegacyBeginner, initLegacyBeginner } from '../benchmarks/beginner-legacy.ts';
const read=(path:string)=>JSON.parse(readFileSync(path,'utf8'));
const sha=(path:string)=>createHash('sha256').update(readFileSync(path)).digest('hex');
const write=(path:string,value:unknown)=>{writeFileSync(path+'.tmp',JSON.stringify(value,null,2)+'\n');renameSync(path+'.tmp',path);};
const confirm=true;
const current=process.argv.includes('--current');
const games=current?100:50;
const active=read('models/beginner-default.json');
const activeModel=read(active.defaultModel);
assert.equal(sha(active.defaultModel),active.modelSha256);
const modelIndex=process.argv.indexOf('--model');
const profiles=modelIndex>=0?[process.argv[modelIndex+1]]:['error80e2-T1'];
const seed=20261117;
const stage=current?'vs-current':'vs-legacy';
const experiments=profiles.map(profile=>{
  const [kind,temp]=profile.split('-T');
  const name=kind==='error80e2lowLR'?'beginner-error-policy-80-e2-lr0003-v1':kind==='puree2lowLR'?'beginner-pure-lowelo-e2-lr0003-v1':kind==='error80e2'?'beginner-error-policy-80-e2-v1':kind.startsWith('error')?`beginner-error-policy-${kind.slice(5)}-v1`:kind==='local'?'beginner-low-elo1300-local4-v1':'beginner-low-elo1300-policy-v1';
  return {profile,temperature:Number(temp),name:`第二档候选 ${profile} vs ${current?'当前入门':'旧入门'} · ${games}局`,
    model:`models/${name}.json`,report:`models/second-level-${profile}-${stage}-${games}.json`,games,focus:'candidate',
    displayNames:{candidate:`低Elo ${profile}`,baseline:current?'当前入门 · 2轮低学习率':'旧入门 · 75%随机+25%一手启发式'},displayOrder:['candidate','baseline']};
});
for(const e of experiments){assert.ok(!existsSync(e.report),`Preserve ${e.report}`);assert.ok(existsSync(e.model));}
const old=read('models/current-experiment.json');
write('models/current-experiment.json',{version:'低Elo模仿策略 · 对旧入门各50局',experiments,
  archivedExperiments:[...old.experiments,...(old.archivedExperiments??[])]});
await initEngine();await initLegacyBeginner();
let random=seed;
const draw=()=>{random^=random<<13;random^=random>>>17;random^=random<<5;return random>>>0;};
const openings=Array.from({length:games/2},()=>{
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
  const metadata={experiment:'candidate-vs-approved-baseline',version:model.version,plannedGames:games,
    configs:[{id:'candidate',name:model.version},{id:'baseline',name:current?active.version:'旧入门 · 75%随机+25%一手启发式'}],
    protocol:{stage,seed,openings,openingPlies:8,pairedColors:true,useBook:false,model:e.model,modelSha256:sha(e.model),
      trainingReport:e.model.replace('.json','-training.json'),trainingReportSha256:sha(e.model.replace('.json','-training.json')),
      baselineModel:current?active.defaultModel:null,baselineModelSha256:current?active.modelSha256:null,baselineTemperature:current?active.temperature:null,
      baseline:current?'Current approved beginner; trained policy sampling with same paired seed formula as candidate':'Exact pre-model browser 75% legal-random/25% top-three one-ply heuristic; seed=(ply+1)*7919; no book',
      baselineSourceSha256:sha('benchmarks/beginner-legacy.ts'),wasmSha256:sha('src/game/wasm/othello.mjs'),
      profile:e.profile,temperature:e.temperature,inferenceSourceSha256:{local:sha('src/game/beginner-local.ts'),policy:sha('src/game/beginner-policy.ts'),value:sha('src/game/beginner.ts'),selection:sha('src/game/engine.ts')},scriptSha256:sha('scripts/select-second-level.ts'),
      selection:'User-approved sampling exclusively from trained low-Elo policy softmax(logits/T); no uniform mixture or search. Candidate PRNG mixes board and pair seed; same profile and openings under paired colors.',
      limits:'Paired-color confirmation games; see plannedGames; internal Elo is not official; no human win-rate or statistical significance claim'}};
  const snapshot=(complete:boolean,currentGame:unknown=null)=>({...metadata,games:records.length,complete,results,diagnostics,
    seconds:(performance.now()-started)/1000,currentGame,records});
  const live=(complete:boolean,currentGame:unknown=null)=>{
    const {records:_records,...state}=snapshot(complete,currentGame);write(e.report.replace('.json','.live.json'),state);
  };
  write(e.report,snapshot(false));
  for(let game=0;game<games;game++){
    const color=game%2===0?1:-1,black=color===1?'candidate':'baseline',white=color===-1?'candidate':'baseline';
    let p=initial();const moves=[...openings[Math.floor(game/2)]];
    for(const move of moves)p=play(p,move);
    const board=()=>({number:game+1,ply:moves.length,black,white,turn:p.turn,board:p.cells,
      blackCount:p.cells.filter(c=>c===1).length,whiteCount:p.cells.filter(c=>c===-1).length});
    live(false,board());
    while(!p.over){
      const candidate=p.turn===color,startedMove=performance.now();
      const result=candidate?samplePolicyResult(model.kernelWeights?analyzeLocalPolicy(model,p):analyzeBeginnerPolicy(model,p),p,seed+Math.floor(game/2)*104729+(moves.length+1)*7919,e.temperature):current?samplePolicyResult(analyzeBeginnerPolicy(activeModel,p),p,seed+Math.floor(game/2)*104729+(moves.length+1)*7919,active.temperature):analyzeLegacyBeginner(p,(moves.length+1)*7919);
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
    write(e.report,snapshot(game===games-1));live(game===games-1);
  }
  summaries.push({profile:e.profile,temperature:e.temperature,model:e.model,report:e.report,results:results.candidate,diagnostics,
    cornerChoiceRate:diagnostics.cornersTaken/diagnostics.cornersAvailable,meanMoveMs:diagnostics.ms/diagnostics.decisions});
  console.log(JSON.stringify(summaries[summaries.length-1]));
}
write(`models/second-level-${profiles.join("_")}-${stage}-summary.json`,{stage,seed,comparisons:summaries,
  limits:'Screening results are used to select low strength; confirmation uses fresh openings. No human win-rate claim.'});

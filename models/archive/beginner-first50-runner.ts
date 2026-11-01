import { readFileSync, writeFileSync, existsSync, renameSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { analyze, initial, initEngine, legal, play, score } from '../src/game/engine.ts';
import { analyzeLegacyBeginner, initLegacyBeginner } from '../benchmarks/beginner-legacy.ts';
import { analyzeBoardCandidate } from '../src/game/beginner-candidates.ts';
import type { Position } from '../src/game/types.ts';

const sha = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
const read = (path: string) => JSON.parse(readFileSync(path,'utf8'));
function write(path: string, value: unknown) {
  writeFileSync(path+'.tmp',JSON.stringify(value,null,2)+'\n');
  renameSync(path+'.tmp',path);
}
const models = {
  tiny: {name:'小价值 64→8→1', model:'models/beginner8-v1.json', parameters:529},
  policy: {name:'直接策略 8×8×3→8×8', model:'models/beginner-board-heads-v1.json', parameters:12352},
  value: {name:'价值 8×8×3→32→1', model:'models/beginner-board-heads-v1.json', parameters:6209},
  legacy: {name:'改动前入门 · 75%随机+25%一手启发式', model:'benchmarks/beginner-legacy.ts', parameters:0},
  v2: {name:'冻结 v2 · 100ms', model:'models/pattern32-global-policy-v2.otpat', parameters:916987},
};
type Player = keyof typeof models;
const pairs: [Player,Player][] = [['tiny','legacy'],['policy','legacy'],['value','legacy']];
const experiments = pairs.map(([candidate,opponent]) => ({candidate,opponent,
  report:`models/beginner-${candidate}-vs-${opponent}-50.json`,
  name:`${models[candidate].name} vs ${models[opponent].name}`,
  focus:'candidate',displayNames:{candidate:models[candidate].name,baseline:models[opponent].name},
  displayOrder:['candidate','baseline'],games:50,
}));
for (const experiment of experiments) if (existsSync(experiment.report)) throw new Error(`Preserve existing results: ${experiment.report}`);
const baseline = read('models/v2-default.json');
if (sha(models.v2.model) !== baseline.modelSha256) throw new Error('Approved v2 model checksum differs');
const current = read('models/current-experiment.json');
write('models/current-experiment.json',{version:'三种入门模型 vs 改动前入门 · 各50局 · 无开局库',experiments,
  archivedExperiments:[...current.experiments.map((e: object)=>({...e,status:'stopped'})),...(current.archivedExperiments??[])]});
await initEngine();
await initLegacyBeginner();
let rng=20261101;
const random = () => {rng^=rng<<13;rng^=rng>>>17;rng^=rng<<5;return rng>>>0;};
const openings = Array.from({length:25}, () => {
  let p=initial();const moves:number[]=[];
  for(let ply=0;ply<8;ply++){const choices=legal(p);const move=choices[random()%choices.length];moves.push(move);p=play(p,move);}
  return moves;
});
function decide(player: Player,p: Position,seed: number) {
  if(player==='legacy')return analyzeLegacyBeginner(p,seed);
  if(player==='tiny')return analyze(p,'easy',seed);
  if(player==='v2')return analyze(p,'normal',seed);
  return analyzeBoardCandidate(p,player);
}
const suiteStarted=performance.now();
for(const experiment of experiments){
  const {candidate,opponent}=experiment, started=performance.now();
  const configs=[{id:'candidate',name:models[candidate].name},{id:'baseline',name:models[opponent].name}];
  const tally = () => ({win:0,draw:0,loss:0,scoreRate:0,meanDiscDifference:0,meanWallMs:0,meanSearchUnits:0});
  const results={candidate:tally(),baseline:tally()};
  const timing={candidate:{ms:0,decisions:0,nodes:0},baseline:{ms:0,decisions:0,nodes:0}};
  const records:unknown[]=[];
  const metadata={experiment:'candidate-vs-approved-baseline',version:'beginner-architecture-v1',plannedGames:50,configs,
    protocol:{seed:20261101,openingPlies:8,openings,pairedColors:true,useBook:false,
      candidate:{...models[candidate],sha256:sha(models[candidate].model)},
      opponent:{...models[opponent],sha256:sha(models[opponent].model)},
      baselineManifest:opponent==='v2'?'models/v2-default.json':null,
      baselineMilliseconds:opponent==='v2'?100:null,
      legacy:'Exact pre-model browser recipe, 75% uniformly sampled legal moves plus 25% seeded one-ply heuristic top-three; book disabled; seed (placement count+1)*7919 as in the browser',
      legacyWasmSha256:sha('src/game/wasm/othello.mjs'),
      selection:'Deterministic legal argmax; tiny/value evaluate only immediate successor; policy directly masks 64 logits. D4 mean. No noise, book or terminal override.',
      scriptSha256:sha('scripts/compare-beginner.ts'),wasmSha256:sha('src/game/wasm/othello.mjs'),
      limits:'50 paired-color screening games per pairing; no Elo, significance, or human win-rate claim. Unequal search budgets are intentional. No default promotion.'}};
  const snapshot = (complete:boolean,currentGame:unknown=null) => ({...metadata,games:records.length,complete,results,
    seconds:(performance.now()-started)/1000,currentGame,records});
  const live = (currentGame:unknown) => {
    const {records:_records,...state}=snapshot(false,currentGame);
    write(experiment.report.replace(/\.json$/,'.live.json'),state);
  };
  write(experiment.report,snapshot(false));
  for(let game=0;game<50;game++){
    const candidateColor=game%2===0?1:-1;
    const black=candidateColor===1?'candidate':'baseline',white=candidateColor===-1?'candidate':'baseline';
    let p=initial();const moves=[...openings[Math.floor(game/2)]];
    for(const move of moves)p=play(p,move);
    const board = () => ({number:game+1,ply:moves.length,black,white,board:p.cells,blackCount:score(p).black,whiteCount:score(p).white,turn:p.turn});
    live(board());
    while(!p.over){
      const role=p.turn===candidateColor?'candidate':'baseline';
      const player=role==='candidate'?candidate:opponent;
      const before=performance.now(),result=decide(player,p,(moves.length+1)*7919);
      timing[role].ms+=performance.now()-before;timing[role].decisions++;timing[role].nodes+=result.nodes;
      if(!legal(p).includes(result.move))throw new Error(`Illegal ${player} move`);
      moves.push(result.move);p=play(p,result.move);live(board());
      if(moves.length>60)throw new Error('Too many placements');
    }
    const difference=(score(p).black-score(p).white)*candidateColor;
    results.candidate[difference>0?'win':difference<0?'loss':'draw']++;
    results.baseline[difference<0?'win':difference>0?'loss':'draw']++;
    for(const role of ['candidate','baseline'] as const){
      const total=results[role],n=game+1;
      total.scoreRate=(total.win+total.draw/2)/n;
      total.meanDiscDifference=(total.meanDiscDifference*game+difference*(role==='candidate'?1:-1))/n;
      total.meanWallMs=timing[role].ms/timing[role].decisions;
      total.meanSearchUnits=timing[role].nodes/timing[role].decisions;
    }
    records.push({number:game+1,black,white,moves,openingPlies:8,finalBoard:p.cells,candidateDiscDifference:difference});
    write(experiment.report,snapshot(game===49));
    const {records:_records,...state}=snapshot(game===49);
    write(experiment.report.replace(/\.json$/,'.live.json'),state);
    console.log(`${candidate} vs ${opponent}: ${game+1}/50 · ${results.candidate.win}W/${results.candidate.draw}D/${results.candidate.loss}L`);
    // Let the live monitor and new user input be observed between games.
    await new Promise(resolve=>setTimeout(resolve,0));
  }
}
write('models/beginner-legacy50-comparison.json',{version:'beginner-architecture-v1',
  trainingReports:['models/beginner8-v1-training.json','models/beginner-board-heads-v1-training.json'],
  comparisons:experiments.map(e=>({name:e.name,report:e.report,results:read(e.report).results.candidate})),
  seconds:(performance.now()-suiteStarted)/1000,selectionPending:true,
  limits:'Screening only. Suitable human beginner win rates require user playtesting; v2 remains the approved general baseline.'});

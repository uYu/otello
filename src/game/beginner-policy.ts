import { legal } from './engine.ts';
import type { Analysis, Color, Position } from './types.ts';
export interface BeginnerPolicyModel {
  version: string;
  policyWeights: number[][];
  policyBias: number[];
  inferenceViews?: number;
}
const transforms=Array.from({length:8},(_,symmetry)=>Array.from({length:64},(_,i)=>{
  let x=i%8,y=Math.floor(i/8);
  if(symmetry&4)[x,y]=[y,x];
  if(symmetry&1)x=7-x;
  if(symmetry&2)y=7-y;
  return y*8+x;
}));
export function beginnerPolicyLogits(model:BeginnerPolicyModel,cells:readonly number[],side:Color){
  const logits=Array<number>(64).fill(0);
  const views=model.inferenceViews===1?transforms.slice(0,1):transforms;
  for(const mapping of views){
    const channels=mapping.map(i=>cells[i]*side===1?0:cells[i]*side===-1?1:2);
    for(let square=0;square<64;square++){
      let value=model.policyBias[square];
      for(let i=0;i<64;i++)value+=model.policyWeights[square][i*3+channels[i]];
      logits[mapping[square]]+=value/views.length;
    }
  }
  return logits;
}
export function analyzeBeginnerPolicy(model:BeginnerPolicyModel,position:Position):Analysis{
  const started=performance.now(),logits=beginnerPolicyLogits(model,position.cells,position.turn);
  const candidates=legal(position).map(move=>({move,score:logits[move]})).sort((a,b)=>b.score-a.score||a.move-b.move);
  return {move:candidates[0]?.move??-1,score:candidates[0]?.score??0,depth:0,nodes:1,
    elapsed:performance.now()-started,ttHits:null,exact:false,candidates,search:'direct-policy',evaluator:'beginner',
    engine:'builtin',engineVersion:model.version,useBook:false,bookHit:false};
}

// Seeded selection from the trained policy distribution.
// Every probability comes from the trained network; there is no uniform-policy mixture.
export function sampleBeginnerPolicy(model:BeginnerPolicyModel,position:Position,seed:number,temperature=1):Analysis{
  return samplePolicyResult(analyzeBeginnerPolicy(model,position),position,seed,temperature);
}

export function samplePolicyResult(result:Analysis,position:Position,seed:number,temperature:number):Analysis{
  if(!Number.isFinite(temperature)||temperature<=0)throw new Error('Invalid policy temperature');
  if(!result.candidates.length)return result;
  const maximum=result.candidates[0].score;
  const weights=result.candidates.map(c=>Math.exp((c.score-maximum)/temperature));
  let random=((seed^0x9e3779b9)>>>0)||1;
  // A board-dependent draw avoids repeating the same probability quantile at
  // every occurrence of a given ply. The supplied game seed keeps replays reproducible.
  for(const cell of position.cells)random=Math.imul(random^(cell*position.turn+1),16777619)>>>0;
  random ||= 1;
  random^=random<<13;random^=random>>>17;random^=random<<5;
  let draw=(random>>>0)/4294967296*weights.reduce((a,b)=>a+b,0);
  let chosen=result.candidates.at(-1)!;
  for(let i=0;i<weights.length;i++){
    draw-=weights[i];if(draw<=0){chosen=result.candidates[i];break;}
  }
  return {...result,move:chosen.move,score:chosen.score};
}

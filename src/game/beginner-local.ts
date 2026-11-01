import { legal } from './engine.ts';
import type { Analysis, Color, Position } from './types.ts';
export interface LocalPolicyModel {
  version: string;
  hidden: number;
  kernelWeights: number[][];
  hiddenBias: number[];
  outputWeights: number[] | null;
  outputBias: number;
}
export function localPolicyLogits(model:LocalPolicyModel,cells:readonly number[],side:Color):number[]{
  return Array.from({length:64},(_,square)=>{
    const x=square%8,y=Math.floor(square/8);
    let output=model.outputBias;
    for(let h=0;h<model.hidden;h++){
      let hidden=model.hiddenBias[h];
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
        const cx=x+dx,cy=y+dy;
        if(cx>=0&&cx<8&&cy>=0&&cy<8)hidden+=cells[cy*8+cx]*side*model.kernelWeights[h][(dy+1)*3+dx+1];
      }
      output+=model.outputWeights?Math.tanh(hidden)*model.outputWeights[h]:hidden;
    }
    return output;
  });
}
export function analyzeLocalPolicy(model:LocalPolicyModel,p:Position):Analysis{
  const started=performance.now(),logits=localPolicyLogits(model,p.cells,p.turn);
  const candidates=legal(p).map(move=>({move,score:logits[move]})).sort((a,b)=>b.score-a.score||a.move-b.move);
  return {move:candidates[0]?.move??-1,score:candidates[0]?.score??0,candidates,depth:0,nodes:1,
    elapsed:performance.now()-started,ttHits:null,exact:false,search:'direct-policy',evaluator:'beginner',
    engine:'builtin',engineVersion:model.version,useBook:false,bookHit:false};
}

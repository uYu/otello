// Frozen pre-model browser beginner: 75% uniform legal sampling, 25% legacy top-three heuristic.
import createModule from '../src/game/wasm/othello.mjs';
import type { Analysis, Position } from '../src/game/types.ts';
let module: Awaited<ReturnType<typeof createModule>>;
export async function initLegacyBeginner() { module = await createModule(); }
export function analyzeLegacyBeginner(p: Position, seed: number): Analysis {
  const started=performance.now(), input=module._ot_input(), output=module._ot_output();
  module.HEAP32.set(p.cells,input/4);
  if(module._ot_search(input,p.turn,80,1,0,seed,output)!==1)throw new Error('Legacy beginner failed');
  const out=Array.from(module.HEAP32.subarray(output/4,output/4+136));
  const candidates=Array.from({length:out[6]},(_,i)=>({move:out[7+i*2],score:out[8+i*2]}));
  let move=out[0],value=out[1];
  if(candidates.length){
    let random=((seed^0x9e3779b9)>>>0)||1;
    const draw=()=>{random^=random<<13;random^=random>>>17;random^=random<<5;return(random>>>0)/4294967296;};
    if(draw()<0.75){
      const choices=[...candidates].sort((a,b)=>a.move-b.move);
      const chosen=choices[Math.floor(draw()*choices.length)];move=chosen.move;value=chosen.score;
    }
  }
  return {move,score:value,depth:out[2],elapsed:performance.now()-started,nodes:out[4],ttHits:out[135],exact:!!out[5],
    search:'ab',evaluator:'heuristic',candidates,useBook:false,bookHit:false};
}

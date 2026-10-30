import assert from 'node:assert/strict';
import createOld from '../src/game/wasm/othello.mjs';
import createNew from '../.build/engineering/othello.mjs';
const a=await createOld(),b=await createNew();let steps=0,passes=0;
function legal(module,cells,turn){const input=module._ot_input(),output=module._ot_output();module.HEAP32.set(cells,input/4);const n=module._ot_legal(input,turn,output);assert.ok(n>=0);return [...module.HEAP32.subarray(output/4,output/4+n)];}
for(let game=0;game<20;game++){
 let cells=Array(64).fill(0);cells[28]=cells[35]=1;cells[27]=cells[36]=-1;let turn=1;
 for(let ply=0;ply<120;ply++){
  const moves=legal(a,cells,turn);assert.deepEqual(legal(b,cells,turn),moves);
  if(!moves.length&&!legal(a,cells,-turn).length)break;
  const move=moves.length?moves[(game*19+ply*7)%moves.length]:-1;if(move<0)passes++;
  for(const module of [a,b])module.HEAP32.set(cells,module._ot_input()/4);
  assert.equal(a._ot_default_evaluate(a._ot_input(),turn,a._ot_output()),1);assert.equal(b._ot_default_evaluate(b._ot_input(),turn,b._ot_output()),1);
  assert.equal(a.HEAP32[a._ot_output()/4],b.HEAP32[b._ot_output()/4]);
  assert.equal(a._ot_play(a._ot_input(),turn,move,a._ot_output()),1);
  const out=b._ot_output();assert.equal(b._ot_play_with_flips(b._ot_input(),turn,move,out,out+67*4),1);
  const old=[...a.HEAP32.subarray(a._ot_output()/4,a._ot_output()/4+67)],next=[...b.HEAP32.subarray(out/4,out/4+67)];assert.deepEqual(old,next);
  const flipped=[...b.HEAP32.subarray(out/4+67,out/4+67+next[66])];
  assert.deepEqual(flipped,cells.flatMap((value,i)=>value!==0&&value!==next[i]?[i]:[]));
  cells=next.slice(0,64);turn=next[64];steps++;
 }
}
console.log(JSON.stringify({wasmParity:true,games:20,steps,passes,checks:'legal masks, value scores, board transitions, pass handling and reused flip lists'}));

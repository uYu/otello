import createModule from './wasm/direct-policy.mjs';
import { legal } from './engine.ts';
import { samplePolicyResult } from './beginner-policy.ts';
import type { Analysis, Position } from './types.ts';
import selected from '../../models/second-level-default.json' with { type: 'json' };
let module: Awaited<ReturnType<typeof createModule>>;
let pending: Promise<void> | undefined;
export function initDirectPolicy() {
  return pending ??= createModule().then(m => {
    module = m;
    const pointer=m._ot_policy_model_sha256();
    const hash=new TextDecoder().decode(m.HEAPU8.subarray(pointer,pointer+64));
    if(hash!==selected.modelSha256)throw new Error('第二档模型校验失败');
  });
}
export function directPolicyLogits(position:Position):number[] {
  module.HEAP32.set(position.cells,module._ot_policy_input()/4);
  if(module._ot_policy_infer(position.turn)!==1)throw new Error('第二档策略评估失败');
  const pointer=module._ot_policy_output()/8;
  return Array.from(module.HEAPF64.subarray(pointer,pointer+64));
}
export function analyzeDirectPolicy(position:Position,seed:number):Analysis {
  const started=performance.now(),logits=directPolicyLogits(position);
  const candidates=legal(position).map(move=>({move,score:logits[move]})).sort((a,b)=>b.score-a.score||a.move-b.move);
  return samplePolicyResult({move:candidates[0]?.move??-1,score:candidates[0]?.score??0,
    depth:0,nodes:1,elapsed:performance.now()-started,ttHits:null,exact:false,candidates,
    engine:'builtin',engineVersion:selected.version,search:'direct-policy',evaluator:'nnue',useBook:false,bookHit:false},
    position,seed,selected.temperature);
}

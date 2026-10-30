import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import createModule from '../src/game/wasm/othello.mjs';
const f=JSON.parse(readFileSync('.build/v2-default-check/fixture.json','utf8'));
const m=await createModule(),input=m._ot_input(),out=m._ot_output(),probs=(out+15)&~7;
const p=m._ot_default_model_sha256();assert.equal(new TextDecoder().decode(m.HEAPU8.subarray(p,p+64)),f.modelSha256);
let maximumPolicyError=0;
for(const r of f.positions){
 m.HEAP32.set(r.board,input/4);
 assert.equal(m._ot_default_evaluate(input,r.turn,out),1);assert.equal(m.HEAP32[out/4],r.score);
 assert.equal(m._ot_default_infer(input,r.turn,out,probs),1);assert.equal(m.HEAP32[out/4],r.score);
 for(let i=0;i<65;i++)maximumPolicyError=Math.max(maximumPolicyError,Math.abs(m.HEAPF64[probs/8+i]-r.probabilities[i]));
}
assert.ok(maximumPolicyError<1e-12,`Wasm/native policy error: ${maximumPolicyError}`);
const result={wasmEmbeddedModelSha256:f.modelSha256,wasmNativeExactlyEqualValuePositions:f.positions.length,wasmNativePolicyPositions:f.positions.length,policyProbabilitiesPerPosition:65,maximumPolicyError};
writeFileSync('.build/v2-default-check/wasm.json',JSON.stringify(result,null,2)+'\n');console.log(result);

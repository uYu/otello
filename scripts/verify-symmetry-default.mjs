import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import createModule from '../src/game/wasm/othello.mjs';
const fixture=JSON.parse(readFileSync('.build/v1.37-default-check/fixture.json','utf8'));
const m=await createModule(),input=m._ot_input(),output=m._ot_output();
const p=m._ot_default_model_sha256();assert.equal(new TextDecoder().decode(m.HEAPU8.subarray(p,p+64)),fixture.modelSha256);
for(const row of fixture.positions){
 m.HEAP32.set(row.board,input/4);assert.equal(m._ot_default_evaluate(input,row.turn,output),1);assert.equal(m.HEAP32[output/4],row.score);
}
const result={wasmEmbeddedModelSha256:fixture.modelSha256,wasmNativeExactlyEqualPositions:fixture.positions.length};
writeFileSync('.build/v1.37-default-check/wasm.json',JSON.stringify(result,null,2)+'\n');console.log(result);

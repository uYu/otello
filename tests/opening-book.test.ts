import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {initEngine,initial,legal,play,analyze} from '../src/game/engine.ts';
import {BOOK_SOURCE_SHA,BOOK_VERSION,bookShard,canonicalBookPosition,createOpeningBookClient,openingBookAssetBase} from '../src/game/opening-book.ts';
import type {Position} from '../src/game/types.ts';
const prefix=`/opening-book/${BOOK_VERSION}/`;
const reads:string[]=[];
const diskFetch:typeof fetch=async (input)=>{
 const name=String(input);reads.push(name);assert.ok(name.startsWith(prefix));
 try{return new Response(readFileSync(resolve('public',name.slice(1))),{headers:{'Content-Type':'application/json'}});}
 catch{return new Response('',{status:404});}
};
await initEngine();
test('bundled worker loads book data beside assets, including subdirectory deployments',()=>{
 assert.equal(openingBookAssetBase('http://example.test/assets/ai.worker-123.js'),`http://example.test${prefix}`);
 assert.equal(openingBookAssetBase('http://example.test/games/otello/assets/ai.worker-123.js'),`http://example.test/games/otello${prefix}`);
});
test('mandatory book serves legal opening moves with fixed source and no search',async()=>{
 const client=createOpeningBookClient(diskFetch),position=initial();
 const source=JSON.parse(readFileSync(`public/opening-book/${BOOK_VERSION}/SOURCE.json`,'utf8'));
 assert.equal(source.sha256,BOOK_SOURCE_SHA);
 for(let seed=1;seed<=32;seed++){
  const result=await client.lookup(position,seed);assert.ok(result);assert.ok(legal(position).includes(result.move));
  assert.equal(result.bookHit,true);assert.equal(result.useBook,true);assert.equal(result.nodes,0);assert.equal(result.search,'book');
 }
 assert.equal(reads.filter(name=>name.endsWith('/manifest.json')).length,1);assert.equal(reads.length,2);
});
test('book keys preserve all board symmetries and side-to-move color renaming',async()=>{
 const p=play(initial(),19);const client=createOpeningBookClient(diskFetch);const key=canonicalBookPosition(p).key;
 for(let sym=0;sym<8;sym++){
  const cells=Array(64).fill(0);
  for(let i=0;i<64;i++){let x=i%8,y=Math.floor(i/8);if(sym&4)[x,y]=[y,x];if(sym&1)x=7-x;if(sym&2)y=7-y;cells[y*8+x]=p.cells[i];}
  const q={...p,cells};assert.equal(canonicalBookPosition(q).key,key);const result=await client.lookup(q,71);assert.ok(result);assert.ok(legal(q).includes(result.move));
 }
 const renamed={...p,cells:p.cells.map(c=>-c),turn:-p.turn as 1|-1};assert.equal(canonicalBookPosition(renamed).key,key);
 assert.deepEqual((await client.lookup(p,81))?.move,(await client.lookup(renamed,81))?.move);
});
test('only a verified miss returns to the approved default search; terminals skip book',async()=>{
 const client=createOpeningBookClient(diskFetch);let p=initial(),miss:Position|undefined;
 for(let ply=0;ply<50&&!p.over;ply++){
  const result=await client.lookup(p,ply+1);if(!result){miss=p;break;}
  const moves=legal(p);p=play(p,moves[(ply*13+3)%moves.length]);
 }
 assert.ok(miss,'An off-book reachable line should be found');const result=analyze(miss,'normal',91);assert.ok(legal(miss).includes(result.move));assert.equal(result.evaluator,'nnue');
 const terminal={...initial(),cells:Array(64).fill(1),over:true};assert.equal(await client.lookup(terminal,1),null);
});
test('missing or corrupt book is an error and a failed load can be retried',async()=>{
 let fail=true;const fetcher:typeof fetch=async (...args)=>{if(fail){fail=false;return new Response('',{status:503});}return diskFetch(...args);};
 const client=createOpeningBookClient(fetcher);await assert.rejects(()=>client.lookup(initial(),1),/开局库加载失败/);assert.ok(await client.lookup(initial(),1));
 const corrupt=createOpeningBookClient(async ()=>new Response(JSON.stringify({format:'otello-web-book-v1',enabled:false,shards:4096,sourceSha256:BOOK_SOURCE_SHA})));
 await assert.rejects(()=>corrupt.lookup(initial(),1),/校验失败/);
 const key=canonicalBookPosition(initial()).key;assert.match(bookShard(key),/^[0-9a-f]{3}$/);
});

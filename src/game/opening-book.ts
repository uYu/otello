import { legal } from './engine.ts';
import type { Analysis, Position } from './types.ts';
export const BOOK_VERSION='0dd454d84aa6-v1';
export const BOOK_SOURCE_SHA='0dd454d84aa656a64329930bad796c4a48a4322b7693d50b7040006383db5012';
export function openingBookAssetBase(workerUrl:string){return new URL(`../opening-book/${BOOK_VERSION}/`,workerUrl).href;}
const base=typeof self==='undefined'?`/opening-book/${BOOK_VERSION}/`:import.meta.env?.DEV
 ?`${import.meta.env.BASE_URL}opening-book/${BOOK_VERSION}/`:openingBookAssetBase(self.location.href);
const bits=Array.from({length:64},(_,i)=>1n<<BigInt(i));
const transforms=Array.from({length:8},(_,symmetry)=>Array.from({length:64},(_,i)=>{
 let x=i%8,y=Math.floor(i/8);if(symmetry&4)[x,y]=[y,x];if(symmetry&1)x=7-x;if(symmetry&2)y=7-y;return y*8+x;
}));
export function canonicalBookPosition(position: Position){
 let key='z',inverse:number[]=[];
 for(const mapping of transforms){
  let own=0n,opponent=0n;
  for(let i=0;i<64;i++){const c=position.cells[i];if(c===position.turn)own|=bits[mapping[i]];else if(c===-position.turn)opponent|=bits[mapping[i]];}
  const candidate=own.toString(16).padStart(16,'0')+opponent.toString(16).padStart(16,'0');
  if(candidate<key){key=candidate;inverse=Array(64);for(let i=0;i<64;i++)inverse[mapping[i]]=i;}
 }
 return {key,inverse};
}
export function bookShard(key:string){let hash=2166136261;for(const c of key)hash=Math.imul(hash^c.charCodeAt(0),16777619);return (hash&4095).toString(16).padStart(3,'0');}
type Shard=Record<string,number[]>;
export function createOpeningBookClient(fetcher:typeof fetch=fetch){
 let ready:Promise<void>|undefined;const cache=new Map<string,Promise<Shard>>();
 async function checkedJson(path:string){const response=await fetcher(base+path,{credentials:'omit',cache:'force-cache'});if(!response.ok)throw new Error('开局库加载失败，请刷新重试。');return response.json();}
 async function prepare(){
  ready??=checkedJson('manifest.json').then(meta=>{if(meta.format!=='otello-web-book-v1'||meta.sourceSha256!==BOOK_SOURCE_SHA||meta.shards!==4096||meta.enabled!==true)throw new Error('开局库版本校验失败。');}).catch(error=>{ready=undefined;throw error;});
  await ready;
 }
 return {async lookup(position:Position,seed:number):Promise<Analysis|null>{
  const moves=legal(position);if(position.over||!moves.length)return null;
  const started=performance.now();await prepare();
  const {key,inverse}=canonicalBookPosition(position),bucket=bookShard(key);
  let pending=cache.get(bucket);
  if(!pending){pending=checkedJson(bucket+'.json').catch(error=>{cache.delete(bucket);throw error;});cache.set(bucket,pending);if(cache.size>128)cache.delete(cache.keys().next().value!);}
  const shard=await pending;if(!Object.hasOwn(shard,key))return null;const row=shard[key];
  if(!Array.isArray(row)||row.length<3||row.length%2!==1||!Number.isInteger(row[0]))throw new Error('开局库走法数据无效。');
  const choices:{move:number;value:number;weight:number}[]=[];
  for(let i=1;i<row.length;i+=2){const canonical=row[i],value=row[i+1],move=inverse[canonical];
   if(!Number.isInteger(canonical)||canonical<0||canonical>63||!Number.isInteger(value)||value< -64||value>64||value>row[0]||value<row[0]-2||!moves.includes(move))throw new Error('开局库返回了非法走法。');
   choices.push({move,value,weight:((Math.exp(value-row[0])+1.5)/3)**9});
  }
  let random=seed>>>0;random=(random^0x9e3779b9)>>>0;random^=random<<13;random^=random>>>17;random^=random<<5;
  let draw=(random>>>0)/4294967296*choices.reduce((sum,c)=>sum+c.weight,0);let choice=choices[choices.length-1];
  for(const candidate of choices){draw-=candidate.weight;if(draw<=0){choice=candidate;break;}}
  return {move:choice.move,score:choice.value,depth:0,elapsed:Math.round(performance.now()-started),nodes:0,ttHits:null,exact:false,candidates:[],useBook:true,bookHit:true,search:'book',evaluator:'book'};
 }};
}

#!/usr/bin/env python3
"""Independently check browser book candidates against raw records and array rules."""
import hashlib,importlib.util,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'public/opening-book/0dd454d84aa6-v1'
spec=importlib.util.spec_from_file_location('book_rules',ROOT/'scripts/compare-resnet.py');rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
perms=[]
for s in range(8):
 p=[]
 for i in range(64):
  x,y=i%8,i//8
  if s&4:x,y=y,x
  if s&1:x=7-x
  if s&2:y=7-y
  p.append(y*8+x)
 perms.append(p)
def transform(bits,p):
 result=0
 while bits:
  bit=bits&-bits;i=bit.bit_length()-1;result|=1<<p[i];bits^=bit
 return result
def variants(own,opp):return [(transform(own,p),transform(opp,p)) for p in perms]
def shard(key):
 h=2166136261
 for c in key:h=((h^ord(c))*16777619)&0xffffffff
 return f'{h&4095:03x}'
def main():
 binary=(ROOT/'.build/data/egaroucid/book.egbk3').read_bytes();meta=json.loads((BASE/'manifest.json').read_text());assert hashlib.sha256(binary).hexdigest()==meta['sourceSha256']
 records={};sample=[]
 for index,(own,opp,value,level,tail) in enumerate(struct.iter_unpack('<QQbB7s',binary[14:])):
  if own&opp or not -64<=value<=64:continue
  records[(own,opp)]=(value,level)
  if index%2604==0:sample.append((own,opp,value))
 def lookup(own,opp):
  found=[records[k] for k in variants(own,opp) if k in records]
  return max(found,key=lambda row:row[1])[0] if found else None
 shards={};checked=hits=0
 for own,opp,node_value in sample:
  own,opp=min(variants(own,opp));key=f'{own:016x}{opp:016x}';name=shard(key)
  if name not in shards:shards[name]=json.loads((BASE/(name+'.json')).read_text())
  board=[1 if own>>i&1 else -1 if opp>>i&1 else 0 for i in range(64)];links=[]
  for move in rules.legal(board,1):
   after,_=rules.play(board,1,move);turn=-1 if rules.legal(after,-1) else 1
   child_own=sum(1<<i for i,c in enumerate(after) if c==turn);child_opp=sum(1<<i for i,c in enumerate(after) if c==-turn)
   value=lookup(child_own,child_opp)
   if value is not None:links.append((move,value*(-1 if turn==-1 else 1)))
  best=max((v for m,v in links),default=-1000)
  expected=None
  if links and best>=node_value-8:
   expected=[best]
   for move,value in links:
    if value>=best-2:expected.extend((move,value))
  assert shards[name].get(key)==expected,(key,shards[name].get(key),expected)
  checked+=1;hits+=expected is not None
 report=dict(sourceSha256=meta['sourceSha256'],rawRecords=len(records),checkedPositions=checked,bookHits=hits,scope='Raw records, independent 8-symmetry coordinate transforms, independent array legal/flips/pass rules, child-value signs, loss cutoff and 2-disc candidate window')
 (ROOT/'.build/web-book/validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

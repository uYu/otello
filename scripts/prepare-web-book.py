#!/usr/bin/env python3
"""Build a sharded, mandatory browser opening book from the verified v1 snapshot."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA='0dd454d84aa656a64329930bad796c4a48a4322b7693d50b7040006383db5012'
VERSION='0dd454d84aa6-v1'
OUTPUT=ROOT/'public/opening-book'/VERSION
FALLBACK='approved default Pattern + learned-prior PUCT MCTS on a verified book miss'
INPUT=ROOT/'.build/data/egaroucid/book.egbk3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if not INPUT.exists():subprocess.run(['python3',str(ROOT/'scripts/data-assets.py'),'restore','--only','egaroucid-book'],cwd=ROOT,check=True)
 assert sha(INPUT)==SOURCE_SHA,'Opening-book source checksum mismatch'
 code=ROOT/'tools/export-web-book.cpp';header=ROOT/'cpp/othello.hpp';signature={'converter':sha(code),'rules':sha(header)}
 manifest_path=OUTPUT/'manifest.json'
 if manifest_path.exists():
  meta=json.loads(manifest_path.read_text())
  if meta.get('sourceSha256')==SOURCE_SHA and meta.get('generatorSha256')==signature:
   checks=json.loads((OUTPUT/'checksums.json').read_text())
   if all((OUTPUT/name).is_file() and sha(OUTPUT/name)==digest for name,digest in checks.items()):
    if meta.get('fallback')!=FALLBACK:
     meta['fallback']=FALLBACK;manifest_path.write_text(json.dumps(meta,indent=2)+'\n')
    print(f"Opening book verified: {meta['entries']} playable positions; always enabled",flush=True);return
 build=ROOT/'.build/web-book';build.mkdir(parents=True,exist_ok=True)
 binary=build/'export-web-book';compiler=os.environ.get('CXX') or shutil.which('c++') or shutil.which('g++')
 if not compiler:raise RuntimeError('C++17 compiler required to generate the browser opening book')
 subprocess.run([compiler,'-std=c++17','-O3','-I',str(ROOT/'cpp'),str(code),'-o',str(binary)],cwd=ROOT,check=True)
 OUTPUT.mkdir(parents=True,exist_ok=True)
 result=subprocess.run([str(binary),str(INPUT),str(OUTPUT)],cwd=ROOT,check=True,capture_output=True,text=True)
 counts=json.loads(result.stdout)
 for source,name in [(ROOT/'.build/data/egaroucid/LICENSE','LICENSE'),(ROOT/'.build/data/egaroucid/SOURCE.json','SOURCE.json'),(INPUT,'original.egbk3'),(code,'export-web-book.cpp'),(header,'othello.hpp')]:shutil.copyfile(source,OUTPUT/name)
 checks={p.name:sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name not in ('manifest.json','checksums.json')}
 (OUTPUT/'checksums.json').write_text(json.dumps(checks,sort_keys=True)+'\n')
 meta=dict(format='otello-web-book-v1',version=VERSION,sourceSha256=SOURCE_SHA,revision='2a65c87860be999373088ad833aa00b0844883b6',license='GPL-3.0-or-later',generatorSha256=signature,**counts,enabled=True,policy='registered child values; loss cutoff 8; fixed variety 1: 2-disc window, upstream weight exponent 9; seeded browser selection',fallback=FALLBACK)
 manifest_path.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

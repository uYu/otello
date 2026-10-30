#!/usr/bin/env python3
"""Audit the immutable v1.38 offline candidate after subsequent source work."""
import hashlib,json,runpy,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    manifest=json.loads((ROOT/'models/archive/v1.38-direct-policy-sources.json').read_text())
    assert sha(ROOT/manifest['archive'])==manifest['sha256']
    with tarfile.open(ROOT/manifest['archive'],'r:gz') as t:source={name:t.extractfile(name).read() for name in manifest['files']}
    for name,data in source.items():assert hashlib.sha256(data).hexdigest()==manifest['files'][name],name
    experiment=json.loads((ROOT/'models/v1.38-direct-policy-experiment.json').read_text())
    for name,digest in {**experiment['artifactSha256'],**experiment['preservedSha256']}.items():assert sha(ROOT/name)==digest,name
    def historical(path):
        path=Path(path).resolve();name=str(path.relative_to(ROOT))
        if name in source and not name.startswith('models/'):return hashlib.sha256(source[name]).hexdigest()
        return sha(path)
    ns=runpy.run_path(str(ROOT/'scripts/finalize-direct-policy.py'));ns['main'].__globals__['sha']=historical;ns['main']()
    print('Original v1.38 offline model, metrics, protocol and source evidence preserved')

if __name__=='__main__':main()

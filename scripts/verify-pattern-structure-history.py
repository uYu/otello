#!/usr/bin/env python3
"""Audit v1.37 candidate-era evidence using its frozen delivery/source bytes."""
import hashlib
import json
from pathlib import Path
import runpy
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    preserved=json.loads((ROOT/'models/archive/v1.37-promotion-preserved.json').read_text())
    for name,digest in preserved['files'].items():assert sha(ROOT/name)==digest,name
    manifest=json.loads((ROOT/'models/archive/v1.37-candidate-era-verification.json').read_text())
    assert sha(ROOT/manifest['archive'])==manifest['sha256']
    with tarfile.open(ROOT/manifest['archive'],'r:gz') as t:
        delivery={name:t.extractfile(name).read() for name in manifest['files']}
    assert {name:hashlib.sha256(data).hexdigest() for name,data in delivery.items()}==manifest['files']
    sources=json.loads((ROOT/manifest['sourceArchiveManifest']).read_text())
    assert sha(ROOT/sources['archive'])==sources['sha256']
    with tarfile.open(ROOT/sources['archive'],'r:gz') as t:
        frozen={name:t.extractfile(name).read() for name in sources['files']}
    assert {name:hashlib.sha256(data).hexdigest() for name,data in frozen.items()}==sources['files']
    def historical_sha(path):
        path=Path(path).resolve();name=str(path.relative_to(ROOT))
        if name in delivery:return hashlib.sha256(delivery[name]).hexdigest()
        # Check original measured sources from their archive. Model/report/protocol
        # evidence is always checked in place, never substituted from an archive.
        if name in frozen and not name.startswith('models/'):
            return hashlib.sha256(frozen[name]).hexdigest()
        return sha(path)
    assert (ROOT/'scripts/finalize-pattern-structure.py').read_bytes()==delivery['scripts/finalize-pattern-structure.py']
    namespace=runpy.run_path(str(ROOT/'scripts/finalize-pattern-structure.py'))
    namespace['main'].__globals__['sha']=historical_sha
    namespace['main']()
    print('v1.37 candidate-era training, timings, 80 games and 4,865 actions preserved after promotion')


if __name__=='__main__':verify()

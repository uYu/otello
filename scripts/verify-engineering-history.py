#!/usr/bin/env python3
"""Replay immutable v1.35/v1.36 audits against their pre-promotion delivery bytes."""
import hashlib
import json
from pathlib import Path
import runpy
import tarfile
ROOT=Path(__file__).resolve().parents[1]

def verify():
    manifest=json.loads((ROOT/'models/archive/v1.36-candidate-era-verification.json').read_text())
    archive=ROOT/manifest['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
    with tarfile.open(archive,'r:gz') as stream:
        frozen={name:stream.extractfile(name).read() for name in manifest['files']}
    for name,data in frozen.items():
        assert hashlib.sha256(data).hexdigest()==manifest['files'][name],name
    mutable={'models/default-model.json','src/game/wasm/othello.mjs'}
    def historical_sha(path):
        path=Path(path).resolve()
        # Recorded candidate model paths may be absolute on the measurement host.
        # Resolve those model bytes in this checkout so clones audit independently.
        if not path.is_relative_to(ROOT):
            prefix,separator,suffix=str(path).rpartition('/models/')
            assert separator and '..' not in Path(suffix).parts,path
            path=ROOT/'models'/suffix
        name=str(path.relative_to(ROOT))
        return hashlib.sha256(frozen[name] if name in mutable else path.read_bytes()).hexdigest()
    for name in ['scripts/finalize-pattern-speed-suite.py','scripts/finalize-pattern-capacity.py']:
        assert (ROOT/name).read_bytes()==frozen[name],name
        namespace=runpy.run_path(str(ROOT/name))
        namespace['main'].__globals__['sha']=historical_sha
        namespace['main']()
    print('Candidate-era audits preserved; current selector and Wasm are verified by the new baseline.')

if __name__=='__main__':verify()

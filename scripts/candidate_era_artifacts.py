"""Resolve mutable delivery files from the checksummed pre-promotion snapshot.

Candidate reports remain immutable. Their checks refer to the selector, browser
engine and verification tools as they existed before user-approved promotion.
Model files, measured reports and every other artifact are checked in place.
"""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'models/archive/v1.25-candidate-era-verification.json'
MUTABLE={'models/default-model.json','src/game/wasm/othello.mjs',
         'scripts/finalize-pattern-outcome.py','scripts/finalize-pattern-outcome-2s.py'}


def historical_sha(path):
    path=Path(path).resolve()
    name=str(path.relative_to(ROOT))
    active=json.loads((ROOT/'models/default-model.json').read_text())
    if name not in MUTABLE or active['version']=='v1.9':
        return hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot=json.loads(MANIFEST.read_text());archive=ROOT/snapshot['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==snapshot['sha256']
    with tarfile.open(archive,'r:gz') as stream:
        data=stream.extractfile(name).read()
    digest=hashlib.sha256(data).hexdigest()
    assert digest==snapshot['files'][name],name
    return digest

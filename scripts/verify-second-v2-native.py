#!/usr/bin/env python3
"""Save native approved-v2 policy references for browser adapter parity."""
import hashlib,json
from pathlib import Path
import numpy as np
from global_policy_native_experiment import load,native
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'models/v2-default.json').read_text())
model=ROOT/manifest['defaultModel']
assert hashlib.sha256(model.read_bytes()).hexdigest()==manifest['modelSha256']
data=np.load(ROOT/'.build/training/beginner-low-elo1300-v1/positions.npz')
boards=np.repeat(data['test_board'][:64],2,axis=0)
turns=np.tile(np.array([1,-1],np.int8),64)
pair=load(ROOT/'.build/native/libothello.dylib',model)
try: predictions=native(pair,boards,turns)
finally:pair[0].ot_nnue_destroy(pair[1])
result=dict(model=manifest['defaultModel'],modelSha256=manifest['modelSha256'],positions=len(boards),fixtures=[dict(board=b.astype(int).tolist(),turn=int(t),logits=p[4:].tolist()) for b,t,p in zip(boards,turns,predictions)])
(ROOT/'models/second-level-v2-policy-parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(positions=len(boards),modelSha256=manifest['modelSha256'])))

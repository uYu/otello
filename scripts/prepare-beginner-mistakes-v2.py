#!/usr/bin/env python3
"""Keep actual low-Elo moves, enriched for disagreement with frozen v2 policy."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from training_consensus import canonical

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/beginner-low-elo-mistakes-v2'
REPORT=ROOT/'models/beginner-low-elo-mistakes-v2-data.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not REPORT.exists() and not (WORK/'positions.npz').exists()
    source=ROOT/'.build/training/beginner-low-elo-v1/positions.npz'
    meta_path=ROOT/'models/beginner-low-elo-v1-data.json'
    meta=json.loads(meta_path.read_text());assert sha(source)==meta['cacheSha256']
    data=dict(np.load(source));board=data['train_board'];moves=data['train_move'];mask=data['train_legal']
    lib_path=ROOT/f'.build/native/libothello.{"dylib" if sys.platform=="darwin" else "so"}'
    lib=ctypes.CDLL(str(lib_path));ptr=ctypes.POINTER(ctypes.c_int)
    lib.ot_default_infer.argtypes=[ptr,ctypes.c_int,ptr,ctypes.POINTER(ctypes.c_double)]
    lib.ot_default_model_sha256.restype=ctypes.c_char_p
    default=json.loads((ROOT/'models/v2-default.json').read_text())
    assert lib.ot_default_model_sha256().decode()==default['modelSha256']
    lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
    lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    handle=lib.ot_nnue_load(str(ROOT/default['defaultModel']).encode());assert handle
    gaps=[]
    for b,move,legal in zip(board,moves,mask):
        cells=(ctypes.c_int*64)(*map(int,b));out=(ctypes.c_double*68)()
        assert lib.ot_pattern_predict(handle,cells,1,out)==1
        logits=np.ctypeslib.as_array(out)[4:]
        gaps.append(float(logits[legal].max()-logits[move]))
    lib.ot_nnue_destroy(handle)
    gaps=np.array(gaps);assert gaps.max()>0 and np.std(gaps)>0;keys=canonical(board,np.ones(len(board),np.int8))[0]
    seen=set();selected=[]
    for index in np.argsort(-gaps,kind='stable'):
        key=keys[index].tobytes()
        if key in seen:
            continue
        seen.add(key);selected.append(int(index))
        if len(selected)==7500:
            break
    rng=np.random.default_rng(20261101)
    for index in rng.permutation(len(board)):
        key=keys[index].tobytes()
        if key in seen:
            continue
        seen.add(key);selected.append(int(index))
        if len(selected)==10000:
            break
    assert len(selected)==10000 and len(seen)==10000
    original_rows=list(map(json.loads,(source.parent/'positions-provenance.jsonl').open()))
    training_rows=[r for r in original_rows if r['split']=='train'];assert len(training_rows)==len(board)
    rows=[{**training_rows[i], 'v2PolicyLogGap':float(gaps[i]), 'sourceIndex':i,
           'stratum':'largest_policy_gap' if n<7500 else 'unfiltered_low_elo'} for n,i in enumerate(selected)]
    for key in list(data):
        if key.startswith('train_'):
            data[key]=data[key][selected]
    WORK.mkdir(parents=True,exist_ok=True)
    output=WORK/'positions.npz';np.savez_compressed(output,**data)
    provenance=WORK/'positions-provenance.jsonl'
    with provenance.open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    report=dict(version='beginner-low-elo-mistakes-v2',threshold=1350,minPriorTrainGames=20,
                trainPositions=10000,uniqueCanonicalTrainPositions=10000,seed=20261101,
                strata=dict(largestPolicyDisagreement=7500,unfilteredLowElo=2500),
                sourceReport=str(meta_path.relative_to(ROOT)),sourceReportSha256=sha(meta_path),sourceCacheSha256=sha(source),
                cacheSha256=sha(output),provenanceSha256=sha(provenance),
                sourceSha256=sha(Path(__file__)),v2ModelSha256=default['modelSha256'],v2LibrarySha256=sha(lib_path),
                target='Unchanged recorded low-Elo actor legal move; no relabeling to engine choice',
                filtering='Prefer large best legal v2 Pattern logit minus actual-move Pattern logit; retain 25% unfiltered low-Elo states for basic play',
                filterOnly='Frozen v2 is used once before training. The new network starts randomly and inference uses only its own logits.',
                policyGapQuantiles=list(map(float,np.quantile(gaps[selected],[0,.25,.5,.75,1]))),
                dataNotices='data/README.md',limits='Disagreement with a learned prior is a mistake proxy, not a proven search blunder. Heldouts unchanged from low-Elo1350 data; excludes their canonical states before selection. Strength/coherence measured separately.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()

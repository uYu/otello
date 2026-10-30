#!/usr/bin/env python3
"""Test reusable soft-label cache masks, temperatures, symmetry, split isolation and hashes."""
import importlib.util,json,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('teacher_targets',ROOT/'scripts/teacher-targets.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    rng=np.random.default_rng(718);n=8
    row=dict(position_index=np.arange(n),board=rng.integers(-1,2,(n,64),dtype=np.int8),turn=np.ones(n,dtype=np.int8),
        policy_logits=rng.standard_normal((n,65)).astype(np.float32),legal=np.zeros((n,65),dtype=bool),
        outcome_value=np.linspace(-1,1,n,dtype=np.float32),recorded_outcome=np.zeros(n,dtype=np.int8),recorded_move=np.array([0,1,2,3,4,5,6,64],dtype=np.uint8))
    for i,move in enumerate(row['recorded_move']):
        row['legal'][i,move]=True
        if move!=64:row['legal'][i,(int(move)+8)%64]=True
    for temp in (.1,1.,2.,10.):
        p=m.probabilities(row['policy_logits'],row['legal'],temp)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1) and (p[~row['legal']]==0).all() and p[-1,64]==1
    for bad in (0.,-1.,float('nan')):
        try:m.probabilities(row['policy_logits'],row['legal'],bad)
        except ValueError:pass
        else:raise AssertionError('Bad temperature accepted')
    grid=np.arange(64).reshape(8,8)
    for rotation in range(4):
        for mirror in (False,True):
            perm=np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(-1)
            out=m.augment(row,perm);restored=m.augment(out,np.argsort(perm))
            assert all(np.array_equal(row[k],restored[k]) for k in row)
            assert out['legal'][np.arange(n),out['recorded_move']].all()
            assert np.array_equal(row['outcome_value'],out['outcome_value'])
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);splits={}
        for split in ('train','validation','test'):
            path=p/(split+'.npz');np.savez_compressed(path,**row)
            splits[split]=dict(positions=n,shards=[dict(path=path.name,sha256=m.sha(path))])
        manifest=dict(complete=True,splits=splits);(p/'manifest.json').write_text(json.dumps(manifest))
        cache=m.TeacherCache(p);assert len(list(cache.iter_train()))==1
        assert np.array_equal(next(cache.iter_train())['board'],row['board'])
        (p/'train.npz').write_bytes((p/'train.npz').read_bytes()+b'changed')
        try:list(cache.iter_train())
        except ValueError:pass
        else:raise AssertionError('Changed shard accepted')
        manifest['complete']=False;(p/'manifest.json').write_text(json.dumps(manifest))
        try:m.TeacherCache(p)
        except ValueError:pass
        else:raise AssertionError('Incomplete cache accepted')
    print(json.dumps(dict(complete=True,d4Permutations=8,temperatures=4,forcedPass=True,splitIsolation=True,checksumRejection=True,incompleteCacheRejection=True)))
if __name__=='__main__':main()

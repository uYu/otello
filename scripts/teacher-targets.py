#!/usr/bin/env python3
"""Numpy-only reader and augmentation for frozen CNN distillation-target caches."""
import hashlib,json
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def probabilities(logits,legal,temperature=1.):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite temperature required')
    logits=np.asarray(logits,dtype=np.float64);legal=np.asarray(legal,dtype=bool)
    if logits.shape!=legal.shape or logits.shape[-1]!=65 or not legal.any(axis=-1).all():raise ValueError('Invalid legal logits')
    if not np.isfinite(logits).all():raise ValueError('Nonfinite teacher logits')
    x=np.where(legal,logits,-np.inf);x-=x.max(axis=-1,keepdims=True)
    with np.errstate(over='ignore',under='ignore'):
        p=np.exp(x/temperature)
    return p/p.sum(axis=-1,keepdims=True)

class TeacherCache:
    """Read a complete cache without torch, the teacher model or a GPU."""
    def __init__(self,path):
        self.root=Path(path);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest.get('complete'):raise ValueError('Teacher cache is incomplete')
    def iter_split(self,split='train',verify=True):
        if split not in self.manifest['splits']:raise ValueError('Unknown split')
        cursor=0
        for shard in self.manifest['splits'][split]['shards']:
            path=self.root/shard['path']
            if verify and sha(path)!=shard['sha256']:raise ValueError('Teacher shard checksum mismatch')
            with np.load(path,allow_pickle=False) as src:row={k:src[k] for k in src.files}
            n=len(row['position_index'])
            if not np.array_equal(row['position_index'],np.arange(cursor,cursor+n)):raise ValueError('Teacher indices are out of order')
            if row['board'].shape!=(n,64) or row['policy_logits'].shape!=(n,65) or row['legal'].shape!=(n,65):raise ValueError('Teacher row dimensions are invalid')
            cursor+=n;yield row
        if cursor!=self.manifest['splits'][split]['positions']:raise ValueError('Teacher row count mismatch')
    def iter_train(self,verify=True):return self.iter_split('train',verify)

def augment(row,permutation):
    """Permutation maps each new board square to its original square; pass stays64."""
    n=len(row['board']);p=np.asarray(permutation,dtype=np.int64)
    if p.ndim==1:p=np.broadcast_to(p,(n,64))
    if p.shape!=(n,64) or not np.array_equal(np.sort(p,axis=1),np.broadcast_to(np.arange(64),(n,64))):raise ValueError('Invalid board permutation')
    result=dict(row);result['board']=np.take_along_axis(row['board'],p,axis=1)
    for key in ('policy_logits','legal'):
        result[key]=np.concatenate((np.take_along_axis(row[key][:,:64],p,axis=1),row[key][:,64:]),axis=1)
    inverse=np.argsort(p,axis=1);move=row['recorded_move']
    result['recorded_move']=np.where(move==64,64,inverse[np.arange(n),np.minimum(move,63)]).astype(np.uint8)
    return result

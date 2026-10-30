"""Align mixed-corpus empirical targets to the original frozen million board rows."""
import numpy as np
from training_consensus import KEY,PERMS


def canonical_transforms(board,turn):
    relative=(board*turn[:,None]).astype(np.int8)
    best=np.empty(len(board),KEY);best['own']=np.iinfo(np.uint64).max;best['opponent']=np.iinfo(np.uint64).max;sym=np.zeros(len(board),np.uint8)
    for i,p in enumerate(PERMS):
        transformed=relative[:,p]
        own=np.packbits(transformed==1,axis=1,bitorder='little').copy().view('<u8').reshape(-1)
        opp=np.packbits(transformed==-1,axis=1,bitorder='little').copy().view('<u8').reshape(-1)
        better=(own<best['own'])|((own==best['own'])&(opp<best['opponent']))
        sym[better]=i;best['own'][better]=own[better];best['opponent'][better]=opp[better]
    return best,sym


def soft_targets(data,mapping,sym):
    mapping=np.asarray(mapping,np.int64);n=len(mapping);count=data['occurrence_count'][mapping]
    offset=data['policy_offsets'];length=offset[mapping+1]-offset[mapping];rows=np.repeat(np.arange(n),length.astype(np.int64))
    starts=np.repeat(offset[mapping],length.astype(np.int64)).astype(np.int64)
    inner=np.arange(len(rows))-np.repeat(np.cumsum(length)-length,length.astype(np.int64)).astype(np.int64);at=starts+inner
    original_action=PERMS[sym[rows],data['policy_action'][at]]
    policy=np.zeros((n,64),np.float32);policy[rows,original_action]=(data['policy_mass'][at]/count[rows]).astype(np.float32)
    wdl=(data['wdl_count'][mapping]/count[:,None]).astype(np.float32)
    return dict(value=(wdl[:,0]-wdl[:,1]).astype(np.float32),wdl=wdl,policy=policy)

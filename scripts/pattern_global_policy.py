"""Global policy architecture: ordered pattern projections -> 16C -> 64.

The v1.37 encoder/readout can be frozen or trained jointly. D4-tied layers
produce one residual 8x8 policy map per board, rather than one MLP per move.
Native deployment is separate from this PyTorch training implementation.
"""
import math
import numpy as np
import torch
from torch import nn
from pattern_model import DIM,SHAPES,quant
from pattern_structure import StructureNet


def permutations(size):
    grid=np.arange(size*size).reshape(size,size)
    return np.stack([np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(-1)
        for mirror in [False,True] for rotation in range(4)])


def pattern_permutations():
    lookup={tuple(sorted(squares)):i for i,(squares,_,_) in enumerate(SHAPES)}
    return np.asarray([[lookup[tuple(sorted(perm[squares]))] for squares,_,_ in SHAPES]
        for perm in permutations(8)])


class EquivariantLinear(nn.Module):
    def __init__(self,input_permutations,output_permutations):
        super().__init__()
        inputs=input_permutations.shape[1];outputs=output_permutations.shape[1]
        representatives=(output_permutations[:,:,None]*inputs+input_permutations[:,None,:]).min(0)
        _,mapping=np.unique(representatives,return_inverse=True)
        _,bias_mapping=np.unique(output_permutations.min(0),return_inverse=True)
        self.register_buffer('weight_mapping',torch.from_numpy(mapping.reshape(outputs,inputs)))
        self.register_buffer('bias_mapping',torch.from_numpy(bias_mapping))
        self.coefficients=nn.Parameter(torch.empty(int(mapping.max())+1))
        self.biases=nn.Parameter(torch.zeros(int(bias_mapping.max())+1))
        nn.init.uniform_(self.coefficients,-1/math.sqrt(inputs),1/math.sqrt(inputs))

    @property
    def weight(self):return self.coefficients[self.weight_mapping]

    @property
    def bias(self):return self.biases[self.bias_mapping]

    def forward(self,x):
        return nn.functional.linear(x,quant(self.weight),quant(self.bias,65536))


class GlobalPolicyNet(StructureNet):
    def __init__(self,channels=1,baseline_state=None,freeze_baseline=True):
        if channels not in [1,2]:raise ValueError('One or two 4x4 hidden maps supported')
        super().__init__('symmetry')
        if baseline_state is not None:self.load_state_dict(baseline_state)
        for parameter in self.parameters():parameter.requires_grad_(not freeze_baseline)
        hidden=np.concatenate([permutations(4)+16*c for c in range(channels)],axis=1)
        self.global_down=EquivariantLinear(pattern_permutations(),hidden)
        self.global_up=EquivariantLinear(hidden,permutations(8))
        nn.init.zeros_(self.global_up.coefficients);nn.init.zeros_(self.global_up.biases)
        self.channels=channels
        self.freeze_baseline=freeze_baseline

    def frozen_features(self,codes,phase):
        e=quant(self.embedding(self.canonical[codes]))
        pooled=torch.einsum('bpd,pg->bgd',e,self.pool)/self.counts[None,:,None]
        x=torch.cat((quant(pooled.flatten(1).clamp(-4,4)),quant(phase[:,None])),1)
        h=quant(nn.functional.linear(x,quant(self.trunk.weight),quant(self.trunk.bias,65536)).clamp(0,1))
        out=quant(nn.functional.linear(h,quant(self.head.weight),quant(self.head.bias,65536)).clamp(-4,4))
        projection=torch.einsum('bpd,bd->bp',e,out[:,4:36])/math.sqrt(DIM)
        base_policy=projection@self.members+out[:,36:46][:,self.bucket]
        return torch.tanh(out[:,0]),out[:,1:4],projection,base_policy

    def policy_correction(self,projection):
        hidden=quant(self.global_down(quant(projection.clamp(-4,4))).clamp(0,1))
        correction=quant(self.global_up(hidden).clamp(-4,4))
        return correction

    def forward(self,codes,phase):
        value,wdl,projection,base_policy=self.frozen_features(codes,phase)
        return value,wdl,base_policy+self.policy_correction(projection)

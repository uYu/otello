"""v1.48: one shared 32->R->32 residual; original factored policy readout."""
import math
import struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pattern_model import DIM,quant
from pattern_structure import StructureNet


class SharedResidualNet(StructureNet):
    def __init__(self, residual_width=16):
        if residual_width not in (8,16):raise ValueError('Residual width must be 8 or 16')
        super().__init__('symmetry')
        self.residual_width=residual_width
        self.residual_down=nn.Linear(32,residual_width)
        self.residual_up=nn.Linear(residual_width,32)
        nn.init.zeros_(self.residual_up.weight);nn.init.zeros_(self.residual_up.bias)

    def forward(self,codes,phase):
        e=quant(self.embedding(self.canonical[codes]))
        pooled=torch.einsum('bpd,pg->bgd',e,self.pool)/self.counts[None,:,None]
        x=torch.cat((quant(pooled.flatten(1).clamp(-4,4)),quant(phase[:,None])),1)
        h=quant(nn.functional.linear(x,quant(self.trunk.weight),quant(self.trunk.bias,65536)).clamp(0,1))
        z=quant(nn.functional.linear(h,quant(self.residual_down.weight),quant(self.residual_down.bias,65536)).clamp(0,1))
        delta=quant(nn.functional.linear(z,quant(self.residual_up.weight),quant(self.residual_up.bias,65536)).clamp(-1,1))
        h=quant((h+delta).clamp(0,1))
        out=quant(nn.functional.linear(h,quant(self.head.weight),quant(self.head.bias,65536)).clamp(-4,4))
        projection=torch.einsum('bpd,bd->bp',e,out[:,4:36])/math.sqrt(DIM)
        return torch.tanh(out[:,0]),out[:,1:4],projection@self.members+out[:,36:46][:,self.bucket]


def export(net,path):
    def array(tensor,dtype,scale):
        value=tensor.detach().cpu().numpy()*scale
        value=np.sign(value)*np.floor(np.abs(value)+.5);bounds=np.iinfo(dtype)
        if not np.isfinite(value).all() or value.min()<bounds.min or value.max()>bounds.max:raise ValueError('Parameters outside integer format')
        return value.astype(dtype).tobytes()
    binary=struct.pack('<8s8I',b'OTPAT8\0\0',28503,32,32,97,46,256,1,net.residual_width)
    for layer in [net.embedding,net.trunk,net.head,net.residual_down,net.residual_up]:
        binary+=array(layer.weight,'<i2',256)
        if getattr(layer,'bias',None) is not None:binary+=array(layer.bias,'<i4',65536)
    Path(path).write_bytes(binary);return binary

"""Untrained direct-policy prototype; separate from frozen v1.37 experiments.

Policy: [local square feature (32), global trunk (32), D4 bucket one-hot (10)]
        -> shared Linear(74,8) -> clipped ReLU -> shared Linear(8,1).
No generated weights, gate multiplication, or separate policy-bias head.
"""
import numpy as np
import struct
from pathlib import Path
import torch
from torch import nn
from pattern_model import DIM, SHAPES, BUCKET, quant
from pattern_structure import StructureNet


class DirectPolicyNet(StructureNet):
    def __init__(self, policy_width=8):
        if policy_width not in (8,16,32):raise ValueError('Unsupported direct-policy width')
        self.policy_width = policy_width
        super().__init__('symmetry')
        self.variant = f'direct-policy-mlp{policy_width}'
        original = self.head
        self.head = nn.Linear(32, 4)  # Outcome plus auxiliary W/L/D only.
        with torch.no_grad():
            self.head.weight.copy_(original.weight[:4])
            self.head.bias.copy_(original.bias[:4])
        self.policy_mlp = nn.Sequential(nn.Linear(74, policy_width), nn.Hardtanh(0, 1), nn.Linear(policy_width, 1))
        membership = np.zeros((38, 64), np.float32)
        for i, (squares, _, _) in enumerate(SHAPES):
            membership[i, squares] = 1
        self.register_buffer('membership', torch.from_numpy(membership))
        self.register_buffer('square_counts', self.membership.sum(0))
        self.register_buffer('position_onehot', torch.eye(10)[torch.tensor(BUCKET)])

    def forward(self, codes, phase):
        e = quant(self.embedding(self.canonical[codes]))
        pooled = torch.einsum('bpd,pg->bgd', e, self.pool)/self.counts[None, :, None]
        trunk_input = torch.cat((quant(pooled.flatten(1).clamp(-4, 4)), quant(phase[:, None])), 1)
        h = quant(nn.functional.linear(trunk_input, quant(self.trunk.weight), quant(self.trunk.bias, 65536)).clamp(0, 1))
        value = quant(nn.functional.linear(h, quant(self.head.weight), quant(self.head.bias, 65536)).clamp(-4, 4))
        # Sum exact quantized entries before division: D4 permutations preserve sums.
        local = torch.einsum('bpd,pa->bad', e, self.membership)/self.square_counts[None, :, None]
        local = quant(local.clamp(-4, 4))
        policy_input = torch.cat((local, h[:, None, :].expand(-1, 64, -1),
                                  self.position_onehot[None, :, :].expand(len(codes), -1, -1)), 2)
        first, _, last = self.policy_mlp
        hidden = quant(nn.functional.linear(policy_input, quant(first.weight), quant(first.bias, 65536)).clamp(0, 1))
        logits = quant(nn.functional.linear(hidden, quant(last.weight), quant(last.bias, 65536)).clamp(-4, 4)).squeeze(-1)
        return torch.tanh(value[:, 0]), value[:, 1:4], logits


def export(net, path):
    """OTPAT5: compact table, 4 value outputs, fixed shared 74->8->1 policy."""
    def array(tensor, dtype, scale):
        value = tensor.detach().cpu().numpy()*scale
        value = np.sign(value)*np.floor(np.abs(value)+.5)
        bounds = np.iinfo(dtype)
        if not np.isfinite(value).all() or value.min() < bounds.min or value.max() > bounds.max:
            raise ValueError('Policy model parameters outside integer format')
        return value.astype(dtype).tobytes()
    binary = struct.pack('<8s8I', b'OTPAT5\0\0', 28503,32,32,97,4,256,74,net.policy_width)
    first, _, last = net.policy_mlp
    for layer in [net.embedding, net.trunk, net.head, first, last]:
        binary += array(layer.weight, '<i2', 256)
        if hasattr(layer, 'bias') and layer.bias is not None:
            binary += array(layer.bias, '<i4', 65536)
    Path(path).write_bytes(binary)
    return binary

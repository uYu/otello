#!/usr/bin/env python3
"""Check large CNN shapes, gradients, eval/color invariance and serialization."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('big_training',ROOT/'scripts/train-big-cnn.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    out=ROOT/'.build/training/big-cnn-v1.24-checks';out.mkdir(exist_ok=True)
    with np.load(ROOT/'.build/training/v1/positions.npz') as data:
        b=torch.from_numpy(data['validation_board'][:2]);t=torch.from_numpy(data['validation_turn'][:2])
    net=m.BigCnn();parameters=sum(p.numel() for p in net.parameters());assert parameters>13000000
    net.train();v,p=net(b,t);assert v.shape==(2,) and p.shape==(2,65)
    loss=(v-1).square().mean()+torch.nn.functional.cross_entropy(p,torch.tensor([0,1]));loss.backward()
    assert torch.isfinite(loss) and net.stem[0].weight.grad.abs().sum()>0 and net.fc.weight.grad.abs().sum()>0
    assert net.tower[-1].bn2.weight.grad.abs().sum()>0
    net.eval()
    with torch.inference_mode():
        a=net(b,t);color=net(-b,-t)
        assert all(torch.equal(x,y) for x,y in zip(a,color))
        assert a[0].abs().max()<=1 and all(torch.isfinite(x).all() for x in a)
        traced=torch.jit.trace(net,(b,t));path=out/'smoke.torchscript.pt';traced.save(str(path));loaded=torch.jit.load(str(path)).eval()
        q=loaded(b,t);error=max(float(abs(x-y).max()) for x,y in zip(a,q));assert error<1e-6
    del traced,loaded
    m.h.write_json(out/'verification.json',dict(complete=True,parameters=parameters,convolutionLayers=sum(isinstance(layer,torch.nn.Conv2d) for layer in net.modules()),
        checkedPositions=2,finiteLossAndGradients=True,colorInversionExact=True,torchscriptMaxError=error))
    print(json.dumps(dict(parameters=parameters,verified=True)),flush=True)
if __name__=='__main__':main()

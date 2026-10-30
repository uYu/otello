#!/usr/bin/env python3
"""Verify paired-product training, integer export, native inference and legacy parity."""
import ctypes,importlib.util,json,struct
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('product_training',ROOT/'scripts/train-product-nnue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec2=importlib.util.spec_from_file_location('engineering',ROOT/'scripts/check-engineering.py')
e=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(e)

def main():
    torch.set_num_threads(2)
    output=ROOT/'.build/training/product-nnue-v1.22-checks';output.mkdir(exist_ok=True)
    library=ROOT/'.build/native-product-nnue-v1.22/libothello.dylib';lib=e.load(str(library))
    data={}
    with np.load(ROOT/'.build/training/v1/positions.npz') as d:
        for k in ('board','turn'):data['validation_'+k]=d['validation_'+k][:256]
    with np.load(ROOT/'.build/training/policy/labels.npz') as d:data['validation_legal']=d['validation_legal'][:256]
    for k in ('board','turn'):data['validation_'+k]=np.concatenate([data['validation_'+k],-data['validation_'+k]])
    data['validation_legal']=np.concatenate([data['validation_legal']]*2)
    checks=[]
    for product in (False,True):
        torch.manual_seed(17);net=m.ProductNet(product);opt=torch.optim.AdamW(net.parameters(),lr=.001)
        b=torch.from_numpy(data['validation_board'][:32]);t=torch.from_numpy(data['validation_turn'][:32])
        for _ in range(3):
            opt.zero_grad();v,p=net(b,t);loss=(v-1).square().mean()+p.square().mean();loss.backward()
            assert torch.isfinite(net.transform.weight.grad).all()
            if product:assert net.transform.weight.grad[:128].abs().sum()>0 and net.transform.weight.grad[128:].abs().sum()>0
            opt.step()
        net=net.double().eval();binary=m.export(net);path=output/('product.otnnue' if product else 'control.otnnue');path.write_bytes(binary)
        py=m.suite.predict(net,data,'validation',torch.device('cpu'));native=m.h.native_predictions(path,data,'validation',library)
        ve,pe=float(abs(py[0]-native[0]).max()),float(abs(py[1]-native[1]).max());assert ve<=.000501 and pe<1e-10,(ve,pe)
        assert all(np.array_equal(a[:256],a[256:]) for a in native)
        bads=[binary[:-1],binary+b'x',binary[:8]+struct.pack('<I',129)+binary[12:]]
        if product:bads.append(binary[:66584]+struct.pack('<I',512)+binary[66588:])
        for i,bad in enumerate(bads):
            badpath=output/f'bad-{product}-{i}.otnnue';badpath.write_bytes(bad);handle=lib.ot_nnue_load(str(badpath).encode())
            if handle:lib.ot_nnue_destroy(handle)
            assert not handle
        checks.append(dict(product=product,parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),positions=512,maxValueError=ve,maxPolicyError=pe,malformedRejected=len(bads)))
    old=e.load('.build/native-light-cnn-v1.20/libothello.dylib');path=str(ROOT/'models/nnue256-outcome-v1.9.otnnue').encode();a=old.ot_nnue_load(path);b=lib.ot_nnue_load(path);assert a and b
    try:
        for i in range(8):
            cells=(ctypes.c_int*64)(*map(int,data['validation_board'][i*31]));turn=int(data['validation_turn'][i*31])
            old.ot_mcts_clear_cache();lib.ot_mcts_clear_cache();assert e.search(old,a,cells,turn)==e.search(lib,b,cells,turn)
    finally:old.ot_nnue_destroy(a);lib.ot_nnue_destroy(b)
    m.h.write_json(output/'report.json',dict(complete=True,checks=checks,legacyFixed512SearchPositions=8,legacyExact=True,librarySha256=m.h.sha(library)))
    print(json.dumps(checks),flush=True)
if __name__=='__main__':main()

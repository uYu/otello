#!/usr/bin/env python3
"""Compact export, native inference, malformed format and training smoke checks."""
import ctypes
import importlib.util
import json
import struct
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('export_light',ROOT/'scripts/export-light-cnn.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
h=m.suite.helpers

def main():
    torch.set_num_threads(2)
    output=ROOT/'.build/training/light-cnn-v1.20-checks';output.mkdir(exist_ok=True)
    library=ROOT/'.build/native-light-cnn-v1.20/libothello.dylib'
    lib=ctypes.CDLL(str(library));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
    with np.load(ROOT/'.build/training/v1/positions.npz') as d:
        data={k:v[:48] for k,v in d.items() if k.startswith('validation_')}
    with np.load(ROOT/'.build/training/policy/labels.npz') as d:
        data['validation_legal']=d['validation_legal'][:48]
    # Color inversion must preserve side-to-move features and both outputs.
    data['validation_board']=np.concatenate([data['validation_board'],-data['validation_board']])
    data['validation_turn']=np.concatenate([data['validation_turn'],-data['validation_turn']])
    data['validation_legal']=np.concatenate([data['validation_legal']]*2)
    rows=[]
    for c in (8,16,32):
        for width in (16,32,64):
            torch.manual_seed(718)
            net=m.suite.CnnPolicyValue(c,width)
            opt=torch.optim.AdamW(net.parameters(),lr=.001)
            b=torch.from_numpy(data['validation_board'][:8]);t=torch.from_numpy(data['validation_turn'][:8])
            v,p=net(b,t);loss=v.square().mean()+p.square().mean();loss.backward();opt.step();net.eval()
            path=output/f'c{c}h{width}.otcnn';m.export(net,path)
            fp=m.suite.predict(net,data,'validation',torch.device('cpu'))
            native=h.native_predictions(path,data,'validation',library)
            ve,pe=float(abs(fp[0]-native[0]).max()),float(abs(fp[1]-native[1]).max())
            assert ve<.00051 and pe<1e-5,(c,width,ve,pe)
            assert all(np.array_equal(a[:48],a[48:]) for a in fp)
            traced=torch.jit.trace(net,(b,t));trace_path=path.with_suffix('.torchscript.pt');traced.save(str(trace_path))
            reloaded=torch.jit.load(str(trace_path))
            loaded=m.suite.predict(reloaded,data,'validation',torch.device('cpu'))
            assert max(float(abs(a-b).max()) for a,b in zip(fp,loaded))<1e-6
            raw=path.read_bytes()
            bads=[raw[:-1],raw+b'x',raw[:20]+struct.pack('<I',128)+raw[24:],raw[:28]+struct.pack('<f',float('nan'))+raw[32:]]
            if width!=64:bads.append(b'OTCNN2\0\0'+raw[8:])
            for i,rawbad in enumerate(bads):
                bad=output/f'bad-{c}-{width}-{i}.otcnn';bad.write_bytes(rawbad)
                handle=lib.ot_nnue_load(str(bad).encode())
                if handle:lib.ot_nnue_destroy(handle)
                assert not handle,(c,width,i)
            rows.append(dict(channels=c,headWidth=width,parameters=sum(p.numel() for p in net.parameters()),
                             positions=96,maxValueError=ve,maxPolicyError=pe,malformedRejected=len(bads)))
    h.write_json(output/'report.json',dict(complete=True,librarySha256=h.sha(library),checks=rows))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()

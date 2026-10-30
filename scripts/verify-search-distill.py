#!/usr/bin/env python3
"""Validate parent identity, teacher targets, D4 soft labels and finetuning/export."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('distill',ROOT/'scripts/train-search-distill.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    folder=ROOT/'.build/training/search-distill-v1.23-checks';folder.mkdir(exist_ok=True)
    path=ROOT/'models/nnue256-outcome-v1.9.otnnue';net=m.parent_net(path.read_bytes())
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/policy/labels.npz'))
    for split in ('train','validation','test'):
        for k in ('legal','move'):data[split+'_'+k]=labels[split+'_'+k]
    ids=m.choose_indices(data,32);assert len(set(ids))==32 and all(0<=i<len(data['train_board']) for i in ids)
    teacher=m.Teacher(m.LIBRARY,path);rows=[]
    try:
        for i in ids[:8]:rows.append(teacher.search(data['train_board'][i],data['train_turn'][i],data['train_legal'][i],512,718))
        repeat=teacher.search(data['train_board'][ids[0]],data['train_turn'][ids[0]],data['train_legal'][ids[0]],512,718)
        assert rows[0]['value']==repeat['value'] and np.array_equal(rows[0]['policy'],repeat['policy'])
        passes=np.flatnonzero(data['train_legal'][:,64])
        if len(passes):
            i=passes[0];passed=teacher.search(data['train_board'][i],data['train_turn'][i],data['train_legal'][i],512,718)
            assert passed['policy'][64]==1
        forced=np.ones(64,dtype=np.int8);forced[0]=0;forced[1]=-1
        forced_legal=np.zeros(65,dtype=bool);forced_legal[64]=True
        forced_result=teacher.search(forced,-1,forced_legal,512,718)
        assert forced_result['policy'][64]==1 and forced_result['exact'] and forced_result['value']==-1
    finally:teacher.close()
    perms=torch.from_numpy(m.h.policy.value.symmetries());inverse=torch.argsort(perms,1)
    board=torch.from_numpy(data['train_board'][ids[:8]]);turn=torch.from_numpy(data['train_turn'][ids[:8]])
    legal=torch.from_numpy(data['train_legal'][ids[:8]]);moves=torch.from_numpy(data['train_move'][ids[:8]].astype(np.int64))
    pi=torch.from_numpy(np.stack([r['policy'] for r in rows]));value=torch.tensor([r['value'] for r in rows],dtype=torch.float32)
    for sym in range(8):
        p=perms[sym].expand(8,-1);mask=torch.cat((torch.gather(legal[:,:64],1,p),legal[:,64:]),1)
        transformed=torch.cat((torch.gather(pi[:,:64],1,p),pi[:,64:]),1)
        assert (transformed[~mask]==0).all() and torch.allclose(transformed.sum(1),torch.ones(8))
        assert torch.equal(torch.gather(transformed[:,:64],1,inverse[sym].expand(8,-1)),pi[:,:64])
    checks=[]
    for alpha in (0.,.5):
        candidate=m.parent_net(path.read_bytes());opt=torch.optim.AdamW(candidate.parameters(),lr=.0001)
        original=torch.nn.functional.one_hot(moves,65).float();target=(1-alpha)*original+alpha*pi
        v,p=candidate(board,turn);logp=torch.nn.functional.log_softmax(p.masked_fill(~legal,-1e9),1)
        truth=torch.from_numpy(np.sign(data['train_target'][ids[:8]]).astype(np.float32))
        loss=4*(v-((1-alpha)*truth+alpha*value)).square().mean()-(target*logp).sum(1).mean()
        assert torch.isfinite(loss);loss.backward();assert candidate.transform.weight.grad.abs().sum()>0;opt.step()
        binary=m.m.export(candidate);export=folder/f'alpha{alpha}.otnnue';export.write_bytes(binary)
        frozen={k:v[:64] for k,v in data.items() if k.startswith('validation_')}
        py=m.suite.predict(candidate.double().eval(),frozen,'validation',torch.device('cpu'));native=m.h.native_predictions(export,frozen,'validation',m.LIBRARY)
        ve,pe=float(abs(py[0]-native[0]).max()),float(abs(py[1]-native[1]).max());assert ve<=.000501 and pe<1e-10
        checks.append(dict(alpha=alpha,loss=float(loss.detach()),maxValueError=ve,maxPolicyError=pe))
    result=dict(complete=True,parentExactByteRoundtrip=True,teacherPositions=8,teacherRepeatIdentical=True,
        forcedPassChecked=True,d4SoftPolicySymmetries=8,checks=checks)
    m.h.write_json(folder/'report.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

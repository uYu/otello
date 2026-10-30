#!/usr/bin/env python3
"""Check Pattern KD loss, real cached targets, D4, gradients and OTPAT2 parity."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('distill_pattern',ROOT/'scripts/distill-pattern-outcome100.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

def main():
    torch.set_num_threads(2);torch.manual_seed(20261029)
    work=ROOT/'.build/training/pattern-cnn-distill-v1.28-checks';work.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'models/pattern32-outcome-v1.25.pt';net=d.PatternNet();net.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True))
    assert d.export(net,work/'roundtrip.otpat',outcome=True)==parent.with_suffix('.otpat').read_bytes()
    cache=d.targets.TeacherCache(ROOT/'.build/training/teacher-cache/big-cnn-v1.24-joint-snapshot1')
    original=next(cache.iter_train());row={k:v[:64].copy() for k,v in original.items()};labels=np.load(ROOT/'.build/training/pattern-outcome-v1.25/policy.npz')['train'][:64]
    grid=np.arange(64).reshape(8,8);perms=[np.rot90(np.fliplr(grid) if mirror else grid,r).ravel() for mirror in (False,True) for r in range(4)]
    for perm in perms:
        augmented=d.targets.augment(row,perm)
        assert np.array_equal(d.legal_mask(augmented['board'],augmented['turn']),augmented['legal'][:,:64])
        soft=d.targets.probabilities(augmented['policy_logits'],augmented['legal'],2.)
        expected=d.targets.probabilities(row['policy_logits'],row['legal'],2.)
        assert np.allclose(soft[:,:64],expected[:,:64][:,perm]) and np.array_equal(augmented['outcome_value'],row['outcome_value'])
        inverse=np.argsort(perm);assert augmented['legal'][np.arange(64),inverse[labels]].all()
    mask=torch.from_numpy(row['legal'][:,:64]);move=torch.from_numpy(labels.astype(np.int64));truth=torch.from_numpy(row['recorded_outcome'].astype(np.float32))
    tv=torch.from_numpy(row['outcome_value']);tp=torch.from_numpy(d.targets.probabilities(row['policy_logits'],row['legal'],2.)[:,:64].astype(np.float32))
    code=torch.from_numpy(d.encode(row['board'],row['turn']).astype(np.int64));phase=torch.from_numpy((64-np.count_nonzero(row['board'],axis=1)).astype(np.float32)/60)
    v,w,p=net(code,phase);loss=d.loss_fn(v,w,p,truth,move,mask,tv,tp,0.,2.)
    target=torch.where(truth>0,0,torch.where(truth<0,1,2)).long()
    original_loss=F.mse_loss(v,truth)+.1*F.cross_entropy(w,target)+.1*F.cross_entropy(p.masked_fill(~mask,-1e9),move)
    assert torch.equal(loss,original_loss)
    losses={}
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0003)
    before={k:v.clone() for k,v in net.state_dict().items()}
    for alpha in (1.,):
        v,w,p=net(code,phase);loss=d.loss_fn(v,w,p,truth,move,mask,tv,tp,alpha,2.)
        expected=F.mse_loss(v,tv)+.1*F.cross_entropy(w,target)+.4*(-tp*F.log_softmax(p.masked_fill(~mask,-1e9)/2.,1)).sum(1).mean()
        assert torch.allclose(loss,expected,atol=1e-7)
        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in net.parameters())
        losses[str(alpha)]=float(loss.detach());optimizer.step()
    assert any(not torch.equal(v,before[k]) for k,v in net.state_dict().items())
    net.eval();path=work/'smoke.otpat';d.export(net,path,outcome=True)
    pred=d.old.predict(net,row['board'],row['turn']);check=d.parity(ROOT/'.build/training/pattern-cnn-distill-v1.27-run1/libothello-pattern-frozen.dylib',path,row['board'],row['turn'],pred)
    check['maxOutcomeError']=check.pop('maxDiscError')
    report=dict(complete=True,parentExactBinaryRoundtrip=True,realTrainingRows=64,d4Symmetries=8,alphaZeroEqualsSupervisedLoss=True,pureTeacherValueAndPolicyFormula=True,finiteGradients=True,optimizerChangesParameters=True,losses=losses,cppParity=check,parentSha256=d.sha(parent.with_suffix('.otpat')),teacherManifestSha256=d.sha(cache.root/'manifest.json'))
    d.write(work/'report.json',report);print(json.dumps(report))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Materialize soft targets and explicit sample weights from v1.40 evidence."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def targets(data):
    count=data['occurrence_count'].astype(np.float64);quality=data['policy_quality_total'];offsets=data['policy_offsets']
    group=np.repeat(np.arange(len(count)),np.diff(offsets).astype(np.int64))
    assert (count>0).all() and (quality>0).all()
    wdl=data['wdl_count']/count[:,None]
    confidence=quality/count
    policy=data['policy_mass']/count[group]
    elo_policy=data['policy_quality_mass']/quality[group]
    for probability in [policy,elo_policy]:np.testing.assert_allclose(np.bincount(group,weights=probability,minlength=len(count)),1,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(wdl.sum(1),1,rtol=1e-12,atol=1e-12)
    return dict(value_mean=wdl[:,0]-wdl[:,1],wdl_probability=wdl,policy_offsets=offsets,policy_action=data['policy_action'],policy_probability=policy,policy_elo_probability=elo_policy,policy_quality_mean=confidence,policy_loss_weight_neutral_one=confidence/.5,matched_train_group=data['matched_train_group'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,default=ROOT/'.build/training/consensus-elo-v1.40');args=parser.parse_args()
    path=args.work/'training-targets.npz';assert not path.exists(),'Derived targets are immutable'
    result=targets(dict(np.load(args.work/'consensus.npz')));np.savez_compressed(path,**result)
    metadata=dict(schema='v1.40-consensus-training-targets',sourceSha256=sha(args.work/'consensus.npz'),artifactSha256=sha(path),scriptSha256=sha(__file__),wdlOrder=['win','loss','draw'],policy='CSR offsets/actions shared by empirical and optional Elo-weighted probabilities',sampleWeight='policy_loss_weight_neutral_one=mean occurrence quality/.5; needed to downweight singleton low-quality moves; use weighted CE=sum(weight*CE)/sum(weight) to retain overall auxiliary-loss scale',value='Use value_mean and soft WDL directly; do not take sign(value_mean). Value observations remain unweighted by Elo.',ablation='Compare first-match labels, unweighted empirical targets, and Elo policy targets+sample weights as separate candidates, using matched_train_group for exact original-million sample control.',modelTrained=False)
    (args.work/'training-targets.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(dict(complete=True,positions=len(result['value_mean']),sampleWeightQuantiles=list(map(float,np.quantile(result['policy_loss_weight_neutral_one'],[0,.25,.5,.75,1]))))))


if __name__=='__main__':main()

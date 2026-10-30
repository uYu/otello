#!/usr/bin/env python3
"""Verify exact original rows and canonical-to-original empirical-policy alignment."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from training_consensus import PERMS
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    work=ROOT/'.build/training/mixed-label-ablation-v1.45';proto=json.loads((ROOT/'models/v1.45-mixed-label-ablation-protocol.json').read_text())
    for name,digest in proto['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    soft=dict(np.load(work/'aligned-soft-labels.npz'));data=dict(np.load(ROOT/'.build/training/consensus-elo-v1.40/consensus.npz'));mapping=soft['canonical_group'];assert len(mapping)==1000000;np.testing.assert_array_equal(mapping,data['matched_train_group'])
    expected=(data['wdl_count'][mapping]/data['occurrence_count'][mapping,None]).astype(np.float32);np.testing.assert_array_equal(soft['wdl'],expected);np.testing.assert_array_equal(soft['value'],expected[:,0]-expected[:,1]);np.testing.assert_allclose(soft['policy'].sum(1),1,atol=1e-6,rtol=0)
    original=dict(np.load(ROOT/'.build/training/v1/positions.npz'));policy=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'));hard=np.sign(original['train_target']);category=np.where(hard>0,0,np.where(hard<0,1,2));assert (soft['wdl'][np.arange(1000000),category]>0).all() and (soft['policy'][np.arange(1000000),policy['train']]>0).all()
    rng=np.random.default_rng(20261030);samples=set(rng.choice(1000000,512,replace=False).tolist());changed=np.flatnonzero(soft['policy'][np.arange(1000000),policy['train']]<1-1e-7);samples.update(changed[:256].tolist())
    for row in samples:
        relative=(original['train_board'][row]*original['train_turn'][row]).astype(np.int8);best=None;first=None
        for s,perm in enumerate(PERMS):
            own=opponent=0
            for j,old in enumerate(perm):
                if relative[old]==1:own|=1<<j
                elif relative[old]==-1:opponent|=1<<j
            pair=(own,opponent)
            if best is None or pair<best:best=pair;first=s
        group=int(mapping[row]);assert best==(int(data['keys'][group]['own']),int(data['keys'][group]['opponent'])) and first==int(soft['canonical_symmetry'][row])
        reference=np.zeros(64,np.float32);lo,hi=map(int,data['policy_offsets'][[group,group+1]])
        for at in range(lo,hi):reference[PERMS[first,int(data['policy_action'][at])]]=np.float32(data['policy_mass'][at]/data['occurrence_count'][group])
        np.testing.assert_array_equal(reference,soft['policy'][row])
    source_counts=data['source_count'][mapping].sum(0);assert (source_counts>0).all()
    heldout_reference=json.loads((ROOT/'models/v1.41-wthor-elo-filter.json').read_text())['frozenHoldouts']
    for split in ['validation','test']:
        for name in ['board','turn','target','policy']:
            array=policy[split] if name=='policy' else original[split+'_'+name];fingerprint=dict(shape=list(array.shape),dtype=array.dtype.str,sha256=hashlib.sha256(array.tobytes()).hexdigest());assert fingerprint==heldout_reference[split][name]
    report=dict(status='passed',originalTrainRows=1000000,allSoftWdlAndValueTargetsVerified=True,allOriginalHardTargetsSupported=True,scalarPolicyOrientationChecks=len(samples),bothCorporaPresent=True,sourceOccurrenceCounts=dict(wthor=int(source_counts[0]),logistello=int(source_counts[1])),frozenValidationRows=6000,frozenTestRows=6000,holdoutArrayAndPolicyHashesVerified=True,alignedSoftLabelsSha256=sha(work/'aligned-soft-labels.npz'),scriptSha256=sha(__file__))
    p=ROOT/'models/v1.45-mixed-label-input-validation.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()

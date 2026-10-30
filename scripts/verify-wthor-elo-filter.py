#!/usr/bin/env python3
"""Independent checks for the WTHOR-only, equally counted v1.41 candidate."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
WORK=ROOT/'.build/training/wthor-elo-v1.41';BASE=ROOT/'.build/training/consensus-elo-v1.40'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    r=json.loads((ROOT/'models/v1.41-wthor-elo-filter.json').read_text());threshold=r['selected']['threshold'];assert threshold==1600
    for name,digest in {**r['artifactSha256'],**r['preservedSha256'],**r['scriptSha256']}.items():assert sha(ROOT/name)==digest,name
    ledger={g['gameId']:g for g in map(json.loads,(BASE/'game-elo.jsonl').open())};index=list(map(json.loads,(BASE/'game-index.jsonl').open()));keep=np.zeros(len(index),bool);removed_log=removed_low=accepted=0
    decisions=list(map(json.loads,(WORK/'game-filter.jsonl').open()));seen=set()
    for d in decisions:
        assert d['index'] not in seen;seen.add(d['index']);i=d['index'];assert index[i]['gameId']==d['gameId'] and index[i]['split']=='train'
        if d['source']=='logistello':want=False;removed_log+=1
        else:
            row=ledger[d['gameId']];assert row['complete'] and row['split']=='train'
            a,b=row['blackTrainYearStartElo'],row['whiteTrainYearStartElo'];low=(a is not None and b is not None and a<threshold and b<threshold)
            want=not low;removed_low+=low
        assert d['keep']==want;keep[i]=want;accepted+=want
    assert removed_log==60775 and removed_low==28707 and accepted==51854 and len(seen)==141336
    with np.load(BASE/'occurrences.npz') as z:old={k:z[k] for k in ['keys','action_bits','outcome','source','game_index']}
    mask=keep[old['game_index']];occ=dict(np.load(WORK/'occurrences.npz'))
    assert set(occ)==set(old)
    for name in old:np.testing.assert_array_equal(old[name][mask],occ[name])
    assert not occ['source'].any()
    data=dict(np.load(WORK/'consensus.npz'));keys,group,count=np.unique(occ['keys'],return_inverse=True,return_counts=True)
    np.testing.assert_array_equal(keys,data['keys']);np.testing.assert_array_equal(count,data['occurrence_count'])
    for c,outcome in enumerate([1,-1,0]):np.testing.assert_array_equal(np.bincount(group,weights=occ['outcome']==outcome,minlength=len(keys)),data['wdl_count'][:,c])
    assert not any('quality' in name or 'elo' in name for name in data);assert (data['policy_loss_weight']==1).all()
    selected=set(np.argsort(count)[-100:].tolist());rng=np.random.default_rng(20261030);selected.update(rng.choice(len(keys),100,replace=False).tolist());reference=collections.defaultdict(collections.Counter)
    for g,bits in zip(group[np.isin(group,list(selected))],occ['action_bits'][np.isin(group,list(selected))]):
        value=int(bits);n=value.bit_count()
        while value:
            bit=value&-value;reference[int(g)][bit.bit_length()-1]+=1/n;value-=bit
    for g,row in reference.items():
        lo,hi=map(int,data['policy_offsets'][[g,g+1]]);actual=dict(zip(map(int,data['policy_action'][lo:hi]),map(float,data['policy_mass'][lo:hi])));assert dict(row)==actual
    for split in ['validation','test']:
        with np.load(ROOT/'.build/training/v1/positions.npz') as frozen,np.load(ROOT/'.build/training/v1.8/policy.npz') as policy:
            for name in ['board','turn','target','policy']:
                arr=policy[split] if name=='policy' else frozen[split+'_'+name];fingerprint=dict(shape=list(arr.shape),dtype=arr.dtype.str,sha256=hashlib.sha256(arr.tobytes()).hexdigest());assert fingerprint==r['frozenHoldouts'][split][name]
    mapping=data['matched_train_group'];rows=data['matched_train_row'];assert len(mapping)==1000000 and (mapping[rows]>=0).all();np.testing.assert_array_equal(mapping[rows],data['matched_retained_group']);np.testing.assert_array_equal(np.flatnonzero(mapping>=0),rows)
    result=dict(status='passed',selectedThreshold=threshold,keptTrainGames=accepted,excludedLogistelloGames=removed_log,excludedBothBelowThresholdGames=removed_low,filteredOccurrencesExactlyVerified=len(group),equalPolicyMassReferenceGroups=len(reference),allWdlCountsVerified=True,allPolicySampleWeightsOne=True,allEloQualityFieldsAbsent=True,unchangedValidationPositions=6000,unchangedTestPositions=6000,holdoutArrayAndPolicyHashesVerified=True,matchedMillionRetained=len(rows),reportSha256=sha(ROOT/'models/v1.41-wthor-elo-filter.json'),scriptSha256=sha(__file__))
    path=ROOT/'models/v1.41-wthor-elo-filter-validation.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Verify the frozen v1.9 scalar-outcome candidate and its 20-game experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    experiment=json.loads((ROOT/'models/v1.9-outcome-experiment.json').read_text())
    assert experiment['status']=='candidate' and not experiment['defaultPromoted']
    for name,digest in experiment['artifactSha256'].items():assert sha(ROOT/name)==digest,name
    snapshot=json.loads((ROOT/experiment['sourceManifest']).read_text())
    assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
    with tarfile.open(ROOT/snapshot['archive'],'r:gz') as archive:
        assert len(archive.getmembers())==len(snapshot['files'])
        for name,digest in snapshot['files'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
    model=(ROOT/experiment['model']).read_bytes()
    baseline=json.loads((ROOT/experiment['baselineManifest']).read_text())
    parent=(ROOT/baseline['defaultModel']).read_bytes()
    assert sha(ROOT/baseline['defaultModel'])==baseline['modelSha256']
    offset=24+128*256*2+256*4
    assert model[:8]==b'OTNNUE4\0' and len(model)==len(parent)==104936
    assert model[8:offset]==parent[8:offset] and model[67612:]==parent[67612:]
    training=json.loads((ROOT/experiment['trainingReport']).read_text())
    assert training['sha256']==sha(ROOT/experiment['model'])
    assert training['parentSha256']==baseline['modelSha256']
    assert training['featuresUnchanged'] and training['policyBytesUnchanged']
    assert training['trainableParameters']==513 and training['cppParityPositions']==1024
    assert training['cppMaxRoundingError']<=0.501 and training['cppPolicyMaxError']==0
    assert training['trainingScriptSha256']==snapshot['files']['scripts/train-outcome.py']
    assert all(snapshot['files'][p]==digest for p,digest in training['helperSha256'].items())
    assert training['dataset']['cacheSha256']==json.loads((ROOT/'models/v1-dataset.json').read_text())['cacheSha256']
    report=json.loads((ROOT/experiment['matchReport']).read_text())
    assert report['complete'] and report['games']==report['plannedGames']==len(report['records'])==20
    assert report['millisecondsPerMove']==1000 and report['start']=='initial' and not report['useBook']
    assert report['baselineManifestSha256']==sha(ROOT/experiment['baselineManifest'])
    assert report['baselineLibrarySha256']==baseline['nativeLibrarySha256']
    assert all(snapshot['files'][p]==digest for p,digest in report['sourceSha256'].items())
    assert report['runnerSha256']==snapshot['files']['scripts/compare-nnue-candidate.py']
    assert [(c['prior'],c['valueTarget']) for c in report['configs']]==[('learned','discDifference'),('learned','outcome')]
    assert [c['modelSha256'] for c in report['configs']]==[baseline['modelSha256'],training['sha256']]
    spec=importlib.util.spec_from_file_location('outcome_match',ROOT/'scripts/compare-nnue-candidate.py')
    comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison);match=comparison.match
    assert report['sharedRunnerSha256']==sha(ROOT/'scripts/compare-nnue-search.py')
    assert report['rulesSha256']==sha(ROOT/'scripts/compare-resnet.py')
    assert report['policyAuditSha256']==sha(ROOT/'scripts/opening_policy.py')
    assert report['openingPolicy']==match.POLICY
    match.CONFIGS=report['configs'];targets={c['id']:c['valueTarget'] for c in report['configs']}
    actions=0
    for i,game in enumerate(report['records']):
        assert game['number']==i+1 and game['opening']==[] and game['openingPolicy']==match.POLICY
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(report['baseSeed']^((i//2+1)*0x9e3779b9))&0x7fffffff
        assert all(d['search']=='mcts' and d['valueTarget']==targets[d['engine']] for d in game['decisions'])
        actions+=match.replay(game)
    totals,matrix=match.aggregate(report['records'])
    assert totals==report['results']==experiment['results'] and matrix==report['matrix']
    assert actions==report['replayedActions']==experiment['replayedActions']
    print(f'Outcome candidate verified: 20 games, {actions} audited actions, frozen features/policy and explicit outcome format; v1.8 baseline preserved')


if __name__=='__main__':verify()

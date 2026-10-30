#!/usr/bin/env python3
"""Audit the candidate experiment without promoting it to the default baseline."""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path):
    experiment=json.loads(path.read_text())
    protocol=experiment.get('protocol',dict(games=100,millisecondsPerMove=100))
    assert experiment['status']=='candidate'
    for name,digest in experiment['artifactSha256'].items():
        assert sha(ROOT/name)==digest,name
    snapshot=json.loads((ROOT/experiment['sourceManifest']).read_text())
    assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
    with tarfile.open(ROOT/snapshot['archive'],'r:gz') as archive:
        assert len(archive.getmembers())==len(snapshot['files'])
        for name,expected in snapshot['files'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
    model=(ROOT/experiment['model']).read_bytes()
    baseline=json.loads((ROOT/experiment['baselineManifest']).read_text())
    parent=(ROOT/baseline['defaultModel']).read_bytes()
    assert hashlib.sha256(parent).hexdigest()==baseline['modelSha256']
    assert model[:8]==b'OTNNUE3\0' and model[8:len(parent)]==parent[8:]
    training=json.loads((ROOT/experiment['trainingReport']).read_text())
    assert training['sha256']==hashlib.sha256(model).hexdigest()
    assert training['parentSha256']==baseline['modelSha256']
    assert training['valueBytesUnchanged'] and training['valueParity']
    assert training['cppParityPositions']==1024 and training['cppMaxProbabilityError']<1e-10
    assert training['trainingScriptSha256']==snapshot['files']['scripts/train-policy.py']
    assert training['labels']['scriptSha256']==snapshot['files']['scripts/prepare-policy-data.py']
    assert training['labels']['positionsSha256']==json.loads((ROOT/'models/v1-dataset.json').read_text())['cacheSha256']
    report=json.loads((ROOT/experiment['matchReport']).read_text())
    assert report['complete'] and report['games']==report['plannedGames']==len(report['records'])==protocol['games']
    assert report['millisecondsPerMove']==protocol['millisecondsPerMove'] and report['start']=='initial' and not report['useBook']
    assert report['baselineManifestSha256']==sha(ROOT/experiment['baselineManifest'])
    assert all(snapshot['files'][name]==digest for name,digest in report['sourceSha256'].items())
    assert report['runnerSha256']==snapshot['files']['scripts/compare-nnue-candidate.py']
    if experiment.get('resumeCheckpoint'):
        checkpoint=ROOT/experiment['resumeCheckpoint']
        previous=json.loads(checkpoint.read_text())
        assert not previous['complete']
        count=previous['games']
        assert previous['records']==report['records'][:count]
        history=report['resumeHistory']
        assert len(history)==1 and history[0]['completedGames']==count
        assert history[0]['previousReportSha256']==sha(checkpoint)
        assert history[0]['previousRunnerSha256']==previous['runnerSha256']
        assert history[0]['restartedPartialGame']==count+1
        prior=json.loads((ROOT/experiment['priorRunnerManifest']).read_text())
        assert previous['runnerSha256']==prior['files']['scripts/compare-nnue-candidate.py']
        assert sha(ROOT/prior['archive'])==prior['sha256']
        assert report['runnerSegments']==[
            dict(firstGame=1,lastGame=count,sha256=previous['runnerSha256']),
            dict(firstGame=count+1,lastGame=protocol['games'],sha256=report['runnerSha256'])]
        for key in ('millisecondsPerMove','plannedGames','baseSeed','openingPolicy',
                    'librarySha256','baselineLibrarySha256','baselineManifestSha256','sourceSha256'):
            assert previous[key]==report[key]
    spec=importlib.util.spec_from_file_location('policy_comparison',ROOT/'scripts/compare-nnue-candidate.py')
    comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
    shared=comparison.match
    assert report['sharedRunnerSha256']==sha(ROOT/'scripts/compare-nnue-search.py')
    assert report['rulesSha256']==sha(ROOT/'scripts/compare-resnet.py')
    assert report['policyAuditSha256']==sha(ROOT/'scripts/opening_policy.py')
    assert report['openingPolicy']==shared.POLICY
    assert report['configs'][0]['modelSha256']==baseline['modelSha256']
    assert report['configs'][1]['modelSha256']==training['sha256']
    assert [c['prior'] for c in report['configs']]==['uniform','learned']
    shared.CONFIGS=report['configs']
    actions=0
    for i,game in enumerate(report['records']):
        assert game['number']==i+1
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(report['baseSeed']^((i//2+1)*0x9e3779b9))&0x7fffffff
        assert game['openingPolicy']==shared.POLICY and game['opening']==[]
        assert all(d['search']=='mcts' for d in game['decisions'])
        actions+=shared.replay(game)
    totals,matrix=shared.aggregate(report['records'])
    assert totals==report['results']==experiment['results'] and matrix==report['matrix']
    assert actions==report['replayedActions']==experiment['replayedActions']
    print(f"Policy candidate verified: {report['games']} games at {report['millisecondsPerMove']}ms, {actions} audited actions, frozen value bytes and source snapshot; no default promotion")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'models/v1.8-policy-experiment.json')
    verify(parser.parse_args().experiment)

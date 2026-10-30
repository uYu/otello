#!/usr/bin/env python3
"""Audit the residual-trunk experiment without changing the approved baseline."""
import argparse,hashlib,importlib.util,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def verify(path=ROOT/'models/v1.10-deep-experiment.json'):
    experiment=json.loads(Path(path).read_text())
    assert experiment['status']=='candidate' and not experiment['defaultPromoted']
    for path,digest in experiment['artifactSha256'].items():assert sha(path)==digest,path
    snapshot=json.loads((ROOT/experiment['sourceManifest']).read_text());assert sha(snapshot['archive'])==snapshot['sha256']
    with tarfile.open(ROOT/snapshot['archive'],'r:gz') as archive:
        assert len(archive.getmembers())==len(snapshot['files'])
        for path,digest in snapshot['files'].items():assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==digest,path
    baseline=json.loads((ROOT/experiment['baselineManifest']).read_text());training=json.loads((ROOT/experiment['trainingReport']).read_text())
    model=(ROOT/experiment['model']).read_bytes();parent=(ROOT/baseline['defaultModel']).read_bytes()
    assert model[:8]==b'OTNNUE5\0' and len(model)==138864 and len(parent)==104936
    assert model[8:66584]==parent[8:66584] and model[67612:104936]==parent[67612:]
    assert training['sha256']==sha(experiment['model']) and training['parentSha256']==baseline['modelSha256']==sha(baseline['defaultModel'])
    assert training['parameters']==68770 and training['trainableParameters']==17185
    assert training['featuresUnchanged'] and training['policyBytesUnchanged']
    assert training['cppParityPositions']==1024 and training['cppMaxRoundingError']<=.501 and training['cppMaxPolicyError']<1e-10
    assert training['trainingScriptSha256']==snapshot['files']['scripts/train-deep.py']
    assert all(snapshot['files'][p]==digest for p,digest in training['helperSha256'].items())
    assert training['dataset']['cacheSha256']==json.loads((ROOT/'models/v1-dataset.json').read_text())['cacheSha256']
    parity=json.loads((ROOT/experiment['legacyParity']).read_text());assert parity['identical'] and parity['positions']==24 and parity['librarySha256'][0]==baseline['nativeLibrarySha256']
    report=json.loads((ROOT/experiment['matchReport']).read_text())
    assert report['plannedGames']==experiment['protocol']['games']==20
    assert report['games']==len(report['records'])==experiment['protocol']['completedGames']
    if experiment.get('stopped'):
        assert not report['complete'] and report['stopped'] and report['games']<report['plannedGames']
    else:
        assert report['complete'] and report['games']==report['plannedGames']
    assert report['millisecondsPerMove']==experiment['protocol']['millisecondsPerMove']
    assert report['millisecondsPerMove'] in (1000,5000) and report['start']=='initial' and not report['useBook']
    assert report['baselineManifestSha256']==sha(experiment['baselineManifest'])
    if experiment.get('engineeringManifest'):
        assert sha(experiment['engineeringManifest'])==experiment['engineeringManifestSha256']
        engineering=json.loads((ROOT/experiment['engineeringManifest']).read_text())
        assert report['baselineLibrarySha256']==report['librarySha256']==engineering['optimizedLibrarySha256']
        for path,digest in engineering['artifactSha256'].items():assert sha(path)==digest,path
        checks=json.loads((ROOT/engineering['parityReport']).read_text())
        assert all(row['identical'] and row['valueError']==row['policyError']==0 and row['optimizedLibrarySha256']==report['librarySha256'] for row in checks['results'])
        assert checks['results'][0]['referenceLibrarySha256']==baseline['nativeLibrarySha256']
    else:
        assert report['baselineLibrarySha256']==baseline['nativeLibrarySha256']
    assert all(snapshot['files'][p]==digest for p,digest in report['sourceSha256'].items())
    assert report['runnerSha256']==snapshot['files']['scripts/compare-nnue-candidate.py']
    assert [(c['prior'],c['valueTarget']) for c in report['configs']]==[('learned','outcome'),('learned','outcome')]
    assert [c['modelSha256'] for c in report['configs']]==[baseline['modelSha256'],training['sha256']]
    spec=importlib.util.spec_from_file_location('deep_comparison',ROOT/'scripts/compare-nnue-candidate.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison);match=comparison.match
    assert report['sharedRunnerSha256']==sha('scripts/compare-nnue-search.py') and report['rulesSha256']==sha('scripts/compare-resnet.py') and report['policyAuditSha256']==sha('scripts/opening_policy.py')
    match.CONFIGS=report['configs'];actions=0
    for i,game in enumerate(report['records']):
        assert game['number']==i+1 and game['opening']==[] and game['openingPolicy']==match.POLICY
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(report['baseSeed']^((i//2+1)*0x9e3779b9))&0x7fffffff
        assert all(d['valueTarget']=='outcome' and d['search']=='mcts' for d in game['decisions'])
        actions+=match.replay(game)
    totals,matrix=match.aggregate(report['records']);assert totals==report['results']==experiment['results'] and matrix==report['matrix']
    assert actions==report['replayedActions']==experiment['replayedActions']
    print(f"Deep candidate verified: {report['games']} completed games, {actions} audited actions, residual trunk and frozen v1.9 baseline")
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'models/v1.10-deep-experiment.json')
    verify(parser.parse_args().experiment)

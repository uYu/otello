#!/usr/bin/env python3
"""Replay/audit the requested v1.49 vs approved v1.37 2s/20 comparison."""
import hashlib,importlib.util,json,sys,tarfile
import global_policy_native_experiment as e
REPORT=e.ROOT/'models/v1.49-global-policy-vs-v1.37-2s-20.json'
SUMMARY=e.ROOT/'models/v1.49-global-policy-2s-20-comparison.json'
AUDIT=e.ROOT/'models/v1.49-global-policy-2s-20-experiment.json'


def immutable(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text,path
    else:path.write_text(text)


def main():
    r=e.read(REPORT);assert r['complete'] and r['games']==r['plannedGames']==20 and r['millisecondsPerMove']==2000
    assert r['useBook'] is False and r['baseSeed']==20261029
    protocol=e.ROOT/'models/v1.49-global-policy-2s-20-protocol.json';p=e.read(protocol)
    assert r['protocolSha256']==e.sha(protocol) and not p['defaultPromoted']
    e.verify_preserved()
    for name,digest in p['libraries'].items():assert e.sha(e.ROOT/name)==digest,name
    manifest=e.ROOT/r['sourceArchiveManifest'];source=e.read(manifest)
    assert e.sha(manifest)==r['sourceArchiveManifestSha256'] and e.sha(e.ROOT/source['archive'])==source['sha256']
    with tarfile.open(e.ROOT/source['archive'],'r:gz') as archive:
        for name,digest in source['files'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
            assert e.sha(e.ROOT/name)==digest,name
    prefix=str((e.WORK/'source').relative_to(e.ROOT))+'/'
    for name,digest in r['sourceSha256'].items():assert source['files'][prefix+name]==digest,name
    adapter=e.WORK/'compare-global-policy-candidate.py';assert r['runnerSha256']==e.sha(adapter)
    assert r['librarySha256']==e.sha(e.LIB) and r['baselineLibrarySha256']==e.sha(e.REFERENCE)
    configs={c['id']:c for c in r['configs']}
    assert configs['candidate']['modelSha256']==e.sha(e.MODEL)==p['candidateModelSha256']
    assert configs['baseline']['modelSha256']==e.sha(e.BASE)==p['baselineModelSha256']
    assert all(c['prior']=='learned' and c['valueTarget']=='outcome' for c in configs.values())
    spec=importlib.util.spec_from_file_location('global_policy_match_audit_runner',e.ROOT/'scripts/run-global-policy-match-2s.py')
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    match=runner.match;match.CONFIGS=r['configs'];actions=0
    for i,game in enumerate(r['records']):
        assert game['number']==i+1
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(20261029^((i//2+1)*0x9e3779b9))&0x7fffffff
        actions+=match.replay(game)
        for decision in game['decisions']:
            assert decision['valueTarget']=='outcome' and decision['mcts']['maxNodes']==4194304
    results,matrix=match.aggregate(r['records'])
    assert results==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']
    stats={}
    for engine in ['candidate','baseline']:
        decisions=[d for game in r['records'] for d in game['decisions'] if d['engine']==engine]
        n=len(decisions)
        stats[engine]=dict(moves=n,meanWallMs=sum(d['wallMs'] for d in decisions)/n,
            meanSimulations=sum(d['mcts']['simulations'] for d in decisions)/n,
            meanNewLeafEvaluations=sum(d['mcts']['evaluations'] for d in decisions)/n,
            memoryLimitedDecisions=sum(d['mcts']['memoryLimited'] for d in decisions),
            proofStoppedDecisions=sum(d['exact'] for d in decisions),
            peakRssMiB=max(d['mcts']['peakRssBytes'] for d in decisions)/(1024*1024))
    colors={}
    for color in ['black','white']:
        games=[g for g in r['records'] if g[color]=='candidate']
        assert len(games)==10
        colors[color]=dict(win=sum(g['winner']=='candidate' for g in games),draw=sum(g['winner'] is None for g in games),loss=sum(g['winner']=='baseline' for g in games))
    c=results['candidate'];training=e.read(e.TRAINING)
    summary=dict(version='v1.49-global-policy-2s-20',complete=True,status='candidate',defaultPromoted=False,
        candidate='v1.49 full-retrained global38->16->64 policy',baseline='frozen approved v1.37 symmetry',
        millisecondsPerMove=2000,games=20,candidateWDL=dict(win=c['win'],draw=c['draw'],loss=c['loss']),
        candidateScoreRate=c['scoreRate'],candidateMeanDiscDifference=c['meanDiscDifference'],candidateWDLByColor=colors,searchStatistics=stats,
        trainingSelectedEpoch=training['bestEpoch'],trainingTestMetrics=training['metrics']['test'],baselineTestMetrics=training['baselineMetrics']['test'],
        replayedActions=actions,allDecisionsAudited=True,sourceArchiveManifest=r['sourceArchiveManifest'],
        limits='20-game paired screening sample, no Elo or significance inference. Separate2s budget; initial-board no-book model-only flow. Per-engine runtime statistics describe different reached positions and are not same-position kernel speed ratios.',
        report=str(REPORT.relative_to(e.ROOT)),reportSha256=e.sha(REPORT))
    immutable(SUMMARY,summary)
    artifacts=[REPORT,SUMMARY,protocol,manifest,e.ROOT/source['archive'],e.MODEL,e.ROOT/'models/v1.49-global-policy-native-validation.json',e.TRAINING]
    immutable(AUDIT,dict(complete=True,defaultPromoted=False,allActionsReplayed=actions,allDecisionsAudited=True,pairedColors=True,sharedPairSeeds=True,
        candidateWDL=summary['candidateWDL'],candidateScoreRate=c['scoreRate'],searchStatistics=stats,
        preservedSha256=p['preservedSha256'],artifactSha256={str(x.relative_to(e.ROOT)):e.sha(x) for x in artifacts},auditorSha256=e.sha(__file__)))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()

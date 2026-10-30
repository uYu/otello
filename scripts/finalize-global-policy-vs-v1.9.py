#!/usr/bin/env python3
"""Audit the separate latest-vs-frozen-v1.9 200ms/50-game comparison."""
import hashlib,importlib.util,json,sys,tarfile
import global_policy_native_experiment as e
REPORT=e.ROOT/'models/v1.49-global-policy-vs-v1.9-200ms-50.json'
SUMMARY=e.ROOT/'models/v1.49-v1.9-200ms-50-comparison.json'
AUDIT=e.ROOT/'models/v1.49-v1.9-200ms-50-experiment.json'


def immutable(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text,path
    else:path.write_text(text)


def verify_archive(path,current=False):
    record=e.read(path);archive=e.ROOT/record['archive'];assert e.sha(archive)==record['sha256']
    with tarfile.open(archive,'r:gz') as source:
        for name,digest in record['files'].items():
            assert hashlib.sha256(source.extractfile(name).read()).hexdigest()==digest,name
            if current:assert e.sha(e.ROOT/name)==digest,name
    return record


def main():
    r=e.read(REPORT);assert r['complete'] and r['games']==r['plannedGames']==50 and r['millisecondsPerMove']==200
    assert not r['useBook'] and r['baseSeed']==20261029
    protocol=e.ROOT/'models/v1.49-v1.9-200ms-50-protocol.json';p=e.read(protocol)
    assert e.sha(protocol)==r['protocolSha256'] and not p['defaultPromoted']
    for name,digest in p['preservedSha256'].items():assert e.sha(e.ROOT/name)==digest,name
    for name,digest in p['libraries'].items():assert e.sha(e.ROOT/name)==digest,name
    manifest=e.ROOT/r['sourceArchiveManifest'];source=verify_archive(manifest,True)
    assert e.sha(manifest)==r['sourceArchiveManifestSha256']
    for name,digest in source['engineSourceManifests'].items():assert e.sha(e.ROOT/name)==digest;verify_archive(e.ROOT/name)
    spec=importlib.util.spec_from_file_location('global_v19_match_auditor',e.ROOT/'scripts/run-global-policy-vs-v1.9.py')
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    match=runner.match;match.CONFIGS=r['configs'];configs={c['id']:c for c in r['configs']}
    assert configs['baseline']['modelSha256']==p['baselineModelSha256']==e.sha(runner.BASEMODEL)
    assert configs['candidate']['modelSha256']==p['candidateModelSha256']==e.sha(e.MODEL)
    assert all(c['prior']=='learned' and c['valueTarget']=='outcome' for c in configs.values())
    assert r['baselineLibrarySha256']==e.sha(runner.BASELIB)==e.read(runner.BASELINE)['nativeLibrarySha256']
    assert r['librarySha256']==e.sha(e.LIB)
    actions=0
    for i,game in enumerate(r['records']):
        assert game['number']==i+1
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(20261029^((i//2+1)*0x9e3779b9))&0x7fffffff
        actions+=match.replay(game)
        for d in game['decisions']:assert d['valueTarget']=='outcome' and d['mcts']['maxNodes']==runner.CAPS[d['engine']]
    results,matrix=match.aggregate(r['records']);assert results==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']
    stats={}
    for engine in ['candidate','baseline']:
        rows=[d for g in r['records'] for d in g['decisions'] if d['engine']==engine];n=len(rows)
        stats[engine]=dict(moves=n,meanWallMs=sum(d['wallMs'] for d in rows)/n,meanSimulations=sum(d['mcts']['simulations'] for d in rows)/n,
            meanNewLeafEvaluations=sum(d['mcts']['evaluations'] for d in rows)/n,peakRssMiB=max(d['mcts']['peakRssBytes'] for d in rows)/1024**2,
            memoryLimitedDecisions=sum(d['mcts']['memoryLimited'] for d in rows),proofStoppedDecisions=sum(d['exact'] for d in rows))
    colors={}
    for color in ['black','white']:
        games=[g for g in r['records'] if g[color]=='candidate'];assert len(games)==25
        colors[color]=dict(win=sum(g['winner']=='candidate' for g in games),draw=sum(g['winner'] is None for g in games),loss=sum(g['winner']=='baseline' for g in games))
    c=results['candidate']
    summary=dict(version='v1.49-vs-frozen-v1.9-200ms-50',complete=True,status='candidate',defaultPromoted=False,candidate='complete global policy v1.49',baseline='exact frozen v1.9 model and library',
        millisecondsPerMove=200,games=50,candidateWDL=dict(win=c['win'],draw=c['draw'],loss=c['loss']),candidateScoreRate=c['scoreRate'],candidateMeanDiscDifference=c['meanDiscDifference'],
        candidateWDLByColor=colors,searchStatistics=stats,maxNodesByEngine=runner.CAPS,v1_9BinaryMatchesOriginalFrozenHash=True,
        replayedActions=actions,allDecisionsAudited=True,report=str(REPORT.relative_to(e.ROOT)),reportSha256=e.sha(REPORT),
        limits='Historical-version comparison with preserved per-version searches: NNUE v1.9 samples every unvisited child,2M cap; Pattern v1.49 finite FPU,4M cap. Equal200ms budget,25 paired colors, no opening book. Keep separate from2s and other opponents. No automatic default promotion or Elo inference. Different reached positions make runtime stats descriptive, not pure kernel speed ratios.')
    immutable(SUMMARY,summary)
    artifacts=[REPORT,SUMMARY,protocol,manifest,e.ROOT/source['archive'],e.MODEL,runner.BASEMODEL,runner.BASELINE,runner.BUILD_RECORD]
    immutable(AUDIT,dict(complete=True,defaultPromoted=False,allActionsReplayed=actions,allDecisionsAudited=True,pairedColors=True,sharedPairSeeds=True,
        candidateWDL=summary['candidateWDL'],candidateScoreRate=c['scoreRate'],preservedSha256=p['preservedSha256'],
        artifactSha256={str(x.relative_to(e.ROOT)):e.sha(x) for x in artifacts},auditorSha256=e.sha(__file__)))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()

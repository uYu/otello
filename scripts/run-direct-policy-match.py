#!/usr/bin/env python3
"""Run the requested 200ms/20-game direct MLP pilot against frozen v1.37."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/direct-policy-match-v1.38'
SUFFIX='dylib' if sys.platform=='darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-direct-policy-v1.38/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.37-reference/libothello.{SUFFIX}'
MODEL=ROOT/'models/pattern32-direct-mlp8-v1.38.otpat'
PROTOCOL=ROOT/'models/v1.38-direct-policy-200ms-20-protocol.json'
RUNTIME=WORK/'match-200ms-20.json'
REPORT=ROOT/'models/v1.38-direct-mlp8-vs-v1.37-200ms-20.json'
SOURCE=ROOT/'models/archive/v1.38-direct-policy-200ms-20-sources.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('direct_match_helpers',ROOT/'scripts/run-pattern-structure.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
match=h.match


def monitor():
    item=dict(name='直连 MLP v1.38 vs 对称共享 v1.37 · 200ms/步 · 20局',focus='candidate',displayNames=dict(candidate='直连 MLP',baseline='v1.37 对称共享'),displayOrder=['candidate','baseline'],games=20,runtimeReport=str(RUNTIME.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)))
    config=read(ROOT/'models/current-experiment.json');history=config.get('archivedExperiments',[])
    for old in reversed(config['experiments']):
        if old.get('report')!=item['report'] and not any(x.get('report')==old.get('report') for x in history):history.insert(0,old)
    config.update(version='v1.38 直连 MLP · 200ms 棋力预检',experiments=[item],archivedExperiments=history)
    write(ROOT/'models/current-experiment.json',config)
    write(ROOT/'models/v1.38-direct-policy-200ms-20-experiments.json',dict(version=config['version'],experiments=[item]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert not REPORT.exists(),'Completed pilot is immutable'
    WORK.mkdir(parents=True,exist_ok=True)
    active=read(ROOT/'models/default-model.json');baseline=read(ROOT/active['baselineManifest'])
    assert active['version']=='v1.37' and baseline['userApproved'] and baseline['status']=='frozen'
    offline=read(ROOT/'models/v1.38-direct-policy-experiment.json')
    for name,digest in {**offline['artifactSha256'],**offline['preservedSha256']}.items():assert sha(ROOT/name)==digest,name
    timing=read(ROOT/'models/v1.38-direct-policy-benchmark.json')
    assert sha(LIB)==timing['libraries'][str(LIB.relative_to(ROOT))]
    assert sha(REFERENCE)==baseline['referenceNativeLibrary']['sha256']
    if not PROTOCOL.exists():
        assert not args.resume and not RUNTIME.exists()
        write(PROTOCOL,dict(version='v1.38-direct-policy-200ms-pilot',status='candidate',defaultPromoted=False,
            userRequest='200ms先打20把试试',baselineManifest=active['baselineManifest'],baselineManifestSha256=sha(ROOT/active['baselineManifest']),
            candidate=str(MODEL.relative_to(ROOT)),candidateModelSha256=sha(MODEL),baselineModel=active['defaultModel'],baselineModelSha256=active['modelSha256'],
            games=20,millisecondsPerMove=200,seed=20261029,start='initial',pairedColors=True,sharedPairSeeds=True,useBook=False,
            mcts=dict(maxNodes=4194304,patternCacheEntries=0,exploration=1.5,finiteFpuReduction=.15,subtreeReuse=True,terminal='outcome +1/0/-1'),openingPolicy=match.POLICY,
            librarySha256={str(p.relative_to(ROOT)):sha(p) for p in [LIB,REFERENCE]},preservedSha256=offline['preservedSha256'],
            offlineExperimentSha256=sha(ROOT/'models/v1.38-direct-policy-experiment.json'),limits='20-game paired-color screening sample; budgets separate; no Elo/statistical significance claim; no automatic default promotion.'))
    protocol=read(PROTOCOL)
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in protocol['librarySha256'].items():assert sha(ROOT/name)==digest,name
    paths=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/name for name in [
        'AGENTS.md','CMakeLists.txt','scripts/run-direct-policy-match.py','scripts/finalize-direct-policy-match.py','scripts/run-pattern-structure.py',
        'scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py',
        'scripts/watch-match.sh','scripts/watch-matches.py','models/v1.38-direct-policy-200ms-20-protocol.json']]))
    archive=SOURCE.with_suffix('.tar.gz')
    if SOURCE.exists():
        frozen=read(SOURCE);assert sha(archive)==frozen['sha256']
        assert {str(p.relative_to(ROOT)):sha(p) for p in paths}==frozen['files']
    else:
        assert not archive.exists()
        with tarfile.open(archive,'w:gz') as t:
            for p in paths:t.add(p,arcname=str(p.relative_to(ROOT)))
        write(SOURCE,dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    monitor();match.worker=h.worker;match.Engine=h.Engine
    original_write=match.rules.write_json
    def annotated(path,value):
        if Path(path).parent==WORK and 'configs' in value:
            value.update(protocolSha256=sha(PROTOCOL),sourceArchiveManifest=str(SOURCE.relative_to(ROOT)),maxNodesByEngine=dict(baseline=4194304,candidate=4194304),patternCacheEntries=0)
        return original_write(path,value)
    match.rules.write_json=annotated
    h.comparison.compare(argparse.Namespace(model=MODEL,baseline_manifest=ROOT/active['baselineManifest'],baseline_library=REFERENCE,label='v1.38 direct MLP8',library=LIB,
        output=RUNTIME,games=20,milliseconds=200,seed=20261029,resume=args.resume))
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    write(REPORT,read(RUNTIME));print(json.dumps(dict(complete=True,report=str(REPORT.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()

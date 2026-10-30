#!/usr/bin/env python3
"""Preregistered 100ms/20 pilot for v1.48 vs frozen approved v1.37."""
import argparse,importlib.util,json,multiprocessing,sys,tarfile
from pathlib import Path
import shared_residual_experiment as e
REPORT=e.ROOT/'models/v1.48-residual-vs-v1.37-100ms-20.json'
RUNTIME=e.WORK/'match-100ms-20.json'
PROTOCOL=e.ROOT/'models/v1.48-shared-residual-match-protocol.json'
ARCHIVE=e.ROOT/'models/archive/v1.48-shared-residual-match-sources.json'


def comparison_module():
    source=(e.ROOT/'scripts/compare-nnue-candidate.py').read_text()
    source=source.replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(e.ROOT)!r})\nCANDIDATE_SOURCE = ROOT/".build/training/pattern-shared-residual-v1.48/source"')
    assert source.count("b'OTPAT7\\0\\0'")==3
    source=source.replace("b'OTPAT7\\0\\0'","b'OTPAT7\\0\\0',b'OTPAT8\\0\\0'")
    old="sourceSha256={str(p.relative_to(ROOT)):match.sha(p) for p in sorted((ROOT/'cpp').rglob('*'))"
    assert old in source
    source=source.replace(old,"candidateSourceRoot=str(CANDIDATE_SOURCE),\n        sourceSha256={str(p.relative_to(CANDIDATE_SOURCE)):match.sha(p) for p in sorted((CANDIDATE_SOURCE/'cpp').rglob('*'))")
    path=e.WORK/'compare-shared-residual-candidate.py'
    if path.exists():assert path.read_text()==source
    else:path.write_text(source)
    spec=importlib.util.spec_from_file_location('shared_residual_comparison',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    spec=importlib.util.spec_from_file_location('shared_residual_worker_helpers',e.ROOT/'scripts/run-pattern-structure.py')
    helpers=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helpers;spec.loader.exec_module(helpers)
    module.match.worker=helpers.worker
    class Engine(module.match.Engine):
        def __init__(self,library,config):super().__init__(library,dict(config,maxNodes=4194304))
    module.match.Engine=Engine
    return module,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert not REPORT.exists(),'Completed reports are immutable'
    training=e.read(e.ROOT/'models/v1.48-pattern-shared-residual.json');assert training['complete']
    checks=e.read(e.ROOT/'models/v1.48-shared-residual-validation.json');assert checks['complete']
    e.verify_preserved();variant=training['variant'];model=e.ROOT/variant['model']
    comparison,adapter=comparison_module();match=comparison.match
    if not PROTOCOL.exists():
        e.write(PROTOCOL,dict(version='v1.48-shared-residual-100ms-pilot',defaultPromoted=False,baseline='models/v1.37-default.json',candidate=variant['model'],candidateModelSha256=e.sha(model),millisecondsPerMove=100,games=20,seed=20261029,start='initial',pairedColors=True,sharedPairSeeds=True,useBook=False,maxNodes=4194304,patternCacheEntries=0,exploration=1.5,finiteFpu=.15,subtreeReuse=True,terminal='outcome +1/0/-1',openingPolicy=match.POLICY,trainingReportSha256=e.sha(e.ROOT/'models/v1.48-pattern-shared-residual.json'),preservedSha256=e.preserved(),libraries={str(p.relative_to(e.ROOT)):e.sha(p) for p in [e.LIB,e.REFERENCE]},adapterSha256=e.sha(adapter),limits='20-game paired screening sample; one time budget, no Elo/significance claim; no default promotion; all training and correctness checks finished before match.'))
    protocol=e.read(PROTOCOL);e.verify_preserved(protocol)
    files={str(p.relative_to(e.ROOT)):p for p in (e.WORK/'source').rglob('*') if p.is_file()}
    for name in ['scripts/run-shared-residual-match.py','scripts/shared_residual_experiment.py','scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/run-pattern-structure.py','scripts/opening_policy.py','scripts/watch-match.sh','scripts/watch-matches.py','models/v1.48-shared-residual-match-protocol.json']:
        files[name]=e.ROOT/name
    files[str(adapter.relative_to(e.ROOT))]=adapter
    if not ARCHIVE.exists():
        archive=ARCHIVE.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for name,path in sorted(files.items()):t.add(path,arcname=name)
        e.write(ARCHIVE,dict(archive=str(archive.relative_to(e.ROOT)),sha256=e.sha(archive),files={name:e.sha(p) for name,p in sorted(files.items())}))
    else:
        assert e.read(ARCHIVE)['files']=={name:e.sha(p) for name,p in sorted(files.items())}
    config=e.read(e.ROOT/'models/current-experiment.json');history=config.setdefault('archivedExperiments',[])
    for item in config.get('experiments',[]):
        if not any(x.get('report')==item.get('report') for x in history):history.insert(0,item)
    config.update(version='v1.48 共享残差 vs v1.37 · 100ms · 20局',experiments=[dict(name='共享残差16 vs v1.37 · 100ms/步 · 20局',focus='candidate',displayNames=dict(candidate='共享残差16',baseline='v1.37 对称共享'),displayOrder=['candidate','baseline'],games=20,runtimeReport=str(RUNTIME.relative_to(e.ROOT)),report=str(REPORT.relative_to(e.ROOT)))])
    e.write(e.ROOT/'models/current-experiment.json',config)
    original=match.rules.write_json
    def annotated(path,value):
        if Path(path).parent==e.WORK and 'configs' in value:
            value.update(protocolSha256=e.sha(PROTOCOL),sourceArchiveManifest=str(ARCHIVE.relative_to(e.ROOT)),maxNodesByEngine=dict(candidate=4194304,baseline=4194304),patternCacheEntries=0)
        return original(path,value)
    match.rules.write_json=annotated
    comparison.compare(argparse.Namespace(model=model,baseline_manifest=e.ROOT/'models/v1.37-default.json',baseline_library=e.REFERENCE,label='v1.48 shared residual16',library=e.LIB,output=RUNTIME,games=20,milliseconds=100,seed=20261029,resume=args.resume))
    e.verify_preserved(protocol);e.write(REPORT,e.read(RUNTIME))
    print(json.dumps(dict(complete=True,results=e.read(REPORT)['results']),ensure_ascii=False))


if __name__=='__main__':main()

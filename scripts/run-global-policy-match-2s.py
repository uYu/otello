#!/usr/bin/env python3
"""Requested v1.49 complete global policy vs frozen v1.37, 2s/20 games."""
import argparse,ctypes,importlib.util,json,multiprocessing,resource,sys,tarfile,time
from pathlib import Path
import global_policy_native_experiment as e
RUNTIME=e.WORK/'match-2s-20.json'
REPORT=e.ROOT/'models/v1.49-global-policy-vs-v1.37-2s-20.json'
MATCH_PROTOCOL=e.ROOT/'models/v1.49-global-policy-2s-20-protocol.json'
SOURCE=e.ROOT/'models/archive/v1.49-global-policy-native-sources.json'


def comparison_module():
    text=(e.ROOT/'scripts/compare-nnue-candidate.py').read_text()
    text=text.replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(e.ROOT)!r})\nCANDIDATE_SOURCE = ROOT/".build/training/pattern-global-policy-native-v1.49/source"')
    assert text.count("b'OTPAT7\\0\\0'")==4
    text=text.replace("b'OTPAT7\\0\\0'","b'OTPAT7\\0\\0',b'OTPAT9\\0\\0'")
    text=text.replace("'nnue_match'","'global_policy_nnue_match'")
    old="sourceSha256={str(p.relative_to(ROOT)):match.sha(p) for p in sorted((ROOT/'cpp').rglob('*'))"
    assert text.count(old)==1
    text=text.replace(old,"candidateSourceRoot=str(CANDIDATE_SOURCE),\n        sourceSha256={str(p.relative_to(CANDIDATE_SOURCE)):match.sha(p) for p in sorted((CANDIDATE_SOURCE/'cpp').rglob('*'))")
    adapter=e.WORK/'compare-global-policy-candidate.py'
    if adapter.exists():assert adapter.read_text()==text
    else:adapter.write_text(text)
    spec=importlib.util.spec_from_file_location('global_policy_candidate_comparison',adapter)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,adapter


comparison,ADAPTER=comparison_module()
match=comparison.match
BaseEngine=match.Engine


def worker(connection,library,config):
    handle=None
    try:
        lib,handle=e.load(Path(library),e.ROOT/config['model'])
        connection.send(dict(ready=True,maxNodes=config['maxNodes']))
        while True:
            command=connection.recv()
            if command['kind']=='quit':break
            if command['kind']=='clear':
                lib.ot_search_clear_cache();lib.ot_mcts_clear_cache();connection.send(dict(cleared=True));continue
            cells=(ctypes.c_int*64)(*command['board'])
            out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
            tick=time.perf_counter_ns();seed=command['seed']
            ok=lib.ot_nnue_mcts_play_budget(handle,cells,command['turn'],command['ms'],command.get('simulations',0),seed,12,32,64,1,0,config['maxNodes'],out,selection,visits,stats,None)
            if ok!=1:raise RuntimeError(lib.ot_nnue_error().decode())
            peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
            connection.send(dict(move=out[0],score=out[1],depth=out[2],elapsedMs=out[3],nodes=out[4],exact=bool(out[5]),ttHits=out[135],wallMs=(time.perf_counter_ns()-tick)/1e6,
                candidates=[dict(move=out[7+2*i],score=out[8+2*i],visits=visits[i],value=stats[8+i]) for i in range(out[6])],search='mcts',
                mcts=dict(simulations=int(stats[0]),evaluations=int(stats[1]),meanDepth=stats[2]/1000,reusedNodes=int(stats[3]),treeNodes=int(stats[4]),memoryLimited=bool(stats[7]),maxNodes=config['maxNodes'],peakRssBytes=peak),
                selection=dict(seed=seed,eligible=selection[0],temperatureMilli=selection[1],lossLimit=selection[2],bestScore=selection[3],scoreLoss=selection[4],sampled=bool(selection[5]))))
    except (EOFError,BrokenPipeError):pass
    except BaseException as error:connection.send(dict(error=repr(error)))
    finally:
        if handle:lib.ot_nnue_destroy(handle)
        connection.close()


class Engine(BaseEngine):
    def __init__(self,library,config):super().__init__(library,dict(config,maxNodes=4194304))


match.worker=worker;match.Engine=Engine


def snapshot():
    files={str(p.relative_to(e.ROOT)):p for p in (e.WORK/'source').rglob('*') if p.is_file()}
    for name in ['AGENTS.md','benchmarks/pattern-global-policy.hpp','scripts/build-pattern-global-policy-native.py','scripts/global_policy_native_experiment.py','scripts/validate-global-policy-native.py','scripts/run-global-policy-match-2s.py',
        'scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py','scripts/watch-match.sh','scripts/watch-matches.py','tests/global_policy_native.cpp',
        'models/v1.49-global-policy-native-protocol.json','models/v1.49-global-policy-2s-20-protocol.json','models/v1.49-global-policy-native-validation.json']:
        files[name]=e.ROOT/name
    files[str(ADAPTER.relative_to(e.ROOT))]=ADAPTER
    digests={name:e.sha(path) for name,path in sorted(files.items())}
    archive=SOURCE.with_suffix('.tar.gz')
    if SOURCE.exists():
        frozen=e.read(SOURCE);assert e.sha(archive)==frozen['sha256'] and frozen['files']==digests
    else:
        assert not archive.exists()
        with tarfile.open(archive,'w:gz') as out:
            for name,path in sorted(files.items()):out.add(path,arcname=name)
        e.write(SOURCE,dict(archive=str(archive.relative_to(e.ROOT)),sha256=e.sha(archive),files=digests))


def monitor():
    config=e.read(e.ROOT/'models/current-experiment.json');history=config.setdefault('archivedExperiments',[])
    for item in config.get('experiments',[]):
        if item.get('report')!=str(REPORT.relative_to(e.ROOT)) and not any(x.get('report')==item.get('report') for x in history):history.insert(0,item)
    item=dict(name='全局8×8策略 v1.49 vs v1.37 · 2秒/步 · 20局',focus='candidate',displayNames=dict(candidate='全局8×8策略 v1.49',baseline='v1.37 对称共享'),displayOrder=['candidate','baseline'],games=20,runtimeReport=str(RUNTIME.relative_to(e.ROOT)),report=str(REPORT.relative_to(e.ROOT)))
    config.update(version='v1.49 全局策略 vs v1.37 · 2s/20局',experiments=[item]);e.write(e.ROOT/'models/current-experiment.json',config)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    validation=e.read(e.ROOT/'models/v1.49-global-policy-native-validation.json');assert validation['complete'] and e.sha(e.MODEL)==validation['modelSha256']
    e.verify_preserved()
    for name,digest in validation['libraries'].items():assert e.sha(e.ROOT/name)==digest,name
    if args.smoke:
        for label,lib,model in [('candidate',e.LIB,e.MODEL),('baseline',e.REFERENCE,e.BASE)]:
            engine=Engine(lib,dict(id=label,model=str(model.relative_to(e.ROOT))))
            try:
                board=match.rules.initial();response=engine.request(kind='search',board=board,turn=1,ms=0,simulations=2048,seed=1832)
                match.audit_decision(board,1,dict(response,valueTarget='outcome'))
                assert response['mcts']['maxNodes']==4194304
            finally:engine.close()
        print(json.dumps(dict(smoke=True,workers=2,passed=True)));return
    assert not REPORT.exists(),'Completed match reports are immutable'
    if not MATCH_PROTOCOL.exists():
        assert not RUNTIME.exists() and not args.resume
        e.write(MATCH_PROTOCOL,dict(version='v1.49-global-policy-2s-20',status='candidate',defaultPromoted=False,userRequest='和基线2s 20局',
            candidate=str(e.MODEL.relative_to(e.ROOT)),candidateModelSha256=e.sha(e.MODEL),baseline='models/v1.37-default.json',baselineModelSha256=e.sha(e.BASE),
            trainingReportSha256=e.sha(e.TRAINING),nativeValidationSha256=e.sha(e.ROOT/'models/v1.49-global-policy-native-validation.json'),
            millisecondsPerMove=2000,games=20,seed=20261029,start='initial',pairedColors=True,sharedPairSeeds=True,useBook=False,
            mcts=dict(maxNodes=4194304,patternCacheEntries=0,exploration=1.5,finiteFpuReduction=.15,subtreeReuse=True,terminal='outcome +1/0/-1'),openingPolicy=match.POLICY,
            libraries=validation['libraries'],preservedSha256=e.read(e.PROTOCOL)['preservedSha256'],nativeProtocolSha256=e.sha(e.PROTOCOL),sourceArchiveManifest=str(SOURCE.relative_to(e.ROOT)),
            limits='20-game paired screening sample. No retraining, no automatic promotion, no Elo/significance claim. Training/build/validation finished before match; external system CPU activity not controlled.'))
    protocol=e.read(MATCH_PROTOCOL)
    for name,digest in protocol['preservedSha256'].items():assert e.sha(e.ROOT/name)==digest,name
    for name,digest in protocol['libraries'].items():assert e.sha(e.ROOT/name)==digest,name
    snapshot();monitor()
    original=match.rules.write_json
    def annotated(path,value):
        if Path(path).parent==e.WORK and 'configs' in value:
            value.update(protocolSha256=e.sha(MATCH_PROTOCOL),sourceArchiveManifest=str(SOURCE.relative_to(e.ROOT)),sourceArchiveManifestSha256=e.sha(SOURCE),maxNodesByEngine=dict(candidate=4194304,baseline=4194304),patternCacheEntries=0)
        return original(path,value)
    match.rules.write_json=annotated
    comparison.compare(argparse.Namespace(model=e.MODEL,baseline_manifest=e.ROOT/'models/v1.37-default.json',baseline_library=e.REFERENCE,label='v1.49 global policy',library=e.LIB,output=RUNTIME,games=20,milliseconds=2000,seed=20261029,resume=args.resume))
    e.verify_preserved();result=e.read(RUNTIME);assert result['complete'] and result['games']==20
    e.write(REPORT,result);print(json.dumps(dict(complete=True,results=result['results']),ensure_ascii=False))


if __name__=='__main__':main()

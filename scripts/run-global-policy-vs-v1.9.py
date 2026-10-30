#!/usr/bin/env python3
"""Latest complete v1.49 vs exact-source frozen v1.9: 200ms, 50 paired games."""
import argparse,ctypes,importlib.util,json,resource,sys,tarfile,time
from pathlib import Path
import global_policy_native_experiment as e
WORK=e.ROOT/'.build/training/global-policy-vs-v1.9-200ms-50'
BASELINE=e.ROOT/'models/v1.9-default.json'
BASEMODEL=e.ROOT/'models/nnue256-outcome-v1.9.otnnue'
BASELIB=e.ROOT/f'.build/native-v1.9-reference/libothello.{e.SUFFIX}'
BUILD_RECORD=BASELIB.parent/'frozen-build-provenance.json'
RUNTIME=WORK/'match.json'
REPORT=e.ROOT/'models/v1.49-global-policy-vs-v1.9-200ms-50.json'
PROTOCOL=e.ROOT/'models/v1.49-v1.9-200ms-50-protocol.json'
SOURCE=e.ROOT/'models/archive/v1.49-v1.9-200ms-50-sources.json'
PREFLIGHT=e.ROOT/'models/v1.49-v1.9-200ms-50-preflight.json'
spec=importlib.util.spec_from_file_location('global_policy_previous_match_helpers',e.ROOT/'scripts/run-global-policy-match-2s.py')
helpers=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helpers;spec.loader.exec_module(helpers)
comparison=helpers.comparison;match=comparison.match
CAPS=dict(baseline=2097152,candidate=4194304)


def worker(connection,library,config):
    handle=None
    try:
        lib=ctypes.CDLL(library)
        lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_error.restype=ctypes.c_char_p
        legacy=config['id']=='baseline'
        if legacy:
            lib.ot_nnue_mcts_play.argtypes=[ctypes.c_void_p,e.ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*3+[e.ptr,e.ptr,e.ptr,ctypes.POINTER(ctypes.c_double)]
        else:
            lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,e.ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[e.ptr,e.ptr,e.ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        handle=lib.ot_nnue_load(str(e.ROOT/config['model']).encode())
        if not handle:raise RuntimeError(lib.ot_nnue_error().decode())
        connection.send(dict(ready=True,maxNodes=config['maxNodes'],legacyCAbi=legacy))
        while True:
            command=connection.recv()
            if command['kind']=='quit':break
            if command['kind']=='clear':
                lib.ot_search_clear_cache();lib.ot_mcts_clear_cache();connection.send(dict(cleared=True));continue
            cells=(ctypes.c_int*64)(*command['board']);out,sel,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
            tick=time.perf_counter_ns();seed=command['seed'];simulations=command.get('simulations',0)
            if legacy:ok=lib.ot_nnue_mcts_play(handle,cells,command['turn'],command['ms'],simulations,seed,12,32,64,out,sel,visits,stats)
            else:ok=lib.ot_nnue_mcts_play_budget(handle,cells,command['turn'],command['ms'],simulations,seed,12,32,64,1,0,config['maxNodes'],out,sel,visits,stats,None)
            if ok!=1:raise RuntimeError(lib.ot_nnue_error().decode())
            peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
            connection.send(dict(move=out[0],score=out[1],depth=out[2],elapsedMs=out[3],nodes=out[4],exact=bool(out[5]),ttHits=out[135],wallMs=(time.perf_counter_ns()-tick)/1e6,
                candidates=[dict(move=out[7+2*i],score=out[8+2*i],visits=visits[i],value=stats[8+i]) for i in range(out[6])],search='mcts',
                mcts=dict(simulations=int(stats[0]),evaluations=int(stats[1]),meanDepth=stats[2]/1000,reusedNodes=int(stats[3]),treeNodes=int(stats[4]),memoryLimited=bool(stats[7]),maxNodes=config['maxNodes'],peakRssBytes=peak),
                selection=dict(seed=seed,eligible=sel[0],temperatureMilli=sel[1],lossLimit=sel[2],bestScore=sel[3],scoreLoss=sel[4],sampled=bool(sel[5]))))
    except (EOFError,BrokenPipeError):pass
    except BaseException as error:connection.send(dict(error=repr(error)))
    finally:
        if handle:lib.ot_nnue_destroy(handle)
        connection.close()


class Engine(helpers.BaseEngine):
    def __init__(self,library,config):super().__init__(library,dict(config,maxNodes=CAPS[config['id']]))


match.worker=worker;match.Engine=Engine


def preserved():
    return {**e.read(e.PROTOCOL)['preservedSha256'],str(BASELINE.relative_to(e.ROOT)):e.sha(BASELINE),str(BASEMODEL.relative_to(e.ROOT)):e.sha(BASEMODEL),
        str(e.MODEL.relative_to(e.ROOT)):e.sha(e.MODEL),str(BASELIB.relative_to(e.ROOT)):e.sha(BASELIB),str(e.LIB.relative_to(e.ROOT)):e.sha(e.LIB)}


def verify(record):
    for name,digest in record['preservedSha256'].items():assert e.sha(e.ROOT/name)==digest,name


def smoke():
    baseline=e.read(BASELINE);assert baseline['status']=='frozen' and baseline['userApproved'] and e.sha(BASEMODEL)==baseline['modelSha256']
    build=e.read(BUILD_RECORD);assert build['complete'] and e.sha(BASELIB)==build['librarySha256'] and build['sourceFilesUnmodified']
    source=e.read(e.ROOT/build['sourceManifest']);assert e.sha(e.ROOT/source['archive'])==source['sha256']
    assert e.sha(e.MODEL)==e.read(e.ROOT/'models/v1.49-global-policy-native-validation.json')['modelSha256']
    for label,lib,model in [('baseline',BASELIB,BASEMODEL),('candidate',e.LIB,e.MODEL)]:
        engine=Engine(lib,dict(id=label,model=str(model.relative_to(e.ROOT))))
        try:
            board=match.rules.initial();result=engine.request(kind='search',board=board,turn=1,ms=0,simulations=2048,seed=1832)
            match.audit_decision(board,1,dict(result,valueTarget='outcome'));assert result['mcts']['maxNodes']==CAPS[label]
        finally:engine.close()
    record=dict(complete=True,v1_9ModelSha256=e.sha(BASEMODEL),v1_9ReferenceBuild=build,modelsLoaded=True,twoWorkersSmokeAudited=True,maxNodesByEngine=CAPS,
        candidateModelSha256=e.sha(e.MODEL),candidateNativeValidationSha256=e.sha(e.ROOT/'models/v1.49-global-policy-native-validation.json'),preservedSha256=preserved())
    if PREFLIGHT.exists():assert e.read(PREFLIGHT)==record
    else:e.write(PREFLIGHT,record)
    print(json.dumps(dict(smoke=True,passed=True,baseline='frozen v1.9',candidate='complete v1.49')))


def snapshot():
    names=['AGENTS.md','scripts/run-global-policy-vs-v1.9.py','scripts/build-frozen-v1.9-reference.py','scripts/run-global-policy-match-2s.py','scripts/global_policy_native_experiment.py',
        'scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py','scripts/watch-match.sh','scripts/watch-matches.py',
        'models/v1.9-default.json','models/v1.49-v1.9-200ms-50-protocol.json','models/v1.49-v1.9-200ms-50-preflight.json']
    files={name:e.sha(e.ROOT/name) for name in names};files[str(helpers.ADAPTER.relative_to(e.ROOT))]=e.sha(helpers.ADAPTER)
    archive=SOURCE.with_suffix('.tar.gz')
    if SOURCE.exists():assert e.read(SOURCE)['files']==files and e.sha(archive)==e.read(SOURCE)['sha256']
    else:
        with tarfile.open(archive,'w:gz') as out:
            for name in files:out.add(e.ROOT/name,arcname=name)
        dependencies=['models/archive/v1.9-default-sources.json','models/archive/v1.49-global-policy-native-sources.json']
        e.write(SOURCE,dict(archive=str(archive.relative_to(e.ROOT)),sha256=e.sha(archive),files=files,engineSourceManifests={p:e.sha(e.ROOT/p) for p in dependencies}))


def monitor():
    config=e.read(e.ROOT/'models/current-experiment.json');history=config.setdefault('archivedExperiments',[])
    for old in config.get('experiments',[]):
        if old.get('report')!=str(REPORT.relative_to(e.ROOT)) and not any(x.get('report')==old.get('report') for x in history):history.insert(0,old)
    item=dict(name='最新v1.49 vs 冻结v1.9 · 200ms/步 · 50局',focus='candidate',displayNames=dict(candidate='全局策略 v1.49',baseline='冻结 v1.9'),displayOrder=['candidate','baseline'],games=50,runtimeReport=str(RUNTIME.relative_to(e.ROOT)),report=str(REPORT.relative_to(e.ROOT)))
    config.update(version='v1.49 vs v1.9 · 200ms/50局',experiments=[item]);e.write(e.ROOT/'models/current-experiment.json',config)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    if args.smoke:smoke();return
    assert not REPORT.exists(),'Completed reports are immutable'
    preflight=e.read(PREFLIGHT);assert preflight['complete'];verify(preflight)
    if not PROTOCOL.exists():
        assert not RUNTIME.exists() and not args.resume
        e.write(PROTOCOL,dict(version='v1.49-vs-frozen-v1.9-200ms-50',status='candidate',defaultPromoted=False,userRequest='v1.9还在吗，在的话，最新模型和他打50局200ms',
            candidate=str(e.MODEL.relative_to(e.ROOT)),candidateModelSha256=e.sha(e.MODEL),baseline=str(BASELINE.relative_to(e.ROOT)),baselineModelSha256=e.sha(BASEMODEL),
            millisecondsPerMove=200,games=50,seed=20261029,start='initial',pairedColors=True,sharedPairSeeds=True,useBook=False,openingPolicy=match.POLICY,
            maxNodesByEngine=CAPS,exploration=1.5,subtreeReuse=True,terminal='outcome +1/0/-1',
            searchDetails=dict(candidate='Current Pattern learned-prior PUCT with finite FPU(parent mean-0.15);4M cap',baseline='Exact frozen v1.9 NNUE learned-prior search; sample each unvisited child once in learned-prior order;2M cap'),
            baselineSourceManifest='models/archive/v1.9-default-sources.json',baselineBuildProvenanceSha256=e.sha(BUILD_RECORD),baselineOriginalRecordedLibrarySha256=e.read(BASELINE)['nativeLibrarySha256'],
            libraries={str(p.relative_to(e.ROOT)):e.sha(p) for p in [BASELIB,e.LIB]},preflightSha256=e.sha(PREFLIGHT),preservedSha256=preflight['preservedSha256'],
            sourceArchiveManifest=str(SOURCE.relative_to(e.ROOT)),limits='Historical-version comparison: each model uses its own preserved search implementation/settings; equal200ms wall budget,25 paired-color seeds. No retraining, no opening book, no default promotion. Rebuilt v1.9 library has exact archived sources, not necessarily original binary hash.'))
    protocol=e.read(PROTOCOL);verify(protocol);snapshot();monitor()
    original=match.rules.write_json
    def annotated(path,value):
        if Path(path).parent==WORK and 'configs' in value:
            value.update(protocolSha256=e.sha(PROTOCOL),sourceArchiveManifest=str(SOURCE.relative_to(e.ROOT)),sourceArchiveManifestSha256=e.sha(SOURCE),maxNodesByEngine=CAPS,patternCacheEntriesByEngine=dict(candidate=0,baseline=None))
        return original(path,value)
    match.rules.write_json=annotated
    comparison.compare(argparse.Namespace(model=e.MODEL,baseline_manifest=BASELINE,baseline_library=BASELIB,label='v1.49 global policy',library=e.LIB,output=RUNTIME,games=50,milliseconds=200,seed=20261029,resume=args.resume))
    verify(protocol);result=e.read(RUNTIME);assert result['complete'] and result['games']==50;e.write(REPORT,result)
    print(json.dumps(dict(complete=True,results=result['results']),ensure_ascii=False))


if __name__=='__main__':main()

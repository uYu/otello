#!/usr/bin/env python3
"""Run every preregistered independent candidate vs frozen v1.36, serially."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import platform
import resource
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'.build/training/pattern-structure-v1.37'
SUFFIX = 'dylib' if sys.platform == 'darwin' else 'so'
LIB = ROOT/f'.build/native-pattern-structure-v1.37/libothello.{SUFFIX}'
REFERENCE = ROOT/f'.build/native-v1.36-reference/libothello.{SUFFIX}'
PROTOCOL = ROOT/'models/v1.37-pattern-structure-protocol.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p,v: Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
spec = importlib.util.spec_from_file_location('structure_comparison',ROOT/'scripts/compare-nnue-candidate.py')
comparison = importlib.util.module_from_spec(spec); sys.modules[spec.name] = comparison; spec.loader.exec_module(comparison)
match = comparison.match
BASE_ENGINE = match.Engine
BASE_WRITE = match.rules.write_json
ptr = ctypes.POINTER(ctypes.c_int)


def peak_rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)


def worker(connection,library,config):
    handle = None
    try:
        lib = ctypes.CDLL(library)
        lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_error.restype=ctypes.c_char_p
        lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        handle=lib.ot_nnue_load(str(ROOT/config['model']).encode())
        if not handle:raise RuntimeError(lib.ot_nnue_error().decode())
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
            connection.send(dict(move=out[0],score=out[1],depth=out[2],elapsedMs=out[3],nodes=out[4],exact=bool(out[5]),ttHits=out[135],wallMs=(time.perf_counter_ns()-tick)/1e6,
                candidates=[dict(move=out[7+2*i],score=out[8+2*i],visits=visits[i],value=stats[8+i]) for i in range(out[6])],search='mcts',
                mcts=dict(simulations=int(stats[0]),evaluations=int(stats[1]),meanDepth=stats[2]/1000,reusedNodes=int(stats[3]),treeNodes=int(stats[4]),memoryLimited=bool(stats[7]),maxNodes=config['maxNodes'],peakRssBytes=peak_rss()),
                selection=dict(seed=seed,eligible=selection[0],temperatureMilli=selection[1],lossLimit=selection[2],bestScore=selection[3],scoreLoss=selection[4],sampled=bool(selection[5]))))
    except (EOFError,BrokenPipeError):pass
    except BaseException as error:connection.send(dict(error=repr(error)))
    finally:
        if handle:lib.ot_nnue_destroy(handle)
        connection.close()


class Engine(BASE_ENGINE):
    def __init__(self,library,config):
        super().__init__(library,dict(config,maxNodes=4194304))


def monitor():
    protocol=read(PROTOCOL);items=[]
    for ms in protocol['matches']['budgetsMs']:
        for variant in protocol['matches']['variants']:
            label={'symmetry':'对称共享','pool10':'十组池化'}[variant]
            items.append(dict(name=f'{label} vs v1.36 · {ms}ms/步 · 20局',displayNames=dict(candidate=label,baseline='v1.36'),displayOrder=['candidate','baseline'],games=20,
                runtimeReport=f'.build/training/pattern-structure-v1.37/{variant}-{ms}ms-20.json',report=f'models/v1.37-{variant}-vs-v1.36-{ms}ms-20.json'))
    config=read(ROOT/'models/current-experiment.json')
    config.update(version='v1.37 Pattern 独立结构对战',experiments=items)
    write(ROOT/'models/current-experiment.json',config)


def snapshot():
    archive=ROOT/'models/archive/v1.37-pattern-structure-sources.tar.gz'
    manifest=archive.with_name('v1.37-pattern-structure-sources.json')
    paths=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/p for p in [
        'AGENTS.md','CMakeLists.txt','tests/native.cpp','tests/pattern_structure_test.py','scripts/pattern_structure.py',
        'scripts/train-pattern-structure.py','scripts/pattern_model.py','scripts/train-pattern-outcome.py','scripts/train-pattern.py',
        'scripts/benchmark-pattern-structure.py','scripts/run-pattern-structure.py','scripts/finalize-pattern-structure.py',
        'scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py',
        'scripts/watch-match.sh','scripts/watch-matches.py','models/v1.37-pattern-structure-protocol.json']]))
    if manifest.exists():
        frozen=read(manifest)
        assert sha(archive)==frozen['sha256']
        assert {str(p.relative_to(ROOT)):sha(p) for p in paths}==frozen['files'],'Source changed after match freeze'
        return
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as out:
        for p in paths:out.add(p,arcname=str(p.relative_to(ROOT)))
    write(manifest,dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in paths},libraries={str(p.relative_to(ROOT)):sha(p) for p in [LIB,REFERENCE]}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    protocol=read(PROTOCOL)
    for p,digest in protocol['preservedSha256'].items():assert sha(ROOT/p)==digest,p
    assert sha(REFERENCE)==protocol['baselineLibrarySha256']
    assert read(ROOT/'models/v1.37-pattern-structure.json')['variants']
    if args.smoke:
        match.worker=worker;match.Engine=Engine
        for variant,library,model in [('baseline',REFERENCE,'models/pattern32-outcome-v1.25.otpat')]+[(v,LIB,f'models/pattern32-{v}-v1.37.otpat') for v in ['symmetry','pool10']]:
            engine=Engine(library,dict(id=variant,model=model))
            try:
                board,turn=match.rules.initial(),1
                result=engine.request(kind='search',board=board,turn=turn,ms=0,simulations=1024,seed=1832)
                match.audit_decision(board,turn,dict(result,valueTarget='outcome'))
                assert result['mcts']['maxNodes']==4194304
            finally:engine.close()
        print(json.dumps(dict(smoke=True,workers=3,passed=True)),flush=True);return
    assert read(ROOT/'models/v1.37-pattern-structure-benchmark.json')['complete']
    snapshot();monitor();match.worker=worker;match.Engine=Engine
    def annotated(path,value):
        if Path(path).parent==WORK and 'configs' in value:
            value.update(protocolSha256=sha(PROTOCOL),maxNodesByEngine=dict(baseline=4194304,candidate=4194304),incrementalCacheEntries=0,sourceArchiveManifest='models/archive/v1.37-pattern-structure-sources.json')
        return BASE_WRITE(path,value)
    match.rules.write_json=annotated
    for ms in protocol['matches']['budgetsMs']:
        for variant in protocol['matches']['variants']:
            path=WORK/f'{variant}-{ms}ms-20.json'
            if path.exists() and read(path)['complete']:
                assert args.resume,'Refusing to replace completed match'
                continue
            comparison.compare(argparse.Namespace(model=ROOT/f'models/pattern32-{variant}-v1.37.otpat',baseline_manifest=ROOT/'models/v1.36-default.json',baseline_library=REFERENCE,
                label=f'v1.37 {variant}',library=LIB,output=path,games=20,milliseconds=ms,seed=protocol['matches']['seed'],resume=path.exists() and args.resume))
            final=ROOT/f'models/v1.37-{variant}-vs-v1.36-{ms}ms-20.json'
            assert not final.exists();write(final,read(path))
    for p,digest in protocol['preservedSha256'].items():assert sha(ROOT/p)==digest,p
    print(json.dumps(dict(complete=True,stage='All four independent matches complete')),flush=True)


if __name__=='__main__':main()

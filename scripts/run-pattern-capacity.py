#!/usr/bin/env python3
"""Pilot larger tree budgets on capped trajectories, then run 2s/20 games."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import resource
import tarfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-capacity-v1.36'
SUFFIX='dylib' if platform.system()=='Darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-capacity-v1.36/libothello.{SUFFIX}'
OLD=ROOT/f'.build/native-pattern-suite-v1.35/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.25-reference/libothello.{SUFFIX}'
MODEL=ROOT/'models/pattern32-outcome-v1.25.otpat'
PILOT=ROOT/'models/v1.36-pattern-capacity-pilot.json'
EXPERIMENT=ROOT/'models/v1.36-pattern-capacity.json'
RUNTIME=WORK/'match-2s-20.json'
REPORT='models/v1.36-capacity-vs-v1.25-2s-20.json'
CONFIG=ROOT/'models/v1.36-pattern-capacity-experiments.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('capacity_comparison',ROOT/'scripts/compare-nnue-candidate.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
match=comparison.match
BASE_ENGINE=match.Engine
BASE_WRITE=match.rules.write_json
SELECTED=None
ptr=ctypes.POINTER(ctypes.c_int)


def peak_rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=='Darwin' else 1024)


def worker(connection,library,config):
    handle=None
    try:
        lib=ctypes.CDLL(library);lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_error.restype=ctypes.c_char_p
        args=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*3
        lib.ot_nnue_mcts_play.argtypes=args+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double)]
        if hasattr(lib,'ot_nnue_mcts_play_budget'):
            lib.ot_nnue_mcts_play_budget.argtypes=args+[ctypes.c_int,ctypes.c_uint,ctypes.c_uint,ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        limit=config['maxNodes'];handle=lib.ot_nnue_load(str(ROOT/config['model']).encode());assert handle,lib.ot_nnue_error().decode()
        connection.send(dict(ready=True,maxNodes=limit))
        while True:
            command=connection.recv()
            if command['kind']=='quit':break
            if command['kind']=='clear':lib.ot_search_clear_cache();lib.ot_mcts_clear_cache();connection.send(dict(cleared=True));continue
            cells=(ctypes.c_int*64)(*command['board']);out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
            seed=command['seed'];tick=time.perf_counter_ns();simulations=command.get('simulations',0)
            if hasattr(lib,'ot_nnue_mcts_play_budget'):
                ok=lib.ot_nnue_mcts_play_budget(handle,cells,command['turn'],command['ms'],simulations,seed,12,32,64,1,0,limit,out,selection,visits,stats,None)
            else:
                assert limit==2097152
                ok=lib.ot_nnue_mcts_play(handle,cells,command['turn'],command['ms'],simulations,seed,12,32,64,out,selection,visits,stats)
            wall=(time.perf_counter_ns()-tick)/1e6
            if ok!=1:raise RuntimeError(lib.ot_nnue_error().decode())
            candidates=[dict(move=out[7+2*i],score=out[8+2*i],visits=visits[i],value=stats[8+i]) for i in range(out[6])]
            connection.send(dict(move=out[0],score=out[1],depth=out[2],elapsedMs=out[3],nodes=out[4],exact=bool(out[5]),ttHits=out[135],wallMs=wall,candidates=candidates,search='mcts',
                mcts=dict(simulations=int(stats[0]),evaluations=int(stats[1]),meanDepth=stats[2]/1000,reusedNodes=int(stats[3]),treeNodes=int(stats[4]),memoryLimited=bool(stats[7]),maxNodes=limit,peakRssBytes=peak_rss()),
                selection=dict(seed=seed,eligible=selection[0],temperatureMilli=selection[1],lossLimit=selection[2],bestScore=selection[3],scoreLoss=selection[4],sampled=bool(selection[5]))))
    except (EOFError,BrokenPipeError):pass
    except BaseException as error:connection.send(dict(error=repr(error)))
    finally:
        if handle:lib.ot_nnue_destroy(handle)
        connection.close()


class Engine(BASE_ENGINE):
    def __init__(self,library,config):
        config=dict(config,maxNodes=SELECTED if config['id']=='candidate' else 2097152)
        super().__init__(library,config)


def monitor(matching=False):
    label=f'策略集 {SELECTED//1048576}M 节点' if SELECTED else '策略集扩容预检'
    item=dict(name=f'{label} vs v1.25 · 2秒/步 · 20局',shortName=f'{label} vs v1.25 · 2秒/步 · 20局',displayNames=dict(candidate=label,baseline='v1.25'),displayOrder=['candidate','baseline'],games=20,runtimeReport=str((RUNTIME if matching else WORK/'progress.json').relative_to(ROOT)),report=REPORT)
    config=dict(version='v1.36 Pattern 节点扩容',experiments=[item]);write(CONFIG,config)
    current=read(ROOT/'models/current-experiment.json');history=current.get('archivedExperiments',[]);known={i.get('report') for i in history}
    for old in current['experiments']:
        if old.get('report')!=REPORT and old.get('report') not in known:history.insert(0,old);known.add(old.get('report'))
    current.update(version=config['version'],experiments=[item],archivedExperiments=history);write(ROOT/'models/current-experiment.json',current)


def spawn_engine(limit,library=LIB):
    context=multiprocessing.get_context('spawn');connection,child=context.Pipe()
    config=dict(id='candidate',model=str(MODEL),maxNodes=limit)
    process=context.Process(target=worker,args=(child,str(library),config));process.start();child.close()
    ready=connection.recv();assert ready.get('ready'),ready
    return connection,process


def request(engine,**command):
    engine[0].send(command);assert engine[0].poll(30),'Search worker timed out'
    result=engine[0].recv();assert 'error' not in result,result
    return result


def close(engine):
    engine[0].send(dict(kind='quit'));engine[1].join(5)
    if engine[1].is_alive():engine[1].terminate();engine[1].join()
    engine[0].close()


def pilot():
    assert not PILOT.exists() and not EXPERIMENT.exists() and not RUNTIME.exists()
    WORK.mkdir(parents=True,exist_ok=True);monitor();started=time.perf_counter()
    previous=read(ROOT/'models/v1.35-speed-suite-vs-v1.25-2s-20.json');sequences=[]
    games=sorted(previous['records'],key=lambda g:sum(d['mcts']['memoryLimited'] for d in g['decisions'] if d['engine']=='candidate'),reverse=True)[:2]
    for game in games:
        decisions=[d for d in game['decisions'] if d['engine']=='candidate'];first=next(i for i,d in enumerate(decisions) if d['mcts']['memoryLimited']);last=decisions[min(first+7,len(decisions)-1)]['ply']
        board,turn=match.rules.initial(),1;positions=[]
        by_ply={d['ply']:d for d in decisions}
        for ply,move in enumerate(game['moves']):
            if ply in by_ply and ply<=last:positions.append(dict(board=board.copy(),turn=turn,ply=ply,originallyCapped=by_ply[ply]['mcts']['memoryLimited']))
            board,turn=match.rules.play(board,turn,move)
        sequences.append(dict(game=game['number'],seed=game['seed'],positions=positions))
    limits=[2097152,4194304,8388608];memory_guard=min(4*1024**3,os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')//6)
    def progress(stage,done=0,total=1,summary='',complete=False):write(WORK/'progress.json',dict(experiment='pattern-capacity-pilot',stage=stage,completed=done,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    # Verify the new API and larger capacities preserve unbound fixed-simulation results.
    old=spawn_engine(2097152,OLD);new=spawn_engine(8388608)
    try:
        for sequence in sequences:
            for position in sequence['positions'][:3]:
                responses=[]
                for engine in [old,new]:
                    request(engine,kind='clear');r=request(engine,kind='search',board=position['board'],turn=position['turn'],ms=0,simulations=2048,seed=sequence['seed'])
                    responses.append({k:r[k] for k in ['move','score','depth','exact','candidates','selection']})
                assert responses[0]==responses[1]
    finally:close(old);close(new)
    results=[];total=len(limits)*sum(len(s['positions']) for s in sequences);done=0
    for limit in limits:
        engine=spawn_engine(limit);samples=[];aborted=False
        try:
            for sequence in sequences:
                request(engine,kind='clear')
                for position in sequence['positions']:
                    progress('连续局面复用树：节点上限/内存/2秒预算',done,total,f'{limit:,} 节点 · 上轮第 {sequence["game"]} 局 · 第 {position["ply"]} 步')
                    response=request(engine,kind='search',board=position['board'],turn=position['turn'],ms=2000,seed=sequence['seed']);done+=1
                    match.audit_decision(position['board'],position['turn'],dict(response,valueTarget='outcome'))
                    samples.append(dict(game=sequence['game'],ply=position['ply'],originallyCapped=position['originallyCapped'],**response))
                    if response['mcts']['peakRssBytes']>memory_guard:aborted=True;break
                if aborted:break
        finally:close(engine)
        not_proven=[s for s in samples if not s['exact']]
        results.append(dict(maxNodes=limit,samples=samples,memoryGuardExceeded=aborted,capped=sum(s['mcts']['memoryLimited'] for s in samples),nonProven=len(not_proven),fullBudget=sum(s['elapsedMs']>=1900 for s in not_proven),evaluations=sum(s['mcts']['evaluations'] for s in samples),peakRssBytes=max(s['mcts']['peakRssBytes'] for s in samples)))
    safe=[r for r in results[1:] if not r['memoryGuardExceeded']]
    assert safe,'No tested larger budget fits memory guard'
    good=[r for r in safe if r['capped']<=.05*len(r['samples']) and r['fullBudget']>=.95*r['nonProven']]
    selected=min(good,key=lambda r:r['maxNodes']) if good else min(safe,key=lambda r:(r['capped']/len(r['samples']),-r['evaluations']))
    report=dict(version='v1.36-pattern-capacity',status='candidate',defaultPromoted=False,results=results,selectedMaxNodes=selected['maxNodes'],selectionRule='Smallest safe larger limit with <=5% capped searches and >=95% non-proven searches using >=1900ms; otherwise safe limit minimizing capped fraction, then maximizing evaluations. RSS guard=min(4GiB, physical RAM/6).',memoryGuardBytes=memory_guard,sequences=sequences,previousReportSha256=sha(ROOT/'models/v1.35-speed-suite-vs-v1.25-2s-20.json'),librarySha256=sha(LIB),modelSha256=sha(MODEL),fixedSimulationParity=True,scriptSha256=sha(Path(__file__)),seconds=time.perf_counter()-started)
    write(PILOT,report)
    config=dict(version='v1.36-pattern-capacity',status='candidate',defaultPromoted=False,maxNodes=selected['maxNodes'],baselineMaxNodes=2097152,pilotReport=str(PILOT.relative_to(ROOT)),pilotSha256=sha(PILOT),library=str(LIB.relative_to(ROOT)),librarySha256=sha(LIB),model=str(MODEL.relative_to(ROOT)),modelSha256=sha(MODEL),preservedSha256={p:sha(ROOT/p) for p in ['models/default-model.json','models/v1.25-default.json','src/game/wasm/othello.mjs',str(MODEL.relative_to(ROOT))]},baselineLibrarySha256=sha(REFERENCE),previousSuite='models/v1.35-pattern-speed-suite.json',match=dict(games=20,millisecondsPerMove=2000,seed=20261029,useBook=False,runtimeReport=str(RUNTIME.relative_to(ROOT)),finalReport=REPORT))
    write(EXPERIMENT,config)
    archive=ROOT/'models/archive/v1.36-pattern-capacity-sources.tar.gz';assert not archive.exists()
    sources=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/p for p in ['AGENTS.md','CMakeLists.txt','data/README.md','scripts/run-pattern-capacity.py','scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py','scripts/watch-matches.py','scripts/watch-match.sh','scripts/test-native.sh','tests/native.cpp','benchmarks/pattern-hotpath.cpp','benchmarks/pattern-legal.cpp']]))
    with tarfile.open(archive,'w:gz') as out:
        for path in sources:out.add(path,arcname=str(path.relative_to(ROOT)))
    write(archive.with_name('v1.36-pattern-capacity-sources.json'),dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    progress('预检完成，即将开始 2秒/步 20局',1,1,f'选择 {selected["maxNodes"]:,} 节点，预检触顶 {selected["capped"]}/{len(selected["samples"])}',True)
    return config


def main():
    global SELECTED
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=read(EXPERIMENT) if args.resume else pilot();SELECTED=config['maxNodes']
    for p,digest in config['preservedSha256'].items():assert sha(ROOT/p)==digest
    assert sha(LIB)==config['librarySha256']
    monitor(True);match.worker=worker;match.Engine=Engine
    def annotated_write(path,value):
        if Path(path).parent==WORK and 'configs' in value:
            value.update(maxNodesByEngine=dict(baseline=2097152,candidate=SELECTED),capacityPilotSha256=sha(PILOT),peakMemory='per-worker peak RSS bytes; macOS bytes, Linux ru_maxrss KiB converted')
        return BASE_WRITE(path,value)
    match.rules.write_json=annotated_write
    comparison.compare(argparse.Namespace(model=MODEL,baseline_manifest=ROOT/'models/v1.25-default.json',baseline_library=REFERENCE,label=f'v1.36 suite {SELECTED//1048576}M nodes',library=LIB,output=RUNTIME,games=20,milliseconds=2000,seed=20261029,resume=args.resume))


if __name__=='__main__':main()

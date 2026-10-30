#!/usr/bin/env python3
"""Benchmark bounded MCTS Pattern state caches with deterministic parity."""
import ctypes,hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-incremental-v1.31'
REPORT=ROOT/'models/v1.31-pattern-incremental-benchmark.json'
MODEL=ROOT/'models/pattern32-outcome-v1.25.otpat'
LIB=ROOT/('.build/native-pattern-incremental-v1.31/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
REFERENCE=ROOT/('.build/native-pattern-h64-v1.29/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
MODES=[0,64,256,1024,4096,16384]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
ptr=ctypes.POINTER(ctypes.c_int)


def load(path):
    lib=ctypes.CDLL(str(path));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_error.restype=ctypes.c_char_p
    args=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*3
    lib.ot_nnue_mcts_play.argtypes=args+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double)]
    if hasattr(lib,'ot_nnue_mcts_play_cached'):
        lib.ot_nnue_mcts_play_cached.argtypes=args+[ctypes.c_int,ctypes.c_uint,ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
    handle=lib.ot_nnue_load(str(MODEL).encode());assert handle,lib.ot_nnue_error().decode()
    return lib,handle


def search(lib,h,board,turn,entries=0,simulations=4096,ms=0):
    cells=(ctypes.c_int*64)(*map(int,board));out,selection,visits,stats,cache=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)(),(ctypes.c_double*8)()
    if hasattr(lib,'ot_nnue_mcts_play_cached'):
        ok=lib.ot_nnue_mcts_play_cached(h,cells,int(turn),ms,simulations,1832,12,32,64,1,entries,out,selection,visits,stats,cache)
    else:ok=lib.ot_nnue_mcts_play(h,cells,int(turn),ms,simulations,1832,12,32,64,out,selection,visits,stats)
    assert ok==1,lib.ot_nnue_error().decode()
    elapsed=out[3];out[3]=0;stats[5]=0
    return [list(out),list(selection),list(visits),list(stats)],list(cache),elapsed


def main():
    assert not REPORT.exists(),'Fresh output required'
    baseline=json.loads((ROOT/'models/v1.25-default.json').read_text());assert sha(MODEL)==baseline['modelSha256']
    active_sha=sha(ROOT/'models/default-model.json');baseline_sha=sha(ROOT/'models/v1.25-default.json')
    data=np.load(ROOT/'.build/training/v1/positions.npz');boards=data['validation_board'][:32];turns=data['validation_turn'][:32]
    spec=importlib.util.spec_from_file_location('incremental_rules',ROOT/'scripts/compare-resnet.py');rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
    started=time.perf_counter();lib,h=load(LIB);reference,rh=load(REFERENCE)
    def progress(stage,completed,total,summary=None,complete=False):
        value=dict(experiment='pattern-incremental-benchmark',complete=complete,stage=stage,completed=completed,total=total,seconds=time.perf_counter()-started)
        if summary:value['summary']=summary
        write(WORK/'progress.json',value)
    try:
        expected=[]
        for board,turn in zip(boards,turns):
            reference.ot_mcts_clear_cache();result,_,_=search(reference,rh,board,turn);expected.append(result)
        # Include actual reached-subtree reuse in addition to independent cold roots.
        warm_boards=[];warm_expected=[];board,turn=rules.initial(),1;reference.ot_mcts_clear_cache()
        for ply in range(16):
            result,_,_=search(reference,rh,board,turn);warm_boards.append((board,turn));warm_expected.append(result)
            board,turn=rules.play(board,turn,result[0][0])
        for entries in MODES:
            progress('固定模拟数逐项一致性',MODES.index(entries),len(MODES),f'缓存 {entries} 项')
            for (board,turn),wanted in zip(zip(boards,turns),expected):
                lib.ot_mcts_clear_cache();got,_,_=search(lib,h,board,turn,entries);assert got==wanted,(entries,'cold')
            lib.ot_mcts_clear_cache()
            for (board,turn),wanted in zip(warm_boards,warm_expected):
                got,_,_=search(lib,h,board,turn,entries);assert got==wanted,(entries,'warm')
        cold={n:[] for n in MODES};warm={n:[] for n in MODES};rng=np.random.default_rng(20261029)
        def measure(entries,fixtures,wanted,reuse):
            # Exclude first allocation and page faults, retain actual model snapshot/clear cost.
            search(lib,h,fixtures[0][0],fixtures[0][1],entries,simulations=1);lib.ot_mcts_clear_cache()
            counters=np.zeros(8,np.float64);nodes=evals=0;max_bytes=0;tick=time.perf_counter_ns()
            for (b,t),target in zip(fixtures,wanted):
                if not reuse:lib.ot_mcts_clear_cache()
                result,cache,_=search(lib,h,b,t,entries);assert result==target
                nodes+=result[3][0];evals+=result[3][1]
                counters+=cache;max_bytes=max(max_bytes,cache[4])
            return dict(milliseconds=(time.perf_counter_ns()-tick)/1e6,simulations=int(nodes),evaluations=int(evals),refreshes=int(counters[0]),updates=int(counters[1]),hits=int(counters[2]),changedPatterns=int(counters[3]),cacheBytes=int(max_bytes),probes=int(counters[6]),ancestorDistance=int(counters[7]))
        rounds=7
        for iteration in range(rounds):
            for entries in rng.permutation(MODES).tolist():
                progress('冷根与复用子树交错计时',iteration*len(MODES)+MODES.index(entries)+1,rounds*len(MODES),f'第 {iteration+1} 轮 · 缓存 {entries} 项')
                cold[entries].append(measure(entries,list(zip(boards,turns)),expected,False))
                warm[entries].append(measure(entries,warm_boards,warm_expected,True))
        def summarize(samples):
            ms=statistics.median(s['milliseconds'] for s in samples);updates=sum(s['updates'] for s in samples);evals=sum(s['evaluations'] for s in samples)
            return dict(medianMilliseconds=ms,simulationsPerSecond=samples[0]['simulations']*1000/ms,evaluationsPerSecond=samples[0]['evaluations']*1000/ms,cacheBytes=max(s['cacheBytes'] for s in samples),incrementalFraction=updates/evals if evals else 0,meanChangedPatterns=sum(s['changedPatterns'] for s in samples)/updates if updates else 0,meanAncestorDistance=sum(s['ancestorDistance'] for s in samples)/updates if updates else 0,samples=samples)
        rows=[dict(entries=n,cold=summarize(cold[n]),warm=summarize(warm[n])) for n in MODES]
        for row in rows:
            row['cold']['throughputRatioVsFused']=rows[0]['cold']['medianMilliseconds']/row['cold']['medianMilliseconds']
            row['warm']['throughputRatioVsFused']=rows[0]['warm']['medianMilliseconds']/row['warm']['medianMilliseconds']
        best=min(rows[1:],key=lambda row:row['cold']['medianMilliseconds']+row['warm']['medianMilliseconds'])['entries']
        # Same wall-clock budget checks the practical consequence of update overhead.
        timed={0:[],best:[]};timed_boards=list(zip(boards[:4],turns[:4]))
        for iteration in range(5):
            for entries in rng.permutation([0,best]).tolist():
                progress('每手 100ms 实际预算复核',iteration*2+(entries!=0)+1,10,f'缓存 {entries} 项')
                for board,turn in timed_boards:
                    lib.ot_mcts_clear_cache();result,cache,elapsed=search(lib,h,board,turn,entries,simulations=0,ms=100)
                    assert result[0][0] in rules.legal(board,int(turn))
                    timed[entries].append(dict(simulations=int(result[3][0]),evaluations=int(result[3][1]),elapsedMs=elapsed,exact=bool(result[0][5]),memoryLimited=bool(result[3][7])))
        timed_rows=[dict(entries=n,totalSimulations=sum(s['simulations'] for s in rows),totalEvaluations=sum(s['evaluations'] for s in rows),samples=rows) for n,rows in timed.items()]
        assert sha(ROOT/'models/default-model.json')==active_sha and sha(ROOT/'models/v1.25-default.json')==baseline_sha
        report=dict(version='v1.31-pattern-incremental',status='candidate',defaultPromoted=False,model=str(MODEL.relative_to(ROOT)),modelSha256=sha(MODEL),baselineManifest='models/v1.25-default.json',baselineManifestSha256=baseline_sha,libraries={str(p.relative_to(ROOT)):sha(p) for p in [LIB,REFERENCE]},scriptSha256=sha(Path(__file__)),rounds=rounds,simulationsPerPosition=4096,coldPositions=len(boards),warmPositions=len(warm_boards),allFixedSimulationOutputsExactlyEqual=True,cacheEntriesLimit=16384,results=rows,bestCachedEntries=best,timed100ms=timed_rows,seconds=time.perf_counter()-started,limits='Native experiment only; no training or strength matches. Same model, priors, FPU and proof rules. All fixed-budget outputs exclude elapsed fields only. Warm-cache allocation excluded from timing; per-root model snapshot and ancestor cache operations included.')
        write(REPORT,report)
        best_row=next(r for r in rows if r['entries']==best)
        progress('已完成',1,1,f"最佳增量缓存 {best} 项；冷根吞吐 {best_row['cold']['throughputRatioVsFused']:.3f}×，复用子树 {best_row['warm']['throughputRatioVsFused']:.3f}×",True)
        print(json.dumps(dict(best=best,results=[dict(entries=r['entries'],cold=r['cold']['throughputRatioVsFused'],warm=r['warm']['throughputRatioVsFused'],bytes=r['cold']['cacheBytes'],changed=r['cold']['meanChangedPatterns'],distance=r['cold']['meanAncestorDistance']) for r in rows],timed=[dict(entries=r['entries'],simulations=r['totalSimulations']) for r in timed_rows]),indent=2))
    finally:lib.ot_nnue_destroy(h);reference.ot_nnue_destroy(rh)


if __name__=='__main__':main()

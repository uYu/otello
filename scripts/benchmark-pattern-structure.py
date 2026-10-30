#!/usr/bin/env python3
"""Parity, symmetry and interleaved latency/search measurements for v1.37."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
from pattern_model import legal_mask

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-structure-v1.37'
SUFFIX='dylib' if sys.platform=='darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-structure-v1.37/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.36-reference/libothello.{SUFFIX}'
PORTABLE=ROOT/f'.build/native-pattern-structure-portable-v1.37/libothello.{SUFFIX}'
REPORT=ROOT/'models/v1.37-pattern-structure-benchmark.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('structure_integer_helpers',ROOT/'scripts/benchmark-pattern-int8.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
ptr=ctypes.POINTER(ctypes.c_int)


def search(lib,handle,board,turn,ms=0,simulations=1024,clear=True):
    if clear:lib.ot_mcts_clear_cache()
    cells=(ctypes.c_int*64)(*map(int,board))
    out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
    tick=time.perf_counter_ns()
    assert lib.ot_nnue_mcts_play_budget(handle,cells,int(turn),ms,simulations,1832,12,32,64,1,0,4194304,out,selection,visits,stats,None)==1,lib.ot_nnue_error().decode()
    wall=(time.perf_counter_ns()-tick)/1e9
    metadata=dict(simulations=int(stats[0]),evaluations=int(stats[1]),exact=bool(out[5]),memoryLimited=bool(stats[7]),elapsedMs=out[3],wallSeconds=wall)
    out[3]=0;stats[5]=0
    return [list(out),list(selection),list(visits),list(stats)],metadata


def load(library,model):
    lib,handle=h.load(library,model)
    lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
    return lib,handle


def symmetry(lib,handle,boards,turns):
    grid=np.arange(64).reshape(8,8);base=h.native(lib,handle,boards,turns);values=[];policies=[]
    for mirror in [False,True]:
        for rotation in range(4):
            perm=np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64)
            prediction=h.native(lib,handle,boards[:,perm],turns)
            values.append(prediction[:,0]);policies.append(prediction[:,4:][:,np.argsort(perm)])
    span=np.ptp(values,axis=0);delta=np.abs(np.asarray(policies)-base[None,:,4:])
    return dict(positions=len(boards),meanValueSpan=float(span.mean()),maximumValueSpan=float(span.max()),maximumAlignedPolicyLogitDifference=float(delta.max()))


def main():
    assert not REPORT.exists(),'Choose a new experiment; benchmark reports are immutable'
    protocol=read(ROOT/'models/v1.37-pattern-structure-protocol.json')
    for p,digest in protocol['preservedSha256'].items():assert sha(ROOT/p)==digest
    assert sha(REFERENCE)==protocol['baselineLibrarySha256']
    started=time.perf_counter();data=dict(np.load(ROOT/'.build/training/v1/positions.npz'))
    definitions=[('frozen v1.36',REFERENCE,'models/pattern32-outcome-v1.25.otpat'),('v1.36 model / new library',LIB,'models/pattern32-outcome-v1.25.otpat')]+[(v,LIB,f'models/pattern32-{v}-v1.37.otpat') for v in ['control','symmetry','pool10']]
    handles=[];loaded=[];latencies=[[] for _ in definitions];timed=[[] for _ in definitions]
    def progress(stage,completed=0,total=1,summary='',complete=False):
        write(WORK/'benchmark-progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=completed,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    config=read(ROOT/'models/current-experiment.json');config.update(experiments=[dict(name='独立结构：一致性与相同局面吞吐',games=0,runtimeReport=str((WORK/'benchmark-progress.json').relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)))])
    write(ROOT/'models/current-experiment.json',config)
    try:
        for name,library,model in definitions:
            lib,handle=load(library,ROOT/model);handles.append((lib,handle));loaded.append((lib,handle))
        progress('原模型输出与固定模拟回归检查')
        # Check all original outputs, not just chosen moves.
        for split in ['validation','test']:
            b,t=data[split+'_board'],data[split+'_turn']
            assert np.array_equal(h.native(*loaded[0],b,t),h.native(*loaded[1],b,t)),split
        for board,turn in zip(data['validation_board'][:32],data['validation_turn'][:32]):
            assert search(*loaded[0],board,turn)[0]==search(*loaded[1],board,turn)[0]
        # Also replay an actually reached subtree with frozen deterministic simulations.
        rule_spec=importlib.util.spec_from_file_location('structure_benchmark_rules',ROOT/'scripts/compare-resnet.py')
        rules=importlib.util.module_from_spec(rule_spec);rule_spec.loader.exec_module(rules)
        warm=[]
        for lib,handle in loaded[:2]:
            board,turn=rules.initial(),1;lib.ot_mcts_clear_cache();trace=[]
            for ply in range(16):
                result,_=search(lib,handle,board,turn,clear=False);trace.append(result);board,turn=rules.play(board,turn,result[0][0])
            warm.append(trace)
        assert warm[0]==warm[1]
        portable_checks=[]
        for variant in ['control','symmetry','pool10']:
            portable,handle=load(PORTABLE,ROOT/f'models/pattern32-{variant}-v1.37.otpat');handles.append((portable,handle))
            selected=loaded[2+['control','symmetry','pool10'].index(variant)]
            b,t=data['validation_board'],data['validation_turn']
            assert np.array_equal(h.native(portable,handle,b,t),h.native(*selected,b,t)),variant
            for board,turn in zip(b[:16],t[:16]):assert search(portable,handle,board,turn)[0]==search(*selected,board,turn)[0]
            portable_checks.append(dict(variant=variant,positions=len(b),fixedSearchRoots=16,allOutputsExactlyEqual=True))
        symmetry_rows=[]
        for (name,_,_),pair in zip(definitions,loaded):
            result=symmetry(*pair,data['validation_board'],data['validation_turn']);symmetry_rows.append(dict(name=name,**result))
            if name=='symmetry':assert result['maximumValueSpan']==0 and result['maximumAlignedPolicyLogitDifference']<2e-12
        cells=[(ctypes.c_int*64)(*map(int,b)) for b in data['validation_board'][:64]];turns=data['validation_turn'][:64]
        calls=8192;rounds=9;rng=np.random.default_rng(37137)
        for iteration in range(rounds):
            for index in rng.permutation(len(loaded)):
                lib,handle=loaded[index];out=(ctypes.c_double*68)();tick=time.perf_counter_ns()
                for j in range(calls):
                    k=j%64;assert lib.ot_pattern_predict(handle,cells[k],int(turns[k]),out)==1
                latencies[index].append((time.perf_counter_ns()-tick)/calls/1000)
        placed=np.count_nonzero(data['validation_board'],axis=1)-4
        roots=[]
        for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:roots.extend(np.flatnonzero((placed>=lo)&(placed<=hi))[:2].tolist())
        assert len(roots)==8
        total=len(loaded)*len(roots)*(3+2);completed=0
        for ms,rounds in [(100,3),(2000,2)]:
            for iteration in range(rounds):
                for root in roots:
                    for index in rng.permutation(len(loaded)):
                        progress('相同局面交错搜索吞吐',completed,total,f'{ms}ms · {definitions[index][0]}')
                        _,metadata=search(*loaded[index],data['validation_board'][root],data['validation_turn'][root],ms=ms,simulations=0)
                        timed[index].append(dict(root=int(root),round=iteration,milliseconds=ms,**metadata));completed+=1
        rows=[]
        for index,(name,library,model) in enumerate(definitions):
            budgets={}
            for ms in [100,2000]:
                common=[root for root in roots if all(not s['exact'] and not s['memoryLimited'] and s['elapsedMs']>=.95*ms for samples in timed for s in samples if s['milliseconds']==ms and s['root']==root)]
                samples=[s for s in timed[index] if s['milliseconds']==ms and s['root'] in common]
                seconds=sum(s['wallSeconds'] for s in samples)
                budgets[str(ms)]=dict(commonRoots=common,samples=len(samples),simulationsPerSecond=sum(s['simulations'] for s in samples)/seconds if seconds else None,evaluationsPerSecond=sum(s['evaluations'] for s in samples)/seconds if seconds else None)
            rows.append(dict(name=name,library=str(library.relative_to(ROOT)),librarySha256=sha(library),model=model,modelSha256=sha(ROOT/model),medianJointMicroseconds=statistics.median(latencies[index]),latencySamples=latencies[index],budgets=budgets,searchSamples=timed[index]))
        for row in rows:
            row['jointThroughputVsFrozen']=rows[0]['medianJointMicroseconds']/row['medianJointMicroseconds']
            for ms,budget in row['budgets'].items():
                ref=rows[0]['budgets'][ms]
                if budget['simulationsPerSecond'] is not None:
                    budget['simulationRatioVsFrozen']=budget['simulationsPerSecond']/ref['simulationsPerSecond'];budget['evaluationRatioVsFrozen']=budget['evaluationsPerSecond']/ref['evaluationsPerSecond']
        report=dict(version='v1.37',complete=True,status='candidate',defaultPromoted=False,protocolSha256=sha(ROOT/'models/v1.37-pattern-structure-protocol.json'),results=rows,symmetry=symmetry_rows,
            regression=dict(validationPositions=6000,testPositions=6000,all68OutputsExactlyEqual=True,coldRoots=32,warmRoots=16,fixedSimulations=1024,searchMetadataExactlyEqual=True),portableParity=portable_checks,
            timing=dict(platform=platform.platform(),machine=platform.machine(),roots=roots,source='frozen validation boards; first two in each placements 8..16/17..24/25..32/33..40 band; no labels used',callsPerRound=calls,jointRounds=9,searchRounds={'100':3,'2000':2},maxNodes=4194304),
            preservedSha256=protocol['preservedSha256'],scriptSha256=sha(Path(__file__)),seconds=time.perf_counter()-started,limits='Single-machine native timing; common-root throughput excludes any root with proof, cap or insufficient budget in any implementation/round. No Linux or browser timing. Model differences also change search trajectories; throughput is not a strength measure.')
        for p,digest in protocol['preservedSha256'].items():assert sha(ROOT/p)==digest
        write(REPORT,report);progress('已完成',total,total,complete=True)
        print(json.dumps([dict(name=r['name'],jointUs=r['medianJointMicroseconds'],budgets=r['budgets']) for r in rows]),flush=True)
    finally:
        for lib,handle in handles:lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Confirm packed encoding + four-row SIMD, excluding slower integer rounding."""
import ctypes
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('selected_helpers',ROOT/'scripts/benchmark-pattern-hotpath.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
REPORT=ROOT/'models/v1.33-pattern-selected.json'
DIRECTORIES=['.build/native-pattern-simd-v1.32','.build/native-pattern-control-v1.33','.build/native-pattern-block4-v1.33','.build/native-pattern-selected-v1.33']
NAMES=['v1.32 SIMD','v1.33 control','four-row SIMD only','packed encoding + four-row SIMD']


def main():
    assert not REPORT.exists()
    start=time.perf_counter();data=np.load(ROOT/'.build/training/v1/positions.npz');loaded={};handles=[];fixed={};samples={};native={};timed={}
    preserved={p:h.sha(ROOT/p) for p in ['models/default-model.json','models/v1.25-default.json','src/game/wasm/othello.mjs']+h.MODELS}
    def progress(stage,n=0,total=1,summary='',complete=False):h.write(h.WORK/'progress.json',dict(experiment='pattern-hotpath-benchmark',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-start))
    cells=[(ctypes.c_int*64)(*map(int,b)) for b in data['validation_board'][:32]];turns=data['validation_turn'][:32]
    try:
        for precision,model in enumerate(h.MODELS):
            for i,directory in enumerate(DIRECTORIES):
                lib,handle=h.helper.load(ROOT/directory/f'libothello.{h.SUFFIX}',ROOT/model);handles.append((lib,handle));key=(precision,i);loaded[key]=(lib,handle)
                if i==0:
                    expected={split:h.helper.native(lib,handle,data[split+'_board'],data[split+'_turn']) for split in ['validation','test']}
                else:
                    for split in expected:assert np.array_equal(h.helper.native(lib,handle,data[split+'_board'],data[split+'_turn']),expected[split])
                fixed[key]=[h.helper.search(lib,handle,b,t) for b,t in zip(cells,turns)]
                if i:assert fixed[key]==fixed[(precision,0)]
                samples[key]=[];native[key]=[];timed[key]=[]
        keys=list(loaded);rng=np.random.default_rng(33124);rounds=15;calls=16384
        for iteration in range(rounds):
            for shuffled in rng.permutation(len(keys)):
                key=keys[int(shuffled)];p,i=key;lib,handle=loaded[key]
                progress('剔除较慢整数舍入后的交错复测',iteration*len(keys)+int(shuffled)+1,rounds*len(keys),f'{h.MODELS[p]} · {NAMES[i]}')
                out=(ctypes.c_double*68)();tick=time.perf_counter_ns()
                for j in range(calls):
                    k=j%len(cells);assert lib.ot_pattern_predict(handle,cells[k],int(turns[k]),out)==1
                us=(time.perf_counter_ns()-tick)/calls/1000;tick=time.perf_counter_ns()
                for b,t,expected in zip(cells,turns,fixed[key]):assert h.helper.search(lib,handle,b,t)==expected
                ms=(time.perf_counter_ns()-tick)/1e6;samples[key].append(dict(jointMicroseconds=us,fixedSearchMilliseconds=ms))
                if i:
                    native[key].append(json.loads(subprocess.check_output([str(ROOT/DIRECTORIES[i]/'otello_pattern_benchmark'),str(ROOT/h.MODELS[p]),'200000'],text=True)))
        for iteration in range(7):
            for shuffled in rng.permutation(len(keys)):
                key=keys[int(shuffled)];lib,handle=loaded[key];progress('每步 100ms 最终复核',iteration*len(keys)+int(shuffled)+1,7*len(keys))
                for b,t in zip(cells[:4],turns[:4]):timed[key].append(int(h.helper.search(lib,handle,b,t,simulations=0,ms=100)[3][0]))
        rows=[]
        for p,model in enumerate(h.MODELS):
            for i,directory in enumerate(DIRECTORIES):
                key=(p,i);library=ROOT/directory/f'libothello.{h.SUFFIX}'
                row=dict(precision='INT16' if p==0 else 'INT8',name=NAMES[i],library=str(library.relative_to(ROOT)),librarySha256=h.sha(library),model=model,jointMicroseconds=statistics.median(s['jointMicroseconds'] for s in samples[key]),fixedSearchMilliseconds=statistics.median(s['fixedSearchMilliseconds'] for s in samples[key]),samples=samples[key],nativeSamples=native[key],timed100msSamples=timed[key],timed100msTotalSimulations=sum(timed[key]))
                if native[key]:row['nativeMedianNanoseconds']={m:statistics.median(s[m] for s in native[key]) for m in ['encodingNs','denseNs','jointNs']}
                rows.append(row)
            group=rows[-len(DIRECTORIES):]
            for row in group:
                row['jointThroughputVsV132']=group[0]['jointMicroseconds']/row['jointMicroseconds']
                row['fixedSearchThroughputVsV132']=group[0]['fixedSearchMilliseconds']/row['fixedSearchMilliseconds']
                row['timedSimulationRatioVsV132']=row['timed100msTotalSimulations']/group[0]['timed100msTotalSimulations']
                row['timedSimulationRatioVsControl']=row['timed100msTotalSimulations']/group[1]['timed100msTotalSimulations']
        # The selected combination also retains reached-subtree reuse exactly.
        for p in range(2):
            wanted=None
            for i in [0,3]:
                lib,handle=loaded[(p,i)];lib.ot_mcts_clear_cache();board,turn=h.rules.initial(),1;actual=[]
                for ply in range(16):
                    b=(ctypes.c_int*64)(*map(int,board));out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
                    assert lib.ot_nnue_mcts_play(handle,b,turn,0,1024,1832,12,32,64,out,selection,visits,stats)==1
                    out[3]=0;stats[5]=0;actual.append([list(out),list(selection),list(visits),list(stats)]);board,turn=h.rules.play(board,turn,out[0])
                if wanted is None:wanted=actual
                else:assert wanted==actual
        cross=[]
        for suffix in ['-portable','-x86']:
            directory=ROOT/f'.build/native-pattern-selected{suffix}-v1.33'
            for p,model in enumerate(h.MODELS):
                sample=json.loads(subprocess.check_output([str(directory/'otello_pattern_benchmark'),str(ROOT/model),'200000'],text=True))
                assert sample['checksum']==native[(p,3)][0]['checksum'];cross.append(dict(target=suffix,model=model,checksum=sample['checksum']))
        for path,digest in preserved.items():assert h.sha(ROOT/path)==digest
        report=dict(version='v1.33-pattern-selected',status='candidate',defaultPromoted=False,selectedFlags=dict(SIMD=True,FAST_CODES=True,BLOCK4=True,INTEGER_ROUNDING=False),results=rows,parity=dict(all68JointOutputsExactlyEqual=True,validationPositions=6000,testPositions=6000,precisions=2,fixedSimulationSearchExactlyEqual=True,reusedSubtreeSearchExactlyEqual=True,coldRoots=32,warmRoots=16,simulationsPerRoot=1024),crossArchitectureChecks=cross,timing=dict(rounds=rounds,cAbiCallsPerRound=calls,nativeCallsPerPhase=200000,timedBudgetMs=100,timedRounds=7,timedRoots=4),preservedSha256=preserved,scriptSha256=h.sha(Path(__file__)),seconds=time.perf_counter()-start,limits='Confirmatory measurement after first-round runtime ablation; no accuracy/model selection or strength claim. Apple Silicon timings, x86-64 correctness under Rosetta; Linux timing unavailable. Slower integer-rounding path retained opt-in and excluded from selected flags.')
        h.write(REPORT,report);selected=[row for row in rows if row['name']==NAMES[-1]]
        progress('已完成',1,1,'；'.join(f"{row['precision']} 100ms 模拟 {row['timedSimulationRatioVsV132']:.3f}×" for row in selected),True)
        print(json.dumps(selected,ensure_ascii=False,indent=2))
    finally:
        for lib,handle in handles:lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()

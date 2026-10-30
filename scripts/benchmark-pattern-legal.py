#!/usr/bin/env python3
"""Benchmark legal-only policy projection without changing softmax/search."""
import ctypes
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('legal_helpers',ROOT/'scripts/benchmark-pattern-hotpath.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
WORK=ROOT/'.build/training/pattern-legal-v1.34';REPORT=ROOT/'models/v1.34-pattern-legal.json'
MODES=[('v1.33 selected','.build/native-pattern-selected-v1.33'),('v1.34 control','.build/native-pattern-legal-control-v1.34'),('legal-only projection','.build/native-pattern-legal-v1.34')]
ptr=ctypes.POINTER(ctypes.c_int)


def cached(lib,handle,board,turn,entries=0,clear=True):
    if clear:lib.ot_mcts_clear_cache()
    cells=(ctypes.c_int*64)(*map(int,board));out,selection,visits,stats,cache=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)(),(ctypes.c_double*8)()
    assert lib.ot_nnue_mcts_play_cached(handle,cells,int(turn),0,2048,1832,12,32,64,1,entries,out,selection,visits,stats,cache)==1
    out[3]=0;stats[5]=0
    return [list(out),list(selection),list(visits),list(stats)]


def main():
    assert not REPORT.exists();WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();handles=[]
    data_path=ROOT/'.build/training/v1/positions.npz';data=np.load(data_path)
    assert h.sha(data_path)==json.loads((ROOT/'models/nnue256-v1.json').read_text())['dataset']['cacheSha256']
    preserved={p:h.sha(ROOT/p) for p in ['models/default-model.json','models/v1.25-default.json','src/game/wasm/othello.mjs']+h.MODELS}
    def progress(stage,n=0,total=1,summary='',complete=False):h.write(WORK/'progress.json',dict(experiment='pattern-legal-benchmark',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    boards,turns=data['validation_board'][:32],data['validation_turn'][:32];cells=[(ctypes.c_int*64)(*map(int,b)) for b in boards]
    loaded={};expected={};fixed={};profiles={};times={};timed={};cross=[]
    pass_board=np.array([1 if ch=='B' else -1 if ch=='W' else 0 for ch in 'WWBBBWBBBBBBBWBBBBWWWWWBBBBWBWBBBBWBWWBBBBWWWBWBBWWWWWBBBW.BBBBB'],dtype=np.int8)
    fixtures=list(zip(boards,turns))+[(pass_board,-1),(np.ones(64,np.int8),-1)]
    try:
        for p,model in enumerate(h.MODELS):
            for i,(name,directory) in enumerate(MODES):
                key=(p,i);lib,handle=h.helper.load(ROOT/directory/f'libothello.{h.SUFFIX}',ROOT/model);handles.append((lib,handle));loaded[key]=(lib,handle)
                lib.ot_nnue_mcts_play_cached.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint,ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
                progress('完整输出及树复用逐项对照',p*3+i+1,6,f'{model} · {name}')
                full=np.concatenate([h.helper.native(lib,handle,data[s+'_board'],data[s+'_turn']) for s in ['validation','test']])
                if i==0:expected[p]=full
                else:assert np.array_equal(full,expected[p]),'Full diagnostic API changed'
                fixed[key]=[cached(lib,handle,b,t) for b,t in fixtures]
                if i:assert fixed[key]==fixed[(p,0)],'Cold fixed-budget search changed'
                for entries in [0,64]:
                    board,turn=h.rules.initial(),1;lib.ot_mcts_clear_cache();sequence=[]
                    for ply in range(16):
                        result=cached(lib,handle,board,turn,entries,False);sequence.append(result);board,turn=h.rules.play(board,turn,result[0][0])
                    if i==0:expected[(p,entries)]=sequence
                    else:assert sequence==expected[(p,entries)],'Reached subtree/cached-state search changed'
                profiles[key]=[];times[key]=[];timed[key]=[]
        # Explicit cached-state cold roots, including pass and terminal fixtures.
        for key,(lib,handle) in loaded.items():
            p,_=key
            for n,(b,t) in enumerate(fixtures):assert cached(lib,handle,b,t,64)==fixed[(p,0)][n]
        rng=np.random.default_rng(3412);keys=list(loaded);rounds=11
        for iteration in range(rounds):
            for shuffled in rng.permutation(len(keys)):
                key=keys[int(shuffled)];p,i=key;lib,handle=loaded[key]
                progress('分阶段原生推理与固定搜索交错计时',iteration*len(keys)+int(shuffled)+1,rounds*len(keys),f'{h.MODELS[p]} · {MODES[i][0]}')
                if i:profiles[key].append(json.loads(subprocess.check_output([str(ROOT/MODES[i][1]/'otello_pattern_legal_benchmark'),str(ROOT/h.MODELS[p]),'100000'],text=True)))
                tick=time.perf_counter_ns()
                for n,(b,t) in enumerate(fixtures[:32]):assert cached(lib,handle,b,t)==fixed[key][n]
                times[key].append((time.perf_counter_ns()-tick)/1e6)
        for iteration in range(7):
            for shuffled in rng.permutation(len(keys)):
                key=keys[int(shuffled)];lib,handle=loaded[key]
                progress('每步 100ms 搜索复核',iteration*len(keys)+int(shuffled)+1,7*len(keys))
                for b,t in zip(cells[:4],turns[:4]):timed[key].append(int(h.helper.search(lib,handle,b,t,simulations=0,ms=100)[3][0]))
        rows=[]
        for p,model in enumerate(h.MODELS):
            for i,(name,directory) in enumerate(MODES):
                key=(p,i);library=ROOT/directory/f'libothello.{h.SUFFIX}'
                row=dict(precision='INT16' if p==0 else 'INT8',name=name,library=str(library.relative_to(ROOT)),librarySha256=h.sha(library),model=model,fixedSearchMedianMilliseconds=statistics.median(times[key]),fixedSearchSamples=times[key],timed100msTotalSimulations=sum(timed[key]),timed100msSamples=timed[key],nativeSamples=profiles[key])
                if profiles[key]:
                    row['stageMedians']=[dict(stage=g,medianNanoseconds=statistics.median(s['groups'][g]['nanoseconds'] for s in profiles[key]),meanActivePatterns=profiles[key][0]['groups'][g]['meanActivePatterns'],meanLegalMoves=profiles[key][0]['groups'][g]['meanLegalMoves']) for g in range(3)]
                rows.append(row)
            group=rows[-3:]
            for row in group:
                row['fixedThroughputVsV133']=group[0]['fixedSearchMedianMilliseconds']/row['fixedSearchMedianMilliseconds']
                row['fixedThroughputVsControl']=group[1]['fixedSearchMedianMilliseconds']/row['fixedSearchMedianMilliseconds']
                row['timedSimulationsVsV133']=row['timed100msTotalSimulations']/group[0]['timed100msTotalSimulations']
                row['timedSimulationsVsControl']=row['timed100msTotalSimulations']/group[1]['timed100msTotalSimulations']
            assert len({s['checksum'] for i in [1,2] for s in profiles[(p,i)]})==1
        for suffix in ['x86','portable']:
            directory=ROOT/f'.build/native-pattern-legal-{suffix}-v1.34'
            for p,model in enumerate(h.MODELS):
                sample=json.loads(subprocess.check_output([str(directory/'otello_pattern_legal_benchmark'),str(ROOT/model),'100000'],text=True));assert sample['checksum']==profiles[(p,2)][0]['checksum'];cross.append(dict(target=suffix,model=model,checksum=sample['checksum']))
        for path,digest in preserved.items():assert h.sha(ROOT/path)==digest,path
        report=dict(version='v1.34-pattern-legal',status='candidate',defaultPromoted=False,implementation='Legal-only Pattern projection: pack only rows touching requested legal mask, project active rows in original shape order, aggregate/average only requested squares. Value pools/head unchanged. Existing legal-only softmax and stored tree priors untouched.',results=rows,parity=dict(fullDiagnosticOutputExactlyEqual=True,legalMaskedLogitsNativeTestExactlyEqual=True,coldSearchExactlyEqual=True,reusedSubtreeSearchExactlyEqual=True,incrementalStateSearchExactlyEqual=True,forcedPassAndTerminal=True,validationPositions=6000,testPositions=6000,precisions=2,coldRoots=34,warmRoots=16,simulationsPerRoot=2048),crossArchitectureChecks=cross,timing=dict(rounds=rounds,nativeCallsPerStage=100000,timedRounds=7,timedRoots=4,timedBudgetMs=100),preservedSha256=preserved,datasetSha256=h.sha(data_path),scriptSha256=h.sha(Path(__file__)),seconds=time.perf_counter()-started,limits='Apple Silicon performance; x86-64 correctness under Rosetta and portable fallback. Linux native timing unavailable. No training, strength match or default promotion. Illegal-square logits are zero in the new masked API only; original full diagnostic API unchanged.')
        h.write(REPORT,report);selected=[row for row in rows if row['name']=='legal-only projection']
        progress('已完成',1,1,'；'.join(f"{row['precision']} 100ms 模拟 {row['timedSimulationsVsV133']:.3f}×" for row in selected),True)
        print(json.dumps(selected,ensure_ascii=False,indent=2))
    finally:
        for lib,handle in handles:lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()

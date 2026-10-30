#!/usr/bin/env python3
"""Ablate exact rounding, packed encoding and four-row integer SIMD."""
import ctypes
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-hotpath-v1.33'
REPORT=ROOT/'models/v1.33-pattern-hotpath.json'
SUFFIX='dylib' if platform.system()=='Darwin' else 'so'
spec=importlib.util.spec_from_file_location('hotpath_helpers',ROOT/'scripts/benchmark-pattern-int8.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
spec=importlib.util.spec_from_file_location('hotpath_rules',ROOT/'scripts/compare-resnet.py')
rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
MODES=[('v1.32 SIMD','.build/native-pattern-simd-v1.32'),('v1.33 control','.build/native-pattern-control-v1.33'),
       ('integer rounding','.build/native-pattern-rounding-v1.33'),('packed encoding','.build/native-pattern-codes-v1.33'),
       ('four-row SIMD','.build/native-pattern-block4-v1.33'),('all three','.build/native-pattern-fast-v1.33')]
MODELS=['models/pattern32-outcome-v1.25.otpat','models/pattern32-int8-step1-v1.32.otpat']


def main():
    assert not REPORT.exists(),'Measured reports are immutable'
    WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();handles=[]
    preserved={p:sha(ROOT/p) for p in ['models/default-model.json','models/v1.25-default.json','src/game/wasm/othello.mjs']+MODELS}
    def progress(stage,n=0,total=1,summary='',complete=False):
        write(WORK/'progress.json',dict(experiment='pattern-hotpath-benchmark',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    data_path=ROOT/'.build/training/v1/positions.npz';data=np.load(data_path)
    assert sha(data_path)==json.loads((ROOT/'models/nnue256-v1.json').read_text())['dataset']['cacheSha256']
    boards,turns=data['validation_board'],data['validation_turn'];cells=[(ctypes.c_int*64)(*map(int,b)) for b in boards[:32]]
    rows=[];loaded={};fixed={};warm={};micro={};latencies={};fixed_times={};timed={}
    try:
        # Build native microbenchmarks before timing; preserve v1.32 reference library.
        for _,directory in MODES[1:]:
            subprocess.run(['cmake','-S',str(ROOT),'-B',str(ROOT/directory)],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['cmake','--build',str(ROOT/directory),'--target','otello_pattern_benchmark','-j','2'],check=True,stdout=subprocess.DEVNULL)
        for precision,model in enumerate(MODELS):
            expected=None;expected_test=None;fixed_ref=None;warm_ref=None
            for index,(name,directory) in enumerate(MODES):
                progress('完整输出与固定模拟逐项对照',precision*len(MODES)+index+1,2*len(MODES),f'{model} · {name}')
                lib,h=helper.load(ROOT/directory/f'libothello.{SUFFIX}',ROOT/model);handles.append((lib,h));key=(precision,index);loaded[key]=(lib,h)
                actual=helper.native(lib,h,boards,turns)
                if expected is None:expected=actual
                else:assert np.array_equal(actual,expected),(model,name,'validation')
                actual_test=helper.native(lib,h,data['test_board'],data['test_turn'])
                if expected_test is None:expected_test=actual_test
                else:assert np.array_equal(actual_test,expected_test),(model,name,'test')
                fixed[key]=[helper.search(lib,h,b,t) for b,t in zip(cells,turns[:32])]
                if fixed_ref is None:fixed_ref=fixed[key]
                else:assert fixed[key]==fixed_ref,(model,name,'fixed search')
                # Test actually reached-subtree reuse, not only independent roots.
                board,turn=rules.initial(),1;lib.ot_mcts_clear_cache();warm_rows=[]
                for ply in range(16):
                    b=(ctypes.c_int*64)(*map(int,board));out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
                    assert lib.ot_nnue_mcts_play(h,b,turn,0,1024,1832,12,32,64,out,selection,visits,stats)==1
                    out[3]=0;stats[5]=0;warm_rows.append([list(out),list(selection),list(visits),list(stats)])
                    board,turn=rules.play(board,turn,out[0])
                if warm_ref is None:warm_ref=warm_rows
                else:assert warm_rows==warm_ref,(model,name,'warm search')
                micro[key]=[];latencies[key]=[];fixed_times[key]=[];timed[key]=[]
        # Dedicated native measurements separate C ABI overhead and individual phases.
        rng=np.random.default_rng(33123);rounds=9;iterations=100000;calls=8192
        keys=list(loaded)
        for iteration in range(rounds):
            for shuffled in rng.permutation(len(keys)):
                key=keys[int(shuffled)];precision,index=key;name,directory=MODES[index];model=MODELS[precision]
                progress('分项与完整推理交错计时',iteration*len(keys)+int(shuffled)+1,rounds*len(keys),f'{model} · {name}')
                if index:
                    sample=json.loads(subprocess.check_output([str(ROOT/directory/'otello_pattern_benchmark'),str(ROOT/model),str(iterations)],text=True))
                    micro[key].append(sample)
                lib,h=loaded[key];out=(ctypes.c_double*68)();tick=time.perf_counter_ns()
                for j in range(calls):
                    k=j%len(cells);assert lib.ot_pattern_predict(h,cells[k],int(turns[k]),out)==1
                latencies[key].append((time.perf_counter_ns()-tick)/calls/1000)
                tick=time.perf_counter_ns()
                for b,t,expected in zip(cells,turns[:32],fixed[key]):assert helper.search(lib,h,b,t)==expected
                fixed_times[key].append((time.perf_counter_ns()-tick)/1e6)
        for precision in range(2):
            sums={sample['checksum'] for index in range(1,len(MODES)) for sample in micro[(precision,index)]}
            assert len(sums)==1,'Native benchmark checksums changed'
        # Interleaved actual clock budget: previous SIMD candidate, new control, all three.
        for iteration in range(5):
            subset=[(p,i) for p in range(2) for i in [0,1,5]]
            for shuffled in rng.permutation(len(subset)):
                key=subset[int(shuffled)];lib,h=loaded[key]
                progress('100ms 搜索预算复核',iteration*len(subset)+int(shuffled)+1,5*len(subset),str(key))
                for b,t in zip(cells[:4],turns[:4]):timed[key].append(int(helper.search(lib,h,b,t,simulations=0,ms=100)[3][0]))
        for precision in range(2):
            for index,(name,directory) in enumerate(MODES):
                key=(precision,index);samples=micro[key];library=ROOT/directory/f'libothello.{SUFFIX}'
                row=dict(precision='INT16' if precision==0 else 'INT8',name=name,library=str(library.relative_to(ROOT)),librarySha256=sha(library),model=MODELS[precision],jointCAbiMicroseconds=statistics.median(latencies[key]),latencySamples=latencies[key],fixedSearchMilliseconds=statistics.median(fixed_times[key]),fixedSearchSamples=fixed_times[key],timed100msSamples=timed[key],nativeSamples=samples)
                if samples:row['nativeMedianNanoseconds']={metric:statistics.median(s[metric] for s in samples) for metric in ['encodingNs','denseNs','jointNs']}
                if timed[key]:row['timed100msTotalSimulations']=sum(timed[key])
                row['jointThroughputVsV132']=statistics.median(latencies[(precision,0)])/row['jointCAbiMicroseconds']
                row['fixedSearchThroughputVsV132']=statistics.median(fixed_times[(precision,0)])/row['fixedSearchMilliseconds']
                rows.append(row)
        # Run portable and translated-x86 binaries on actual frozen models too.
        cross=[]
        for suffix in ['portable','x86']:
            directory=ROOT/f'.build/native-pattern-fast-{suffix}-v1.33'
            subprocess.run(['cmake','-S',str(ROOT),'-B',str(directory)],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['cmake','--build',str(directory),'--target','otello_pattern_benchmark','-j','2'],check=True,stdout=subprocess.DEVNULL)
            for precision,model in enumerate(MODELS):
                sample=json.loads(subprocess.check_output([str(directory/'otello_pattern_benchmark'),str(ROOT/model),str(iterations)],text=True))
                assert sample['checksum']==micro[(precision,5)][0]['checksum']
                cross.append(dict(target=suffix,model=model,checksum=sample['checksum']))
        for p,digest in preserved.items():assert sha(ROOT/p)==digest,p
        report=dict(version='v1.33-pattern-hotpath',status='candidate',defaultPromoted=False,platform=platform.platform(),machine=platform.machine(),preservedSha256=preserved,datasetSha256=sha(data_path),results=rows,crossArchitectureChecks=cross,parity=dict(validationPositions=6000,testPositions=6000,precisions=2,implementations=len(MODES),all68JointOutputsExactlyEqual=True,fixedSimulationSearchExactlyEqual=True,reusedSubtreeSearchExactlyEqual=True,coldRoots=32,warmRoots=16,simulationsPerRoot=1024,nativeChecksumsExactlyEqual=True),timing=dict(rounds=rounds,nativeCallsPerPhase=iterations,cAbiCallsPerRound=calls,timedBudgetMs=100,timedRounds=5,timedRoots=4),scriptSha256=sha(Path(__file__)),seconds=time.perf_counter()-started,limits='Apple Silicon performance. x86-64 correctness under Rosetta and portable fallback checked; no Linux native timing. No training, strength match, model-byte changes or default promotion. All timing samples retained; same-model comparisons only.')
        write(REPORT,report)
        selected=[row for row in rows if row['name']=='all three']
        progress('已完成',1,1,'；'.join(f"{v['precision']} 联合吞吐 {v['jointThroughputVsV132']:.3f}×，固定搜索 {v['fixedSearchThroughputVsV132']:.3f}×" for v in selected),True)
        print(json.dumps(selected,ensure_ascii=False,indent=2))
    finally:
        for lib,h in handles:lib.ot_nnue_destroy(h)


if __name__=='__main__':main()

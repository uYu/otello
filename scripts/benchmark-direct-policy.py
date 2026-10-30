#!/usr/bin/env python3
"""Audit the trained direct-policy export and measure same-position throughput."""
import hashlib,importlib.util,json,platform,statistics,sys,time
from pathlib import Path
import ctypes
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-direct-policy-v1.38'
REPORT=ROOT/'models/v1.38-direct-policy-benchmark.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('direct_policy_benchmark_helpers',ROOT/'scripts/benchmark-pattern-structure.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
SUFFIX='dylib' if sys.platform=='darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-direct-policy-v1.38/libothello.{SUFFIX}'
PORTABLE=ROOT/f'.build/native-pattern-direct-policy-portable-v1.38/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.37-reference/libothello.{SUFFIX}'


def main():
    assert not REPORT.exists(),'Reports are immutable'
    protocol=read(ROOT/'models/v1.38-direct-policy-protocol.json')
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    training=read(ROOT/'models/v1.38-pattern-direct-policy.json');candidate=ROOT/training['variants'][0]['model']
    baseline=ROOT/'models/pattern32-symmetry-v1.37.otpat'
    assert sha(candidate)==training['variants'][0]['modelSha256']
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'));started=time.perf_counter();handles=[]
    def progress(stage,n=0,total=1,summary='',complete=False):
        write(WORK/'benchmark-progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    config=read(ROOT/'models/current-experiment.json');config['experiments']=[dict(name='直连 MLP：导出、对称性与吞吐核验',focus='direct-mlp8',games=0,runtimeReport=str((WORK/'benchmark-progress.json').relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)))];write(ROOT/'models/current-experiment.json',config)
    try:
        for library,model in [(REFERENCE,baseline),(LIB,candidate),(LIB,baseline),(PORTABLE,candidate)]:handles.append(b.load(library,model))
        for split in ['validation','test']:
            boards,turns=data[split+'_board'],data[split+'_turn']
            assert np.array_equal(b.h.native(*handles[0],boards,turns),b.h.native(*handles[2],boards,turns))
            assert np.array_equal(b.h.native(*handles[1],boards,turns),b.h.native(*handles[3],boards,turns))
        for board,turn in zip(data['validation_board'][:16],data['validation_turn'][:16]):
            assert b.search(*handles[0],board,turn)[0]==b.search(*handles[2],board,turn)[0]
            assert b.search(*handles[1],board,turn)[0]==b.search(*handles[3],board,turn)[0]
        symmetry=b.symmetry(*handles[1],data['validation_board'],data['validation_turn'])
        assert symmetry['maximumValueSpan']==0 and symmetry['maximumAlignedPolicyLogitDifference']==0
        names=['v1.37 symmetry baseline','v1.38 direct MLP8'];latencies=[[],[]];timed=[[],[]];rng=np.random.default_rng(3818)
        cells=[(ctypes.c_int*64)(*map(int,row)) for row in data['validation_board'][:64]];turns=data['validation_turn'][:64]
        for iteration in range(9):
            for k in rng.permutation(2):
                lib,handle=handles[k];out=(ctypes.c_double*68)();tick=time.perf_counter_ns()
                for j in range(8192):
                    index=j%64;assert lib.ot_pattern_predict(handle,cells[index],int(turns[index]),out)==1
                latencies[k].append((time.perf_counter_ns()-tick)/8192/1000)
        placed=np.count_nonzero(data['validation_board'],axis=1)-4;roots=[]
        for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:roots.extend(np.flatnonzero((placed>=lo)&(placed<=hi))[:2].tolist())
        total=2*len(roots)*(3+2);done=0
        for ms,rounds in [(100,3),(1000,2)]:
            for iteration in range(rounds):
                for root in roots:
                    for k in rng.permutation(2):
                        progress('相同局面交错吞吐',done,total,f'{ms}ms · {names[k]}')
                        _,sample=b.search(*handles[k],data['validation_board'][root],data['validation_turn'][root],ms=ms,simulations=0)
                        timed[k].append(dict(root=root,round=iteration,budgetMs=ms,**sample));done+=1
        results=[]
        for k,name in enumerate(names):
            budgets={}
            for ms in [100,1000]:
                common=[root for root in roots if all(not s['exact'] and not s['memoryLimited'] and s['elapsedMs']>=.95*ms for samples in timed for s in samples if s['root']==root and s['budgetMs']==ms)]
                samples=[s for s in timed[k] if s['root'] in common and s['budgetMs']==ms];seconds=sum(s['wallSeconds'] for s in samples)
                budgets[str(ms)]=dict(commonRoots=common,samples=len(samples),simulationsPerSecond=sum(s['simulations'] for s in samples)/seconds if seconds else None,evaluationsPerSecond=sum(s['evaluations'] for s in samples)/seconds if seconds else None)
            results.append(dict(name=name,medianJointMicroseconds=statistics.median(latencies[k]),latencySamples=latencies[k],budgets=budgets,searchSamples=timed[k]))
        for ms in ['100','1000']:
            for metric in ['simulationsPerSecond','evaluationsPerSecond']:
                old,new=results[0]['budgets'][ms][metric],results[1]['budgets'][ms][metric]
                results[1]['budgets'][ms][metric+'Ratio']=new/old if old and new else None
        r=dict(version='v1.38-direct-policy',complete=True,status='candidate',defaultPromoted=False,baseline='models/v1.37-default.json',results=results,symmetry=symmetry,
            parity=dict(baselineRegressionPositions=12000,directScalarSimdPositions=12000,all68OutputsExactlyEqual=True,fixedSimulationRoots=16,simulationsPerRoot=1024),
            libraries={str(p.relative_to(ROOT)):sha(p) for p in [LIB,PORTABLE,REFERENCE]},models={str(p.relative_to(ROOT)):sha(p) for p in [baseline,candidate]},
            preservedSha256=protocol['preservedSha256'],protocolSha256=sha(ROOT/'models/v1.38-direct-policy-protocol.json'),trainingReportSha256=sha(ROOT/'models/v1.38-pattern-direct-policy.json'),
            sourceArchiveManifestSha256=sha(ROOT/'models/archive/v1.38-direct-policy-sources.json'),scriptSha256=sha(Path(__file__)),platform=platform.platform(),seconds=time.perf_counter()-started,
            limits='Offline candidate. No strength matches or Elo claim. Same-position wall-clock throughput only; common roots exclude proof/cap/short searches in either model/round. Native macOS ARM64 timing; Linux and browser performance unmeasured.')
        for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
        write(REPORT,r);progress('已完成',total,total,complete=True);print(json.dumps(results,ensure_ascii=False),flush=True)
    finally:
        for lib,handle in handles:lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()

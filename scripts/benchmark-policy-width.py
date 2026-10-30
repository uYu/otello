#!/usr/bin/env python3
"""Check policy-width exports and interleave same-position cost measurements."""
import hashlib,importlib.util,json,platform,statistics,sys,time,ctypes
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-policy-width-v1.39';REPORT=ROOT/'models/v1.39-policy-width-benchmark.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('policy_width_benchmark_helpers',ROOT/'scripts/benchmark-pattern-structure.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
SUFFIX='dylib' if sys.platform=='darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-policy-width-v1.39/libothello.{SUFFIX}'
PORTABLE=ROOT/f'.build/native-pattern-policy-width-portable-v1.39/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.37-reference/libothello.{SUFFIX}'
MLP8=ROOT/f'.build/native-pattern-direct-policy-v1.38/libothello.{SUFFIX}'


def main():
    assert not REPORT.exists(),'Reports are immutable'
    match=read(ROOT/'models/v1.38-direct-mlp8-vs-v1.37-2s-20.json')
    assert match['complete'] and match['games']==20,'Timing starts after the concurrent strength match has finished'
    protocol=read(ROOT/'models/v1.39-policy-width-protocol.json')
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    training=read(ROOT/'models/v1.39-pattern-policy-width.json')
    models=[ROOT/'models/pattern32-symmetry-v1.37.otpat',ROOT/'models/pattern32-direct-mlp8-v1.38.otpat']+[ROOT/v['model'] for v in training['variants']]
    for v in training['variants']:assert sha(ROOT/v['model'])==v['modelSha256']
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'));started=time.perf_counter();handles=[]
    names=['v1.37 symmetry baseline','v1.38 direct MLP8','v1.39 direct MLP16','v1.39 direct MLP32']
    def progress(stage,n=0,total=1,summary='',complete=False):
        write(WORK/'benchmark-progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    config=read(ROOT/'models/current-experiment.json');config['experiments']=[v for v in config['experiments'] if v.get('focus')!='direct-mlp16/32']
    config['experiments'].append(dict(name='MLP16 / MLP32：导出与相同局面吞吐核验',focus='policy-width',games=0,runtimeReport=str((WORK/'benchmark-progress.json').relative_to(ROOT)),report='models/v1.39-policy-width-monitor.json'));write(ROOT/'models/current-experiment.json',config)
    try:
        for library,model in [(REFERENCE,models[0]),(MLP8,models[1]),(LIB,models[2]),(LIB,models[3]),(LIB,models[0]),(LIB,models[1]),(PORTABLE,models[2]),(PORTABLE,models[3])]:handles.append(b.load(library,model))
        for split in ['validation','test']:
            boards,turns=data[split+'_board'],data[split+'_turn']
            for a,c in [(0,4),(1,5),(2,6),(3,7)]:assert np.array_equal(b.h.native(*handles[a],boards,turns),b.h.native(*handles[c],boards,turns))
        for board,turn in zip(data['validation_board'][:16],data['validation_turn'][:16]):
            for a,c in [(0,4),(1,5),(2,6),(3,7)]:assert b.search(*handles[a],board,turn)[0]==b.search(*handles[c],board,turn)[0]
        symmetry={}
        for k in [2,3]:
            symmetry[names[k]]=b.symmetry(*handles[k],data['validation_board'],data['validation_turn'])
            assert symmetry[names[k]]['maximumValueSpan']==0 and symmetry[names[k]]['maximumAlignedPolicyLogitDifference']==0
        latencies=[[] for _ in names];timed=[[] for _ in names];rng=np.random.default_rng(39139)
        cells=[(ctypes.c_int*64)(*map(int,row)) for row in data['validation_board'][:64]];turns=data['validation_turn'][:64]
        for iteration in range(9):
            for k in rng.permutation(len(names)):
                lib,handle=handles[k];out=(ctypes.c_double*68)();tick=time.perf_counter_ns()
                for j in range(8192):
                    index=j%64;assert lib.ot_pattern_predict(handle,cells[index],int(turns[index]),out)==1
                latencies[k].append((time.perf_counter_ns()-tick)/8192/1000)
        placed=np.count_nonzero(data['validation_board'],axis=1)-4;roots=[]
        for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:roots.extend(np.flatnonzero((placed>=lo)&(placed<=hi))[:2].tolist())
        total=4*len(roots)*5;done=0
        for ms,rounds in [(100,3),(2000,2)]:
            for iteration in range(rounds):
                for root in roots:
                    for k in rng.permutation(len(names)):
                        progress('相同局面交错吞吐',done,total,f'{ms}ms · {names[k]}')
                        _,sample=b.search(*handles[k],data['validation_board'][root],data['validation_turn'][root],ms=ms,simulations=0)
                        timed[k].append(dict(root=root,round=iteration,budgetMs=ms,**sample));done+=1
        results=[]
        for k,name in enumerate(names):
            budgets={}
            for ms in [100,2000]:
                common=[root for root in roots if all(not s['exact'] and not s['memoryLimited'] and s['elapsedMs']>=.95*ms for samples in timed for s in samples if s['root']==root and s['budgetMs']==ms)]
                samples=[s for s in timed[k] if s['root'] in common and s['budgetMs']==ms];seconds=sum(s['wallSeconds'] for s in samples)
                budgets[str(ms)]=dict(commonRoots=common,samples=len(samples),simulationsPerSecond=sum(s['simulations'] for s in samples)/seconds if seconds else None,evaluationsPerSecond=sum(s['evaluations'] for s in samples)/seconds if seconds else None)
            results.append(dict(name=name,medianJointMicroseconds=statistics.median(latencies[k]),latencySamples=latencies[k],budgets=budgets,searchSamples=timed[k]))
        for ms in ['100','2000']:
            for metric in ['simulationsPerSecond','evaluationsPerSecond']:
                for row in results:
                    val=row['budgets'][ms][metric]
                    for ref,k in [('VsApproved',0),('VsMlp8',1)]:
                        old=results[k]['budgets'][ms][metric];row['budgets'][ms][metric+ref+'Ratio']=val/old if old and val else None
        r=dict(version='v1.39-policy-width',complete=True,status='candidate',defaultPromoted=False,baseline='models/v1.37-default.json',results=results,symmetry=symmetry,
            parity=dict(baselineRegressionPositions=12000,mlp8RegressionPositions=12000,eachWidthScalarSimdPositions=12000,all68OutputsExactlyEqual=True,fixedSimulationRoots=16,simulationsPerRoot=1024),
            libraries={str(p.relative_to(ROOT)):sha(p) for p in [LIB,PORTABLE,REFERENCE,MLP8]},models={str(p.relative_to(ROOT)):sha(p) for p in models},
            preservedSha256=protocol['preservedSha256'],protocolSha256=sha(ROOT/'models/v1.39-policy-width-protocol.json'),trainingReportSha256=sha(ROOT/'models/v1.39-pattern-policy-width.json'),
            sourceArchiveManifestSha256=sha(ROOT/'models/archive/v1.39-policy-width-sources.json'),scriptSha256=sha(Path(__file__)),platform=platform.platform(),seconds=time.perf_counter()-started,
            limits='Timing after training and 2s match finished. Common roots exclude proof/cap/short searches in any model/round. Native macOS ARM64 only. Offline metrics and throughput do not establish strength; no MLP16/32 matches in this experiment.')
        for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
        write(REPORT,r);progress('已完成',total,total,complete=True)
        write(ROOT/'models/v1.39-policy-width-monitor.json',read(WORK/'benchmark-progress.json'))
        print(json.dumps(dict(complete=True,seconds=r['seconds'],metrics=[dict(name=v['name'],budgets=v['budgets']) for v in results]),ensure_ascii=False),flush=True)
    finally:
        for lib,handle in handles:lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Post-training INT8 embedding quantization and exact integer SIMD comparison."""
import ctypes
import hashlib
import json
import platform
import statistics
import struct
import time
from pathlib import Path
import numpy as np
from pattern_model import encode, legal_mask, SHAPES, BUCKET
MEMBERS=np.zeros((38,64),np.float64)
for i,(squares,_,_) in enumerate(SHAPES):MEMBERS[i,squares]=1
MEMBERS/=MEMBERS.sum(0)

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-int8-v1.32'
REPORT=ROOT/'models/v1.32-pattern-int8.json'
MODEL=ROOT/'models/pattern32-outcome-v1.25.otpat'
SUFFIX='dylib' if platform.system()=='Darwin' else 'so'
SCALAR=ROOT/f'.build/native-pattern-int8-v1.32/libothello.{SUFFIX}'
SIMD=ROOT/f'.build/native-pattern-simd-v1.32/libothello.{SUFFIX}'
FUSED=ROOT/f'.build/native-pattern-h64-v1.29/libothello.{SUFFIX}'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
ptr=ctypes.POINTER(ctypes.c_int)


def rounded(a):return np.sign(a)*np.floor(np.abs(a)+.5)


def arrays(binary):
    _,rows,dim,width,inputs,outputs,scale=struct.unpack_from('<8s6I',binary)
    assert (rows,dim,inputs,outputs,scale)==(55728,32,97,46,256)
    offset=32;result=[]
    for dtype,size,shape in [('<i2',rows*dim,(rows,dim)),('<i2',width*inputs,(width,inputs)),('<i4',width,(width,)),('<i2',outputs*width,(outputs,width)),('<i4',outputs,(outputs,))]:
        x=np.frombuffer(binary,dtype=dtype,count=size,offset=offset).reshape(shape).astype(np.int64)
        offset+=size*np.dtype(dtype).itemsize;result.append(x)
    assert offset==len(binary)
    return result


def reference(parameters,board,turn):
    table,trunk,tbias,head,hbias=parameters
    e=table[encode(board,turn)]
    sums=np.zeros((len(board),3,32),np.int64)
    for i,(_,_,group) in enumerate(SHAPES):sums[:,group]+=e[:,i]
    x=np.column_stack((np.clip(rounded(sums/np.array([4,30,4])[None,:,None]),-1024,1024).reshape(-1,96),rounded((64-np.count_nonzero(board,axis=1))*256/60))).astype(np.int64)
    h=np.clip(rounded((x@trunk.T+tbias)/256),0,256).astype(np.int64)
    out=np.clip(rounded((h@head.T+hbias)/256),-1024,1024).astype(np.int64)
    projection=np.einsum('bpd,bd->bp',e,out[:,4:36])/(256*256*np.sqrt(32))
    logits=projection@MEMBERS.astype(np.float64)+out[:,36:46][:,BUCKET]/256
    wdl=out[:,1:4]/256;wdl=np.exp(wdl-wdl.max(1,keepdims=True));wdl/=wdl.sum(1,keepdims=True)
    return np.column_stack((np.tanh(out[:,0]/256),wdl,logits))


def metrics(prediction,target,labels,mask):
    target=np.sign(target);value=prediction[:,0];prob=prediction[:,1:4];logits=np.where(mask,prediction[:,4:],-1e9)
    truth=np.where(target>0,0,np.where(target<0,1,2));non_draw=target!=0
    logp=logits-logits.max(1,keepdims=True);logp-=np.log(np.exp(logp).sum(1,keepdims=True))
    return dict(mse=float(np.mean((value-target)**2)),nonDrawSignAccuracy=float(np.mean(np.sign(value[non_draw])==target[non_draw])),wdlCrossEntropy=float(-np.log(np.maximum(1e-12,prob[np.arange(len(target)),truth])).mean()),policyTop1=float(np.mean(logits.argmax(1)==labels)),policyCrossEntropy=float(-logp[np.arange(len(target)),labels].mean()))


def load(path,model):
    lib=ctypes.CDLL(str(path));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_error.restype=ctypes.c_char_p
    lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    lib.ot_nnue_mcts_play.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*3+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double)]
    h=lib.ot_nnue_load(str(model).encode());assert h,lib.ot_nnue_error().decode()
    return lib,h


def native(lib,h,board,turn):
    out=(ctypes.c_double*68)();rows=[]
    for b,t in zip(board,turn):
        cells=(ctypes.c_int*64)(*map(int,b));assert lib.ot_pattern_predict(h,cells,int(t),out)==1
        rows.append(list(out))
    return np.array(rows)


def search(lib,h,cells,turn,simulations=1024,ms=0):
    lib.ot_mcts_clear_cache()
    out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
    assert lib.ot_nnue_mcts_play(h,cells,int(turn),ms,simulations,1832,12,32,64,out,selection,visits,stats)==1
    out[3]=0;stats[5]=0
    return [list(out),list(selection),list(visits),list(stats)]


def main():
    assert not REPORT.exists(),'Refusing to overwrite measured report'
    WORK.mkdir(parents=True,exist_ok=True);start=time.perf_counter();handles=[]
    preserved={p:sha(ROOT/p) for p in ['models/default-model.json','models/v1.25-default.json',str(MODEL.relative_to(ROOT)),'src/game/wasm/othello.mjs']}
    def progress(stage,n=0,total=1,summary='',complete=False):
        write(WORK/'progress.json',dict(experiment='pattern-int8-benchmark',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-start))
    data_path=ROOT/'.build/training/v1/positions.npz';data=dict(np.load(data_path));policy_path=ROOT/'.build/training/pattern-outcome-v1.25/policy.npz';labels=dict(np.load(policy_path))
    assert sha(data_path)==json.loads((ROOT/'models/nnue256-v1.json').read_text())['dataset']['cacheSha256']
    provenance=json.loads(policy_path.with_suffix('.json').read_text());assert sha(policy_path)==provenance['policySha256']
    binary=MODEL.read_bytes();parameters=arrays(binary);variants=[]
    base={};predictions={}
    board,turn=data['validation_board'],data['validation_turn'];mask=legal_mask(board,turn)
    wanted=reference(parameters,board,turn)
    base['validation']=metrics(wanted,data['validation_target'],labels['validation'],mask)
    try:
        for step in [1,2,4]:
            progress('INT8 验证集量化步长比较',step,4)
            raw=rounded(parameters[0]/step);q=np.clip(raw,-128,127).astype(np.int8)
            path=ROOT/f'models/pattern32-int8-step{step}-v1.32.otpat';assert not path.exists()
            path.write_bytes(b'OTPAT3\0\0'+binary[8:32]+struct.pack('<I',step)+q.tobytes()+binary[32+55728*32*2:])
            qp=[q.astype(np.int64)*step]+parameters[1:]
            predicted=reference(qp,board,turn);predictions[step]=predicted
            lib,h=load(SCALAR,path);handles.append((lib,h));actual=native(lib,h,board,turn)
            error=np.max(np.abs(actual-predicted),axis=0);assert error.max()<2e-6,error.max()
            delta=actual-wanted
            variants.append(dict(step=step,model=str(path.relative_to(ROOT)),modelSha256=sha(path),bytes=path.stat().st_size,clippedEntries=int(np.sum(raw!=q)),embeddingRmseInOriginalUnits=float(np.sqrt(np.mean((qp[0]-parameters[0])**2))),validation=metrics(actual,data['validation_target'],labels['validation'],mask),parity=dict(positions=len(board),maximumError=float(error.max())),valueDeltaRmse=float(np.sqrt(np.mean(delta[:,0]**2))),maximumValueDelta=float(np.max(np.abs(delta[:,0])))))
        best=min(variants,key=lambda row:row['validation']['mse']);step=best['step'];candidate=ROOT/best['model']
        # All step candidates frozen before selecting by validation; test only selected step.
        qp=[np.clip(rounded(parameters[0]/step),-128,127).astype(np.int64)*step]+parameters[1:]
        tb,tt=data['test_board'],data['test_turn'];tm=legal_mask(tb,tt)
        progress('已冻结量化步长；测试与原生对照')
        base['test']=metrics(reference(parameters,tb,tt),data['test_target'],labels['test'],tm)
        wanted_test=reference(qp,tb,tt)
        modes=[('v1.29 fused INT16',FUSED,MODEL),('portable INT16',SCALAR,MODEL),('SIMD INT16',SIMD,MODEL),('portable INT8',SCALAR,candidate),('SIMD INT8',SIMD,candidate)]
        loaded=[];outputs=[]
        for name,path,model in modes:
            lib,h=load(path,model);handles.append((lib,h));loaded.append((lib,h))
            outputs.append(native(lib,h,tb,tt));assert np.max(np.abs(outputs[-1]-(wanted_test if model==candidate else reference(parameters,tb,tt))))<2e-6
        assert np.array_equal(outputs[0],outputs[1]) and np.array_equal(outputs[1],outputs[2])
        assert np.array_equal(outputs[3],outputs[4])
        best['test']=metrics(outputs[3],data['test_target'],labels['test'],tm)
        best['testParity']=dict(positions=len(tb),maximumError=float(np.max(np.abs(outputs[3]-wanted_test))))
        fixed=[];cells=[(ctypes.c_int*64)(*map(int,row)) for row in board[:32]]
        for lib,h in loaded:fixed.append([search(lib,h,b,t) for b,t in zip(cells,turn[:32])])
        assert fixed[0]==fixed[1]==fixed[2] and fixed[3]==fixed[4]
        latency=[[] for _ in modes];search_times=[[] for _ in modes];timed=[[] for _ in modes];rng=np.random.default_rng(3218);calls=8192;rounds=9
        for iteration in range(rounds):
            for index in rng.permutation(len(modes)):
                progress('联合推理及搜索交错计时',iteration*len(modes)+int(index)+1,rounds*len(modes),modes[index][0])
                lib,h=loaded[index];out=(ctypes.c_double*68)();tick=time.perf_counter_ns()
                for j in range(calls):
                    k=j%len(cells);assert lib.ot_pattern_predict(h,cells[k],int(turn[k]),out)==1
                latency[index].append((time.perf_counter_ns()-tick)/calls/1000)
                tick=time.perf_counter_ns()
                for b,t,want in zip(cells,turn[:32],fixed[index]):assert search(lib,h,b,t)==want
                search_times[index].append((time.perf_counter_ns()-tick)/1e6)
        for iteration in range(5):
            for index in rng.permutation(len(modes)):
                progress('100ms 实际搜索预算复核',iteration*len(modes)+int(index)+1,5*len(modes),modes[index][0])
                lib,h=loaded[index]
                for b,t in zip(cells[:4],turn[:4]):
                    result=search(lib,h,b,t,simulations=0,ms=100);timed[index].append(int(result[3][0]))
        rows=[]
        for i,(name,path,model) in enumerate(modes):
            rows.append(dict(name=name,library=str(path.relative_to(ROOT)),librarySha256=sha(path),model=str(model.relative_to(ROOT)),medianMicroseconds=statistics.median(latency[i]),latencySamples=latency[i],fixedSearchMedianMilliseconds=statistics.median(search_times[i]),searchSamples=search_times[i],timed100msSimulations=sum(timed[i]),timed100msSamples=timed[i]))
        for row in rows:
            row['jointThroughputVsFused']=rows[0]['medianMicroseconds']/row['medianMicroseconds']
            row['fixedSearchThroughputVsFused']=rows[0]['fixedSearchMedianMilliseconds']/row['fixedSearchMedianMilliseconds']
            row['timedSimulationRatioVsFused']=row['timed100msSimulations']/rows[0]['timed100msSimulations']
        for p,digest in preserved.items():assert sha(ROOT/p)==digest,p
        report=dict(version='v1.32-pattern-int8-simd',status='candidate',defaultPromoted=False,platform=platform.platform(),machine=platform.machine(),quantization='Embedding only INT8; shared integer step in original 1/256 units; round away from zero; clip [-128,127]. Trunk/head INT16 and biases/accumulators INT32 retained. No retraining.',selection='Minimum validation outcome MSE over steps 1,2,4; test evaluated only after all candidates exported and selected step frozen',selectedStep=step,baselineMetrics=base,variants=variants,results=rows,parity=dict(validationPositions=len(board),testPositions=len(tb),simdInt16ExactlyEqual=True,simdInt8ExactlyEqual=True,fixedSimulationMctsExactlyEqualWithinEachPrecision=True,searchPositions=32,simulationsPerPosition=1024),timing=dict(rounds=rounds,callsPerRound=calls,timedRounds=5,timedPositions=4,budgetMs=100),preservedSha256=preserved,datasetSha256=sha(data_path),policyDatasetSha256=sha(policy_path),scriptSha256=sha(Path(__file__)),seconds=time.perf_counter()-start,limits='Apple Silicon native measurement; SSE2 correctness separately under Rosetta, not Linux performance. No strength match. Python/C ABI prediction overhead included. Comparator v1.29 fused INT16; INT8 changes values and search trajectories, fixed-simulation parity asserted only within same precision.')
        write(REPORT,report);progress('已完成',1,1,f"INT8 步长 {step}；SIMD INT8 联合吞吐 {rows[-1]['jointThroughputVsFused']:.2f}×，100ms 模拟 {rows[-1]['timedSimulationRatioVsFused']:.2f}×",True)
        print(json.dumps(dict(selected=best,baseline=base,results=rows),ensure_ascii=False,indent=2))
    finally:
        for lib,h in handles:lib.ot_nnue_destroy(h)


if __name__=='__main__':main()

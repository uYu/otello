#!/usr/bin/env python3
"""Supplement the MLP study with equal-work timing and phase-wise offline metrics."""
import ctypes
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('mlp_study',Path(__file__).with_name('investigate-mlp-inference.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
REPORT=s.ROOT/'models/mlp-search-cost-2026-10-30.json'
ptr=ctypes.POINTER(ctypes.c_int)


def main():
    assert not REPORT.exists(), 'Reports are immutable'
    base=json.loads(s.REPORT.read_text());assert base['complete']
    data=np.load(s.ROOT/'.build/training/v1/positions.npz')
    labels=np.load(s.ROOT/'.build/training/v1.8/policy.npz')
    started=time.perf_counter();handles={};metrics=[];samples=[];parity=[]
    # Every possible local sum for up to six arbitrary INT16 entries, including
    # negative half ties. Integer normalization precedes the unchanged clamp.
    values=np.arange(-6*32768,6*32768+1,dtype=np.int64)
    for divisor in range(2,7):
        exact=np.sign(values)*((np.abs(values)+divisor//2)//divisor)
        floating=np.sign(values)*np.floor(np.abs(values/divisor)+.5)
        assert np.array_equal(exact,floating)
    def progress(stage,n=0,total=1,complete=False):
        s.write(s.PROGRESS,dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,complete=complete,seconds=time.perf_counter()-started))
    def load(v,m):
        lib=ctypes.CDLL(str(s.WORK/v/f'libstudy.{s.SUFFIX}'))
        lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
        lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
        lib.ot_legal.argtypes=[ptr,ctypes.c_int,ptr]
        lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        handle=lib.ot_nnue_load(str(s.ROOT/f'models/{s.MODELS[m]}.otpat').encode());assert handle
        handles[v,m]=(lib,handle)
    try:
        for m in range(4):
            for v in ['original','combined']:load(v,m)
            lib,handle=handles['original',m]
            for split in ['validation','test']:
                progress('补充：按阶段核验离线收益',m,4)
                boards,turns=data[split+'_board'],data[split+'_turn']
                prediction=[];masks=[];out=(ctypes.c_double*68)();moves=(ctypes.c_int*64)()
                for board,turn in zip(boards,turns):
                    cells=(ctypes.c_int*64)(*map(int,board))
                    assert lib.ot_pattern_predict(handle,cells,int(turn),out)==1
                    prediction.append(list(out));mask=np.zeros(64,bool)
                    n=lib.ot_legal(cells,int(turn),moves);assert n>=0
                    mask[list(moves)[:n]]=True;masks.append(mask)
                prediction=np.asarray(prediction);masks=np.asarray(masks);target=np.sign(data[split+'_target'])
                placed=np.count_nonzero(boards,axis=1)-4
                logits=np.where(masks,prediction[:,4:],-1e9);logp=logits-logits.max(1,keepdims=True);logp-=np.log(np.exp(logp).sum(1,keepdims=True))
                for lo,hi in [(0,20),(20,44),(44,61),(0,61)]:
                    selected=(placed>=lo)&(placed<hi);non_draw=selected&(target!=0)
                    metrics.append(dict(model=s.MODELS[m],split=split,placements=[lo,hi-1],positions=int(selected.sum()),
                        mse=float(np.mean((prediction[selected,0]-target[selected])**2)),
                        nonDrawSignAccuracy=float(np.mean(np.sign(prediction[non_draw,0])==target[non_draw])),
                        policyTop1=float(np.mean(logits[selected].argmax(1)==labels[split][selected])),
                        policyCrossEntropy=float(-logp[np.arange(len(boards)),labels[split]][selected].mean()),
                        policyEntropy=float(-np.sum(np.exp(logp[selected])*logp[selected],axis=1).mean())))
        placed=np.count_nonzero(data['validation_board'],axis=1)-4;roots=[]
        for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:roots.extend(np.flatnonzero((placed>=lo)&(placed<=hi))[:2].tolist())
        rng=random.Random(20261031)
        for round_ in range(3):
            for i in roots:
                for m in rng.sample(range(4),4):
                    results=[]
                    for v in rng.sample(['original','combined'],2):
                        progress('补充：相同模拟次数搜索成本',len(samples),3*len(roots)*8)
                        lib,handle=handles[v,m];lib.ot_mcts_clear_cache()
                        cells=(ctypes.c_int*64)(*map(int,data['validation_board'][i]))
                        out,sel,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
                        tick=time.perf_counter_ns()
                        assert lib.ot_nnue_mcts_play_budget(handle,cells,int(data['validation_turn'][i]),0,131072,1832,12,32,64,1,0,4194304,out,sel,visits,stats,None)==1
                        wall=(time.perf_counter_ns()-tick)/1e9
                        samples.append(dict(model=s.MODELS[m],variant=v,root=i,round=round_,wallSeconds=wall,simulations=int(stats[0]),evaluations=int(stats[1]),exact=bool(out[5]),memoryLimited=bool(stats[7])))
                        out[3]=0;stats[5]=0
                        results.append([list(out),list(sel),list(visits),list(stats)])
                    assert results[0]==results[1],(m,i,round_)
                    parity.append(dict(model=s.MODELS[m],root=i,round=round_,exact=True))
        summary=[]
        for m in s.MODELS:
            a=[r for r in samples if r['model']==m and r['variant']=='original'];b=[r for r in samples if r['model']==m and r['variant']=='combined']
            summary.append(dict(model=m,searchesPerVariant=len(a),originalSeconds=sum(r['wallSeconds'] for r in a),combinedSeconds=sum(r['wallSeconds'] for r in b),speedup=sum(r['wallSeconds'] for r in a)/sum(r['wallSeconds'] for r in b)))
        s.write(REPORT,dict(complete=True,defaultPromoted=False,baseReportSha256=s.sha(s.REPORT),scriptSha256=s.sha(Path(__file__)),
            policyLabelsSha256=s.sha(s.ROOT/'.build/training/v1.8/policy.npz'),localRoundingCheck=dict(numeratorBounds=[-6*32768,6*32768],divisors=[2,3,4,5,6],exact=True),metrics=metrics,fixedWorkSamples=samples,fixedWorkParity=parity,summary=summary,seconds=time.perf_counter()-started,
            limits='Exploratory native macOS measurements; background experiments may still be running. Equal-weight pairs use identical complete search results, evaluation counts and simulation counts; timings are interleaved. Phase offline metrics describe the existing held-out splits, not independent strength evidence. No additional training or matches.'))
        progress('推理效率研究及补充核验完成',1,1,True)
        print(json.dumps(summary,ensure_ascii=False))
    finally:
        for lib,handle in handles.values():lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Batch legal moves within one board, without changing MCTS scheduling."""
import ctypes
import hashlib
import importlib.util
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('mlp_study',Path(__file__).with_name('investigate-mlp-inference.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
PREPARED='--prepared' in sys.argv
GEMM='--gemm' in sys.argv or PREPARED
KIND='prepared' if PREPARED else 'gemm' if GEMM else 'batch'
WORK=s.ROOT/f'.build/mlp-move-{KIND}-study-20261030'
REPORT=s.ROOT/f'models/mlp-move-{KIND}-study-2026-10-30.json'
VARIANTS=['original','gemm_prepared','gemm_fused'] if PREPARED else ['original','gemm','gemm_norm'] if GEMM else ['original','batch_moves','norm_simd','batch_moves_norm']
ptr=ctypes.POINTER(ctypes.c_int)


def main():
    assert not REPORT.exists(),'Reports are immutable'
    base=json.loads(s.REPORT.read_text());assert base['complete']
    WORK.mkdir(exist_ok=True);started=time.perf_counter()
    source=(s.WORK/'source-snapshot/pattern.cpp').read_text()
    header=s.ROOT/'benchmarks/mlp-move-batch.hpp'
    shutil.copy2(Path(__file__),WORK/'measured-runner.py')
    shutil.copy2(header,WORK/'measured-kernels.hpp')
    progressfile=WORK/'progress.json'
    def progress(stage,n=0,total=1,summary='',complete=False):
        s.write(progressfile,dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    config=json.loads((s.ROOT/'models/current-experiment.json').read_text())
    config['experiments']=[x for x in config.get('experiments',[]) if x.get('runtimeReport')!=str(progressfile.relative_to(s.ROOT))]
    config['experiments'].append(dict(name='MLP：一次打包权重 + 融合 batch' if PREPARED else 'MLP：全部合法着一次 GEMM' if GEMM else 'MLP：同一棋盘合法着 batch / SIMD 归一化',games=0,report=str(REPORT.relative_to(s.ROOT)),runtimeReport=str(progressfile.relative_to(s.ROOT))))
    s.write(s.ROOT/'models/current-experiment.json',config)
    flags=base['flags'];compiler=base['compiler']
    folders={'original':s.WORK/'original'}
    for v in VARIANTS[1:]:
        progress('构建合法着批处理隔离原型',VARIANTS.index(v)-1,len(VARIANTS)-1,v)
        folder=WORK/v;folder.mkdir(exist_ok=True);folders[v]=folder
        shutil.copytree(s.WORK/'source-snapshot',folder/'cpp',dirs_exist_ok=True)
        shutil.copy2(header,folder/'cpp/mlp-move-batch.hpp')
        patched='#define STUDY_DEFINE_CHECK 1\n#include "mlp-move-batch.hpp"\n'+source
        if v=='norm_simd':
            patched=s.replace_once(patched,'for(int j=0;j<32;++j)feature[j]=std::int16_t(std::clamp(rounded(double(local[square][j])/geometry.counts[square]),-1024,1024));',
                'study::normalize(local[square].data(),geometry.counts[square],feature.data());')
        else:
            begin=patched.index('        std::array<std::int64_t,32> global{};')
            end=patched.index('        return result;\n    }\n    maskedPolicy',begin)
            patched=patched[:begin]+'        study::policyBatch(*this,local,geometry.counts,geometry.buckets,h.data(),mask,result);\n'+patched[end:]
        (folder/'cpp/pattern.cpp').write_text(patched)
        ownflags=flags+(['-DSTUDY_VECTOR_NORMALIZE=1'] if v in ['norm_simd','batch_moves_norm','gemm_norm','gemm_prepared','gemm_fused'] else [])
        if GEMM:ownflags+=['-DSTUDY_GEMM=1','-framework','Accelerate']
        if PREPARED:
            ownflags+=['-DSTUDY_PREPACK=1']
            if v=='gemm_fused':ownflags+=['-DSTUDY_FUSED=1']
            mcts=(folder/'cpp/mcts.cpp').read_text()
            mcts='#include "mlp-move-batch.hpp"\n'+mcts
            mcts=s.replace_once(mcts,'MctsResult mcts(const Position& position,MctsOptions options){',
                '''MctsResult mcts(const Position& position,MctsOptions options){
    std::optional<nnue::study::Prepared> studyPrepared;
    std::optional<nnue::study::Scope> studyScope;
    if(options.network&&options.network->pattern&&options.network->pattern->variant==3){
        studyPrepared.emplace(*options.network->pattern);studyScope.emplace(*studyPrepared);}
''')
            (folder/'cpp/mcts.cpp').write_text(mcts)
            api=(folder/'cpp/api.cpp').read_text()
            api='#include "mlp-move-batch.hpp"\n'+api
            api+='''
extern "C" void ot_study_prepare(const othello::nnue::Pattern* model){
    static thread_local std::optional<othello::nnue::study::Prepared> prepared;
    prepared.emplace(*model);othello::nnue::study::activePrepared=&*prepared;
}
'''
            (folder/'cpp/api.cpp').write_text(api)
        files=['search','move_policy','nnue','pattern','cnn','resnet','mcts','default_engine','api']
        command=[compiler,*ownflags,'-dynamiclib' if sys.platform=='darwin' else '-shared',*[str(folder/f'cpp/{f}.cpp') for f in files],'-o',str(folder/f'libstudy.{s.SUFFIX}')]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        harness=s.ROOT/'benchmarks/mlp-inference-study.cpp'
        if PREPARED:
            text=harness.read_text()
            text=text.replace('using namespace othello;', 'using namespace othello;\nextern "C" void ot_study_prepare(const nnue::Pattern*);')
            text=s.replace_once(text,'const auto& model=*network.pattern;', 'const auto& model=*network.pattern;ot_study_prepare(&model);')
            harness=folder/'bench.cpp';harness.write_text(text)
        command=[compiler,*ownflags,'-I'+str(folder/'cpp'),str(harness),str(folder/f'libstudy.{s.SUFFIX}'),'-o',str(folder/'bench')]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
    data=np.load(s.ROOT/'.build/training/v1/positions.npz')
    handles={};parity=[];timings=[];searches=[];fixedParity=[]
    def load(v,m):
        lib=ctypes.CDLL(str(folders[v]/f'libstudy.{s.SUFFIX}'))
        lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
        lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
        lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        if v!='original':assert lib.ot_study_check_normalize()==1
        handle=lib.ot_nnue_load(str(s.ROOT/f'models/{s.MODELS[m]}.otpat').encode());assert handle
        handles[v,m]=(lib,handle)
    try:
        boards=np.concatenate([data['validation_board'],data['test_board']]);turns=np.concatenate([data['validation_turn'],data['test_turn']])
        for m in range(4):
            for v in VARIANTS:
                progress('全部68项输出及归一化穷举核验',m*len(VARIANTS)+VARIANTS.index(v),4*len(VARIANTS),f'{s.MODELS[m]} · {v}')
                load(v,m);lib,handle=handles[v,m];digest=hashlib.sha256();out=(ctypes.c_double*68)()
                for board,turn in zip(boards,turns):
                    cells=(ctypes.c_int*64)(*map(int,board));assert lib.ot_pattern_predict(handle,cells,int(turn),out)==1
                    digest.update(bytes(out))
                assert digest.hexdigest()==base['parity'][m]['all68OutputSha256'],(v,m)
                parity.append(dict(variant=v,model=s.MODELS[m],positions=len(boards),all68OutputSha256=digest.hexdigest(),exact=True))
        rng=random.Random(30102026);jobs=[(v,m,mode) for v in VARIANTS for m in range(4) for mode in [1,2]]
        for round_ in range(7):
            rng.shuffle(jobs)
            for v,m,mode in jobs:
                progress('原生合法着/完整策略交错计时',len(timings),7*len(jobs),f'{v} · {s.MODELS[m]}')
                output=subprocess.check_output([str(folders[v]/'bench'),str(s.ROOT/f'models/{s.MODELS[m]}.otpat'),str(s.WORK/'boards.txt'),'32768',str(mode)],text=True)
                timings.append(dict(variant=v,model=m,mode=mode,round=round_,**json.loads(output)))
        placed=np.count_nonzero(data['validation_board'],axis=1)-4;roots=[]
        for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:roots.extend(np.flatnonzero((placed>=lo)&(placed<=hi))[:2].tolist())
        for round_ in range(3):
            for i in roots:
                for m in rng.sample(range(4),4):
                    results=[]
                    for v in rng.sample(VARIANTS,len(VARIANTS)):
                        progress('相同131072次模拟：批处理搜索成本',len(searches),3*len(roots)*4*len(VARIANTS),f'{v} · {s.MODELS[m]}')
                        lib,handle=handles[v,m];lib.ot_mcts_clear_cache()
                        cells=(ctypes.c_int*64)(*map(int,data['validation_board'][i]))
                        out,sel,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
                        tick=time.perf_counter_ns()
                        assert lib.ot_nnue_mcts_play_budget(handle,cells,int(data['validation_turn'][i]),0,131072,1832,12,32,64,1,0,4194304,out,sel,visits,stats,None)==1
                        wall=(time.perf_counter_ns()-tick)/1e9
                        searches.append(dict(variant=v,model=s.MODELS[m],root=i,round=round_,wallSeconds=wall,simulations=int(stats[0]),evaluations=int(stats[1]),exact=bool(out[5]),memoryLimited=bool(stats[7])))
                        out[3]=0;stats[5]=0;results.append([list(out),list(sel),list(visits),list(stats)])
                    assert all(x==results[0] for x in results),(i,m,round_)
                    fixedParity.append(dict(model=s.MODELS[m],root=i,round=round_,exact=True))
        summary=[]
        for m,model in enumerate(s.MODELS):
            originalSeconds=sum(x['wallSeconds'] for x in searches if x['model']==model and x['variant']=='original')
            for v in VARIANTS:
                modes={}
                for mode in [1,2]:modes[str(mode)]=[statistics.median(row['groups'][g]['nanoseconds'] for row in timings if row['model']==m and row['variant']==v and row['mode']==mode) for g in range(3)]
                wall=sum(x['wallSeconds'] for x in searches if x['model']==model and x['variant']==v)
                summary.append(dict(model=model,variant=v,medianNativeNanosecondsByPhase=modes,fixedSearchSeconds=wall,fixedWorkSpeedup=originalSeconds/wall))
        for path,digest in base['preservedSha256'].items():assert s.sha(s.ROOT/path)==digest,path
        report=dict(complete=True,defaultPromoted=False,date='2026-10-30',platform=platform.platform(),models=s.MODELS,variants=VARIANTS,
            baseReportSha256=s.sha(s.REPORT),scriptSha256=s.sha(Path(__file__)),headerSha256=s.sha(header),flags=flags,
            librarySha256={v:s.sha(folders[v]/f'libstudy.{s.SUFFIX}') for v in VARIANTS},allOutputParity=parity,
            exhaustiveNormalization=dict(divisors=[2,3,4,5,6],numeratorBounds=[-196608,196608],matchesDoubleRoundAndClamp=True),
            timingSamples=timings,fixedWorkSamples=searches,fixedWorkParity=fixedParity,summary=summary,seconds=time.perf_counter()-started,
            preservedSha256=base['preservedSha256'],limits='Exploratory single-machine macOS ARM64 with nonexclusive CPU. SIMD batch lanes are four legal moves from the SAME board; unused tail lanes are zero padded. Embedding/trunk remain single-board and MCTS is unchanged. Equal-weight fixed-work searches have identical complete outputs, new-leaf evaluations and simulations. All 64-logit parity includes partial SIMD tails; real searches exercise legal-only masks. No training or strength matches; Linux/SSE2 path is implemented but unmeasured and unvalidated here.')
        s.write(REPORT,report);progress('合法着batch研究完成',1,1,'原始报告 '+str(REPORT.relative_to(s.ROOT)),True)
        print(json.dumps(summary,ensure_ascii=False))
    finally:
        for lib,handle in handles.values():lib.ot_nnue_destroy(handle)


if __name__=='__main__':main()

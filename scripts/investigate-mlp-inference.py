#!/usr/bin/env python3
"""Reproduce an isolated, weight-preserving MLP inference investigation.

Copies sources into .build; default sources/models and historical reports are
never rewritten. No training or strength match. See the JSON for raw samples.
"""
import ctypes
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.build/mlp-inference-study-20261030'
REPORT = ROOT / 'models/mlp-inference-study-2026-10-30.json'
PROGRESS = WORK / 'progress.json'
MODELS = ['pattern32-symmetry-v1.37', 'pattern32-direct-mlp8-v1.38',
          'pattern32-direct-mlp16-v1.39', 'pattern32-direct-mlp32-v1.39']
VARIANTS = ['original', 'dot4', 'integer', 'combined', 'profile']
SUFFIX = 'dylib' if sys.platform == 'darwin' else 'so'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def patched(source, variant):
    if variant in ('dot4', 'combined'):
        source = replace_once(source,
            'for(int k=0;k<policyWidth;++k)global[k]=policyHiddenBias[k]+pattern_kernels::dot(policyHidden.data()+k*74+32,h.data(),32);',
            '''for(int k=0;k<policyWidth;k+=4){std::int64_t totals[4];
            pattern_kernels::dot4(policyHidden.data()+k*74+32,h.data(),32,74,totals);
            for(int j=0;j<4;++j)global[k+j]=policyHiddenBias[k+j]+totals[j];}''')
        source = replace_once(source,
            '''for(int k=0;k<policyWidth;++k){auto total=global[k]+std::int64_t(policyHidden[k*74+64+geometry.buckets[square]])*256+
                    pattern_kernels::dot(policyHidden.data()+k*74,feature.data(),32);
                hiddenPolicy[k]=std::int16_t(std::clamp(rounded(double(total)/256),0,256));}''',
            '''for(int k=0;k<policyWidth;k+=4){std::int64_t totals[4];
                pattern_kernels::dot4(policyHidden.data()+k*74,feature.data(),32,74,totals);
                for(int j=0;j<4;++j){auto total=global[k+j]+std::int64_t(policyHidden[(k+j)*74+64+geometry.buckets[square]])*256+totals[j];
                    hiddenPolicy[k+j]=std::int16_t(std::clamp(rounded(double(total)/256),0,256));}}''')
    if variant in ('integer', 'combined'):
        # Quantized sums are integers. For odd d there are no half ties; adding
        # floor(d/2) to the absolute numerator also matches std::round exactly.
        source = replace_once(source, 'int bucket(int square){', '''template<int D> void studyFeatures(const std::int32_t* local,std::int16_t* out){
    for(int j=0;j<32;++j){int v=local[j];int r=v>=0?(v+D/2)/D:-((-v+D/2)/D);out[j]=std::int16_t(std::clamp(r,-1024,1024));}}
int bucket(int square){''')
        source = replace_once(source,
            'for(int j=0;j<32;++j)feature[j]=std::int16_t(std::clamp(rounded(double(local[square][j])/geometry.counts[square]),-1024,1024));',
            '''switch(geometry.counts[square]){
            case 2:studyFeatures<2>(local[square].data(),feature.data());break;
            case 3:studyFeatures<3>(local[square].data(),feature.data());break;
            case 4:studyFeatures<4>(local[square].data(),feature.data());break;
            case 5:studyFeatures<5>(local[square].data(),feature.data());break;
            case 6:studyFeatures<6>(local[square].data(),feature.data());break;
            default:throw std::runtime_error("Unexpected membership count");}''')
        # Only policy scaling changes; trunk/default rounding stays untouched.
        start = source.index('    if(variant==3){\n        // This is one fixed')
        policy = source[start:]
        policy = policy.replace('rounded(double(total)/256)', 'pattern_kernels::roundedDivide<256>(total)')
        policy = policy.replace('std::clamp(pattern_kernels::roundedDivide<256>(total),', 'std::clamp<std::int64_t>(pattern_kernels::roundedDivide<256>(total),')
        source = source[:start] + policy
    if variant == 'profile':
        source = '#include <chrono>\nunsigned long long study_ns[5]{};\n' + source
        source = replace_once(source,
            '    alignas(64) std::array<std::int32_t,320> sums{};',
            '''    auto study_tick=std::chrono::steady_clock::now();
    auto study_mark=[&](int stage){auto now=std::chrono::steady_clock::now();study_ns[stage]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-study_tick).count();study_tick=now;};
    alignas(64) std::array<std::int32_t,320> sums{};''')
        source = replace_once(source, '    Prediction result;alignas(64) std::array<std::int16_t,MaxHidden> h{};',
                              '    study_mark(0);\n    Prediction result;alignas(64) std::array<std::int16_t,MaxHidden> h{};')
        source = replace_once(source, '    if(variant==3){\n        // This is one fixed',
                              '    study_mark(1);\n    if(variant==3){\n        // This is one fixed')
        source = replace_once(source, '        std::array<std::int64_t,32> global{};',
                              '        study_mark(2);\n        std::array<std::int64_t,32> global{};')
        source = replace_once(source, '        while(mask){int square=first(mask);mask&=mask-1;\n            alignas(64)',
                              '        study_mark(3);\n        while(mask){int square=first(mask);mask&=mask-1;\n            alignas(64)')
        source = replace_once(source, '        return result;\n    }\n    maskedPolicy',
                              '        study_mark(4);\n        return result;\n    }\n    maskedPolicy')
    return source


def main():
    assert not REPORT.exists(), 'Research reports are immutable; choose a new run name'
    WORK.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    preserved = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'models/default-model.json', ROOT/'models/v1.37-default.json', ROOT/'src/game/wasm/othello.mjs'] + [ROOT/f'models/{m}.otpat' for m in MODELS]}
    snapshot = WORK/'source-snapshot'
    shutil.copytree(ROOT/'cpp', snapshot, dirs_exist_ok=True)
    sources = {'cpp/'+str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    def progress(stage, done=0, total=1, summary='', complete=False):
        write(PROGRESS, dict(experiment='pattern-suite-sampling', stage=stage, completed=done, total=total,
                             summary=summary, complete=complete, seconds=time.perf_counter()-started))
    config = json.loads((ROOT/'models/current-experiment.json').read_text())
    config['experiments'] = [v for v in config.get('experiments', []) if v.get('runtimeReport') != str(PROGRESS.relative_to(ROOT))]
    concurrent_experiments = list(config.get('experiments', []))
    config.setdefault('experiments', []).append(dict(name='MLP8/16/32：合法着推理、分段计时与精确优化原型', games=0,
        report=str(REPORT.relative_to(ROOT)), runtimeReport=str(PROGRESS.relative_to(ROOT))))
    write(ROOT/'models/current-experiment.json', config)
    progress('准备隔离源码')
    compiler = shutil.which('c++')
    flags = ['-O3', '-DNDEBUG', '-std=c++17', '-fPIC']
    if sys.platform == 'darwin':
        sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
        flags += ['-isysroot', sdk, '-isystem', str(Path(sdk)/'usr/include/c++/v1')]
    flags += ['-DOTELLO_PATTERN_' + v + '=1' for v in ['SIMD', 'FAST_CODES', 'BLOCK4', 'LEGAL_POLICY']]
    original = (snapshot/'pattern.cpp').read_text()
    for v in VARIANTS:
        progress('构建隔离原型', VARIANTS.index(v), len(VARIANTS), v)
        folder = WORK/v; folder.mkdir(exist_ok=True)
        shutil.copytree(snapshot, folder/'cpp', dirs_exist_ok=True)
        (folder/'cpp/pattern.cpp').write_text(patched(original, v))
        files = ['search', 'move_policy', 'nnue', 'pattern', 'cnn', 'resnet', 'mcts', 'default_engine', 'api']
        lib = folder/f'libstudy.{SUFFIX}'
        command = [compiler, *flags, '-dynamiclib' if sys.platform=='darwin' else '-shared', *[str(folder/f'cpp/{f}.cpp') for f in files], '-o', str(lib)]
        subprocess.run(command, check=True, capture_output=True)
        command = [compiler, *flags, '-I'+str(folder/'cpp'), str(ROOT/'benchmarks/mlp-inference-study.cpp'), str(lib), '-o', str(folder/'bench')]
        if v == 'profile': command.insert(1, '-DSTUDY_PROFILE=1')
        subprocess.run(command, check=True, capture_output=True)
    data = np.load(ROOT/'.build/training/v1/positions.npz')
    boards = np.concatenate([data['validation_board'], data['test_board']])
    turns = np.concatenate([data['validation_turn'], data['test_turn']])
    # Timing: fixed stratified 128 per phase, labels never read.
    placed = np.count_nonzero(data['validation_board'], axis=1)-4
    indices = np.concatenate([np.flatnonzero((placed>=lo)&(placed<hi))[:128] for lo,hi in [(0,20),(20,44),(44,61)]])
    boardfile = WORK/'boards.txt'
    boardfile.write_text(''.join(''.join('B' if x==1 else 'W' if x==-1 else '.' for x in data['validation_board'][i])+f' {int(data["validation_turn"][i])}\n' for i in indices))
    ptr = ctypes.POINTER(ctypes.c_int)
    handles = []
    def load(v, m):
        lib = ctypes.CDLL(str(WORK/v/f'libstudy.{SUFFIX}'))
        lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
        lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
        lib.ot_nnue_mcts_play_budget.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*4+[ctypes.c_uint]*2+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        handle=lib.ot_nnue_load(str(ROOT/f'models/{MODELS[m]}.otpat').encode());assert handle
        handles.append((lib,handle));return lib,handle
    def search(pair, i, ms=0, simulations=2048):
        lib,handle=pair;lib.ot_mcts_clear_cache()
        cells=(ctypes.c_int*64)(*map(int,data['validation_board'][i]))
        out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
        tick=time.perf_counter_ns()
        assert lib.ot_nnue_mcts_play_budget(handle,cells,int(data['validation_turn'][i]),ms,simulations,1832,12,32,64,1,0,4194304,out,selection,visits,stats,None)==1
        wall=(time.perf_counter_ns()-tick)/1e9
        record=dict(root=int(i),simulations=int(stats[0]),evaluations=int(stats[1]),exact=bool(out[5]),memoryLimited=bool(stats[7]),elapsedMs=out[3],wallSeconds=wall)
        out[3]=0;stats[5]=0
        return [list(out),list(selection),list(visits),list(stats)],record
    pairs = {(v,m):load(v,m) for v in VARIANTS[:-1] for m in range(4)}
    parity = []
    try:
        for m in range(4):
            progress('核验原型的全部输出和固定模拟搜索',m,4,MODELS[m])
            digests=[]
            for v in VARIANTS[:-1]:
                lib,handle=pairs[v,m];digest=hashlib.sha256();out=(ctypes.c_double*68)()
                for board,turn in zip(boards,turns):
                    cells=(ctypes.c_int*64)(*map(int,board))
                    assert lib.ot_pattern_predict(handle,cells,int(turn),out)==1
                    digest.update(bytes(out))
                digests.append(digest.hexdigest())
            assert len(set(digests))==1,(MODELS[m],digests)
            for i in range(16):
                results=[search(pairs[v,m],i)[0] for v in VARIANTS[:-1]]
                assert all(r==results[0] for r in results)
            parity.append(dict(model=MODELS[m],positions=len(boards),all68OutputSha256=digests[0],exact=True,fixedSimulationRoots=16,simulations=2048))
        rng=random.Random(20261030);timings=[]
        jobs=[(v,m,mode) for v in VARIANTS[:-1] for m in range(4) for mode in range(3)]
        for round_ in range(7):
            rng.shuffle(jobs)
            for j,(v,m,mode) in enumerate(jobs):
                progress('交错原生推理计时',round_*len(jobs)+j,7*len(jobs),f'{v} · {MODELS[m]} · mask {mode}')
                output=subprocess.check_output([str(WORK/v/'bench'),str(ROOT/f'models/{MODELS[m]}.otpat'),str(boardfile),'32768',str(mode)],text=True)
                timings.append(dict(variant=v,model=m,mode=mode,round=round_,**json.loads(output)))
        profiles=[]
        for m in range(1,4):
            for round_ in range(5):
                output=subprocess.check_output([str(WORK/'profile/bench'),str(ROOT/f'models/{MODELS[m]}.otpat'),str(boardfile),'32768','1'],text=True)
                profiles.append(dict(model=m,round=round_,**json.loads(output)))
        roots=[]
        for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:roots.extend(np.flatnonzero((placed>=lo)&(placed<=hi))[:2].tolist())
        search_samples=[]
        # Compare identical weights first, then baseline, using the same roots.
        jobs=[(v,m) for m in range(4) for v in ['original','combined']]
        for ms,rounds in [(100,3),(2000,2)]:
            for round_ in range(rounds):
                for i in roots:
                    rng.shuffle(jobs)
                    for v,m in jobs:
                        progress('相同局面交错搜索吞吐',len(search_samples),len(jobs)*len(roots)*5,f'{ms}ms · {v} · {MODELS[m]}')
                        _,sample=search(pairs[v,m],i,ms,0)
                        search_samples.append(dict(variant=v,model=m,budgetMs=ms,round=round_,**sample))
        summaries=[]
        for v in VARIANTS[:-1]:
            for m in range(4):
                modes={}
                for mode in range(3):
                    rows=[r for r in timings if r['variant']==v and r['model']==m and r['mode']==mode]
                    modes[str(mode)]=[dict(phase=g,medianNanoseconds=statistics.median(r['groups'][g]['nanoseconds'] for r in rows),meanLegalMoves=rows[0]['groups'][g]['meanLegalMoves']) for g in range(3)]
                budgets={}
                for ms in [100,2000]:
                    common=[i for i in roots if all(not s['exact'] and not s['memoryLimited'] and s['elapsedMs']>=.95*ms for s in search_samples if s['budgetMs']==ms and s['root']==i)]
                    samples=[s for s in search_samples if s['model']==m and s['variant']==v and s['budgetMs']==ms and s['root'] in common]
                    seconds=sum(s['wallSeconds'] for s in samples)
                    if seconds:budgets[str(ms)]=dict(commonRoots=common,samples=len(samples),simulationsPerSecond=sum(s['simulations'] for s in samples)/seconds,evaluationsPerSecond=sum(s['evaluations'] for s in samples)/seconds)
                summaries.append(dict(variant=v,model=MODELS[m],modes=modes,budgets=budgets))
        for p,digest in preserved.items():assert sha(ROOT/p)==digest,p
        source_changes = [p for p,digest in sources.items() if sha(ROOT/p)!=digest]
        report=dict(complete=True,defaultPromoted=False,date='2026-10-30',platform=platform.platform(),compiler=compiler,compilerVersion=subprocess.check_output([compiler,'--version'],text=True),flags=flags,
            models=MODELS,variants=VARIANTS,modeLegend={'0':'value/trunk, zero policy mask','1':'actual legal moves','2':'all 64 squares'},phaseLegend=['placements 0..19','placements 20..43','placements 44..60'],
            datasetSha256=sha(ROOT/'.build/training/v1/positions.npz'),timingValidationIndices=indices.tolist(),preservedSha256=preserved,sourceSha256=sources,externalSourceChangesDuringRun=source_changes,concurrentExperimentsAtStart=concurrent_experiments,
            scriptSha256=sha(Path(__file__)),harnessSha256=sha(ROOT/'benchmarks/mlp-inference-study.cpp'),
            librarySha256={v:sha(WORK/v/f'libstudy.{SUFFIX}') for v in VARIANTS},parity=parity,summary=summaries,timingSamples=timings,profileSamples=profiles,
            profileStages=['encode/gather','pool/trunk/value head','local feature accumulation','global MLP term','per-square normalization + MLP'],searchSamples=search_samples,
            seconds=time.perf_counter()-started,limits='Native macOS ARM64 only; no strength matches or training. Profile clock overhead is included; absolute inference latency comes from uninstrumented builds. Equal-weight output/fixed-search parity isolates implementation effects; cross-model search throughput also depends on traversal. Synthetic zero-mask timing is an ablation, not MCTS runtime. Concurrent phase-value training is registered in the workspace; CPU is not exclusive and system-wide background activity could not be inspected in the sandbox. Treat new timings as exploratory relative measurements, not an idle-machine release benchmark.')
        write(REPORT,report);progress('已完成',1,1,'报告 '+str(REPORT.relative_to(ROOT)),True)
        print(json.dumps(dict(report=str(REPORT),seconds=report['seconds'],summary=summaries),ensure_ascii=False))
    finally:
        for lib,handle in handles:lib.ot_nnue_destroy(handle)


if __name__=='__main__':
    main()

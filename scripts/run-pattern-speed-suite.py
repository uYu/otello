#!/usr/bin/env python3
"""Select an exact optimization suite, measure 2s throughput, run 20 paired games."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import platform
import tarfile
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-speed-suite-v1.35'
SUFFIX='dylib' if platform.system()=='Darwin' else 'so'
LIB=ROOT/f'.build/native-pattern-suite-v1.35/libothello.{SUFFIX}'
REFERENCE=ROOT/f'.build/native-v1.25-reference/libothello.{SUFFIX}'
MODEL=ROOT/'models/pattern32-outcome-v1.25.otpat'
INT8=ROOT/'models/pattern32-int8-step1-v1.32.otpat'
SAMPLING=ROOT/'models/v1.35-pattern-speed-suite-sampling.json'
SUITE=ROOT/'models/v1.35-pattern-speed-suite.json'
RUNTIME=WORK/'match-2s-20.json'
REPORT='models/v1.35-speed-suite-vs-v1.25-2s-20.json'
CONFIG=ROOT/'models/v1.35-pattern-speed-suite-experiments.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('suite_comparison',ROOT/'scripts/compare-nnue-candidate.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
spec=importlib.util.spec_from_file_location('suite_helpers',ROOT/'scripts/benchmark-pattern-int8.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def monitor(matching=False,label='策略集'):
    entry=dict(name=f'{label} vs v1.25 · 2秒/步 · 20局',shortName=f'{label} vs v1.25 · 2秒/步 · 20局',
        displayNames=dict(candidate=label,baseline='v1.25'),displayOrder=['candidate','baseline'],games=20,
        runtimeReport=str((RUNTIME if matching else WORK/'progress.json').relative_to(ROOT)),report=REPORT)
    value=dict(version='v1.35 Pattern 提速策略集',experiments=[entry]);write(CONFIG,value)
    path=ROOT/'models/current-experiment.json';current=read(path)
    current['version']=value['version'];current['experiments']=[entry]+[item for item in current['experiments'] if item.get('report')!=REPORT]
    write(path,current)


def sampling():
    assert not SAMPLING.exists() and not SUITE.exists() and not RUNTIME.exists()
    WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();monitor()
    baseline=read(ROOT/'models/v1.25-default.json');assert sha(REFERENCE)==baseline['nativeLibrarySha256'] and sha(MODEL)==baseline['modelSha256']
    cache=ROOT/'.build/training/v1/positions.npz';assert sha(cache)==read(ROOT/'models/nnue256-v1.json')['dataset']['cacheSha256']
    data=np.load(cache);boards,turns=data['validation_board'],data['validation_turn'];occupied=np.count_nonzero(boards,axis=1)
    fixtures=[]
    for low,high in [(4,13),(14,23),(24,33),(34,43)]:
        indices=[int(i) for i in np.flatnonzero((occupied>=low)&(occupied<=high)) if comparison.match.rules.legal(boards[i].tolist(),int(turns[i]))][:2]
        assert len(indices)==2
        fixtures.extend(indices)
    modes=[('approved v1.25',REFERENCE,MODEL),('suite INT16',LIB,MODEL),('suite INT8',LIB,INT8)];loaded=[]
    def progress(stage,n=0,total=1,summary='',complete=False):
        write(WORK/'progress.json',dict(experiment='pattern-suite-sampling',stage=stage,completed=n,total=total,summary=summary,complete=complete,seconds=time.perf_counter()-started))
    try:
        for _,library,model in modes:loaded.append(helpers.load(library,model))
        progress('与批准基线核对固定模拟结果')
        for index in fixtures:
            b=(ctypes.c_int*64)(*map(int,boards[index]));t=int(turns[index])
            a=helpers.search(*loaded[0],b,t,simulations=2048);c=helpers.search(*loaded[1],b,t,simulations=2048)
            assert a==c,'INT16 suite changed approved baseline search semantics'
        rows=[[] for _ in modes];rng=np.random.default_rng(35123)
        for iteration in range(2):
            for shuffled in rng.permutation(len(modes)*len(fixtures)):
                mode,fixture=divmod(int(shuffled),len(fixtures));index=fixtures[fixture];lib,handle=loaded[mode]
                progress('同局面 2秒采样吞吐对照',iteration*24+int(shuffled)+1,48,f'第 {iteration+1}/2 轮 · {modes[mode][0]} · 占据 {occupied[index]} 格；INT8 仅在吞吐领先 INT16 >5% 时选择')
                cells=(ctypes.c_int*64)(*map(int,boards[index]));out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
                lib.ot_mcts_clear_cache();tick=time.perf_counter_ns()
                assert lib.ot_nnue_mcts_play(handle,cells,int(turns[index]),2000,0,1832,12,32,64,out,selection,visits,stats)==1
                wall=(time.perf_counter_ns()-tick)/1e6
                assert out[0] in comparison.match.rules.legal(boards[index].tolist(),int(turns[index]))
                rows[mode].append(dict(round=iteration,fixture=fixture,datasetIndex=index,occupied=int(occupied[index]),simulations=int(stats[0]),evaluations=int(stats[1]),elapsedMs=out[3],wallMs=wall,exact=bool(out[5]),memoryLimited=bool(stats[7]),move=out[0]))
        common=[fixture for fixture in range(len(fixtures)) if all(not s['exact'] and not s['memoryLimited'] and s['elapsedMs']>=1900 for samples in rows for s in samples if s['fixture']==fixture)]
        assert len(common)>=4,'Insufficient non-proven time-limited fixtures; investigate before selection'
        results=[]
        for (name,library,model),samples in zip(modes,rows):
            selected=[s for s in samples if s['fixture'] in common];nodes=sum(s['simulations'] for s in selected);wall=sum(s['wallMs'] for s in selected)
            results.append(dict(name=name,library=str(library.relative_to(ROOT)),librarySha256=sha(library),model=str(model.relative_to(ROOT)),modelSha256=sha(model),modelBytes=model.stat().st_size,commonTimedSimulations=nodes,commonWallMs=wall,simulationsPerSecond=nodes*1000/wall,samples=samples))
        choose8=results[2]['simulationsPerSecond']>1.05*results[1]['simulationsPerSecond'];chosen=2 if choose8 else 1
        for row in results:row['throughputVsApproved']=row['simulationsPerSecond']/results[0]['simulationsPerSecond']
        report=dict(version='v1.35-pattern-speed-suite',status='candidate',defaultPromoted=False,results=results,selected=results[chosen]['name'],selectedIndex=chosen,selectionRule='INT8 only if >5% higher 2s sampling throughput than INT16 on common uncapped/non-proven fixtures; otherwise retain exact approved INT16 weights. Rule specified before measurement.',commonFixtureIndices=common,fixtureDatasetIndices=fixtures,datasetSha256=sha(cache),millisecondsPerRoot=2000,rounds=2,scriptSha256=sha(Path(__file__)),limits='Fixed validation fixtures, no outcome labels used for selection; actual game trajectories and proof stopping differ. No throughput multiplication across earlier reports.')
        write(SAMPLING,report)
        suite=dict(version='v1.35-pattern-speed-suite',status='candidate',defaultPromoted=False,baselineManifest='models/v1.25-default.json',baselineManifestSha256=sha(ROOT/'models/v1.25-default.json'),selected=results[chosen],
            enabled=['fused current-side value/policy inference','integer NEON on ARM64 / SSE2 on x86-64 with portable fallback','packed ternary encoding','four-row SIMD with shared input loads','legal-only policy projection']+(['INT8 embedding table'] if choose8 else []),
            excluded=['incremental state cache (slower)','integer rounding (slower)','H64 width (no approved-baseline benefit)']+([] if choose8 else ['INT8 embedding (no >5% sampling speed benefit)']),
            buildFlags=dict(OTELLO_PATTERN_SIMD='ON',OTELLO_PATTERN_FAST_CODES='ON',OTELLO_PATTERN_BLOCK4='ON',OTELLO_PATTERN_LEGAL_POLICY='ON',OTELLO_PATTERN_INTEGER_ROUNDING='OFF'),patternCacheEntries=0,
            match=dict(millisecondsPerMove=2000,games=20,seed=20261029,start='initial',useBook=False,pairedColors=True,openingPolicy=comparison.match.POLICY,runtimeReport=str(RUNTIME.relative_to(ROOT)),finalReport=REPORT),
            samplingReport=str(SAMPLING.relative_to(ROOT)),samplingSha256=sha(SAMPLING),preservedSha256={p:sha(ROOT/p) for p in ['models/default-model.json','models/v1.25-default.json','src/game/wasm/othello.mjs',str(MODEL.relative_to(ROOT)),str(INT8.relative_to(ROOT))]})
        write(SUITE,suite)
        # Freeze exactly the implementation and runner used before beginning the match.
        archive=ROOT/'models/archive/v1.35-pattern-speed-suite-sources.tar.gz';assert not archive.exists()
        sources=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/p for p in ['AGENTS.md','CMakeLists.txt','.github/workflows/check.yml','data/README.md','scripts/run-pattern-speed-suite.py','scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py','scripts/benchmark-pattern-int8.py','scripts/pattern_model.py','scripts/watch-matches.py','scripts/watch-match.sh','scripts/test-native.sh','tests/native.cpp','tests/pattern_training_test.py','benchmarks/pattern-hotpath.cpp','benchmarks/pattern-legal.cpp']]))
        with tarfile.open(archive,'w:gz') as out:
            for path in sources:out.add(path,arcname=str(path.relative_to(ROOT)))
        write(archive.with_name('v1.35-pattern-speed-suite-sources.json'),dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in sources}))
        progress('采样完成，开始 2秒/步 20局',1,1,f"选中 {report['selected']}；吞吐 {results[chosen]['throughputVsApproved']:.3f}×批准基线",True)
        return suite
    finally:
        for lib,handle in loaded:lib.ot_nnue_destroy(handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.resume:
        suite=read(SUITE)
        for p,digest in suite['preservedSha256'].items():assert sha(ROOT/p)==digest
        assert sha(LIB)==suite['selected']['librarySha256']
    else:suite=sampling()
    label='策略集 '+('INT8' if 'INT8' in suite['selected']['name'] else 'INT16');monitor(True,label)
    comparison.compare(argparse.Namespace(model=ROOT/suite['selected']['model'],baseline_manifest=ROOT/'models/v1.25-default.json',baseline_library=REFERENCE,label='v1.35 speed suite '+label.split()[-1],library=LIB,output=RUNTIME,games=20,milliseconds=2000,seed=20261029,resume=args.resume))


if __name__=='__main__':main()

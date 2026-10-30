#!/usr/bin/env python3
"""Run a fresh 5s/20-game deep-vs-v1.9 comparison and publish audited results."""
import hashlib,json,signal,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME='.build/training/deep/v1.10-deep-vs-v1.9-5s-run1.json'
REPORT='models/v1.10-deep-vs-v1.9-5s-run1.json'
MANIFEST='models/v1.10-deep-5s-run1-experiment.json'
DOC='models/v1.10-deep-5s-run1-comparison.md'

def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def save(path,value):
    target=ROOT/path;temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(target)

def main():
    for name in (RUNTIME,REPORT,MANIFEST,DOC):
        if (ROOT/name).exists():raise ValueError('Choose a fresh run; refusing to overwrite '+name)
    baseline=json.loads((ROOT/'models/v1.9-default.json').read_text())
    training=json.loads((ROOT/'models/nnue256-deep-v1.10.json').read_text())
    previous=json.loads((ROOT/'models/v1.10-deep-vs-v1.9-1s.json').read_text())
    engineering=json.loads((ROOT/'models/engineering-rerun1.json').read_text())
    snapshot=json.loads((ROOT/engineering['sourceManifest']).read_text())
    assert sha(baseline['defaultModel'])==baseline['modelSha256']
    assert sha('models/nnue256-deep-v1.10.otnnue')==training['sha256']
    assert sha('.build/native-v1.9-reference/libothello.dylib')==baseline['nativeLibrarySha256']
    assert sha('.build/native-engineering/libothello.dylib')==engineering['optimizedLibrarySha256']
    measured={p:digest for p,digest in snapshot['files'].items() if p.startswith('cpp/') or p in ('scripts/compare-nnue-candidate.py','scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py')}
    assert all(sha(p)==digest for p,digest in measured.items())
    command=[sys.executable,'scripts/compare-nnue-candidate.py','--model','models/nnue256-deep-v1.10.otnnue',
        '--label','v1.10-deep-5s-run1','--baseline-manifest','models/v1.9-default.json',
        '--library','.build/native-engineering/libothello.dylib','--baseline-library','.build/native-engineering/libothello.dylib',
        '--games','20','--milliseconds','5000','--seed',str(previous['baseSeed']),'--output',RUNTIME]
    child=subprocess.Popen(command,cwd=ROOT)
    (ROOT/'.build/training/deep/5s-run1.pid').write_text(str(child.pid)+'\n')
    try:
        code=child.wait()
    except KeyboardInterrupt:
        child.send_signal(signal.SIGINT);child.wait();raise
    if code:raise subprocess.CalledProcessError(code,command)
    report=json.loads((ROOT/RUNTIME).read_text());assert report['complete'] and report['games']==20
    report['baselineManifest']='models/v1.9-default.json'
    report['configs'][1]['model']='models/nnue256-deep-v1.10.otnnue'
    report['publication']=dict(runtimeReportSha256=sha(RUNTIME),separateFrom='models/v1.10-deep-vs-v1.9-1s.json',
        notes='Fresh 5s/20-game comparison using the same optimized engine on BOTH models, original weights, seeds and budgets; excludes all earlier stopped games.',publicationScriptSha256=sha('scripts/run-deep-5s.py'))
    save(REPORT,report);c=report['results']['candidate'];b=report['results']['baseline']
    (ROOT/DOC).write_text(f'''# 更深主干 vs v1.9：独立重跑\n\n双方每步各 5 秒，20 局交换黑白，从标准初始局面开始，不用开局库。更深模型 **{c['win']} 胜、{c['draw']} 平、{c['loss']} 负，得分率 {c['scoreRate']:.1%}，平均子差 {c['meanDiscDifference']:+.2f}**。\n\n使用原模型权重、对战运行器、预算与随机种子；双方均使用同一个优化引擎，已接入合并双头推理、复用合法掩码、倍增合法点与射线表翻子，没有重新训练。优化引擎已在两份模型上验证与旧库的固定模拟走法、评分和根访问完全一致。之前的 1 秒组不计入本报告；时限控制会使不同运行的搜索量和走子产生变化。默认仍为 v1.9。\n\n| 指标 | 更深主干 | v1.9 |\n| --- | ---: | ---: |\n| 胜／平／负 | {c['win']}／{c['draw']}／{c['loss']} | {b['win']}／{b['draw']}／{b['loss']} |\n| 得分率 | {c['scoreRate']:.1%} | {b['scoreRate']:.1%} |\n| 平均子差 | {c['meanDiscDifference']:+.2f} | {b['meanDiscDifference']:+.2f} |\n| 平均每步耗时 | {c['meanWallMs']:.1f}ms | {b['meanWallMs']:.1f}ms |\n| 平均每步模拟数 | {c['meanSearchUnits']:,.0f} | {b['meanSearchUnits']:,.0f} |\n\n完整棋谱共 {report['replayedActions']} 步，已逐步审计。20 局且起点相同，不推算 Elo；原训练与结构说明见 `v1.10-deep-comparison.md`。实测代码见 `archive/engineering-rerun1-sources.*`，工程校验见 `engineering-rerun1.json`。\n\n统一监控 `sh scripts/watch-match.sh`，棋盘使用 `*` 黑、`o` 白、`.` 空；完成后保留终盘。核对命令：`python3 scripts/verify-deep-experiment.py --experiment {MANIFEST}`。\n''')
    files=['models/nnue256-deep-v1.10.otnnue','models/nnue256-deep-v1.10.pt','models/nnue256-deep-v1.10.json',REPORT,DOC,
        'models/v1.10-deep-legacy-parity.json','models/archive/engineering-rerun1-sources.json','models/archive/engineering-rerun1-sources.tar.gz','models/engineering-rerun1.json']
    save(MANIFEST,dict(version='v1.10-deep-5s-run1',status='candidate',defaultPromoted=False,
        protocol=dict(games=20,completedGames=20,millisecondsPerMove=5000),baselineManifest='models/v1.9-default.json',
        model=files[0],trainingReport=files[2],matchReport=REPORT,sourceManifest=engineering['sourceManifest'],engineeringManifest='models/engineering-rerun1.json',engineeringManifestSha256=sha('models/engineering-rerun1.json'),
        legacyParity='models/v1.10-deep-legacy-parity.json',artifactSha256={p:sha(p) for p in files},
        results=report['results'],replayedActions=report['replayedActions'],runtimeReportSha256=sha(RUNTIME),
        separateFrom='models/v1.10-deep-experiment.json'))
    subprocess.run([sys.executable,'scripts/verify-deep-experiment.py','--experiment',MANIFEST],cwd=ROOT,check=True)
    print(json.dumps(dict(complete=True,games=20,results=c,report=REPORT),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

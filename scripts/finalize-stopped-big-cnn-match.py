#!/usr/bin/env python3
"""Audit and preserve the interrupted v1.26 match after its engines have exited."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('light_comparison',ROOT/'scripts/compare-big-cnn-candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
m=c.match
runtime=ROOT/'.build/training/big-cnn-match-v1.26/bigcnn-vs-v1.9-b32-2s-20-run1.json'
target=ROOT/'models/v1.26-bigcnn-vs-v1.9-b32-2s-20-run1.json'
assert not target.exists()
r=json.loads(runtime.read_text());assert not r['complete']
live=runtime.with_suffix('.live.json');l=json.loads(live.read_text())
assert r['games']==len(r['records'])<r['plannedGames']
m.CONFIGS=r['configs'];actions=sum(m.replay(g) for g in r['records'])
totals,matrix=m.aggregate(r['records']);assert totals==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']
archive=ROOT/r['batchProtocol']['sourceManifest'];source=json.loads(archive.read_text())
assert m.sha(ROOT/source['archive'])==source['sha256']
assert all(m.sha(ROOT/p)==digest for p,digest in source['files'].items())
assert m.sha(ROOT/'.build/native-product-nnue-v1.22/libothello.dylib')==r['librarySha256']==r['baselineLibrarySha256']
for conf in r['configs']:assert m.sha(ROOT/conf['model'])==conf['modelSha256']
original=runtime.parent/'stopped-original-report.json';original.write_bytes(runtime.read_bytes())
original_live=runtime.parent/'stopped-original-live.json';original_live.write_bytes(live.read_bytes())
r.update(complete=False,stopped=True,stopReason='User requested stopping: CNN search sampling rate is much lower than v1.9',
         currentGame=None,interruptedGame=l.get('currentGame'),
         publication=dict(auditedActions=actions,defaultPromoted=False,finalizerSha256=m.sha(Path(__file__)),
             originalReportSha256=m.sha(original),originalLiveSha256=m.sha(original_live)))
m.rules.write_json(target,r)
x=totals['candidate'];b=totals['baseline'];doc=ROOT/'models/v1.26-big-cnn-2s-20-comparison.md'
doc.write_text(f'''# 大 CNN vs v1.9：按用户要求提前停止

原定双方每步2秒、batch32、20局交换黑白，不使用开局库；完成{r['games']}局后按用户要求停止。
大 CNN **{x['win']}胜、{x['draw']}平、{x['loss']}负，得分率{x['scoreRate']:.1%}，平均子差{x['meanDiscDifference']:+.2f}**。
首局平均每步模拟数：v1.9为{b['meanSearchUnits']:,.0f}，CNN为{x['meanSearchUnits']:,.0f}，相差{b['meanSearchUnits']/max(1,x['meanSearchUnits']):.1f}倍。这是整局平均，包含提前证明与内存限额，不等同于纯推理吞吐基准。
未完成的当前局不计入结果；原始中断现场单独保留。最后一局完整终盘保留在统一监控。

完整棋谱共{actions}步已重放审计，战绩重新汇总；这是提前停止的小样本，不能称为完成20局。
候选为v1.24 C256/R8/H256第11轮联合最优权重，双方同一原生MCTS与batch32调度；v1.9使用单线程Accelerate，CNN使用FP32 MPS回调。默认保持v1.9。
''')
manifest=dict(status='candidate',stopped=True,defaultPromoted=False,
    protocol=dict(plannedGames=r['plannedGames'],completedGames=r['games'],millisecondsPerMove=2000),
    batchProtocol=r['batchProtocol'],matchReport=str(target.relative_to(ROOT)),results=totals,replayedActions=actions,
    artifactSha256={str(p.relative_to(ROOT)):m.sha(p) for p in (target,doc,archive,ROOT/source['archive'],Path(__file__))})
m.rules.write_json(ROOT/'models/v1.26-big-cnn-2s-20-experiment.json',manifest)
config_path=ROOT/'models/current-experiment.json';config=json.loads(config_path.read_text())
for item in config['experiments']:
    if item.get('runtimeReport')==str(runtime.relative_to(ROOT)):item['status']='stopped'
m.rules.write_json(config_path,config)
# Preserve the interrupted runtime as stopped; training is managed separately.
m.rules.write_json(runtime,r)
l=dict(r);l.pop('records');m.rules.write_json(live,l)
print(json.dumps(dict(stopped=True,completed=r['games'],candidate=x)),flush=True)

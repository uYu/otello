#!/usr/bin/env python3
"""Finalize completed KD run while auditing an externally changed default pointer."""
import importlib.util,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('distill100',ROOT/'scripts/distill-pattern-outcome100.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
work=ROOT/'.build/training/pattern-cnn-distill-v1.28-run1'
p=json.loads((work/'protocol.json').read_text())
assert all(d.sha(ROOT/f)==h for f,h in p['sources'].items())
changes=[]
for f,h in p['preservedSha256'].items():
    actual=d.sha(ROOT/f)
    if actual!=h:
        assert f=='models/default-model.json',f
        changes.append(dict(path=f,beforeSha256=h,afterSha256=actual,currentContent=json.loads((ROOT/f).read_text())))
assert len(changes)==1
# The new pointer still selects exactly the frozen parent used by this experiment.
assert changes[0]['currentContent']['modelSha256']==p['parentSha256']
c=json.loads((ROOT/'models/pattern32-cnn-distill100-v1.28.json').read_text())
assert d.sha(ROOT/c['model'])==c['sha256']
assert all(x['positions']==6000 and x['maxOutcomeError']<2e-5 and x['maxPolicyLogitError']<2e-5 for x in c['cppParity'].values())
training=json.loads((work/'training-checkpoints.json').read_text());assert len(training)==1 and len(training[0]['history'])==11
parent=json.loads((ROOT/'models/pattern32-outcome-v1.25.json').read_text());prior=json.loads((ROOT/'models/v1.27-pattern-cnn-distill.json').read_text())
progress=json.loads((work/'progress.json').read_text())
r=dict(**p,complete=True,experiment='cnn-offline-training',candidates=[c],trainingReports=training,parentMetrics=parent['metrics'],baselineMetrics=prior['baselineMetrics'],seconds=progress['seconds'],concurrentExternalChanges=changes,finalizationSourceSha256=d.sha(Path(__file__)))
d.write(work/'report.json',r);d.write(ROOT/'models/v1.28-pattern-cnn-distill.json',r)
lines=['# 100% CNN教师 → outcome Pattern：离线指标','','价值与策略100%使用固定CNN第11轮软标签，温度2；WDL辅助头沿用原始赛果。学生从原v1.25权重微调10轮，其他配置与v1.27三组一致。按原始赛果验证MSE选第8轮，测试集不参与选模。验证/测试各6000局面均通过原生C++一致性检查。','',
'训练期间共享默认指针被外部工作更新为原v1.25 Pattern；本实验未修改或回退该配置。下表v1.9保留作历史比较。','',
'| 模型 | 测试胜负MSE ↓ | 非和局准确率 ↑ | 走法命中率 ↑ | 走法交叉熵 ↓ |','|---|---:|---:|---:|---:|']
b=prior['baselineMetrics']['test'];lines.append(f"| v1.9 | {b['outcomeMse']:.6f} | {b['nonDrawSignAccuracy']:.2%} | {b['policyTop1']:.2%} | {b['policyCrossEntropy']:.6f} |")
for name,s in [('原 outcome Pattern',parent['metrics']['test'])]+[(x['name'],x['metrics']['test']) for x in prior['candidates']+[c]]:
    lines.append(f"| {name} | {s['mse']:.6f} | {s['nonDrawSignAccuracy']:.2%} | {s['policyTop1']:.2%} | {s['policyCrossEntropy']:.6f} |")
doc=ROOT/'models/v1.28-pattern-cnn-distill-comparison.md';doc.write_text('\n'.join(lines)+'\n')
archive=ROOT/'models/archive/v1.28-pattern-cnn-distill-sources.tar.gz'
artifacts=[ROOT/'models/v1.28-pattern-cnn-distill.json',doc]+list((ROOT/'models').glob('pattern32-*-v1.28.*'))
d.write(ROOT/'models/archive/v1.28-pattern-cnn-distill-sources.json',dict(archive=str(archive.relative_to(ROOT)),sha256=d.sha(archive),sources=p['sources'],finalizationSource=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=d.sha(Path(__file__))),artifactSha256={str(x.relative_to(ROOT)):d.sha(x) for x in artifacts},preservedSha256=p['preservedSha256'],concurrentExternalChanges=changes))
shown=dict(c,channels=32,metrics={split:dict(value=dict(outcomeMse=s['mse'],nonDrawSignAccuracy=s['nonDrawSignAccuracy']),policy=dict(top1=s['policyTop1'])) for split,s in c['metrics'].items()})
progress.pop('error',None);progress.update(stage='complete',complete=True,candidates=[shown],concurrentExternalChanges=changes);d.write(work/'progress.json',progress)
print(doc.read_text())

#!/usr/bin/env python3
"""Freeze the independent Pattern inference/width experiment and its validation."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
r=read(ROOT/'models/v1.29-pattern-width.json');speed=read(ROOT/'models/v1.29-pattern-inference-benchmark.json');widthspeed=read(ROOT/'models/v1.29-pattern-width-benchmark.json')
assert r['baselineManifestSha256']==sha(ROOT/r['baselineManifest'])
assert r['status']=='candidate' and not r['defaultPromoted']
a,b=r['variants'];assert a['width']==32 and b['width']==64
assert a['dataOrderAndAugmentationSha256']==b['dataOrderAndAugmentationSha256'] and a['initialEmbeddingSha256']==b['initialEmbeddingSha256']
assert b['parameters']-a['parameters']==4608 and b['bytes']-a['bytes']==9280
for v in r['variants']:
 assert sha(ROOT/v['model'])==v['modelSha256']
 assert v['bestEpoch']==min(v['history'],key=lambda h:h['mse'])['epoch']
 for key in ['validationParity','testParity']:assert v[key]['positions']==6000 and v[key]['maxOutcomeError']<2e-5 and v[key]['maxPolicyLogitError']<2e-5
old_manifest=ROOT/'models/v1.29-pattern-width-experiment.json'
archived_files=read(ROOT/read(old_manifest)['sourceManifest'])['files'] if old_manifest.exists() else {}
for p,digest in r['scriptSha256'].items():assert (archived_files[p] if archived_files else sha(ROOT/p))==digest,p
assert speed['jointOutputsExactlyEqual'] and speed['fixedSimulationMctsExactlyEqual']
assert speed['results'][-1]['librarySha256']==widthspeed['librarySha256']==r['librarySha256']
base=read(ROOT/'models/pattern32-outcome-v1.25.json');baseline=read(ROOT/r['baselineManifest'])
summary=ROOT/'models/v1.29-pattern-width-comparison.md'
text='''# Pattern 简单结构与融合推理实验（v1.29）

算法来源沿用原棋形实验：受 Rapfi 启发、独立实现的黑白棋网络；参考链接见 `v1.8-comparison.md`。
默认仍为已批准、冻结的 v1.25，候选未自动提升。

## 数据增强已经启用

训练从四种旋转 × 是否镜像中为每个采样局面随机选择一种，同时变换棋盘、实战下一手和合法着掩码。
每轮仍遍历百万原始训练局面；八种变换全部遍历会令训练工作量约增至八倍，不会增加独立棋谱数量。
本轮 H32/H64 使用相同样本顺序和相同对称序列，其流式 SHA-256 相同；不在此次对照中更换增强方法。

## 推理实现

新联合路径只汇总当前行棋方的棋形，保留查表结果供策略投影复用，避免价值/策略重复编码与对方无用汇总；格点计数与位置桶预计算。
64 个冻结验证局面的全部 68 项联合输出逐项完全相同；固定每局面 1,024 次模拟时，落点、候选值、访问数和选择元数据完全一致。
这项优化不改变模型权重或搜索规则。九轮交错测量，每轮 8,192 次，包含 Python/C ABI 开销。

| 同一批准模型 | 联合推理中位数 |
| --- | ---: |
'''
for row in speed['results']:text+=f"| {row['name']} | {row['medianMicroseconds']:.3f}µs |\n"
text+=f"\n相对原批准库，测得吞吐为 {speed['speedupVsApproved']:.3f} 倍；相对同版未融合实现为 {speed['speedupVsPreFusion']:.3f} 倍。固定模拟的单次计时仅作诊断，不作为稳定吞吐或棋力结论。\n"
text+='''
## 主干宽度对照

仅把共享主干从 32 维扩大到 64 维，查表维度仍为 32、棋形与聚合方式不变。
两组从随机参数训练 15 轮，初始 embedding 完全相同；AdamW lr 0.001、batch 2048，
胜负 MSE + 0.1 WDL CE + 0.1 实战策略 CE。每组按最低验证胜负 MSE 选权重；两组均选定并保存后才评估测试集。
验证和测试分别 6,000 个局面，均通过三头 Python/C++ 对照。

| 模型 | 参数 | 字节 | 验证胜负 MSE | 测试胜负 MSE | 测试下一手命中率 | 融合推理 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
rows=[dict(name='已批准 v1.25',parameters=base['parameters'],bytes=base['bytes'],metrics=base['metrics'])]+r['variants']
for row,timing in zip(rows,widthspeed['results']):
 text+=f"| {row['name']} | {row['parameters']:,} | {row['bytes']:,} | {row['metrics']['validation']['mse']:.6f} | {row['metrics']['test']['mse']:.6f} | {row['metrics']['test']['policyTop1']:.2%} | {timing['medianMicroseconds']:.3f}µs |\n"
text+='''
H64 比 H32 多 4,608 个参数、9,280 字节；int32 偏置使字节增量不等于参数增量乘二。
这是单种子离线结构对照，推理速度以本机交错计时为准；还未进行候选棋力对战。
新原生库单独保存在 `.build/native-pattern-h64-v1.29/`，当前默认原生参考库和浏览器 Wasm 保持冻结。
复核：`python3 scripts/finalize-pattern-width.py`；该命令会重新审计报告，但不覆盖已冻结源码归档。
'''
if summary.exists():assert summary.read_text()==text,'Summary would change; investigate before replacing'
else:summary.write_text(text)
manifest=ROOT/'models/v1.29-pattern-width-experiment.json'
artifacts=[ROOT/'models/v1.29-pattern-width.json',ROOT/'models/v1.29-pattern-inference-benchmark.json',ROOT/'models/v1.29-pattern-width-benchmark.json',summary,ROOT/'models/v1.29-pattern-width-experiments.json']
for v in r['variants']:
 p=ROOT/v['model'];artifacts.extend([p,p.with_suffix('.json'),p.with_suffix('.pt')])
archive=ROOT/'models/archive/v1.29-pattern-width-sources.tar.gz';source_manifest=archive.with_name('v1.29-pattern-width-sources.json')
if not manifest.exists():
 assert not archive.exists()
 sources=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/p for p in ['AGENTS.md','CMakeLists.txt','data/README.md','models/training-data-manifest.json','requirements-training.txt','scripts/pattern_model.py','scripts/train-pattern-width.py','scripts/train-pattern.py','scripts/train-pattern-outcome.py','scripts/benchmark-pattern-inference.py','scripts/benchmark-pattern-width.py','scripts/finalize-pattern-width.py','scripts/watch-matches.py','scripts/watch-match.sh','tests/native.cpp','tests/pattern_training_test.py']]))
 with tarfile.open(archive,'w:gz') as out:
  for p in sources:out.add(p,arcname=str(p.relative_to(ROOT)))
 write(source_manifest,dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in sources}))
 write(manifest,dict(version='v1.29-pattern-width',status='candidate',defaultPromoted=False,baselineManifest=r['baselineManifest'],baselineManifestSha256=r['baselineManifestSha256'],sourceManifest=str(source_manifest.relative_to(ROOT)),artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},preservedSha256={p:sha(ROOT/p) for p in ['models/default-model.json','models/v1.25-default.json','models/pattern32-outcome-v1.25.otpat','src/game/wasm/othello.mjs']},checks=dict(native=True,sanitizers=True,patternTests=4,fullJointParity=True,fixedSimulationParity=True)))
frozen=read(manifest);snapshot=read(ROOT/frozen['sourceManifest']);assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
with tarfile.open(ROOT/snapshot['archive'],'r:gz') as stream:
 for p,digest in snapshot['files'].items():assert hashlib.sha256(stream.extractfile(p).read()).hexdigest()==digest,p
for p,digest in {**frozen['artifactSha256'],**frozen['preservedSha256']}.items():assert sha(ROOT/p)==digest,p
print('Verified H32/H64 shared D4 sequence, 6000+6000 parity per model, fused inference equality and preserved v1.25 default')

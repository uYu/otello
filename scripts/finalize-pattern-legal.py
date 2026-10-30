#!/usr/bin/env python3
"""Freeze legal-only policy projection, measurements and untouched baseline."""
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
report=ROOT/'models/v1.34-pattern-legal.json';r=read(report)
assert not r['defaultPromoted']
for key in ['fullDiagnosticOutputExactlyEqual','legalMaskedLogitsNativeTestExactlyEqual','coldSearchExactlyEqual','reusedSubtreeSearchExactlyEqual','incrementalStateSearchExactlyEqual','forcedPassAndTerminal']:assert r['parity'][key]
for p,digest in r['preservedSha256'].items():assert sha(ROOT/p)==digest,p
summary=ROOT/'models/v1.34-pattern-legal-comparison.md'
text='''# 合法着策略投影稀疏计算 v1.34

检查发现，原 MCTS 已经只对合法着计算 softmax，先验也已经保存于树中；没有额外的非法着指数或重复归一化可去除。
本轮只优化其上游：Pattern 仍对所有 38 行汇总价值特征，但仅保留触及请求掩码的棋形行用于策略投影，
把这些行按原棋形顺序紧凑排列后交给原四行 SIMD，只向请求落点汇总并平均 logits。
棋形掩码、每格的完整棋形计数和位置桶静态缓存；不增加逐节点持久缓存或动态分配。

新增 `predictLegal(Position, mask)` 及增量状态版本；所有 46 个值/门控输出不变，请求外 logits 为零。
原完整 `predict`、策略诊断 C ABI 和全部 64 格原输出不变。
MCTS 通过实验开关 `OTELLO_PATTERN_LEGAL_POLICY` 使用已有合法着掩码调用新方法，
保持原 softmax 的最大值、指数、累加及归一化顺序，以及原先验存储不变。
强制停着没有投影或 logits 汇总，仍保留相同的价值输出和先验 1；终局处理保持原样。

这是一种按需求跳算的稀疏计算。原棋形输入本来就只查 38 行；此次减少的是策略投影与落点汇总，
并未声称可跳过价值头需要的棋形表读取。模型、数据、权重、价值目标和搜索规则没有改变。

## 分阶段原生推理

512 个确定性随机对局局面；初盘为占据数 <24，中盘 24..47，后盘 >=48。
11 轮交错计时，每阶段 100,000 次预测，合法着预先生成，符合 MCTS 在调用网络前已知合法着的条件。
表中是预测加合法 logits 读取的中位耗时，单位 ns；控制组是同源码关闭新开关。
计时前预热，静态几何缓存的首次初始化不作为稳定吞吐计入。

| 精度 | 阶段 | 平均合法着 | 涉及棋形/38 | 完整策略 ns | 合法着策略 ns | 耗时减少 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
'''
for precision in ['INT16','INT8']:
    rows=[v for v in r['results'] if v['precision']==precision];control,candidate=rows[1:]
    for base,chosen in zip(control['stageMedians'],candidate['stageMedians']):
        text+=f"| {precision} | {['初盘','中盘','后盘'][chosen['stage']]} | {chosen['meanLegalMoves']:.2f} | {chosen['meanActivePatterns']:.2f} | {base['medianNanoseconds']:.1f} | {chosen['medianNanoseconds']:.1f} | {1-chosen['medianNanoseconds']/base['medianNanoseconds']:.1%} |\n"
text+='''
## 搜索结果与吞吐

固定预算为 32 个独立根 × 2048 次模拟，11 轮交错测量；实际时限为 7 轮 × 4 根 × 100ms。
v1.33 对照使用上轮选中的“位压缩编码 + 四行 SIMD”。不把不同轮次报告的倍率相乘。

| 精度 | 实现 | 固定搜索耗时 ms | 固定吞吐/v1.33 | 100ms 模拟/v1.33 | 100ms 模拟/同版控制 |
| --- | --- | ---: | ---: | ---: | ---: |
'''
for row in r['results']:text+=f"| {row['precision']} | {row['name']} | {row['fixedSearchMedianMilliseconds']:.2f} | {row['fixedThroughputVsV133']:.3f} | {row['timedSimulationsVsV133']:.3f} | {row['timedSimulationsVsControl']:.3f} |\n"
text+='''
实际 100ms 模拟次数约增加 3.5%，同版控制的增幅约为 INT16 5.0%、INT8 3.4%。
这是小幅本机收益，单轮有负向波动，不能把原生预测耗时下降直接等同于整棵搜索提速，也不推断棋力提升。
逐轮原始数据保存在 JSON 中；默认开关保持 OFF，未替换批准的 v1.25 基线或 Wasm。

## 正确性与平台

两种精度各 6000 验证 + 6000 测试局面的原完整诊断输出逐项一致。
原生测试针对 H32/H64、四个 INT8 步长，以及合法、空、全、单格、随机掩码，检查 46 项值/门控输出、
掩码内 logits 与完整计算精确相同、掩码外为零；完整刷新和增量状态均覆盖。
原生小基准还对 512 个实际随机对局局面的掩码 logits 逐项验证。
固定模拟覆盖 32 冷根、强制停着、终局、16 个到达的复用子树，并分别比较无增量缓存与 64 项缓存；
除耗时字段外，所有落点、值、访问数、证明和统计逐项完全相同。

ARM64 NEON、x86-64 SSE2（Rosetta）及普通回退通过 CTest，实际两个冻结模型的微基准校验和一致。
ARM64 候选通过地址/未定义行为检查和 4 项 Python Pattern 测试。
Linux GCC/Clang CI 加入此路径，尚未远程执行；没有 Linux 实机性能结果。
不增加 ISA 要求，沿用 NEON/SSE2/普通回退。原生实验构建：

```sh
cmake -S . -B .build/native-pattern-legal -DCMAKE_BUILD_TYPE=Release \\
  -DOTELLO_PATTERN_SIMD=ON -DOTELLO_PATTERN_FAST_CODES=ON \\
  -DOTELLO_PATTERN_BLOCK4=ON -DOTELLO_PATTERN_LEGAL_POLICY=ON
cmake --build .build/native-pattern-legal -j 2
ctest --test-dir .build/native-pattern-legal --output-on-failure
```

以上原生加速不自动改变网页浏览器的冻结 Wasm。实时入口 `sh scripts/watch-match.sh`；
复核 `python3 scripts/finalize-pattern-legal.py`。没有重训、对战或默认提升。
更进一步的模型稀疏化可研究结构化通道剪枝，但需另做激活统计、微调与棋力对照，本轮没有实施。
'''
if summary.exists():assert summary.read_text()==text
else:summary.write_text(text)
manifest=ROOT/'models/v1.34-pattern-legal-experiment.json'
archive=ROOT/'models/archive/v1.34-pattern-legal-sources.tar.gz'
source_manifest=archive.with_name('v1.34-pattern-legal-sources.json')
if not manifest.exists():
    assert sha(ROOT/'scripts/benchmark-pattern-legal.py')==r['scriptSha256']
    for row in r['results']:assert sha(ROOT/row['library'])==row['librarySha256']
    logs=['.build/pattern-legal-v1.34-sanitizers.log','.build/pattern-legal-v1.34-python-tests.log']
    assert 'PASS:' in (ROOT/logs[0]).read_text() and 'OK' in (ROOT/logs[1]).read_text()
    for suffix in ['', '-control','-x86','-portable']:
        assert 'Test Passed' in (ROOT/f'.build/native-pattern-legal{suffix}-v1.34/Testing/Temporary/LastTest.log').read_text()
    sources=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/p for p in [
        'AGENTS.md','CMakeLists.txt','.github/workflows/check.yml','data/README.md',
        'benchmarks/pattern-legal.cpp','benchmarks/pattern-hotpath.cpp',
        'scripts/benchmark-pattern-legal.py','scripts/benchmark-pattern-hotpath.py',
        'scripts/benchmark-pattern-int8.py','scripts/pattern_model.py','scripts/compare-resnet.py',
        'scripts/finalize-pattern-legal.py','scripts/watch-matches.py','scripts/watch-match.sh',
        'scripts/test-native.sh','tests/native.cpp','tests/pattern_training_test.py']]))
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as out:
        for p in sources:out.add(p,arcname=str(p.relative_to(ROOT)))
    write(source_manifest,dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    artifacts=[report,summary,ROOT/'models/v1.34-pattern-legal-experiments.json']
    write(manifest,dict(version=r['version'],status='candidate',defaultPromoted=False,sourceManifest=str(source_manifest.relative_to(ROOT)),artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},preservedSha256=r['preservedSha256'],localEvidenceSha256={p:sha(ROOT/p) for p in logs},checks=dict(nativeNeon=True,translatedX86Sse2=True,portable=True,arm64Sanitizers=True,patternTests=4,linuxCiAddedNotExecuted=True)))
frozen=read(manifest);snapshot=read(ROOT/frozen['sourceManifest'])
assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
with tarfile.open(ROOT/snapshot['archive'],'r:gz') as stream:
    for p,digest in snapshot['files'].items():assert hashlib.sha256(stream.extractfile(p).read()).hexdigest()==digest,p
for p,digest in {**frozen['artifactSha256'],**frozen['preservedSha256']}.items():assert sha(ROOT/p)==digest,p
print('Verified legal-only policy projection, search/cache parity, archived sources and preserved default')

#!/usr/bin/env python3
"""Freeze the quantization/SIMD measurements, sources and platform limitations."""
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
report=ROOT/'models/v1.32-pattern-int8.json';r=read(report)
assert not r['defaultPromoted']
assert all(r['parity'][key] for key in ['simdInt16ExactlyEqual','simdInt8ExactlyEqual','fixedSimulationMctsExactlyEqualWithinEachPrecision'])
best=min(r['variants'],key=lambda row:row['validation']['mse'])
assert best['step']==r['selectedStep']
assert sum('test' in v for v in r['variants'])==1
for v in r['variants']:
    assert sha(ROOT/v['model'])==v['modelSha256']
    assert (ROOT/v['model']).stat().st_size==v['bytes']==1792796
for p,digest in r['preservedSha256'].items():assert sha(ROOT/p)==digest,p
summary=ROOT/'models/v1.32-pattern-int8-comparison.md'
base=r['baselineMetrics']
text='''# Pattern INT8 与整数 SIMD 实验 v1.32

候选基于批准的 v1.25 Pattern 胜负模型，不训练、不改变搜索规则，不自动提升默认。
tiny_nn 目前用于其他模型的 FP32 推理，Pattern 没有调用；本实验直接实现整数 NEON/SSE2 内核。

## INT8 棋形表

新增 OTPAT3：沿用 OTPAT2 的 32 字节头，随后一个小端 uint32 量化步长，
然后是 55,728 × 32 个有符号 INT8 棋形表条目；后续主干/输出层 INT16 权重及 INT32 偏置不变。
原始单位为 1/256；解码条目等于 INT8 值乘共享步长。支持步长 1/2/4/8，
本实验只比较 1/2/4，以验证集胜负 MSE 选择，所有候选冻结后仅评估选中者的测试集。
量化使用远离零的半数舍入并截断到 [-128,127]。内存保留压缩表，仅解码当前用到的 32 项行，
累加器保留 INT32；未把整张表重新展开为 INT16。主干与策略点积使用 INT64 等价结果。

| 步长 | 截断条目 | 验证胜负 MSE | 验证下一手命中率 | 模型字节 |
| ---: | ---: | ---: | ---: | ---: |
'''
for v in r['variants']:text+=f"| {v['step']} | {v['clippedEntries']:,} | {v['validation']['mse']:.6f} | {v['validation']['policyTop1']:.2%} | {v['bytes']:,} |\n"
text+=f'''
选中步长 {best['step']}。文件从 3,576,088 字节降到 1,792,796 字节（约减半），压缩表占用同样减半；
这不是整个进程内存减半。选中版本相对原模型的验证局面价值差 RMSE 为 {best['valueDeltaRmse']:.6f}，
最大绝对差为 {best['maximumValueDelta']:.6f}。虽然多数表项保留原整数值，截断仍会改变部分局面的搜索。

| 测试指标（6000 局面） | 原 INT16 | 选中 INT8 |
| --- | ---: | ---: |
| 胜负 MSE | {base['test']['mse']:.6f} | {best['test']['mse']:.6f} |
| 非和局符号准确率 | {base['test']['nonDrawSignAccuracy']:.2%} | {best['test']['nonDrawSignAccuracy']:.2%} |
| 下一手命中率 | {base['test']['policyTop1']:.2%} | {best['test']['policyTop1']:.2%} |
| 策略交叉熵 | {base['test']['policyCrossEntropy']:.6f} | {best['test']['policyCrossEntropy']:.6f} |

误差指标总体接近，微小差异不足以证明棋力提升；策略交叉熵略退。本轮没有棋力对战。
各步长 6000 个验证局面、选中者 6000 个测试局面通过独立 NumPy 整数参考对照，最大误差小于 2e-6。

## SIMD 与计时

对行解码/累加、主干、输出层与策略投影实现整数向量运算。
点积的输入被限制到 ±1024、长度不超过 128，INT16 权重的每个 INT32 部分和不会溢出，最终以 INT64 汇总。
尾部按标量处理，使用未对齐加载。ARM64 选 NEON，x86-64 选 SSE2，其余平台走普通实现。
编译期开关 `OTELLO_PATTERN_SIMD` 默认 OFF；无 `-march=native`、无 AVX2 CPU 假设。

本机 ARM64/macOS，9 轮交错测量，每组每轮 8192 次完整联合推理，包含 Python/C ABI 开销。
固定搜索为 32 局面 × 1024 模拟；实际预算为 5 轮 × 4 局面 × 100ms。
比较参照为此前 v1.29 融合 INT16 库。INT16 的各实现输出/固定模拟搜索逐项一致；
INT8 的普通/SIMD 路径同样逐项一致。不同精度之间会改变数值和树轨迹，不做逐项相等断言。

| 实现 | 联合推理 µs | 联合吞吐比 | 固定模拟搜索吞吐比 | 100ms 模拟数比 |
| --- | ---: | ---: | ---: | ---: |
'''
for v in r['results']:text+=f"| {v['name']} | {v['medianMicroseconds']:.3f} | {v['jointThroughputVsFused']:.3f} | {v['fixedSearchThroughputVsFused']:.3f} | {v['timedSimulationRatioVsFused']:.3f} |\n"
text+='''
普通路径存在吞吐退步，实验结论不能概括成“INT8 本身必然更快”。
本轮主要收益来自整数 SIMD，INT8 再减小表尺寸。计时波动较大，逐轮样本完整保留，
100ms 下 INT8 + SIMD 没有超过 INT16 + SIMD；不同精度树轨迹也会影响搜索速度。

## Linux 适配与验证边界

原生 Linux ARM64 使用 NEON，Linux x86-64 使用 SSE2，不把 ARM 指令编入 x86 产物。
本机 ARM64 NEON、普通路径以及 x86-64 SSE2（Rosetta）均通过 CTest，
NEON 路径通过地址/未定义行为检查，4 项 Python Pattern 测试通过。
增加 Linux CI 的 GCC/Clang × SIMD ON/OFF 构建与测试任务；这些新增任务尚未在远程 CI 执行，
当前环境没有 Linux 容器运行时，因此不宣称已测得 Linux 性能或完成 Linux 实机验证。

Linux 构建命令：

```sh
cmake -S . -B .build/native-pattern-simd -DCMAKE_BUILD_TYPE=Release -DOTELLO_PATTERN_SIMD=ON
cmake --build .build/native-pattern-simd -j 2
ctest --test-dir .build/native-pattern-simd --output-on-failure
```

部署前在目标架构重新编译并测量，不能复制本机 dylib。
当前 Docker 网页部署承载的是已冻结浏览器 Wasm；服务端 Linux CPU 的 NEON/SSE2 不会加速浏览器里的 Wasm。
本次未替换批准模型、默认原生参考库或浏览器产物。复核：`python3 scripts/finalize-pattern-int8.py`。
'''
if summary.exists():assert summary.read_text()==text
else:summary.write_text(text)
manifest=ROOT/'models/v1.32-pattern-int8-experiment.json'
archive=ROOT/'models/archive/v1.32-pattern-int8-sources.tar.gz'
source_manifest=archive.with_name('v1.32-pattern-int8-sources.json')
if not manifest.exists():
    assert sha(ROOT/'scripts/benchmark-pattern-int8.py')==r['scriptSha256']
    for row in r['results']:assert sha(ROOT/row['library'])==row['librarySha256']
    logs=['.build/pattern-int8-v1.32-sanitizers.log','.build/pattern-int8-v1.32-python-tests.log']
    assert 'PASS:' in (ROOT/logs[0]).read_text() and 'OK' in (ROOT/logs[1]).read_text()
    tests=['.build/native-pattern-int8-v1.32/Testing/Temporary/LastTest.log','.build/native-pattern-simd-v1.32/Testing/Temporary/LastTest.log','.build/native-pattern-sse2-v1.32/Testing/Temporary/LastTest.log']
    assert all('Test Passed' in (ROOT/p).read_text() for p in tests)
    assert not archive.exists()
    sources=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/p for p in [
        'CMakeLists.txt','.github/workflows/check.yml','AGENTS.md','data/README.md',
        'scripts/benchmark-pattern-int8.py','scripts/finalize-pattern-int8.py','scripts/pattern_model.py',
        'scripts/watch-matches.py','scripts/watch-match.sh','scripts/test-native.sh',
        'tests/native.cpp','tests/pattern_training_test.py']]))
    with tarfile.open(archive,'w:gz') as out:
        for p in sources:out.add(p,arcname=str(p.relative_to(ROOT)))
    write(source_manifest,dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    artifacts=[report,summary,ROOT/'models/v1.32-pattern-int8-experiments.json']+[ROOT/v['model'] for v in r['variants']]
    write(manifest,dict(version=r['version'],status='candidate',defaultPromoted=False,sourceManifest=str(source_manifest.relative_to(ROOT)),artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},preservedSha256=r['preservedSha256'],checks=dict(arm64Neon=True,x86Sse2UnderRosetta=True,arm64Sanitizers=True,patternTests=4,linuxCiAddedButNotExecuted=True),localEvidenceSha256={p:sha(ROOT/p) for p in logs+tests}))
frozen=read(manifest);snapshot=read(ROOT/frozen['sourceManifest'])
assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
with tarfile.open(ROOT/snapshot['archive'],'r:gz') as stream:
    for p,digest in snapshot['files'].items():assert hashlib.sha256(stream.extractfile(p).read()).hexdigest()==digest,p
for p,digest in {**frozen['artifactSha256'],**frozen['preservedSha256']}.items():assert sha(ROOT/p)==digest,p
print('Verified INT8 artifacts, SIMD parity, source archive, platform limits and preserved v1.25 default')

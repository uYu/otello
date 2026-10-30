#!/usr/bin/env python3
"""Freeze exact Pattern hot-path ablations and the selected combination."""
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
a=read(ROOT/'models/v1.33-pattern-hotpath.json');r=read(ROOT/'models/v1.33-pattern-selected.json')
for report in [a,r]:
    assert not report['defaultPromoted']
    for field in ['all68JointOutputsExactlyEqual','fixedSimulationSearchExactlyEqual','reusedSubtreeSearchExactlyEqual']:assert report['parity'][field]
    for p,digest in report['preservedSha256'].items():assert sha(ROOT/p)==digest,p
assert r['selectedFlags']==dict(SIMD=True,FAST_CODES=True,BLOCK4=True,INTEGER_ROUNDING=False)
summary=ROOT/'models/v1.33-pattern-hotpath-comparison.md'
text='''# Pattern 精确热点优化 v1.33

基于批准的 v1.25 模型及 v1.32 整数 SIMD 候选，分别测试：
1. 整数舍入，匹配负数半值远离零的原规则。
2. 完整联合推理路径的棋形编码：按行/列/对角线位压缩，镜像角部，用 512 项表生成相同三进制编号。
3. SIMD 同时计算四个输出，复用输入加载；ARM64 分开低/高半累加，x86-64 用 SSE2，最后 INT64 等价汇总。
   输入与暂存行按 64 字节对齐，尾部单独处理；没有重排持久权重，也没有引入需要同步的派生权重副本。

模型、文件格式、价值目标、策略、搜索规则均不变；未训练、未做棋力对战、未提升默认。
整数舍入通过一致性但本机较慢，因此最终组合只启用位压缩编码与四行 SIMD。
全部实验开关默认 OFF，批准的默认参考库和浏览器 Wasm 保持冻结。

## 分项消融

9 轮交错计时。原生小基准按同一确定性序列生成 128 个合法局面，预热后每阶段 100,000 次；
编码、聚合后的主干/输出层、完整联合推理分别测量，包含各自函数边界成本，阶段时间不宜直接相加。
表中时间均为中位数，单位 ns，完整逐轮样本见 `v1.33-pattern-hotpath.json`。

| 精度 | 单项/组合 | 编码 ns | 主干/输出层 ns | 联合推理 ns |
| --- | --- | ---: | ---: | ---: |
'''
for row in a['results']:
    if 'nativeMedianNanoseconds' not in row:continue
    m=row['nativeMedianNanoseconds'];text+=f"| {row['precision']} | {row['name']} | {m['encodingNs']:.1f} | {m['denseNs']:.1f} | {m['jointNs']:.1f} |\n"
text+='''
整数舍入的单项主干/输出层耗时增加，不能因为去掉浮点就假定更快。
原来的编译器浮点实现已较有效，本版整数实现没有获得收益，予以保留和明确排除。

## 排除整数舍入后的组合复测

15 轮交错计时，原生每阶段 200,000 次，C ABI 每轮 16,384 次。
独立冷根搜索为 32 局面 × 1024 模拟；实际预算为 7 轮 × 4 局面 × 100ms。
计时对照是上一轮 v1.32 SIMD 候选，不把此前不同环境下的倍率相乘。

| 精度 | 实现 | 原生联合 ns | C ABI 联合 µs | 固定搜索吞吐/v1.32 | 100ms 模拟/v1.32 | 100ms 模拟/同版对照 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
'''
for row in r['results']:
    native=row.get('nativeMedianNanoseconds',{}).get('jointNs')
    value=f'{native:.1f}' if native is not None else '未测'
    text+=f"| {row['precision']} | {row['name']} | {value} | {row['jointMicroseconds']:.3f} | {row['fixedSearchThroughputVsV132']:.3f} | {row['timedSimulationRatioVsV132']:.3f} | {row['timedSimulationRatioVsControl']:.3f} |\n"
text+='''
原生联合推理相对同版对照约提高 24%～25%；100ms 搜索相对 v1.32 提高约 13%/14%（INT16/INT8），
相对同版对照约提高 11%/8%。单次 C ABI 的 INT16 中位数没有改善，不能声称所有计时口径均提速；
C ABI 还包括棋盘解码和 Python 调用开销，测量也有调度波动。完整样本及不同口径均保留。
没有据此推断棋力或进行默认提升。

## 一致性与平台边界

两种精度各 6000 个验证和 6000 个测试局面，所有 68 项联合输出与各自 v1.32 模型逐项完全相同。
32 冷根与 16 个实际到达的复用子树局面，固定 1024 模拟的落点、分值、访问数、证明和统计逐项相同，
只排除耗时字段。两种精度彼此仍有量化差异，本实验不要求 INT8 与 INT16 相等。

原生测试新增 20,000 对掩码 × 38 棋形（含重叠掩码的 own 优先语义），池化整数舍入穷举
-983040..983040、阶段与点积边界、四行点积的极端权重/奇数步长/尾部检查。
ARM64 NEON、x86-64 SSE2（Rosetta）、普通回退的全部启用/选中组合均通过 CTest；
全部启用与选中组合通过地址/未定义行为检查，选中库通过 4 项 Python Pattern 测试。
原生微基准的实际批准模型/INT8 候选校验和也跨三条路径一致。

保持 Linux ARM64 NEON / x86-64 SSE2 / 其他目标普通回退，无 `-march=native` 或 AVX2 依赖。
Linux CI 增加 GCC/Clang × SIMD ON/OFF × 整数舍入 ON/OFF，编码与四行布局开启。
新增 CI 尚未远程执行，本机没有 Linux 运行时，未验证 Linux 实机性能。
原生候选构建：

```sh
cmake -S . -B .build/native-pattern-selected -DCMAKE_BUILD_TYPE=Release \\
  -DOTELLO_PATTERN_SIMD=ON -DOTELLO_PATTERN_FAST_CODES=ON \\
  -DOTELLO_PATTERN_BLOCK4=ON -DOTELLO_PATTERN_INTEGER_ROUNDING=OFF
cmake --build .build/native-pattern-selected -j 2
ctest --test-dir .build/native-pattern-selected --output-on-failure
.build/native-pattern-selected/otello_pattern_benchmark models/pattern32-outcome-v1.25.otpat 200000
```

服务器应在目标架构重编译；以上原生指令不会自动加速网页浏览器中的冻结 Wasm。
实时入口仍为 `sh scripts/watch-match.sh`。复核冻结记录：`python3 scripts/finalize-pattern-hotpath.py`。
'''
if summary.exists():assert summary.read_text()==text
else:summary.write_text(text)
manifest=ROOT/'models/v1.33-pattern-hotpath-experiment.json'
archive=ROOT/'models/archive/v1.33-pattern-hotpath-sources.tar.gz'
source_manifest=archive.with_name('v1.33-pattern-hotpath-sources.json')
if not manifest.exists():
    assert sha(ROOT/'scripts/benchmark-pattern-hotpath.py')==a['scriptSha256']
    assert sha(ROOT/'scripts/benchmark-pattern-selected.py')==r['scriptSha256']
    for report in [a,r]:
        for row in report['results']:assert sha(ROOT/row['library'])==row['librarySha256']
    logs=['.build/pattern-hotpath-v1.33-sanitizers.log','.build/pattern-selected-v1.33-sanitizers.log','.build/pattern-selected-v1.33-python-tests.log']
    assert all('PASS:' in (ROOT/p).read_text() for p in logs[:2])
    assert 'OK' in (ROOT/logs[-1]).read_text()
    directories={str(Path(row['library']).parent) for row in a['results']+r['results'] if 'v1.33' in row['library']}
    directories.update(f'.build/native-pattern-{name}-v1.33' for name in ['fast-x86','fast-portable','selected-x86','selected-portable'])
    configurations={}
    for directory in sorted(directories):
        log=ROOT/directory/'Testing/Temporary/LastTest.log';assert 'Test Passed' in log.read_text()
        cache=(ROOT/directory/'CMakeCache.txt').read_text().splitlines()
        configurations[directory]={line.split('=',1)[0]:line.split('=',1)[1] for line in cache if '=' in line and (line.startswith('OTELLO_PATTERN_') or line.startswith('CMAKE_OSX_ARCHITECTURES:'))}
    sources=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+[ROOT/p for p in [
        'AGENTS.md','CMakeLists.txt','.github/workflows/check.yml','data/README.md',
        'benchmarks/pattern-hotpath.cpp','scripts/benchmark-pattern-hotpath.py',
        'scripts/benchmark-pattern-selected.py','scripts/benchmark-pattern-int8.py',
        'scripts/pattern_model.py','scripts/compare-resnet.py','scripts/finalize-pattern-hotpath.py',
        'scripts/watch-matches.py','scripts/watch-match.sh','scripts/test-native.sh',
        'tests/native.cpp','tests/pattern_training_test.py']]))
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as out:
        for p in sources:out.add(p,arcname=str(p.relative_to(ROOT)))
    write(source_manifest,dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),files={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    artifacts=[ROOT/'models/v1.33-pattern-hotpath.json',ROOT/'models/v1.33-pattern-selected.json',ROOT/'models/v1.33-pattern-hotpath-experiments.json',summary]
    write(manifest,dict(version=r['version'],status='candidate',defaultPromoted=False,sourceManifest=str(source_manifest.relative_to(ROOT)),artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},preservedSha256=r['preservedSha256'],buildConfigurations=configurations,localEvidenceSha256={p:sha(ROOT/p) for p in logs},checks=dict(nativeNeon=True,translatedX86Sse2=True,portable=True,sanitizers=True,patternTests=4,linuxCiAddedNotExecuted=True)))
frozen=read(manifest);snapshot=read(ROOT/frozen['sourceManifest'])
assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
with tarfile.open(ROOT/snapshot['archive'],'r:gz') as stream:
    for p,digest in snapshot['files'].items():assert hashlib.sha256(stream.extractfile(p).read()).hexdigest()==digest,p
for p,digest in {**frozen['artifactSha256'],**frozen['preservedSha256']}.items():assert sha(ROOT/p)==digest,p
print('Verified exact Pattern hot-path ablations, selected flags, source archive and unchanged default')

#!/usr/bin/env python3
"""Freeze and verify the opt-in incremental Pattern MCTS experiment."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p, d: Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
report = ROOT / 'models/v1.31-pattern-incremental-benchmark.json'
r = read(report)
assert r['allFixedSimulationOutputsExactlyEqual'] and not r['defaultPromoted']
assert sha(ROOT / r['model']) == r['modelSha256']
assert sha(ROOT / r['baselineManifest']) == r['baselineManifestSha256']
for row in r['results']:
    assert row['entries'] <= r['cacheEntriesLimit'] == 16384
    for mode in ['cold', 'warm']:
        assert row[mode]['cacheBytes'] <= row['entries'] * 1024

summary = ROOT / 'models/v1.31-pattern-incremental-comparison.md'
text = '''# Pattern MCTS 增量推理实验 v1.31

使用已批准 v1.25 模型，基于 v1.29 融合完整推理实现接入可选增量缓存。
缓存保存双视角棋形编码及聚合和，从搜索路径中最近的缓存祖先更新至叶节点。
固定容量直接映射；模型、策略或树重新编号时失效。默认容量为零，继续完整融合推理。
仅新增原生实验入口 `ot_nnue_mcts_play_cached`，未替换批准的原生参考库或浏览器 Wasm。

## 一致性与测量

32 个独立验证局面和 16 个连续子树复用局面，每个最多 4096 次模拟；
六档缓存均与完整推理逐项一致，比较落点、价值、访问数、证明及搜索统计，排除时间字段。
完整路径也与 v1.29 融合库一致。原生测试覆盖逐步和跨多步更新、强制停着、换模型、H64、缓存失效及非法容量；
CTest、地址/未定义行为检查及 4 项 Python Pattern 测试通过。

七轮交错计时，预热及缓存分配不计时；根局面处理、模型快照及祖先查找计时。
冷根组逐局清树，连续组复用可达子树。缓存字节仅指额外缓存表，并非进程总内存。

| 缓存条目 | 额外字节 | 冷根耗时 ms | 连续耗时 ms | 冷根吞吐比 | 连续吞吐比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
'''
for row in r['results']:
    a, b = row['cold'], row['warm']
    text += f"| {row['entries']} | {a['cacheBytes']:,} | {a['medianMilliseconds']:.2f} | {b['medianMilliseconds']:.2f} | {a['throughputRatioVsFused']:.3f} | {b['throughputRatioVsFused']:.3f} |\n"
a, b = r['timed100ms']
text += f'''
选择冷根与连续组耗时之和最小的缓存档 {r['bestCachedEntries']}，再测 5 轮 × 4 个局面、每步 100ms：
完整推理共 {a['totalSimulations']:,} 次模拟，增量路径共 {b['totalSimulations']:,} 次，比例 {b['totalSimulations']/a['totalSimulations']:.3f}。
计时受本机负载影响；此处不据此推断棋力。

## 结论

本实现没有提速，保留为默认关闭的实验，不提升基线。
冷根高命中档平均每次改变约 10.7/38 个棋形；维护双视角、读取旧行与新行、复制状态及探测缓存抵消了减少重算的收益。
策略投影与共享主干仍需完整计算。没有训练或棋力对战。
复核：`python3 scripts/finalize-pattern-incremental.py`；原始逐轮数据保留在 benchmark JSON。
'''
if summary.exists():
    assert summary.read_text() == text
else:
    summary.write_text(text)
manifest = ROOT / 'models/v1.31-pattern-incremental-experiment.json'
archive = ROOT / 'models/archive/v1.31-pattern-incremental-sources.tar.gz'
source_manifest = archive.with_name('v1.31-pattern-incremental-sources.json')
if not manifest.exists():
    for p, digest in r['libraries'].items():
        assert sha(ROOT / p) == digest
    assert sha(ROOT / 'scripts/benchmark-pattern-incremental.py') == r['scriptSha256']
    assert not archive.exists()
    sources = sorted(set([p for p in (ROOT / 'cpp').rglob('*') if p.is_file()] + [ROOT / p for p in [
        'AGENTS.md', 'CMakeLists.txt', 'data/README.md', 'tests/native.cpp',
        'tests/pattern_training_test.py', 'scripts/test-native.sh', 'scripts/pattern_model.py',
        'scripts/benchmark-pattern-incremental.py', 'scripts/finalize-pattern-incremental.py',
        'scripts/watch-matches.py', 'scripts/watch-match.sh']]))
    with tarfile.open(archive, 'w:gz') as out:
        for p in sources:
            out.add(p, arcname=str(p.relative_to(ROOT)))
    write(source_manifest, dict(archive=str(archive.relative_to(ROOT)), sha256=sha(archive),
        files={str(p.relative_to(ROOT)): sha(p) for p in sources}))
    artifacts = [report, summary, ROOT / 'models/v1.31-pattern-incremental-experiments.json']
    preserved = ['models/default-model.json', r['baselineManifest'], r['model'], 'src/game/wasm/othello.mjs']
    write(manifest, dict(version=r['version'], status='candidate', defaultPromoted=False,
        sourceManifest=str(source_manifest.relative_to(ROOT)),
        artifactSha256={str(p.relative_to(ROOT)): sha(p) for p in artifacts},
        preservedSha256={p: sha(ROOT / p) for p in preserved},
        checks=dict(native=True, sanitizers=True, patternTests=4, fixedSimulationParity=True)))
frozen = read(manifest)
snapshot = read(ROOT / frozen['sourceManifest'])
assert sha(ROOT / snapshot['archive']) == snapshot['sha256']
with tarfile.open(ROOT / snapshot['archive'], 'r:gz') as stream:
    for p, digest in snapshot['files'].items():
        assert hashlib.sha256(stream.extractfile(p).read()).hexdigest() == digest, p
for p, digest in {**frozen['artifactSha256'], **frozen['preservedSha256']}.items():
    assert sha(ROOT / p) == digest, p
print('Verified incremental parity, bounded cache, negative speed result and preserved default')

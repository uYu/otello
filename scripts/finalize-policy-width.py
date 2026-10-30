#!/usr/bin/env python3
"""Audit and freeze two matched direct-policy width candidates, without promotion."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def immutable(path,value):
    content=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==content,f'Refusing to replace {path}'
    else:path.write_text(content)


def main():
    protocol=read(ROOT/'models/v1.39-policy-width-protocol.json')
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/protocol['referenceCandidate'])==protocol['referenceCandidateSha256']
    source=read(ROOT/'models/archive/v1.39-policy-width-sources.json');assert sha(ROOT/source['archive'])==source['sha256']
    with tarfile.open(ROOT/source['archive'],'r:gz') as t:
        for name,digest in source['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
    r=read(ROOT/'models/v1.39-pattern-policy-width.json');bench=read(ROOT/'models/v1.39-policy-width-benchmark.json')
    assert bench['complete'] and r['baselineManifest']=='models/v1.37-default.json' and not r['defaultPromoted']
    assert bench['trainingReportSha256']==sha(ROOT/'models/v1.39-pattern-policy-width.json')
    assert bench['protocolSha256']==sha(ROOT/'models/v1.39-policy-width-protocol.json')
    assert bench['sourceArchiveManifestSha256']==sha(ROOT/'models/archive/v1.39-policy-width-sources.json')
    assert bench['scriptSha256']==source['files']['scripts/benchmark-policy-width.py']
    assert bench['parity']['all68OutputsExactlyEqual']
    for name,digest in bench['libraries'].items():assert sha(ROOT/name)==digest,name
    for name,digest in r['scriptSha256'].items():assert source['files'][name]==digest,name
    original=read(ROOT/'models/pattern32-direct-mlp8-v1.38.json')
    baseline=read(ROOT/'models/pattern32-symmetry-v1.37.json')
    assert [v['policyHead'] for v in r['variants']]==[[74,16,1],[74,32,1]]
    for v in r['variants']:
        path=ROOT/v['model'];assert sha(path)==v['modelSha256']
        record=read(path.with_suffix('.json'));assert not record['defaultPromoted'] and record['status']=='candidate'
        assert v['bestEpoch']==min(v['history'],key=lambda x:x['mse'])['epoch']
        assert len(v['history'])==15 and v['validationParity']['positions']==v['testParity']['positions']==6000
        assert v['initialEmbeddingSha256']==original['initialEmbeddingSha256']==baseline['initialEmbeddingSha256']
        assert v['dataOrderAndAugmentationSha256']==original['dataOrderAndAugmentationSha256']==baseline['dataOrderAndAugmentationSha256']
    lines=['# v1.39：直连策略 MLP16 / MLP32 独立训练','',
        '用户要求沿 MLP 有效果的方向先训练。两组候选只把 v1.38 直连共享策略头的隐藏宽度 8 分别改为 16、32；没有组合十组池化，没有改价值主干。默认仍是用户批准的 v1.37 对称共享。','',
        '每个落点仍为局部均值 32 维 + 全局主干 32 维 + D4 位置桶 one-hot 10 维 → 共享 74→H→1，H 为 16 或 32，中间 clipped ReLU。全网随机重训，保持 38 个棋形、28,503×32 共享棋形表、三组均值、97→32 价值主干和 4 个价值输出。','',
        '与 v1.37 / v1.38 使用相同表、主干和价值初值、百万训练局面、15 轮、batch 2048、AdamW 0.001、weight decay 0.0001、胜负 MSE +0.1 WDL CE +0.1 合法走法 CE；样本和 D4 增强顺序哈希完全相同。各自按验证胜负 MSE 最低轮次选定，冻结两个权重后才统一评估测试集。','',
        '| 模型 | 选定轮次 | 参数 | 验证 MSE | 测试 MSE | 测试策略命中率 | 测试策略 CE |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in [('批准 v1.37',baseline),('直连 MLP8 v1.38',original)]+[(v['name']+' v1.39',v) for v in r['variants']]:
        vm,tm=v['metrics']['validation'],v['metrics']['test']
        lines.append(f"| {name} | {v['bestEpoch']} | {v['parameters']:,} | {vm['mse']:.6f} | {tm['mse']:.6f} | {tm['policyTop1']:.2%} | {tm['policyCrossEntropy']:.6f} |")
    lines+=['','| 模型 | 完整 64 格推理 µs | 100ms 模拟吞吐 / MLP8 | 100ms 评估吞吐 / MLP8 | 2s 模拟吞吐 / MLP8 | 2s 评估吞吐 / MLP8 |','|---|---:|---:|---:|---:|---:|']
    for row in bench['results']:
        ratios=[]
        for ms in ['100','2000']:
            for metric in ['simulationsPerSecond','evaluationsPerSecond']:
                x=row['budgets'][ms][metric+'VsMlp8Ratio'];ratios.append(f'{x:.3f}×' if x is not None else '无共同样本')
        lines.append(f"| {row['name']} | {row['medianJointMicroseconds']:.3f} | "+' | '.join(ratios)+' |')
    lines+=['','计时在训练和 2s 对照结束后进行。四模型使用相同验证局面交错运行；每预算各保留所有模型、所有轮次均未证明、未触顶且用满至少 95% 预算的共同局面。完整推理含 Python/C ABI 和全部 64 格输出；实际搜索只算合法着。策略变化会改变搜索路径，因此模拟/评估吞吐不同，不是纯内核倍率。原始逐次样本保留在 benchmark JSON。','',
        '两个新模型各通过验证/测试合计 12,000 局面的 Torch/C++ 导出对照，以及各 12,000 局面的 SIMD/portable 全部 68 项输出精确对照；16 个固定 1,024 模拟搜索结果也相同。v1.37 和 v1.38 MLP8 分别在新库与冻结库上通过相同输出及固定搜索回归。原生独立标量参考覆盖三种策略宽度、合法着投影、D4、文件损坏和状态刷新，ASan/UBSan 通过。旧 MLP8 的 PT 重新导出与原二进制逐字节一致。','',
        '训练使用两个 CPU 线程，与用户要求的 v1.38 vs v1.37 2s/20 对照并行；同机还有候选代码构建和一致性检查，没有独占 CPU。该对照保留所有实测时间与结果，协议已注明并行负载；它评价的是 MLP8，不是本轮新宽度。','',
        '单训练种子、沿用历史固定测试集。离线指标用于分析容量收益，不能证明棋力提升；MLP16/32 尚未进行棋力对战，均保持候选。Linux 与浏览器性能未测，默认选择器、默认模型和浏览器 Wasm 的哈希未变。','',
        '训练记录：`v1.39-pattern-policy-width.json`；耗时与输出核验：`v1.39-policy-width-benchmark.json`；协议：`v1.39-policy-width-protocol.json`；源码快照：`archive/v1.39-policy-width-sources.json`。审计：`python3 scripts/finalize-policy-width.py`。监控：`sh scripts/watch-match.sh`。']
    summary=ROOT/'models/v1.39-policy-width-comparison.md';immutable(summary,'\n'.join(lines)+'\n')
    paths=[ROOT/'models/v1.39-pattern-policy-width.json',ROOT/'models/v1.39-policy-width-benchmark.json',ROOT/'models/v1.39-policy-width-protocol.json',ROOT/'models/archive/v1.39-policy-width-sources.json',ROOT/'models/v1.39-policy-width-validation.json',summary]
    for v in r['variants']:
        p=ROOT/v['model'];paths.extend([p,p.with_suffix('.pt'),p.with_suffix('.json')])
    record=dict(version='v1.39-policy-width',status='candidate',defaultPromoted=False,matchesRun=False,artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in paths},preservedSha256=protocol['preservedSha256'])
    immutable(ROOT/'models/v1.39-policy-width-experiment.json',record)
    print(json.dumps(dict(complete=True,variants=[dict(name=v['name'],bestEpoch=v['bestEpoch'],test=v['metrics']['test']) for v in r['variants']]),ensure_ascii=False))


if __name__=='__main__':main()

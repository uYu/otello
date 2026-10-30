#!/usr/bin/env python3
"""Audit controlled label ablations and preserve WTHOR-only as a separate side result."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def immutable(path,value):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',choices=['wthor','mixed'],required=True);args=parser.parse_args()
    mixed=args.experiment=='mixed';prefix='v1.45-mixed-label-ablation' if mixed else 'v1.44-wthor-label-ablation';r=read(ROOT/f'models/{prefix}.json');protocol=read(ROOT/f'models/{prefix}-protocol.json');source=read(ROOT/f'models/archive/{prefix}-sources.json')
    assert r['complete'] and not r['defaultPromoted'] and not r['matchesRun'];assert [v['name'] for v in r['variants']]==['hard','soft']
    a,b=r['variants'];assert a['initialParametersSha256']==b['initialParametersSha256'] and a['inputFingerprints']==b['inputFingerprints']==r['controlledInputs']
    assert a['dataOrderAndAugmentationSha256']==b['dataOrderAndAugmentationSha256']
    assert r['epochs']==len(a['history'])==len(b['history'])==15 and r['batchSize']==2048 and r['threads']==2
    assert r['protocolSha256']==sha(ROOT/f'models/{prefix}-protocol.json')
    assert r['librarySha256']==read(ROOT/'models/v1.37-default.json')['referenceNativeLibrary']['sha256']
    for name,digest in {**protocol['preservedSha256'],**r['scriptSha256']}.items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/source['archive'])==source['sha256']
    with tarfile.open(ROOT/source['archive'],'r:gz') as t:
        for name,digest in source['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
    for name,digest in r['scriptSha256'].items():assert source['files'][name]==digest,name
    for v in r['variants']:
        assert sha(ROOT/v['model'])==v['modelSha256'];assert v['parameters']==916750 and v['bytes']==1833692
        assert v['bestEpoch']==min(v['history'],key=lambda x:x['mse'])['epoch'];assert v['validationParity']['positions']==v['testParity']['positions']==6000
    replay=r['hardBaselineReplay'] if mixed else r['softReferenceReplay']
    title='原百万混合语料：原基线硬标签 vs 聚合软标签' if mixed else 'WTHOR-only：首条硬标签 vs 聚合软标签（旁证）'
    lines=[f'# {prefix}：{title}','']
    if mixed:
        lines+=['这是用户明确的主对照：WTHOR 与 Logistello 都保留，使用原始冻结百万训练棋盘、行棋方和行顺序；不是前一轮 WTHOR-only 局面集。硬标签直接使用 v1.37 冻结结果和走法，不重新挑首条记录。软标签来自两库原始 train split 的全部同局面记录，限制到这同一百万行。','',
            '软策略先在相对颜色/D4 规范化坐标聚合，再反向映射回每一行原始棋盘坐标；稳定子等价落点均分质量。每行每轮采样一次，没有按重复频率增加采样/损失权重，没有 Elo 筛选、Elo 加权或棋力筛选。','']
    else:
        lines+=['本轮固定 1,054,132 个 WTHOR-only 局面，首条标签按冻结 occurrence 文件中的 WTHOR 出现顺序确定。它已完成，但用户随后明确要加上 Logistello；因此它只是旁证，主对照另见 `v1.45-mixed-label-ablation-comparison.md`。','',
            '硬标签从同一首条记录取结果和走法，规范化稳定子动作并列时选择最小落点；软标签保留全部 WTHOR 次数分布。此处的硬标签不是原 v1.37 的混合语料标签，也不声称完全重建历史首条选择细节。','']
    lines+=['两版都用批准的 v1.37 对称共享原结构、相同完整参数初值、同样的样本/增强序列、AdamW、15 轮、batch 2048、两个 CPU 线程、lr 0.001、weight decay 0.0001。输入棋盘、行棋方、phase 和合法掩码哈希均相同。数学损失为价值 MSE +0.1 W/L/D CE +0.1 合法策略 CE；只切换目标。','',
        '原验证和测试各 6,000 个棋盘、结果、走法标签保持原值。两个 checkpoint 都按验证胜负 MSE 选定并冻结后，才评估任一测试集。训练 15 轮的最后验证指标也单独保留，便于观察相同优化步数下的结果。','',
        '| 标签 | 选定轮次 | 验证 MSE | 测试 MSE | 测试走法命中率 | 测试策略 CE |','|---|---:|---:|---:|---:|---:|']
    for v in r['variants']:
        vm,tm=v['metrics']['validation'],v['metrics']['test'];lines.append(f"| {v['name']} | {v['bestEpoch']} | {vm['mse']:.6f} | {tm['mse']:.6f} | {tm['policyTop1']:.2%} | {tm['policyCrossEntropy']:.6f} |")
    lines+=['','| 标签 | 第15轮验证 MSE | 第15轮验证走法命中率 | 第15轮验证策略 CE |','|---|---:|---:|---:|']
    for v in r['variants']:
        x=v['epoch15Validation'];lines.append(f"| {v['name']} | {x['mse']:.6f} | {x['policyTop1']:.2%} | {x['policyCrossEntropy']:.6f} |")
    lines+=['',f"控制项：完整初值 SHA `{a['initialParametersSha256']}`；样本/增强序列 SHA `{a['dataOrderAndAugmentationSha256']}`。",'',
        f"{'硬标签对原 v1.37' if mixed else '软标签对原 v1.42'}复现核验：模型二进制逐字节相同={replay['modelBytesIdentical']}；全部轮次损失与验证指标相同={replay['allEpochMetricsIdentical']}；逐轮验证 MSE 最大差={replay['maxHistoryMseDifference']:.9g}。",'',
        '两版各通过验证/测试合计 12,000 局面的 PyTorch/冻结原生库导出对照。没有重建或修改 C++ 推理、默认模型、浏览器 Wasm；没有启动棋力对局。','',
        '这是单种子的离线目标构造对照；它比较价值/WLD/策略同时从硬变软的总效果，不能把效果单独归因于某一头。与原来不同来源的数据集对比也不能替代这一固定行实验。默认仍为 v1.37，不因离线分数自动提升候选。','',
        f'协议：`{prefix}-protocol.json`；完整训练和测试：`{prefix}.json`；源码档案：`archive/{prefix}-sources.json`。审计：`python3 scripts/finalize-label-ablations.py --experiment {args.experiment}`。监控：`sh scripts/watch-match.sh`。']
    summary=ROOT/f'models/{prefix}-comparison.md';immutable(summary,'\n'.join(lines)+'\n')
    paths=[ROOT/f'models/{prefix}.json',ROOT/f'models/{prefix}-protocol.json',ROOT/f'models/archive/{prefix}-sources.json',summary]
    for v in r['variants']:
        p=ROOT/v['model'];paths.extend([p,p.with_suffix('.json'),p.with_suffix('.pt')])
    immutable(ROOT/f'models/{prefix}-experiment.json',dict(status='candidate-offline-complete',primaryUserRequestedComparison=mixed,defaultPromoted=False,matchesRun=False,artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in paths},preservedSha256=protocol['preservedSha256'],replay=replay))
    print(json.dumps(dict(complete=True,experiment=args.experiment,replay=replay,results=[dict(name=v['name'],test=v['metrics']['test']) for v in r['variants']]),ensure_ascii=False))


if __name__=='__main__':main()

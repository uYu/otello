#!/usr/bin/env python3
"""Freeze a WTHOR baseline-architecture training report; never promote defaults."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def immutable(path,value):
    content=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==content,f'Refusing to replace {path}'
    else:path.write_text(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',choices=['v1.42','v1.43'],required=True);args=parser.parse_args()
    version=args.version;name='pattern-wthor' if version=='v1.42' else 'pattern-wthor1600';prefix=version+'-'+name
    path=ROOT/f'models/{prefix}.json';r=read(path);candidate=r['variants'][0];baseline=read(ROOT/'models/pattern32-symmetry-v1.37.json');protocol=read(ROOT/f'models/{prefix}-protocol.json');source=read(ROOT/f'models/archive/{prefix}-sources.json')
    assert not r['defaultPromoted'] and r['baselineManifest']=='models/v1.37-default.json' and candidate['formatMagic']=='OTPAT4'
    assert candidate['parameters']==baseline['parameters']==916750 and candidate['bytes']==baseline['bytes']==1833692
    assert candidate['initialEmbeddingSha256']==baseline['initialEmbeddingSha256'] and candidate['bestEpoch']==min(candidate['history'],key=lambda e:e['mse'])['epoch']
    assert r['epochs']==len(candidate['history'])==15 and r['seed']==20261029 and r['dataSeed']==20261030
    assert r['dataset']['sources']==['wthor'] and not r['dataset']['eloWeighting'] and r['dataset']['allOccurrencesEqual']
    assert r['dataset']['eloFiltering']==(version=='v1.43') and candidate['validationParity']['positions']==candidate['testParity']['positions']==6000
    assert r['protocolSha256']==sha(ROOT/f'models/{prefix}-protocol.json')
    assert sha(ROOT/candidate['model'])==candidate['modelSha256']
    assert r['librarySha256']==read(ROOT/'models/v1.37-default.json')['referenceNativeLibrary']['sha256']
    assert sha(ROOT/source['archive'])==source['sha256']
    with tarfile.open(ROOT/source['archive'],'r:gz') as t:
        for name,digest in source['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
    for name,digest in r['scriptSha256'].items():assert source['files'][name]==digest,name
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    models=[('批准 v1.37',baseline)]
    if version=='v1.43':models.append(('WTHOR 全量 v1.42',read(ROOT/'models/v1.42-pattern-wthor.json')['variants'][0]))
    models.append((version+' WTHOR '+('1600 过滤' if version=='v1.43' else '全量'),candidate))
    lines=[f'# {version}：WTHOR '+('1600 过滤版' if version=='v1.43' else '全量版')+'，对称共享原结构','',
        '使用用户批准的 v1.37 对称共享原结构：38 个棋形、28,503×32 INT16 表、三组均值、97→32 主干、原条件策略头及 46 输出；未使用直连 MLP。默认仍为 v1.37。','',
        f"训练数据包含 {r['dataset']['trainGames']:,} 盘合格原始 train split WTHOR、{r['dataset']['retainedOccurrences']:,} 次保留出现，聚合成 {r['dataset']['trainPositions']:,} 个相对颜色/D4 规范化局面。Logistello 独立语料全部排除；WTHOR 中的程序参赛记录按 WTHOR 来源处理，没有凭名称另删。",'',
        ('过滤仅使用已冻结 train-only 年初评级；双方已知且均严格低于 1600 才剔除，评级恰好 1600 保留，缺失评级单独计数并保留。没有在过滤后重算 Elo。' if version=='v1.43' else '没有 Elo 阈值、棋力筛选或 Elo 权重。'),'',
        '每次出现等权贡献策略质量与胜/负/平计数。对称稳定子等价落点均分质量；每个唯一局面每轮采样一次，次数用于形成软标签，不用于增加该局面的采样/损失权重。标量目标为实战分布的 P(胜)-P(负)，辅助 W/L/D 为软分布；它们并非精确博弈价值。','',
        '全网从随机参数重训；seed、原始随机表/主干/完整价值策略头初值与 v1.37 相同。15 轮、batch 2048、AdamW lr 0.001、weight decay 0.0001、两个 CPU 线程；软价值 MSE +0.1 软 W/L/D CE +0.1 等权经验合法走法 CE。数据量变化会改变每轮优化步数和具体采样序列，不能声称与原百万训练行顺序完全相同。','',
        f"按原始冻结 6,000 局面验证胜负 MSE 选中第 {candidate['bestEpoch']} 轮，然后才统一计算测试指标。验证/测试各 6,000 个棋盘、行棋方、原结果和单走法标签完全保留；没有把 holdout 也改成 WTHOR-only。",'',
        '| 模型 | 验证 MSE | 测试 MSE | 测试走法命中率 | 测试策略 CE |','|---|---:|---:|---:|---:|']
    for label,model in models:
        vm,tm=model['metrics']['validation'],model['metrics']['test'];lines.append(f"| {label} | {vm['mse']:.6f} | {tm['mse']:.6f} | {tm['policyTop1']:.2%} | {tm['policyCrossEntropy']:.6f} |")
    lines+=['','验证和测试合计 12,000 局面通过 PyTorch/冻结原生库导出一致性对照；网络结构和推理实现均沿用冻结基线，本轮没有修改 C++ 内核或浏览器 Wasm。','',
        '单训练种子。全量版同时改变数据来源和多标签聚合，不能把其影响分别归因；1600 版进一步改变训练子集。离线指标不能替代实战，也不自动授权提升默认。','',
        ('v1.42 的 2s/步 20 局对照独立记录于 `v1.42-pattern-wthor-2s-20-comparison.md`；双方用同一冻结原生库、独立进程，训练结束后才开始。' if version=='v1.42' else '本候选按用户要求在 v1.42 的 2s/20 对局结束后开始训练，避免并行训练干扰其计时。本轮没有额外安排 1600 版棋力对战。'),'',
        f'模型：`{candidate["model"]}`。训练记录：`{prefix}.json`；协议：`{prefix}-protocol.json`；源码快照：`archive/{prefix}-sources.json`。审计：`python3 scripts/finalize-wthor-training.py --version {version}`。监控仍为 `sh scripts/watch-match.sh`。']
    summary=ROOT/f'models/{prefix}-comparison.md';immutable(summary,'\n'.join(lines)+'\n')
    p=ROOT/candidate['model'];files=[path,summary,p,p.with_suffix('.pt'),p.with_suffix('.json'),ROOT/f'models/{prefix}-protocol.json',ROOT/f'models/archive/{prefix}-sources.json']
    immutable(ROOT/f'models/{prefix}-experiment.json',dict(version=version,status='candidate',defaultPromoted=False,artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in files},preservedSha256=protocol['preservedSha256']))
    print(json.dumps(dict(complete=True,version=version,bestEpoch=candidate['bestEpoch'],test=candidate['metrics']['test']),ensure_ascii=False))


if __name__=='__main__':main()

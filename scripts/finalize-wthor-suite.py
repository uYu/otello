#!/usr/bin/env python3
"""Summarize the two same-architecture WTHOR training candidates and first match."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def immutable(path,value):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)


def main():
    base=read(ROOT/'models/pattern32-symmetry-v1.37.json');full=read(ROOT/'models/v1.42-pattern-wthor.json');filtered=read(ROOT/'models/v1.43-pattern-wthor1600.json');match=read(ROOT/'models/v1.42-wthor-vs-v1.37-2s-stopped.json');assert match['userStopped'] and not match['complete'] and match['games']==12
    for p in ['models/v1.42-pattern-wthor-experiment.json','models/v1.43-pattern-wthor1600-experiment.json','models/v1.42-pattern-wthor-2s-stopped-experiment.json','models/v1.41-wthor-elo-filter-experiment.json']:
        r=read(ROOT/p)
        for name,digest in {**r['artifactSha256'],**r['preservedSha256']}.items():assert sha(ROOT/name)==digest,name
    assert full['dataset']['sources']==filtered['dataset']['sources']==['wthor']
    assert not full['dataset']['eloFiltering'] and filtered['dataset']['eloFiltering'] and filtered['dataset']['eloThreshold']==1600
    assert not full['dataset']['eloWeighting'] and not filtered['dataset']['eloWeighting']
    assert full['dataset']['frozenCacheSha256']==filtered['dataset']['frozenCacheSha256']==sha(ROOT/'.build/training/v1/positions.npz')
    assert full['dataset']['frozenPolicySha256']==filtered['dataset']['frozenPolicySha256']==sha(ROOT/'.build/training/v1.8/policy.npz')
    for r in [full,filtered]:
        assert r['epochs']==15 and r['seed']==20261029 and r['dataSeed']==20261030
        assert r['variants'][0]['parameters']==base['parameters'] and r['variants'][0]['initialEmbeddingSha256']==base['initialEmbeddingSha256']
    score=match['results']['candidate']
    lines=['# WTHOR 数据对照：全量 v1.42 与 1600 过滤 v1.43','',
        '按用户要求，两版均使用已批准 v1.37 对称共享原结构，不使用直连 MLP；训练来源仅为 WTHOR，独立 Logistello 语料排除，聚合按原始次数等权。两版都从相同随机初值训练 15 轮，按原冻结验证胜负 MSE 选权重。','',
        '1600 版仅剔除双方已知年初 train-only Elo 都严格低于 1600 的对局；没有 Elo 聚合或损失权重，未知评级另计并保留。所有验证/测试棋盘、行棋方、结果和策略标签保持原值，每组各 6,000 局面，没有改成仅 WTHOR 的 holdout。','',
        '| 模型 | 训练唯一局面 | 选中轮次 | 验证 MSE | 测试 MSE | 测试走法命中率 | 测试策略 CE |','|---|---:|---:|---:|---:|---:|---:|']
    for label,data,count in [('批准 v1.37',base,1000000),('WTHOR 全量 v1.42',full['variants'][0],full['dataset']['trainPositions']),('WTHOR 1600 v1.43',filtered['variants'][0],filtered['dataset']['trainPositions'])]:
        vm,tm=data['metrics']['validation'],data['metrics']['test'];lines.append(f"| {label} | {count:,} | {data['bestEpoch']} | {vm['mse']:.6f} | {tm['mse']:.6f} | {tm['policyTop1']:.2%} | {tm['policyCrossEntropy']:.6f} |")
    lines+=['','全量版同时改变训练来源与多标签聚合，不能分别归因；1600 版与全量版之间还改变了训练子集和每轮优化步数。单种子、固定训练轮数，指标按实际测量记录，不自动授权更换默认。','',
        f"WTHOR 全量 v1.42 对冻结 v1.37 的 2s/步、用户终止后实际 {match['games']} 局结果：{score['win']} 胜 / {score['draw']} 平 / {score['loss']} 负，得分率 {score['scoreRate']:.1%}，平均子差 {score['meanDiscDifference']:+.2f}。双方使用同一冻结原生库与搜索设置，独立进程、交替执色、成对共享种子、无开局库。用户要求停止这组对局、原生进程退出后，1600 版才开始训练。",'',
        '1600 版本轮仅完成训练和冻结验证/测试，没有额外安排棋力对局。提前停止的 12 局是小样本，不据此推断 Elo 或统计显著性，不合并其他预算结果。默认仍为用户批准的 v1.37，浏览器 Wasm 和开局库保持原配置。','',
        '两版各通过验证/测试合计 12,000 局面的 PyTorch/冻结原生库导出对照。v1.42 已完成的实战记录逐行动独立回放，终盘与访问数选着均审计。没有改动 C++ 推理实现。','',
        '详细报告：`v1.42-pattern-wthor-comparison.md`、`v1.43-pattern-wthor1600-comparison.md`、`v1.42-pattern-wthor-2s-20-comparison.md`。数据筛选表：`v1.41-wthor-elo-filter.md`。稳定监控入口为 `sh scripts/watch-match.sh`，保留 v1.42 终盘及 v1.43 训练结果。']
    path=ROOT/'models/v1.43-wthor-data-comparison.md';immutable(path,'\n'.join(lines)+'\n')
    paths=[path]+[ROOT/p for p in ['models/v1.42-pattern-wthor.json','models/v1.43-pattern-wthor1600.json','models/v1.42-wthor-vs-v1.37-2s-stopped.json','models/v1.42-pattern-wthor-experiment.json','models/v1.43-pattern-wthor1600-experiment.json','models/v1.42-pattern-wthor-2s-stopped-experiment.json']]
    immutable(ROOT/'models/v1.43-wthor-data-suite.json',dict(status='complete',defaultPromoted=False,filteredCandidateStrengthMatchRun=False,fullCandidateResults=score,fullCandidateMatchUserStopped=True,fullCandidateCompletedGames=match['games'],artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in paths},preservedSha256=read(ROOT/'models/v1.42-pattern-wthor-protocol.json')['preservedSha256'],scriptSha256=sha(Path(__file__))))
    config=read(ROOT/'models/current-experiment.json');config['version']='WTHOR 全量/1600 同结构对照完成';(ROOT/'models/current-experiment.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(complete=True,fullMatch=score,filteredTest=filtered['variants'][0]['metrics']['test']),ensure_ascii=False))


if __name__=='__main__':main()

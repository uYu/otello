#!/usr/bin/env python3
"""Replay and freeze the user-requested WTHOR symmetry 2s/20-game pilot."""
import hashlib,importlib.util,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
spec=importlib.util.spec_from_file_location('direct_pilot_audit',ROOT/'scripts/compare-nnue-candidate.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)


def main():
    protocol=read(ROOT/'models/v1.42-pattern-wthor-2s-20-protocol.json')
    path=ROOT/'models/v1.42-wthor-vs-v1.37-2s-20.json';r=read(path)
    assert r['complete'] and r['games']==r['plannedGames']==len(r['records'])==20 and r['millisecondsPerMove']==2000
    assert not protocol['concurrentTraining'] and r['librarySourceManifest']==protocol['librarySourceManifest']
    assert r['librarySourceManifestSha256']==protocol['librarySourceManifestSha256']==sha(ROOT/protocol['librarySourceManifest'])
    assert r['baselineVersion']=='v1.37' and not r['useBook'] and r['baseSeed']==20261029
    assert r['protocolSha256']==sha(ROOT/'models/v1.42-pattern-wthor-2s-20-protocol.json')
    assert r['maxNodesByEngine']==dict(baseline=4194304,candidate=4194304) and r['patternCacheEntries']==0
    assert r['baselineManifestSha256']==protocol['baselineManifestSha256']
    assert [v['modelSha256'] for v in r['configs']]==[protocol['baselineModelSha256'],protocol['candidateModelSha256']]
    assert all(v['prior']=='learned' and v['valueTarget']=='outcome' for v in r['configs'])
    source=read(ROOT/r['sourceArchiveManifest']);assert sha(ROOT/source['archive'])==source['sha256']
    with tarfile.open(ROOT/source['archive'],'r:gz') as t:
        for name,digest in source['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
    for name,digest in r['sourceSha256'].items():assert source['files'][name]==digest,name
    for name,digest in {**protocol['preservedSha256'],**protocol['librarySha256']}.items():assert sha(ROOT/name)==digest,name
    c.match.CONFIGS=r['configs'];actions=0
    for i,game in enumerate(r['records']):
        assert game['number']==i+1 and (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(20261029^((i//2+1)*0x9e3779b9))&0x7fffffff
        assert game['opening']==[] and game['openingPolicy']==c.match.POLICY
        assert all(d['valueTarget']=='outcome' and d['mcts']['maxNodes']==4194304 for d in game['decisions'])
        actions+=c.match.replay(game)
    totals,matrix=c.match.aggregate(r['records']);assert totals==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']
    score=totals['candidate']
    lines=['# v1.42 WTHOR 全量：同结构基线对照，2s/步、20 局','',
        '用户要求先用 WTHOR 全量、不筛 Elo、聚合不加权，并使用对称共享基线原结构，不使用直连 MLP。候选为重新随机训练的 v1.42；对手为用户批准、冻结的 v1.37。训练完成后沿用上一轮 2s/步、20 局预算比较；未提升默认。','',
        '双方使用同一个冻结 v1.37 原生库、各自独立进程；训练已结束，无并行训练；每步 2s、4,194,304 节点；相同 PUCT/FPU、v1.3 开局多样性、子树复用和胜负终局证明。初始棋盘出发，无开局库；交替执黑白，每对颜色共享种子，共 10 对、20 局。','',
        '| 候选胜 | 平 | 负 | 得分率 | 平均子差 |','|---:|---:|---:|---:|---:|',
        f"| {score['win']} | {score['draw']} | {score['loss']} | {score['scoreRate']:.1%} | {score['meanDiscDifference']:+.2f} |",'',
        '| 引擎 | 思考步数 | 平均耗时 ms | 平均模拟/步 | 平均新叶评估/步 | 节点触顶 | 已证明步数 | 峰值 RSS MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for identity in ['candidate','baseline']:
        decisions=[d for game in r['records'] for d in game['decisions'] if d['engine']==identity];n=len(decisions);v=totals[identity]
        lines.append(f"| {identity} | {n} | {v['meanWallMs']:.2f} | {v['meanSearchUnits']:,.0f} | {sum(d['mcts']['evaluations'] for d in decisions)/n:,.0f} | {v['memoryLimitedSearches']} | {v['exactMoves']} | {max(d['mcts']['peakRssBytes'] for d in decisions)/1024**2:.1f} |")
    lines+=['',f'20 局共 {actions:,} 个行动均已独立回放，每步合法着、访问数选着、开局随机选择、终局与胜负均审计。各引擎遇到的局面不同，上表为实战描述性统计，不是同局面速度对照。','',
        '这是小规模配对预检，不据此推断 Elo 或统计显著性，不与不同预算结果合并。训练与原始冻结验证/测试指标独立保存在 `v1.42-pattern-wthor-comparison.md`。默认选择器、v1.37 权重、冻结记录和浏览器 Wasm 保持原哈希。','',
        '监控入口：`sh scripts/watch-match.sh`，候选视角 W/D/L 仅显示一次，逐步更新 ASCII 棋盘并保留终盘。原始报告：`v1.42-wthor-vs-v1.37-2s-20.json`。协议与开赛前源码快照分别为 `v1.42-pattern-wthor-2s-20-protocol.json` 和 `archive/v1.42-pattern-wthor-2s-20-sources.json`。复核：`python3 scripts/finalize-pattern-wthor-match.py`。']
    summary=ROOT/'models/v1.42-pattern-wthor-2s-20-comparison.md';text='\n'.join(lines)+'\n'
    if summary.exists():assert summary.read_text()==text
    else:summary.write_text(text)
    manifest=ROOT/'models/v1.42-pattern-wthor-2s-20-experiment.json'
    files=[path,summary,ROOT/'models/v1.42-pattern-wthor-2s-20-protocol.json',ROOT/r['sourceArchiveManifest']]
    record=dict(version='v1.42-pattern-wthor-2s-pilot',status='candidate',defaultPromoted=False,candidateResults=score,replayedActions=actions,artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in files},preservedSha256=protocol['preservedSha256'])
    content=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
    if manifest.exists():assert manifest.read_text()==content
    else:manifest.write_text(content)
    print(json.dumps(dict(complete=True,win=score['win'],draw=score['draw'],loss=score['loss'],scoreRate=score['scoreRate'],replayedActions=actions),ensure_ascii=False))


if __name__=='__main__':main()

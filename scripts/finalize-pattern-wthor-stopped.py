#!/usr/bin/env python3
"""Audit completed v1.42 games after explicit user stop; never mark 20 complete."""
import hashlib,importlib.util,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
spec=importlib.util.spec_from_file_location('wthor_stopped_audit',ROOT/'scripts/compare-nnue-candidate.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)


def immutable(path,value):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)


def main():
    path=ROOT/'models/v1.42-wthor-vs-v1.37-2s-stopped.json';r=read(path);protocol=read(ROOT/'models/v1.42-pattern-wthor-2s-20-protocol.json')
    assert r['userStopped'] and r['stopped'] and not r['complete'] and not r['partialGameIncludedInResults']
    assert r['games']==len(r['records'])<r['plannedGames']==20 and r['millisecondsPerMove']==2000
    assert r['protocolSha256']==sha(ROOT/'models/v1.42-pattern-wthor-2s-20-protocol.json') and not r['useBook']
    assert r['librarySha256']==r['baselineLibrarySha256']==next(iter(protocol['librarySha256'].values()))
    for name,digest in {**protocol['preservedSha256'],**protocol['librarySha256']}.items():assert sha(ROOT/name)==digest,name
    source=read(ROOT/r['sourceArchiveManifest']);assert sha(ROOT/source['archive'])==source['sha256']
    with tarfile.open(ROOT/source['archive'],'r:gz') as t:
        for name,digest in source['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
    for name,digest in r['sourceSha256'].items():assert source['files'][name]==digest,name
    c.match.CONFIGS=r['configs'];actions=0
    for i,game in enumerate(r['records']):
        assert game['number']==i+1 and (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(20261029^((i//2+1)*0x9e3779b9))&0x7fffffff
        assert all(d['mcts']['maxNodes']==4194304 and d['valueTarget']=='outcome' for d in game['decisions'])
        actions+=c.match.replay(game)
    results,matrix=c.match.aggregate(r['records']);assert results==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']
    score=results['candidate'];text=f'''# v1.42 WTHOR 全量 vs v1.37：用户停止的 2s 对照

原计划 2s/步、20 局；用户要求停止，实际只完成 {r['games']} 局。统计仅包含已完成对局，第 {(r.get('abortedPartialGame') or {}).get('number')} 局部分过程保留作追溯，不计入结果。`complete=false`、`userStopped=true`，不得写成完成 20 局。

候选视角为 {score['win']} 胜 / {score['draw']} 平 / {score['loss']} 负，得分率 {score['scoreRate']:.1%}，平均子差 {score['meanDiscDifference']:+.2f}。全部 {actions:,} 个已完成行动独立回放；交替执色、每对共享种子、无开局库、双方 4M 节点和同一冻结原生库。训练在对局开始前结束，1600 版在用户终止这组对局且原生进程退出后才开始。

这是按用户要求提前停止的小样本，尤其不能据此推断 Elo 或统计显著性。当前测量呈现回归迹象，默认仍保留 v1.37。该实验同时改变来源为 WTHOR-only、训练唯一局面数和经验多标签聚合，不能将回归单独归因于数据量或聚合。新全量训练局面为 1,054,132，原基线为 1,000,000；总行数没有减少，但有效来源覆盖发生变化。

原始已完成记录：`v1.42-wthor-vs-v1.37-2s-stopped.json`；原计划协议与开赛源码快照仍保持原哈希。源代码档案中的 C++ 来自双方实际使用的冻结 v1.37 原生库；Python 运行脚本单独捕获。稳定监控入口 `sh scripts/watch-match.sh` 保留最后已完成的终盘。

训练/冻结验证与测试报告：`v1.42-pattern-wthor-comparison.md`。审计：`python3 scripts/finalize-pattern-wthor-stopped.py`。
'''
    summary=ROOT/'models/v1.42-pattern-wthor-2s-20-comparison.md';immutable(summary,text)
    files=[path,summary,ROOT/'models/v1.42-pattern-wthor-2s-20-protocol.json',ROOT/r['sourceArchiveManifest']]
    immutable(ROOT/'models/v1.42-pattern-wthor-2s-stopped-experiment.json',dict(status='user-stopped',complete=False,plannedGames=20,completedGames=r['games'],partialGameIncluded=False,candidateResults=score,replayedActions=actions,artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in files},preservedSha256=protocol['preservedSha256'],scriptSha256=sha(Path(__file__))))
    print(json.dumps(dict(stopped=True,completedGames=r['games'],results=score,replayedActions=actions),ensure_ascii=False))


if __name__=='__main__':main()

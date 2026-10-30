#!/usr/bin/env python3
"""Audit larger-capacity pilot, 2s/20-game outcome and memory evidence."""
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
CONFIG=ROOT/'models/v1.36-pattern-capacity.json'
REPORT=ROOT/'models/v1.36-capacity-vs-v1.25-2s-20.json'
SUMMARY=ROOT/'models/v1.36-pattern-capacity-comparison.md'
MANIFEST=ROOT/'models/v1.36-pattern-capacity-experiment.json'


def main():
    config=read(CONFIG);pilot=read(ROOT/config['pilotReport'])
    assert not config['defaultPromoted'] and sha(ROOT/config['pilotReport'])==config['pilotSha256']
    assert pilot['fixedSimulationParity'] and pilot['selectedMaxNodes']==config['maxNodes']
    safe=[v for v in pilot['results'][1:] if not v['memoryGuardExceeded']]
    good=[v for v in safe if v['capped']<=.05*len(v['samples']) and v['fullBudget']>=.95*v['nonProven']]
    chosen=min(good,key=lambda v:v['maxNodes']) if good else min(safe,key=lambda v:(v['capped']/len(v['samples']),-v['evaluations']))
    assert chosen['maxNodes']==config['maxNodes']
    r=read(REPORT if REPORT.exists() else ROOT/config['match']['runtimeReport'])
    assert r['complete'] and r['games']==r['plannedGames']==len(r['records'])==20
    assert r['millisecondsPerMove']==2000 and r['baseSeed']==20261029 and r['start']=='initial' and not r['useBook']
    assert r['maxNodesByEngine']==dict(baseline=2097152,candidate=config['maxNodes'])
    assert r['capacityPilotSha256']==config['pilotSha256']
    assert r['librarySha256']==config['librarySha256'] and r['baselineLibrarySha256']==config['baselineLibrarySha256']
    assert r['baselineManifestSha256']==sha(ROOT/'models/v1.25-default.json')
    assert [(c['prior'],c['valueTarget']) for c in r['configs']]==[('learned','outcome')]*2
    assert all(c['modelSha256']==config['modelSha256'] for c in r['configs'])
    for p,digest in config['preservedSha256'].items():assert sha(ROOT/p)==digest,p
    snapshot=read(ROOT/'models/archive/v1.36-pattern-capacity-sources.json')
    assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
    with tarfile.open(ROOT/snapshot['archive'],'r:gz') as stream:
        for p,digest in snapshot['files'].items():assert hashlib.sha256(stream.extractfile(p).read()).hexdigest()==digest,p
    for p,digest in r['sourceSha256'].items():assert snapshot['files'][p]==digest,p
    previous_sources=read(ROOT/'models/archive/v1.35-pattern-speed-suite-sources.json')['files']
    for p,digest in r['sourceSha256'].items():
        if p not in ['cpp/api.h','cpp/api.cpp','cpp/mcts.cpp']:assert previous_sources[p]==digest,p
    for field,path in [('runnerSha256','scripts/compare-nnue-candidate.py'),('sharedRunnerSha256','scripts/compare-nnue-search.py'),('rulesSha256','scripts/compare-resnet.py'),('policyAuditSha256','scripts/opening_policy.py')]:assert r[field]==snapshot['files'][path]
    spec=importlib.util.spec_from_file_location('capacity_final_audit',ROOT/'scripts/compare-nnue-candidate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);match=module.match;match.CONFIGS=r['configs'];actions=0
    for i,game in enumerate(r['records']):
        assert game['number']==i+1 and game['opening']==[] and game['openingPolicy']==match.POLICY
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(r['baseSeed']^((i//2+1)*0x9e3779b9))&0x7fffffff
        for d in game['decisions']:
            assert d['valueTarget']=='outcome' and d['mcts']['maxNodes']==r['maxNodesByEngine'][d['engine']]
            assert d['mcts']['treeNodes']<=d['mcts']['maxNodes'] and d['mcts']['peakRssBytes']>0
        actions+=match.replay(game)
    totals,matrix=match.aggregate(r['records']);assert totals==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']
    if REPORT.exists():assert read(REPORT)==r
    else:write(REPORT,r)
    score=totals['candidate'];old=read(ROOT/'models/v1.35-speed-suite-vs-v1.25-2s-20.json');oldscore=old['results']['candidate']
    text='''# 策略集节点扩容 v1.36

检验上一轮更快的策略集是否被 2,097,152 节点上限过早截断。本轮保持原 INT16 权重与所有提速开关，
新增显式容量实验 C ABI；旧接口及默认仍为 2,097,152 节点，仅实验接口可请求最多 16,777,216。
库不预分配整个上限，只随搜索按需增长。批准默认模型、参考库与浏览器 Wasm 保持冻结。

## 连续局面预检与内存

从上轮按候选触顶次数排序选取前两局，从每局开局连续重放至第一次触顶后的七个候选思考局面，
保留实际路径与子树复用。所有容量使用相同已记录局面，每步 2000ms；逐容量使用独立进程测峰值 RSS。
选择规则在预检前固定：优先选触顶率 <=5%、且 >=95% 未证明局面用时至少 1900ms 的最小安全容量；
否则在安全容量中先最小化触顶率，再最大化评估数。内存保护阈值为 min(4GiB, 物理内存/6)。
这些局面专为诊断触顶选择，不能当作代表性棋力/速度基准，峰值 RSS 包含解释器、表和树。
新 8M 容量与旧库在不触顶的 2048 固定模拟下，落点、分值、全部候选和选择元数据精确相同。

| 节点上限 | 思考数 | 触顶数 | 未证明用满预算 | 新叶评估总数 | 峰值 RSS MiB |
| ---: | ---: | ---: | --- | ---: | ---: |
'''
    for row in pilot['results']:text+=f"| {row['maxNodes']:,} | {len(row['samples'])} | {row['capped']} | {row['fullBudget']}/{row['nonProven']} | {row['evaluations']:,} | {row['peakRssBytes']/1048576:.1f} |\n"
    text+=f"\n选中 **{config['maxNodes']:,} 节点**；本轮候选容量变大，基线仍为原批准的 2,097,152 节点。这是不同内存预算下的整套实现对照，不是只改变指令内核。\n"
    text+='''
## 2秒/步、20局

双方相同时间，保留原开局多样性、PUCT/FPU、终局证明与配对颜色/种子；无开局库。
下表上轮与本轮是两次独立墙钟实验，不能把胜负差额当作确定的容量因果效应。

| 实验 | 胜 | 平 | 负 | 得分率 | 平均子差 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for label,v in [('上轮策略集 2M',oldscore),(f'本轮策略集 {config["maxNodes"]//1048576}M',score)]:text+=f"| {label} | {v['win']} | {v['draw']} | {v['loss']} | {v['scoreRate']:.1%} | {v['meanDiscDifference']:+.2f} |\n"
    text+='''
## 实战预算利用与采样

各引擎遇到的局面不同，此表是描述性统计；模拟包含树内重复访问，网络评估更接近新叶数量。
已证明的提前停止是正常行为，用时不足 2 秒不能都归因于节点上限。

| 实验/引擎 | 思考数 | 触顶数 | 触顶平均 ms | 未证明且未触顶平均 ms | 网络评估/步 | 峰值 RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    for label,report in [('上轮',old),('本轮',r)]:
        for identity in ['baseline','candidate']:
            decisions=[d for g in report['records'] for d in g['decisions'] if d['engine']==identity]
            capped=[d for d in decisions if d['mcts']['memoryLimited']];normal=[d for d in decisions if not d['exact'] and not d['mcts']['memoryLimited']]
            rss=max((d['mcts'].get('peakRssBytes',0) for d in decisions),default=0)
            rss_label=f'{rss/1048576:.1f}' if rss else '未记录'
            text+=f"| {label}/{identity} | {len(decisions)} | {len(capped)} | {sum(d['wallMs'] for d in capped)/len(capped) if capped else 0:.1f} | {sum(d['wallMs'] for d in normal)/len(normal) if normal else 0:.1f} | {sum(d['mcts']['evaluations'] for d in decisions)/len(decisions):,.0f} | {rss_label} |\n"
    text+=f"\n20 局共 {actions} 个行动完整独立回放，所有决策逐步核验。样本有限，不推断 Elo 或统计显著性；本机 ARM64 的内存/性能不代表 Linux 服务器。\n"
    text+='''
原生测试检查放宽上限后的固定模拟一致性、小容量精确触顶及非法上限拒绝；旧默认值未变。
每个思考记录 maxNodes、treeNodes、memoryLimited 及进程峰值 RSS（macOS 原字节，Linux KiB 转字节）。
源码、库、模型和脚本哈希、原始预检与对战记录均冻结，没有提升默认或部署。
监控 `sh scripts/watch-match.sh` 保留终盘；复核 `python3 scripts/finalize-pattern-capacity.py`。
'''
    if SUMMARY.exists():assert SUMMARY.read_text()==text
    else:SUMMARY.write_text(text)
    if not MANIFEST.exists():
        artifacts=[CONFIG,ROOT/config['pilotReport'],ROOT/'models/v1.36-pattern-capacity-experiments.json',REPORT,SUMMARY,Path(__file__).resolve()]
        write(MANIFEST,dict(version=config['version'],status='candidate',defaultPromoted=False,sourceManifest='models/archive/v1.36-pattern-capacity-sources.json',sourceManifestSha256=sha(ROOT/'models/archive/v1.36-pattern-capacity-sources.json'),artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},preservedSha256=config['preservedSha256'],checks=dict(games=20,replayedActions=actions,pairedColors=True,budgetMs=2000,perDecisionCapacityAndRss=True)))
    frozen=read(MANIFEST);assert sha(ROOT/frozen['sourceManifest'])==frozen['sourceManifestSha256']
    for p,digest in {**frozen['artifactSha256'],**frozen['preservedSha256']}.items():assert sha(ROOT/p)==digest,p
    print(f"Verified capacity experiment: {score['win']}W/{score['draw']}D/{score['loss']}L, {actions} replayed actions, default preserved")


if __name__=='__main__':main()

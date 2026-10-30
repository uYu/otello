#!/usr/bin/env python3
"""Audit and freeze the combined 2s/20-game Pattern optimization suite."""
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
SUITE=ROOT/'models/v1.35-pattern-speed-suite.json'
RUNTIME=ROOT/'.build/training/pattern-speed-suite-v1.35/match-2s-20.json'
REPORT=ROOT/'models/v1.35-speed-suite-vs-v1.25-2s-20.json'
MANIFEST=ROOT/'models/v1.35-pattern-speed-suite-experiment.json'
SUMMARY=ROOT/'models/v1.35-pattern-speed-suite-comparison.md'


def main():
    suite=read(SUITE);sampling=read(ROOT/suite['samplingReport']);r=read(REPORT if REPORT.exists() else RUNTIME)
    assert not suite['defaultPromoted'] and sha(ROOT/suite['samplingReport'])==suite['samplingSha256']
    for p,digest in suite['preservedSha256'].items():assert sha(ROOT/p)==digest,p
    expected=2 if sampling['results'][2]['simulationsPerSecond']>1.05*sampling['results'][1]['simulationsPerSecond'] else 1
    assert expected==sampling['selectedIndex'] and suite['selected']==sampling['results'][expected]
    assert r['complete'] and r['games']==r['plannedGames']==len(r['records'])==20
    assert r['millisecondsPerMove']==2000 and r['baseSeed']==20261029 and r['start']=='initial' and not r['useBook']
    assert r['baselineVersion']=='v1.25' and r['baselineManifestSha256']==suite['baselineManifestSha256']
    baseline=read(ROOT/suite['baselineManifest']);assert r['baselineLibrarySha256']==baseline['nativeLibrarySha256']
    assert r['librarySha256']==suite['selected']['librarySha256']
    assert [(c['prior'],c['valueTarget']) for c in r['configs']]==[('learned','outcome')]*2
    assert r['configs'][1]['modelSha256']==suite['selected']['modelSha256']
    for c in r['configs']:assert sha(ROOT/c['model'])==c['modelSha256']
    snapshot=read(ROOT/'models/archive/v1.35-pattern-speed-suite-sources.json')
    assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
    with tarfile.open(ROOT/snapshot['archive'],'r:gz') as stream:
        for p,digest in snapshot['files'].items():assert hashlib.sha256(stream.extractfile(p).read()).hexdigest()==digest,p
    for p,digest in r['sourceSha256'].items():assert snapshot['files'][p]==digest,p
    for field,path in [('runnerSha256','scripts/compare-nnue-candidate.py'),('sharedRunnerSha256','scripts/compare-nnue-search.py'),('rulesSha256','scripts/compare-resnet.py'),('policyAuditSha256','scripts/opening_policy.py')]:assert r[field]==snapshot['files'][path]
    spec=importlib.util.spec_from_file_location('suite_final_audit',ROOT/'scripts/compare-nnue-candidate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);match=module.match;match.CONFIGS=r['configs'];actions=0
    for i,game in enumerate(r['records']):
        assert game['number']==i+1 and game['opening']==[] and game['openingPolicy']==match.POLICY
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(r['baseSeed']^((i//2+1)*0x9e3779b9))&0x7fffffff
        assert all(d['valueTarget']=='outcome' for d in game['decisions'])
        actions+=match.replay(game)
    results,matrix=match.aggregate(r['records']);assert results==r['results'] and matrix==r['matrix'] and actions==r['replayedActions']
    if REPORT.exists():assert read(REPORT)==r
    else:write(REPORT,r)
    score=results['candidate']
    text='''# 提速策略集 v1.35：总体采样与 2秒/20局

批准基线为冻结 v1.25 Pattern outcome，使用原参考库；候选组合启用当前行棋方融合推理、
整数 SIMD（ARM64 NEON / x86-64 SSE2）、位压缩编码、四行 SIMD、合法着相关策略投影。
排除已测得较慢的增量缓存与整数舍入，不增大主干宽度。没有重训或自动提升默认。

## 同局面总体采样

从冻结验证缓存按占据数 4..13、14..23、24..33、34..43 各取两个局面，
每个实现各测两轮、每根 2000ms，交错顺序；每次清树。
只在所有实现/轮次均未证明、未触及内存限制、实际用时至少 1900ms 的共同局面上比较模拟/秒。
选择规则在测量前固定：INT8 必须比 INT16 吞吐高 >5% 才选；否则保留原批准 INT16 权重。
没有用测试集标签或本轮胜负选择精度；完整原始样本、证明状态及壁钟开销均保留。

| 实现 | 模型字节 | 模拟/秒 | 网络评估/秒 | 模拟倍率 | 评估倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    def evaluation_rate(row):
        samples=[sample for sample in row['samples'] if sample['fixture'] in sampling['commonFixtureIndices']]
        return sum(sample['evaluations'] for sample in samples)*1000/sum(sample['wallMs'] for sample in samples)
    for row in sampling['results']:text+=f"| {row['name']} | {row['modelBytes']:,} | {row['simulationsPerSecond']:,.0f} | {evaluation_rate(row):,.0f} | {row['throughputVsApproved']:.3f}× | {evaluation_rate(row)/evaluation_rate(sampling['results'][0]):.3f}× |\n"
    text+=f"\n共同计时局面 {len(sampling['commonFixtureIndices'])}/8，选中 **{sampling['selected']}**。这是整套实现的直接测量，没有把历史单项倍率相乘。\n"
    text+='''
## 对战

双方 2000ms/步，初始棋盘，20 局交替执黑白，每对颜色共享种子；
保留原 v1.3 开局温度/多样性、PUCT、FPU、子树复用与 outcome 终局证明。
这是无开局库的模型/搜索对照，与网页部署的强制开局库流程分开。
每步由独立规则与选着审计验证，完成后再次完整回放。

| 候选胜 | 平 | 负 | 得分率 | 平均子差 |
| ---: | ---: | ---: | ---: | ---: |
'''
    text+=f"| {score['win']} | {score['draw']} | {score['loss']} | {score['scoreRate']:.1%} | {score['meanDiscDifference']:+.2f} |\n"
    text+=f"\n独立回放 {actions} 个行动。样本仅 20 局，不推断 Elo 或统计显著性；配对种子不能消除墙钟预算和证明停止带来的路径差异。\n"
    text+='''
## 对战中的实际采样（描述性统计）

下表来自各自遇到的局面，不能视为同局面速度对照；已证明的提前停止局面也会影响平均值。
受控吞吐对照应以上一节的同局面采样为准。

| 引擎 | 思考步数 | 平均模拟/步 | 平均网络评估/步 | 触及节点上限 | 触顶平均耗时 ms | 已证明步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    for identity in ['baseline','candidate']:
        decisions=[d for game in r['records'] for d in game['decisions'] if d['engine']==identity]
        nodes=sum(d['mcts']['simulations'] for d in decisions)
        capped=[d for d in decisions if d['mcts']['memoryLimited']]
        cap_ms=sum(d['wallMs'] for d in capped)/len(capped) if capped else 0
        text+=f"| {identity} | {len(decisions)} | {nodes/len(decisions):,.0f} | {sum(d['mcts']['evaluations'] for d in decisions)/len(decisions):,.0f} | {len(capped)} | {cap_ms:.1f} | {sum(d['exact'] for d in decisions)} |\n"
    text+='''
双方保留相同的 2,097,152 节点上限；触及上限会按现有逻辑提前停止，以上记录次数。运行中只修正了监控显示：默认聚焦本轮，历史可用 `sh scripts/watch-match.sh --all` 查看；对战实现和参数没有改变。模拟数包含树内重复访问，不能等同于独立新局面；网络评估次数另列。

所有模型、原批准库、候选库和源码/脚本 SHA-256 已记录，候选源码在开赛前归档。
当前默认选择器、v1.25 模型与浏览器 Wasm 保持原哈希，未部署或提升候选。
测量平台为本机 ARM64；此前 SSE2/Rosetta 和普通回退已通过一致性，Linux 实机性能仍待验证。
实时/最终棋盘：`sh scripts/watch-match.sh`；复核：`python3 scripts/finalize-pattern-speed-suite.py`。
'''
    if SUMMARY.exists():assert SUMMARY.read_text()==text
    else:SUMMARY.write_text(text)
    if not MANIFEST.exists():
        artifacts=[SUITE,ROOT/suite['samplingReport'],ROOT/'models/v1.35-pattern-speed-suite-experiments.json',REPORT,SUMMARY,Path(__file__).resolve(),ROOT/'scripts/watch-matches.py']
        write(MANIFEST,dict(version=suite['version'],status='candidate',defaultPromoted=False,sourceManifest='models/archive/v1.35-pattern-speed-suite-sources.json',sourceManifestSha256=sha(ROOT/'models/archive/v1.35-pattern-speed-suite-sources.json'),artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},preservedSha256=suite['preservedSha256'],checks=dict(games=20,replayedActions=actions,pairedColors=True,sharedSeeds=True,budgetMs=2000,modelOnly=True)))
    frozen=read(MANIFEST)
    assert sha(ROOT/frozen['sourceManifest'])==frozen['sourceManifestSha256']
    for p,digest in {**frozen['artifactSha256'],**frozen['preservedSha256']}.items():assert sha(ROOT/p)==digest,p
    print(f"Verified strategy suite: {score['win']}W/{score['draw']}D/{score['loss']}L, {actions} replayed actions, unchanged approved default")


if __name__=='__main__':main()

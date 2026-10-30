#!/usr/bin/env python3
"""Freeze and verify the separate 2s/20-game Pattern outcome comparison."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from candidate_era_artifacts import historical_sha

ROOT=Path(__file__).resolve().parents[1]
PREFIX='v1.25-pattern-outcome-2s-20-run1'
RUNTIME=ROOT/'.build/training/pattern-outcome-v1.25/match-2s-20-run1.json'
REPORT=ROOT/'models/v1.25-pattern-outcome-vs-v1.9-2s-20-run1.json'
MANIFEST=ROOT/f'models/{PREFIX}-experiment.json'
SUMMARY=ROOT/f'models/{PREFIX}-comparison.md'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return historical_sha(path)
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def audit(report):
    previous=read(ROOT/'models/v1.25-pattern-outcome-experiment.json')
    snapshot=read(ROOT/previous['sourceManifest'])
    baseline=read(ROOT/'models/v1.9-default.json')
    for name,digest in {**previous['artifactSha256'],**previous['preservedSha256']}.items():
        assert sha(ROOT/name)==digest,name
    assert report['complete'] and report['games']==report['plannedGames']==len(report['records'])==20
    assert report['millisecondsPerMove']==2000 and report['start']=='initial' and not report['useBook']
    assert report['baselineVersion']=='v1.9' and report['baselineManifestSha256']==sha(ROOT/'models/v1.9-default.json')
    assert report['librarySha256']==previous['librarySha256']
    assert report['baselineLibrarySha256']==baseline['nativeLibrarySha256']
    assert [(c['prior'],c['valueTarget']) for c in report['configs']]==[('learned','outcome')]*2
    for config in report['configs']:assert sha(ROOT/config['model'])==config['modelSha256']
    for name,digest in report['sourceSha256'].items():assert snapshot['files'][name]==digest,name
    for key,name in [('runnerSha256','scripts/compare-nnue-candidate.py'),('sharedRunnerSha256','scripts/compare-nnue-search.py'),('rulesSha256','scripts/compare-resnet.py'),('policyAuditSha256','scripts/opening_policy.py')]:
        assert report[key]==snapshot['files'][name],name
    spec=importlib.util.spec_from_file_location('pattern_two_second_match',ROOT/'scripts/compare-nnue-candidate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    match=module.match;match.CONFIGS=report['configs'];actions=0
    for index,game in enumerate(report['records']):
        assert game['number']==index+1 and game['opening']==[] and game['openingPolicy']==match.POLICY
        assert (game['black'],game['white'])==(('baseline','candidate') if index%2==0 else ('candidate','baseline'))
        assert game['seed']==(report['baseSeed']^((index//2+1)*0x9e3779b9))&0x7fffffff
        assert all(d['valueTarget']=='outcome' for d in game['decisions'])
        actions+=match.replay(game)
    assert actions==report['replayedActions']
    results,matrix=match.aggregate(report['records'])
    assert results==report['results'] and matrix==report['matrix']
    return previous,actions


def verify():
    manifest=read(MANIFEST)
    assert manifest['status']=='candidate' and not manifest['defaultPromoted']
    for name,digest in manifest['artifactSha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/manifest['previousExperiment'])==manifest['previousExperimentSha256']
    assert sha(ROOT/manifest['sourceManifest'])==manifest['sourceManifestSha256']
    report=read(REPORT);_,actions=audit(report);r=report['results']['candidate']
    print(f"Verified 2s/20 games: {r['win']}W/{r['draw']}D/{r['loss']}L; {actions} independently replayed actions; 100ms artifacts and v1.9 unchanged")


def freeze():
    assert not MANIFEST.exists(),'Refusing to overwrite a frozen experiment'
    report=read(RUNTIME);previous,actions=audit(report)
    shutil.copyfile(RUNTIME,REPORT)
    old=read(ROOT/'models/v1.25-pattern-outcome-vs-v1.9-100ms-20.json')
    r=report['results']['candidate'];b=report['results']['baseline'];o=old['results']['candidate']
    unique=len({tuple(g['moves']) for g in report['records']})
    SUMMARY.write_text(f'''# Pattern outcome vs v1.9：每步 2 秒、20 局补充对照

沿用 v1.25 已训练的 OTPAT2 胜负价值权重，不重训、不更换搜索实现。
双方分别使用与 100ms 实验完全相同的候选原生库和冻结 v1.9 参考库。
20 局，10 组共享种子交换黑白，从标准初始棋盘开始，无指定开局、无开局库，
前 12 次落子保留共同的开局多样性策略。每步时间上限均为 2,000ms，
已证明结果或达到 2,097,152 个节点上限时可提前停止。
候选保留 Pattern 学习先验与父均值减 0.15 的有限 FPU；双方终局均回传胜 +1、平 0、负 −1。

| 独立时限组（候选视角） | 胜 | 平 | 负 | 得分率 | 平均子差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 100ms / 20 局 | {o['win']} | {o['draw']} | {o['loss']} | {o['scoreRate']:.1%} | {o['meanDiscDifference']:+.2f} |
| 本次 2s / 20 局 | {r['win']} | {r['draw']} | {r['loss']} | {r['scoreRate']:.1%} | {r['meanDiscDifference']:+.2f} |

两组分开记录，不合并计算胜率。两组种子相同，用于控制配对，不表示时限不同的棋谱相同。

| 本次搜索统计 | Pattern outcome | v1.9 |
| --- | ---: | ---: |
| 平均实际耗时／手 | {r['meanWallMs']:.1f}ms | {b['meanWallMs']:.1f}ms |
| 平均模拟次数／手 | {r['meanSearchUnits']:,.0f} | {b['meanSearchUnits']:,.0f} |
| 触及节点上限的搜索次数 | {r['memoryLimitedSearches']} | {b['memoryLimitedSearches']} |

总运行时间 {report['seconds']:.1f} 秒；{unique} 种完整棋谱；全部 {actions} 个动作已通过独立规则、
落点选择、停着、终盘和配对种子审计。
本轮比较完整候选系统，不能单独归因于棋形结构或 FPU；20 局样本不能确定稳定胜率或 Elo。
默认继续为用户批准的 v1.9。

- 逐手记录：`{REPORT.name}`
- 实验校验：`{MANIFEST.name}`
- 模型／实测源码快照：沿用 `v1.25-pattern-outcome-experiment.json` 中的冻结记录。
- 复核：`python3 scripts/finalize-pattern-outcome-2s.py --verify`
- 监控：`sh scripts/watch-match.sh`，结束后保留第 20 局终盘。
''')
    artifacts=[REPORT,SUMMARY,ROOT/f'models/{PREFIX}-experiments.json',Path(__file__).resolve()]
    source_manifest=ROOT/previous['sourceManifest']
    write(MANIFEST,dict(version=PREFIX,status='candidate',defaultPromoted=False,
        previousExperiment='models/v1.25-pattern-outcome-experiment.json',
        previousExperimentSha256=sha(ROOT/'models/v1.25-pattern-outcome-experiment.json'),
        sourceManifest=previous['sourceManifest'],sourceManifestSha256=sha(source_manifest),
        artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts}))
    verify()


if __name__=='__main__':
    verify() if '--verify' in sys.argv else freeze()

#!/usr/bin/env python3
"""Freeze and independently verify the Pattern outcome vs v1.9 experiment."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
from candidate_era_artifacts import historical_sha

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'v1.25-pattern-outcome'
MODEL = ROOT/'models/pattern32-outcome-v1.25.otpat'
REPORT = ROOT/f'models/{PREFIX}-vs-v1.9-100ms-20.json'
MANIFEST = ROOT/f'models/{PREFIX}-experiment.json'


def sha(path):
    return historical_sha(path)


def load(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def comparison():
    spec = importlib.util.spec_from_file_location('pattern_outcome_match', ROOT/'scripts/compare-nnue-candidate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def offline(library):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT/'scripts'))
    from pattern_model import legal_mask
    torch.set_num_threads(2)
    data = np.load(ROOT/'.build/training/v1/positions.npz')
    labels = np.load(ROOT/'.build/training/pattern-outcome-v1.25/policy.npz')
    lib = ctypes.CDLL(str(library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.ot_nnue_load.argtypes = [ctypes.c_char_p]; lib.ot_nnue_load.restype = ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
    lib.ot_nnue_infer.argtypes = [ctypes.c_void_p,ptr,ctypes.c_int,ptr,ctypes.POINTER(ctypes.c_double)]
    baseline = load(ROOT/'models/v1.9-default.json')
    handle = lib.ot_nnue_load(str(ROOT/baseline['defaultModel']).encode())
    assert handle
    output = {}
    try:
        for split in ('validation','test'):
            board, turn = data[split+'_board'], data[split+'_turn']
            values, probabilities = [], []
            for row, actor in zip(board,turn):
                cells = (ctypes.c_int*64)(*map(int,row)); score = ctypes.c_int(); probs = (ctypes.c_double*65)()
                assert lib.ot_nnue_infer(handle,cells,int(actor),ctypes.byref(score),probs)==1
                values.append(score.value/1000); probabilities.append(list(probs)[:64])
            target = np.sign(data[split+'_target']); values = np.asarray(values); probs = np.asarray(probabilities)
            mask = legal_mask(board,turn); truth = labels[split]; non_draw = target!=0
            assert np.allclose(probs.sum(1),1) and not np.any(probs[~mask])
            output[split] = dict(positions=len(board),outcomeMse=float(np.mean((values-target)**2)),
                nonDrawSignAccuracy=float(np.mean(np.sign(values[non_draw])==target[non_draw])),
                policyTop1=float(np.mean(probs.argmax(1)==truth)),
                policyCrossEntropy=float(-np.log(np.maximum(1e-12,probs[np.arange(len(board)),truth])).mean()))
    finally:
        lib.ot_nnue_destroy(handle)
    return output


def verify():
    experiment = load(MANIFEST)
    assert experiment['status']=='candidate' and not experiment['defaultPromoted']
    for name,digest in experiment['artifactSha256'].items():
        assert sha(ROOT/name)==digest,name
    snapshot = load(ROOT/experiment['sourceManifest'])
    assert sha(ROOT/snapshot['archive'])==snapshot['sha256']
    with tarfile.open(ROOT/snapshot['archive'],'r:gz') as archive:
        assert len(archive.getmembers())==len(snapshot['files'])
        for name,digest in snapshot['files'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
    training = load(MODEL.with_suffix('.json')); baseline = load(ROOT/'models/v1.9-default.json')
    assert training['formatMagic']=='OTPAT2' and MODEL.read_bytes()[:8]==b'OTPAT2\0\0'
    assert sha(MODEL)==training['sha256'] and training['trainPositions']==1000000
    assert training['epochs']==15 and training['status']=='candidate' and not training['defaultPromoted']
    assert training['dataset']['cacheSha256']==load(ROOT/'models/nnue256-v1.json')['dataset']['cacheSha256']
    assert training['policyDataset']['labels']==dict(train=1000000,validation=6000,test=6000)
    assert training['baselineModelSha256']==baseline['modelSha256']
    assert training['bestEpoch']==min(training['history'],key=lambda r:r['mse'])['epoch']
    assert training['cppParity']['positions']==6000 and training['cppParity']['maxOutcomeError']<2e-5
    for name,digest in training['scriptSha256'].items():
        assert snapshot['files']['scripts/'+name]==digest,name
    report = load(REPORT)
    assert report['complete'] and report['games']==report['plannedGames']==len(report['records'])==20
    assert report['millisecondsPerMove']==100 and not report['useBook']
    assert report['baselineVersion']=='v1.9' and report['baselineLibrarySha256']==baseline['nativeLibrarySha256']
    assert report['baselineManifestSha256']==sha(ROOT/'models/v1.9-default.json')
    assert [(c['prior'],c['valueTarget']) for c in report['configs']]==[('learned','outcome')]*2
    for c in report['configs']:
        assert sha(ROOT/c['model'])==c['modelSha256']
    for name,digest in report['sourceSha256'].items():
        assert snapshot['files'][name]==digest,name
    assert report['runnerSha256']==snapshot['files']['scripts/compare-nnue-candidate.py']
    for key,name in [('sharedRunnerSha256','scripts/compare-nnue-search.py'),('rulesSha256','scripts/compare-resnet.py'),('policyAuditSha256','scripts/opening_policy.py')]:
        assert report[key]==snapshot['files'][name]
    module = comparison(); module.match.CONFIGS = report['configs']
    actions = 0
    for i,game in enumerate(report['records']):
        assert game['number']==i+1 and game['opening']==[]
        assert (game['black'],game['white'])==(('baseline','candidate') if i%2==0 else ('candidate','baseline'))
        assert game['seed']==(report['baseSeed']^((i//2+1)*0x9e3779b9))&0x7fffffff
        assert all(d['valueTarget']=='outcome' for d in game['decisions'])
        actions += module.match.replay(game)
    assert actions==report['replayedActions']
    results,matrix = module.match.aggregate(report['records'])
    assert results==report['results'] and matrix==report['matrix']
    for name,digest in experiment['preservedSha256'].items():
        assert sha(ROOT/name)==digest,name
    r=results['candidate']
    print(f"Verified Pattern outcome: {r['win']}W/{r['draw']}D/{r['loss']}L, 20 games, {actions} replayed actions; frozen v1.9 preserved")


def freeze(args):
    assert not MANIFEST.exists(), 'Experiment already frozen'
    report=load(args.match)
    assert report['complete'] and report['games']==20
    assert sha(args.library)==report['librarySha256']
    shutil.copyfile(args.match,REPORT)
    baseline_metrics=offline(args.library)
    metrics_path=ROOT/f'models/{PREFIX}-offline.json'
    write(metrics_path,dict(baselineVersion='v1.9',librarySha256=sha(args.library),
        rounding='v1.9 integer C ABI scores / 1000; candidate Python quantized float output',baseline=baseline_metrics,
        candidate=load(MODEL.with_suffix('.json'))['metrics']))
    sources=sorted(set([p for p in (ROOT/'cpp').rglob('*') if p.is_file()]+
        [ROOT/p for p in ['CMakeLists.txt','AGENTS.md','package.json','scripts/train-pattern-outcome.py',
        'scripts/train-pattern.py','scripts/pattern_model.py','scripts/compare-nnue-candidate.py',
        'scripts/compare-nnue-search.py','scripts/compare-resnet.py','scripts/opening_policy.py',
        'scripts/watch-matches.py','scripts/watch-match.sh','scripts/finalize-pattern-outcome.py',
        'tests/native.cpp','tests/pattern_training_test.py']]))
    archive_path=ROOT/f'models/archive/{PREFIX}-sources.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        for p in sources:
            archive.add(p,arcname=str(p.relative_to(ROOT)))
    source_manifest=archive_path.with_name(f'{PREFIX}-sources.json')
    write(source_manifest,dict(archive=str(archive_path.relative_to(ROOT)),sha256=sha(archive_path),
        files={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    training=load(MODEL.with_suffix('.json')); r=report['results']['candidate']; bm=baseline_metrics['test']; cm=training['metrics']['test']
    summary_path=ROOT/f'models/{PREFIX}-comparison.md'
    summary_path.write_text(f'''# Pattern 胜负价值重训与 v1.9 对照（v1.25）

候选维持独立实验状态，默认仍为用户批准的 v1.9。历史 OTPAT1 模型和原报告保持原样。
本轮从随机参数重训 Pattern 全网络，沿用冻结百万局面，验证／测试各 6,000 个局面。
标量标签为行棋方记录终局胜 +1、平 0、负 −1；WDL 分类辅助头不参与搜索。
15 轮中按最低验证胜负 MSE 选择第 {training['bestEpoch']} 轮。格式 OTPAT2，训练耗时 {training['seconds']:.1f} 秒。

| 测试指标 | 冻结 v1.9 | Pattern outcome |
| --- | ---: | ---: |
| 胜负 MSE | {bm['outcomeMse']:.6f} | {cm['mse']:.6f} |
| 非和局符号准确率 | {bm['nonDrawSignAccuracy']:.2%} | {cm['nonDrawSignAccuracy']:.2%} |
| 实战下一手命中率 | {bm['policyTop1']:.2%} | {cm['policyTop1']:.2%} |
| 策略交叉熵 | {bm['policyCrossEntropy']:.6f} | {cm['policyCrossEntropy']:.6f} |

v1.9 为 C ABI 整数分值／1000，候选为 Python 量化浮点输出，取整误差不超过 0.5/1000。
候选全部 6,000 个测试局面的三头 Python/C++ 一致性已验证。

每步 100ms、20 局交换黑白，10 组共享种子，初始棋盘，不用开局库，沿用前 12 次落子的开局多样性。
v1.9 使用冻结参考库，候选使用合并后的原生库；两者哈希均在逐手报告中。
双方使用各自学习的合法策略先验；Pattern 额外保留父均值减 0.15 的有限 FPU。
因此本轮比较完整候选系统，不能单独归因于棋形结构、价值目标或 FPU。

候选视角 **{r['win']} 胜、{r['draw']} 平、{r['loss']} 负，得分率 {r['scoreRate']:.1%}，平均子差 {r['meanDiscDifference']:+.2f}**。
全部 {report['replayedActions']} 个动作通过独立规则和选择审计重放。
样本仅 20 局，不能据此推断稳定胜率或 Elo；100ms 结果不与历史 1 秒组混算。

逐手报告：`{REPORT.name}`；离线指标：`{metrics_path.name}`；冻结清单：`{MANIFEST.name}`。
复核：`python3 scripts/finalize-pattern-outcome.py --verify`。
监控：`sh scripts/watch-match.sh`，完成后仍显示最后一局终盘。
''')
    integration_path=ROOT/f'models/{PREFIX}-integration.json'
    integration=load(integration_path)
    assert integration['positions']==64 and integration['allOutputsEqualExceptElapsed']
    assert integration['libraries'][str(args.library.resolve().relative_to(ROOT))]==report['librarySha256']
    artifacts=[integration_path,MODEL,MODEL.with_suffix('.pt'),MODEL.with_suffix('.json'),REPORT,metrics_path,summary_path]
    preserved=['models/default-model.json','models/v1.9-default.json','models/nnue256-outcome-v1.9.otnnue',
        'src/game/wasm/othello.mjs','models/v1.8-baseline.json','models/pattern32-v1.8.otpat','models/v1.8-pattern-matches.json']
    write(MANIFEST,dict(version=PREFIX,status='candidate',defaultPromoted=False,
        sourceManifest=str(source_manifest.relative_to(ROOT)),librarySha256=report['librarySha256'],
        artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},
        preservedSha256={p:sha(ROOT/p) for p in preserved}))
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--match',type=Path,default=ROOT/'.build/training/pattern-outcome-v1.25/match.json')
    parser.add_argument('--library',type=Path,default=ROOT/('.build/native/libothello.'+('dylib' if sys.platform=='darwin' else 'so')))
    args=parser.parse_args()
    verify() if args.verify else freeze(args)

#!/usr/bin/env python3
"""Audit and summarize the isolated v1.46 staged-value candidate."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def immutable(path,value):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)


def main():
    report=ROOT/'models/v1.46-pattern-phase-value.json';r=read(report);protocol=ROOT/'models/v1.46-phase-value-protocol.json';p=read(protocol)
    assert r['complete'] and not r['defaultPromoted'] and not r['matchesRun']
    assert r['protocolSha256']==sha(protocol)
    for name,digest in p['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    manifest=ROOT/r['sourceArchiveManifest'];m=read(manifest);assert sha(manifest)==p['sourceArchiveManifestSha256'] and sha(ROOT/m['archive'])==m['sha256']
    with tarfile.open(ROOT/m['archive'],'r:gz') as t:
        for name,digest in m['files'].items():
            assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
            assert sha(ROOT/name)==digest,name
    v=r['variants'][0];base=read(ROOT/'models/pattern32-symmetry-v1.37.json')
    assert sha(ROOT/v['model'])==v['modelSha256'] and v['parameters']==916882 and v['bytes']==1833972
    assert len(v['history'])==15 and v['bestEpoch']==min(v['history'],key=lambda x:x['mse'])['epoch']
    assert v['dataOrderAndAugmentationSha256']==base['dataOrderAndAugmentationSha256']
    assert v['sharedInitialParametersIdentical'] and v['initialPredictionsIdentical']
    for split in ['validation','test']:
        assert sum(x['positions'] for x in v['phaseMetrics'][split])==6000
        assert v[split+'Parity']['positions']==6000
        assert r['baselineInferenceRegression'][split]['positions']==6000
    b=base['metrics']['test'];c=v['metrics']['test'];bv=base['metrics']['validation'];cv=v['metrics']['validation']
    lines=['# v1.46：分阶段价值头，策略结构沿用 v1.37','',
        '在对称共享基线上，保留共享棋形表、三组池化、97→32 主干和原条件策略读出，新增开局 32→4 价值/WLD 读出。两套读出均预测行棋方胜/平/负：标量目标 +1/0/-1，辅助分类顺序为胜/负/平。第 12 手及以前完全使用开局读出，第 24 手起完全使用中后盘读出，中间按整数空格数线性混合量化 logits，再计算 tanh/WLD。主干仍接收原 empty/60 特征。','',
        '策略头结构及损失保持原样，参数随共享主干联合训练，并非冻结策略权重。初始共享参数与基线完全相同，新增开局头复制原四行价值参数，初始全部预测一致。总参数 916,882（增加 132），OTPAT6 文件 1,833,972 字节（增加 280）。','',
        '原百万 WTHOR + Logistello 棋盘和原硬标签，不筛 Elo、不加权、不聚合；验证/测试各 6,000 局面不变。相同随机种子、样本/增强顺序、AdamW 和 15 轮，按原验证 MSE 选权重后才评估候选测试集。soft 对局按用户要求停止、工作进程退出后才启动训练。','',
        f'候选选第 {v["bestEpoch"]} 轮。以下为固定留出集离线指标；没有运行棋力对局，没有提升默认。','',
        '| 指标 | 冻结 v1.37 | 分阶段价值头 | 差值（候选−基线） |','|---|---:|---:|---:|',
        f'| 验证胜负 MSE | {bv["mse"]:.6f} | {cv["mse"]:.6f} | {cv["mse"]-bv["mse"]:+.6f} |',
        f'| 测试胜负 MSE | {b["mse"]:.6f} | {c["mse"]:.6f} | {c["mse"]-b["mse"]:+.6f} |',
        f'| 测试 WLD CE | {b["wdlCrossEntropy"]:.6f} | {c["wdlCrossEntropy"]:.6f} | {c["wdlCrossEntropy"]-b["wdlCrossEntropy"]:+.6f} |',
        f'| 测试策略命中率 | {b["policyTop1"]:.2%} | {c["policyTop1"]:.2%} | {(c["policyTop1"]-b["policyTop1"])*100:+.2f} pp |',
        f'| 测试策略 CE | {b["policyCrossEntropy"]:.6f} | {c["policyCrossEntropy"]:.6f} | {c["policyCrossEntropy"]-b["policyCrossEntropy"]:+.6f} |','',
        '| 测试阶段（已落子手数） | 局面数 | 基线价值 MSE | 候选价值 MSE | 差值 |','|---|---:|---:|---:|---:|']
    for phase in v['phaseMetrics']['test']:
        lo,hi=phase['placements'];bm=phase['baseline']['mse'];cm=phase['candidate']['mse'];lines.append(f'| {lo}–{hi} | {phase["positions"]:,} | {bm:.6f} | {cm:.6f} | {cm-bm:+.6f} |')
    lines+=['', '分阶段指标用于描述结果；初始阶段误差更高也包含结果不可预测性，不能单独证明拆头有效。本次仅一个随机种子，离线差异不能代替相同计时的棋力评测。', '',
        '检查已通过：独立 Release 和关闭 SIMD 的可移植原生 CTest；两版各 2 项 Python 测试（512 局面原生对照、全部阶段边界、初始基线等价、梯度、策略隔离、D4/换色对称）；候选验证/测试合计 12,000 局面的 Torch/原生导出对照；原基线在新原生库上的 12,000 局面对照。原生标量参考另外覆盖完整随机对局、合法策略投影、传入已有价值读出、缓存状态和坏文件拒绝。', '',
        '默认选择器、默认清单、基线模型、冻结原生参考库和浏览器权重哈希均保持原值。新模型需要支持 OTPAT6 的原生库，旧冻结库仍只用于原基线；没有更新浏览器嵌入权重。', '',
        '训练数据来源和使用说明继续遵循 data/README.md；候选不改变数据许可。', '',
        '权重：models/pattern32-symmetry-phase-value-v1.46.otpat；完整结果：models/v1.46-pattern-phase-value.json；协议：models/v1.46-phase-value-protocol.json；开训源码：models/archive/v1.46-phase-value-sources.json；审计：python3 scripts/finalize-phase-value.py。', '']
    summary=ROOT/'models/v1.46-phase-value-comparison.md';immutable(summary,'\n'.join(lines))
    artifacts=[report,protocol,summary,manifest,ROOT/m['archive'],ROOT/v['model'],(ROOT/v['model']).with_suffix('.pt'),(ROOT/v['model']).with_suffix('.json'),ROOT/'scripts/compare-nnue-candidate.py']
    immutable(ROOT/'models/v1.46-phase-value-experiment.json',dict(status='candidate-complete',defaultPromoted=False,matchesRun=False,baseline='v1.37',candidateTest=c,baselineTest=b,artifactSha256={str(x.relative_to(ROOT)):sha(x) for x in artifacts},preservedSha256=p['preservedSha256'],scriptSha256=sha(__file__),checks=dict(releaseCTestPassed=True,portableCTestPassed=True,pythonPhaseTestsPerBuild=2,candidateTorchNativePositions=12000,baselineRegressionPositions=12000)))
    print(json.dumps(dict(complete=True,bestEpoch=v['bestEpoch'],baselineTest=b,candidateTest=c,phaseTest=v['phaseMetrics']['test']),ensure_ascii=False))


if __name__=='__main__':main()

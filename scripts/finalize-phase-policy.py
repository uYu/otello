#!/usr/bin/env python3
"""Audit the independent v1.47 phase-policy ablation and middle-game metrics."""
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
    report=ROOT/'models/v1.47-pattern-phase-policy.json';r=read(report);protocol=ROOT/'models/v1.47-phase-policy-protocol.json';p=read(protocol)
    assert r['complete'] and not r['defaultPromoted'] and not r['matchesRun'] and r['protocolSha256']==sha(protocol)
    for name,digest in p['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    manifest=ROOT/r['sourceArchiveManifest'];m=read(manifest);assert sha(manifest)==p['sourceArchiveManifestSha256'] and sha(ROOT/m['archive'])==m['sha256']
    with tarfile.open(ROOT/m['archive'],'r:gz') as t:
        for name,digest in m['files'].items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest,name
    base=read(ROOT/'models/pattern32-symmetry-v1.37.json');control,candidate=r['variants']
    assert all(control['baselineReplay'].values())
    assert len(control['history'])==len(candidate['history'])==15
    assert control['sharedInitialParametersSha256']==candidate['sharedInitialParametersSha256']
    for v in r['variants']:
        assert sha(ROOT/v['model'])==v['modelSha256']
        assert v['bestEpoch']==min(v['history'],key=lambda x:x['middle']['policyCrossEntropy'])['epoch']
        assert v['dataOrderAndAugmentationSha256']==base['dataOrderAndAugmentationSha256']
        for split in ['validation','test']:
            assert sum(x['positions'] for x in v['phaseMetrics'][split])==6000
            assert v[split+'Parity']['positions']==6000 and r['baselineInferenceRegression'][split]['positions']==6000
    assert candidate['parameters']==918136 and candidate['bytes']==1836556
    def mid(v,split):return v['phaseMetrics'][split][1]['metrics']
    bm=r['baselinePhaseMetrics']['test'][1]['metrics'];cm=mid(control,'test');pm=mid(candidate,'test')
    names=['冻结默认 v1.37','单头对照（同中盘规则选模）','分阶段策略头']
    test_metrics=[base['metrics']['test'],control['metrics']['test'],candidate['metrics']['test']]
    middle_metrics=[bm,cm,pm]
    lines=['# v1.47：独立分阶段策略头，中盘策略优先','',
        '用户指出部署有开局库和后盘精确搜索，希望重点看中盘。此候选只把原条件策略的门控向量 g[32] 和位置偏置 b[10] 分成两套阶段读出；保留对称棋形表、三组池化、97→32 主干和单个原价值/WLD 读出。没有叠加 v1.46 分阶段价值头，没有改为直连 MLP。价值、策略及共享特征仍联合训练，所谓价值头不动指结构和目标不变，并非冻结权重。','',
        '阶段边界沿用上一候选，开训前固定：已落子 ≤12 手为开局头，≥24 手为中后盘头，中间按整数空格数线性混合量化后的门控/偏置，再进行原局部特征投影和合法 softmax。原共享初值相同，新开局头复制原策略读出，初始函数与基线完全相同。新增 1,386 参数，OTPAT7 为 918,136 参数、1,836,556 字节（比原基线增加 2,864 字节）。','',
        '数据仍为原百万 WTHOR + Logistello 硬标签，不筛 Elo、不加权、不聚合；验证/测试各 6,000 局面与标签不变。同种子、同初值、同样本/增强顺序、同 AdamW 与 15 轮损失。训练未对中盘样本加权。','',
        f'为避免选模偏差，补训单头对照，两组都按同一个固定中盘验证子集（已落子 21–36 手，{r["middleValidationPositions"]:,} 局面）的合法策略交叉熵选权重；单头选第 {control["bestEpoch"]} 轮，分阶段选第 {candidate["bestEpoch"]} 轮。二者冻结后才评估测试。冻结默认 v1.37 仍是原来按全验证价值 MSE 选择的权重，另列作部署参照。单头训练的全部原有每轮指标及全验证 MSE 最佳权重逐字节复现 v1.37，验证了训练对照一致。','',
        '| 中盘测试（21–36 手） | 策略 CE ↓ | 策略 Top-1 ↑ | 价值 MSE ↓ |','|---|---:|---:|---:|']
    for name,score in zip(names,middle_metrics):lines.append(f'| {name} | {score["policyCrossEntropy"]:.6f} | {score["policyTop1"]:.2%} | {score["mse"]:.6f} |')
    lines+=['',f'相对同选模单头对照，中盘策略 CE 差值 {pm["policyCrossEntropy"]-cm["policyCrossEntropy"]:+.6f}，Top-1 差值 {(pm["policyTop1"]-cm["policyTop1"])*100:+.2f} 个百分点。中盘验证 CE：单头 {mid(control,"validation")["policyCrossEntropy"]:.6f}，分阶段 {mid(candidate,"validation")["policyCrossEntropy"]:.6f}。','',
        '| 全测试集 | 策略 CE ↓ | 策略 Top-1 ↑ | 价值 MSE ↓ |','|---|---:|---:|---:|']
    for name,score in zip(names,test_metrics):lines.append(f'| {name} | {score["policyCrossEntropy"]:.6f} | {score["policyTop1"]:.2%} | {score["mse"]:.6f} |')
    lines+=['','| 测试阶段 | 单头策略 CE | 分阶段策略 CE | 单头 Top-1 | 分阶段 Top-1 |','|---|---:|---:|---:|---:|']
    for a,b in zip(control['phaseMetrics']['test'],candidate['phaseMetrics']['test']):
        low,high=a['placements'];x,y=a['metrics'],b['metrics'];lines.append(f'| {low}–{high} 手 | {x["policyCrossEntropy"]:.6f} | {y["policyCrossEntropy"]:.6f} | {x["policyTop1"]:.2%} | {y["policyTop1"]:.2%} |')
    lines+=['',
        '本轮仅一个随机种子，指标衡量人类棋谱走法拟合；策略用于 PUCT 搜索先验，离线 CE/命中率不等于相同计时棋力。开局库可能脱库，精确搜索前的后盘仍会用到网络，因此完整留出集指标仍保留。本次没有运行棋力对局，也没有提升默认或与分阶段价值头组合。','',
        '检查通过：Release 与关闭 SIMD 两组原生 CTest、两组各两项 Python 阶段策略测试；阶段边界、初始基线函数一致、价值输出隔离、梯度、D4/换色对称、512 局面原生对照；两份选中模型合计 24,000 局面 Torch/原生对照，原基线在新原生库另有 12,000 局面回归对照。原生独立标量参考覆盖随机完整对局、合法策略投影、已有读出复用、缓存状态、坏文件和参数变化导致缓存失效。','',
        '默认选择器、默认清单、基线模型、冻结原生参考库、浏览器嵌入权重、v1.46 权重和报告均保留原哈希。候选仅由隔离构建目录中的新库加载，旧冻结库不支持 OTPAT7。数据来源与许可继续遵循 data/README.md。','',
        '权重：models/pattern32-symmetry-phase-policy-v1.47.otpat；完整结果：models/v1.47-pattern-phase-policy.json；协议：models/v1.47-phase-policy-protocol.json；源码快照：models/archive/v1.47-phase-policy-sources.json；复核：python3 scripts/finalize-phase-policy.py。','']
    summary=ROOT/'models/v1.47-phase-policy-comparison.md';immutable(summary,'\n'.join(lines))
    artifacts=[report,protocol,summary,manifest,ROOT/m['archive']]
    for v in r['variants']:artifacts += [ROOT/v['model'],(ROOT/v['model']).with_suffix('.pt'),(ROOT/v['model']).with_suffix('.json')]
    immutable(ROOT/'models/v1.47-phase-policy-experiment.json',dict(status='candidate-complete',defaultPromoted=False,matchesRun=False,controlMiddleTest=cm,candidateMiddleTest=pm,middlePolicyCeDelta=pm['policyCrossEntropy']-cm['policyCrossEntropy'],middlePolicyTop1Delta=pm['policyTop1']-cm['policyTop1'],artifactSha256={str(x.relative_to(ROOT)):sha(x) for x in artifacts},preservedSha256=p['preservedSha256'],scriptSha256=sha(__file__),checks=dict(releaseCTestPassed=True,portableCTestPassed=True,pythonPhaseTestsPerBuild=2,selectedTorchNativePositions=24000,baselineRegressionPositions=12000,originalControlReplay=control['baselineReplay'])))
    print(json.dumps(dict(complete=True,controlEpoch=control['bestEpoch'],candidateEpoch=candidate['bestEpoch'],controlMiddle=cm,candidateMiddle=pm),ensure_ascii=False))


if __name__=='__main__':main()

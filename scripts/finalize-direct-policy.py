#!/usr/bin/env python3
"""Audit the direct MLP training candidate; never change an approved default."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    protocol=read(ROOT/'models/v1.38-direct-policy-protocol.json')
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    r=read(ROOT/'models/v1.38-pattern-direct-policy.json');bench=read(ROOT/'models/v1.38-direct-policy-benchmark.json')
    assert bench['complete'] and r['baselineManifest']=='models/v1.37-default.json'
    candidate=r['variants'][0];assert sha(ROOT/candidate['model'])==candidate['modelSha256']
    baseline=read(ROOT/'models/pattern32-symmetry-v1.37.json')
    assert candidate['dataOrderAndAugmentationSha256']==baseline['dataOrderAndAugmentationSha256']
    assert candidate['initialEmbeddingSha256']==baseline['initialEmbeddingSha256']
    assert candidate['validationParity']['positions']==candidate['testParity']['positions']==6000
    assert candidate['bestEpoch']==min(candidate['history'],key=lambda x:x['mse'])['epoch']
    for name,digest in bench['libraries'].items():assert sha(ROOT/name)==digest,name
    assert bench['trainingReportSha256']==sha(ROOT/'models/v1.38-pattern-direct-policy.json')
    archive=read(ROOT/'models/archive/v1.38-direct-policy-sources.json');assert sha(ROOT/archive['archive'])==archive['sha256']
    for name,digest in archive['files'].items():assert sha(ROOT/name)==digest,name
    for name,digest in r['scriptSha256'].items():assert sha(ROOT/name)==digest,name
    lines=['# v1.38：直连共享 MLP 策略头训练','',
        '默认仍为用户批准的 v1.37 对称共享模型。本候选只作为离线训练实验，未提升默认，未进行棋力对战。','',
        '策略头把每个落点的局部 32 维、全局主干 32 维和 D4 位置桶 one-hot 10 维直接拼接，经共享 74→8→1 MLP 输出分数；中间为 clipped ReLU。没有动态门控乘法或独立位置偏置分支。规则引擎的合法掩码与 softmax 保持独立。','',
        '保持 38 个棋形、28,503×32 对称表、三组均值和 97→32 价值主干。百万训练局面、15 轮、batch 2048、AdamW lr 0.001、weight decay 0.0001、胜负 MSE +0.1 WDL CE +0.1 策略 CE。与 v1.37 使用相同随机表、主干及四个价值输出的初值，样本/增强顺序 SHA-256 一致；从随机参数重训全网，没有从已训练基线微调。','',
        f"按最低验证胜负 MSE 选择第 {candidate['bestEpoch']} 轮；选定后才统一评估测试集。新文件为 OTPAT5，{candidate['parameters']:,} 个参数、{candidate['bytes']:,} 字节。",'',
        '| 模型 | 验证胜负 MSE | 测试胜负 MSE | 测试走法命中率 | 测试策略 CE |','|---|---:|---:|---:|---:|']
    for name,v in [('批准 v1.37',baseline),('直连 MLP8',candidate)]:
        vm,tm=v['metrics']['validation'],v['metrics']['test'];lines.append(f"| {name} | {vm['mse']:.6f} | {tm['mse']:.6f} | {tm['policyTop1']:.2%} | {tm['policyCrossEntropy']:.6f} |")
    lines+=['','双方单训练种子。测试集沿用历史冻结划分，离线指标不能替代同时间棋力对战。','',
        '新模型通过验证/测试各 6,000 局面的 PyTorch/C++ 三头导出一致性；普通标量和 SIMD 路径 12,000 局面的全部 68 项输出精确一致，16 个固定 1,024 模拟搜索结果也一致。独立标量参考测试覆盖完整策略、合法着策略、缓存刷新、D4 对称、换色、损坏文件和参数变更后的缓存失效。新增路径另通过 ASan/UBSan。','',
        '| 模型 | 联合推理 µs | 100ms 模拟吞吐倍率 | 100ms 评估吞吐倍率 | 1s 模拟吞吐倍率 | 1s 评估吞吐倍率 |','|---|---:|---:|---:|---:|---:|']
    for i,row in enumerate(bench['results']):
        ratios=[]
        for ms in ['100','1000']:
            budget=row['budgets'][ms]
            for metric in ['simulationsPerSecond','evaluationsPerSecond']:
                v=budget.get(metric+'Ratio') if i else 1
                ratios.append(f'{v:.3f}×' if v is not None else '无共同样本')
        lines.append(f"| {row['name']} | {row['medianJointMicroseconds']:.3f} | "+' | '.join(ratios)+' |')
    lines+=['','联合推理计时含完整 64 格输出及 Python/C ABI 开销；实际搜索只计算合法落点。对双方使用相同验证局面交错计时，100ms 三轮、1s 两轮、同为 4M 节点；吞吐汇总只保留双方所有轮次均未证明、未触顶且用满至少 95% 预算的共同局面。原始逐次样本全部保留。模拟和网络评估不是同一指标；不同策略先验会改变搜索路径，吞吐倍率不能解释为纯内核倍率或棋力提升。','',
        '原生实现只缓存固定 MLP 第一层的全局加法项；它等价于直接拼接 74 维后计算，没有按局面生成权重。原 v1.37 模型在候选新库与冻结参考库上 12,000 局面的 68 项输出、16 个固定模拟结果精确一致。v1.37 默认选择器、模型、冻结记录及浏览器 Wasm 哈希保持原值。','',
        '监控：`sh scripts/watch-match.sh`。审计：`python3 scripts/finalize-direct-policy.py`。模型：`models/pattern32-direct-mlp8-v1.38.otpat`；训练记录：`v1.38-pattern-direct-policy.json`；计时与一致性：`v1.38-direct-policy-benchmark.json`；协议：`v1.38-direct-policy-protocol.json`。Linux/浏览器性能未测；没有对战胜率或 Elo 结论。']
    path=ROOT/'models/v1.38-direct-policy-comparison.md';text='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==text,'Refusing to replace a frozen report'
    else:path.write_text(text)
    print(json.dumps(dict(complete=True,bestEpoch=candidate['bestEpoch'],test=candidate['metrics']['test'],baselineTest=baseline['metrics']['test']),ensure_ascii=False))


if __name__=='__main__':main()

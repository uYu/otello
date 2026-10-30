#!/usr/bin/env python3
"""Audit frozen v1.37 training, timing and all independently replayed matches."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
spec=importlib.util.spec_from_file_location('structure_final_comparison',ROOT/'scripts/compare-nnue-candidate.py')
comparison=importlib.util.module_from_spec(spec);sys.modules[spec.name]=comparison;spec.loader.exec_module(comparison)


def immutable(path,text):
    if path.exists():assert path.read_text()==text,'Refusing to replace frozen report: '+str(path)
    else:path.write_text(text)


def main():
    protocol=read(ROOT/'models/v1.37-pattern-structure-protocol.json')
    for path,digest in protocol['preservedSha256'].items():assert sha(ROOT/path)==digest,path
    training=read(ROOT/'models/v1.37-pattern-structure.json')
    benchmark=read(ROOT/'models/v1.37-pattern-structure-benchmark.json')
    assert training['baselineManifest']=='models/v1.36-default.json'
    assert benchmark['complete'] and benchmark['regression']['all68OutputsExactlyEqual']
    assert len({v['dataOrderAndAugmentationSha256'] for v in training['variants']})==1
    archive=read(ROOT/'models/archive/v1.37-pattern-structure-sources.json')
    assert sha(ROOT/archive['archive'])==archive['sha256']
    for path,digest in archive['files'].items():assert sha(ROOT/path)==digest,path
    for path,digest in archive['libraries'].items():assert sha(ROOT/path)==digest,path
    models=[]
    for variant in training['variants']:
        path=ROOT/variant['model'];binary=path.read_bytes();assert sha(path)==variant['modelSha256']
        magic,rows,dim,width,inputs,outputs,scale=struct.unpack_from('<8s6I',binary)
        assert dim==width==32 and outputs==46 and scale==256
        assert variant['validationParity']['positions']==variant['testParity']['positions']==6000
        models.append(dict(name=variant['name'],model=variant['model'],sha256=sha(path),bytes=len(binary),embeddingRows=rows,trunkInputs=inputs,format=magic.rstrip(b'\0').decode(),parameters=variant['parameters'],bestEpoch=variant['bestEpoch'],metrics=variant['metrics']))
    matches=[];artifacts={};actions=0
    for ms in protocol['matches']['budgetsMs']:
        for variant in protocol['matches']['variants']:
            path=ROOT/f'models/v1.37-{variant}-vs-v1.36-{ms}ms-20.json';report=read(path)
            assert report['complete'] and report['games']==20 and report['millisecondsPerMove']==ms
            assert report['baselineVersion']=='v1.36' and report['useBook'] is False
            assert report['protocolSha256']==sha(ROOT/'models/v1.37-pattern-structure-protocol.json')
            assert report['maxNodesByEngine']==dict(baseline=4194304,candidate=4194304)
            assert report['baselineLibrarySha256']==protocol['baselineLibrarySha256']
            assert report['configs'][0]['modelSha256']==protocol['preservedSha256']['models/pattern32-outcome-v1.25.otpat']
            assert report['configs'][1]['modelSha256']==next(v['sha256'] for v in models if v['name']==variant)
            assert all(c['valueTarget']=='outcome' and c['prior']=='learned' for c in report['configs'])
            comparison.match.CONFIGS=report['configs']
            for game in report['records']:comparison.match.replay(game)
            replayed=sum(len(g['moves']) for g in report['records']);assert replayed==report['replayedActions'];actions+=replayed
            counts,_=comparison.match.aggregate(report['records']);assert counts==report['results']
            candidate=counts['candidate'];baseline=counts['baseline']
            decisions=[d for g in report['records'] for d in g['decisions'] if d['engine']=='candidate']
            assert all(d['mcts']['maxNodes']==4194304 for d in decisions)
            matches.append(dict(variant=variant,milliseconds=ms,games=20,report=str(path.relative_to(ROOT)),sha256=sha(path),candidate=candidate,baseline=baseline,replayedActions=replayed,peakCandidateRssMiB=max(d['mcts']['peakRssBytes'] for d in decisions)/1024**2))
            artifacts[str(path.relative_to(ROOT))]=sha(path)
    for name in ['models/v1.37-pattern-structure-protocol.json','models/v1.37-pattern-structure.json','models/v1.37-pattern-structure-benchmark.json','models/archive/v1.37-pattern-structure-sources.json']:
        artifacts[name]=sha(ROOT/name)
    summary=dict(version='v1.37',status='candidate',defaultPromoted=False,baseline='models/v1.36-default.json',models=models,matches=matches,replayedActions=actions,artifactsSha256=artifacts,preservedSha256=protocol['preservedSha256'],limits='Single training seed and 20 games (10 paired seeds) per candidate/time budget. Offline and match results are separate; no Elo/statistical significance claim. No combined model trained, no default or browser changes. Native macOS ARM64 timings; Linux performance unmeasured.')
    immutable(ROOT/'models/v1.37-pattern-structure-summary.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# v1.37：对称共享与十组池化独立对照','',
        '默认仍为用户批准的 v1.36，使用原 v1.25 INT16 Pattern outcome 权重。本轮候选未提升默认，也未组合两项改动。','',
        '三组从随机参数训练 15 轮，使用同一百万局面、相同样本顺序和 D4 增强序列；batch 2048、AdamW lr 0.001、weight decay 0.0001，胜负 MSE + 0.1 WDL CE + 0.1 合法落点 CE。每组独立按验证胜负 MSE 选权重，三组冻结后才统一评估测试集。单训练种子，训练/验证/测试沿用原棋谱划分。','',
        'control 是同期重训的原结构；symmetry 只共享直线反序/角部转置表项，使用 28,503 行压缩表；pool10 只将池化改成四种距边深度的行列、五种长度的对角线及角部，共十组。三组先生成相同随机原表；对称版取各轨道代表行作为共享参数初值，不从批准权重微调。','',
        '候选格式 OTPAT4：原 32 字节维度头后追加 uint32 variant（1=对称共享，2=十组池化）；不支持组合标记。对称表在文件及运行内存均保持压缩；十组池化保留 55,728 行，主干输入为 321，主干宽度及输出头保持不变。两个候选均为 INT16 权重、INT32 偏置与胜负价值语义。','',
        '| 模型 | 参数 | 字节 | 验证胜负 MSE | 测试胜负 MSE | 测试走法命中率 | 测试策略 CE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    frozen=read(ROOT/'models/pattern32-outcome-v1.25.json')
    fv,ft=frozen['metrics']['validation'],frozen['metrics']['test']
    lines.append(f"| v1.36 原权重 | {frozen['parameters']:,} | {frozen['bytes']:,} | {fv['mse']:.6f} | {ft['mse']:.6f} | {ft['policyTop1']:.2%} | {ft['policyCrossEntropy']:.6f} |")
    for v in models:
        vm,tm=v['metrics']['validation'],v['metrics']['test'];lines.append(f"| {v['name']} | {v['parameters']:,} | {v['bytes']:,} | {vm['mse']:.6f} | {tm['mse']:.6f} | {tm['policyTop1']:.2%} | {tm['policyCrossEntropy']:.6f} |")
    lines+=['','原模型在新库与冻结 v1.36 库上的验证/测试各 6,000 局面全部 68 项输出逐项相同，32 个冷根和 16 步实际子树复用的固定 1,024 模拟结果与元数据相同。候选各通过验证/测试 6,000 局面的 PyTorch/C++ 三头一致性，普通标量回退与 SIMD 的 6,000 局面输出和固定搜索也一致。新增原生测试覆盖独立标量参考、321 维点积、完整/合法策略、状态更新、D4 与损坏文件。','',
        '| 模型 | 联合推理 µs | 验证平均价值对称跨度 | 最大对应策略差 |','|---|---:|---:|---:|']
    for row,sym in zip(benchmark['results'],benchmark['symmetry']):lines.append(f"| {row['name']} | {row['medianJointMicroseconds']:.3f} | {sym['meanValueSpan']:.9f} | {sym['maximumAlignedPolicyLogitDifference']:.3g} |")
    lines+=['','价值对称跨度是在同一局面的八种 D4 变换间计算，策略先逆变换对齐。联合推理计时含 Python/C ABI 开销。搜索吞吐在相同验证局面交错测量，100ms 三轮、2s 两轮；只汇总所有实现/轮次均未证明、未触顶、用满至少 95% 预算的共同局面，完整原始样本保留。','',
        '| 模型 | 预算 | 共同局面 | 模拟/秒 | 评估/秒 | 模拟倍率 vs v1.36 |','|---|---:|---:|---:|---:|---:|']
    for row in benchmark['results']:
        for ms,b in row['budgets'].items():
            if b['simulationsPerSecond'] is not None:lines.append(f"| {row['name']} | {ms}ms | {len(b['commonRoots'])} | {b['simulationsPerSecond']:,.0f} | {b['evaluationsPerSecond']:,.0f} | {b['simulationRatioVsFrozen']:.3f}× |")
            else:lines.append(f"| {row['name']} | {ms}ms | 0 | 不汇总 | 不汇总 | 不汇总 |")
    lines+=['','双方各自独立进程，同为 4,194,304 节点、原开局多样性、PUCT/FPU、子树复用和终局证明；无开局库。每个候选/预算 20 局，交换颜色，每对颜色共享种子。下面按候选视角列出，预算之间不合并。','',
        '| 候选 | 每步预算 | 胜 | 平 | 负 | 得分率 | 平均子差 | 峰值 RSS MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in matches:
        c=row['candidate'];lines.append(f"| {row['variant']} | {row['milliseconds']}ms | {c['win']} | {c['draw']} | {c['loss']} | {c['scoreRate']:.1%} | {c['meanDiscDifference']:+.2f} | {row['peakCandidateRssMiB']:.1f} |")
    lines+=['',f'80 局共 {actions:,} 个行动均独立回放，各决策选着规则逐步审计。样本规模不足以据此声明统计显著棋力提升；离线准确率、搜索吞吐与对战结果分别解释。阶段/特征覆盖、训练噪声及计时路径均可能影响结果。普通状态缓存接口为新结构使用正确的完整刷新回退；正式对战的增量缓存关闭，未扩大既有每节点状态布局。','',
        '稳定监控入口：`sh scripts/watch-match.sh`，每步更新，保留各组终盘。原始结果、模型/库/源码哈希及开赛前源码归档见 `v1.37-pattern-structure-summary.json`。审计：`python3 scripts/finalize-pattern-structure.py`。复现顺序：独立构建冻结 v1.36 参考库及候选 SIMD/普通库，再运行 `train-pattern-structure.py`、`benchmark-pattern-structure.py`、`run-pattern-structure.py`；已存在模型/报告会拒绝覆盖。Linux 使用 `.so`，应在目标平台重新编译和计时。','',
        'WTHOR/Logistello 数据来源、归属和使用限制保持 `data/README.md` 与原 manifest 的说明。']
    immutable(ROOT/'models/v1.37-pattern-structure-comparison.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,replayedActions=actions,matches=[dict(variant=r['variant'],milliseconds=r['milliseconds'],win=r['candidate']['win'],draw=r['candidate']['draw'],loss=r['candidate']['loss']) for r in matches]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

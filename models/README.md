> 当前默认：**v1.25 Pattern 策略／胜负价值模型＋MCTS**，已由用户批准为新基线。入口见 `default-model.json`，冻结记录见 `v1.25-default.json`。以下候选实验文字记录的是提升前的历史状态，不覆盖原始报告。

# 模型与实验记录

补充棋形 NNUE 实验（本地标识 v1.8，独立于其他机器的版本编号）使用 `pattern32-v1.8.otpat`（独立的 `OTPAT1` 格式），同时训练策略、胜负和、子差三个输出；该历史实验的对手为 v1.3 NNUE＋MCTS；当前批准的默认基线是 v1.25。结构、验证、速度及策略先验消融对战见 [v1.8 实验报告](v1.8-comparison.md)。

| 文件 | 用途 |
| --- | --- |
| `nnue256-v1.otnnue` | 原生推理模型，128→256→1，67,096 字节 |
| `nnue256-v1.pt` | 对应 PyTorch 权重 `state_dict`，不包含优化器状态 |
| `nnue256-v1.json` | 训练配置、数据摘要、验证/测试指标和模型校验值 |
| `v1-baseline.json` | 已冻结基线：源码、模型、报告哈希与重放验证摘要 |
| `v1-baseline-protocol.json` / `v1-experiments.json` | 两组对战的固定设置 |
| `v1-search-vs-model.json` | 纯搜索 vs 模型搜索的 100 局完整记录 |
| `v1-model-vs-book.json` | 模型搜索 vs 模型＋开局库的 100 局完整记录 |
| `v1-dataset.json` / `training-data-manifest.json` | 数据抽样元数据和上游原始文件清单 |
| `archive/` | 48,000 训练局面的历史原型、元数据和实验记录 |
| `archive/v1-sources.tar.gz` / `archive/v1-sources.json` | v1 实测源码快照、逐文件校验值及原始基线校验值 |
| `v1.1-search-baseline.json` / `v1.1-search.md` | v1.1 搜索实现的固定深度性能对比；沿用 v1 模型权重 |
| `v1.1-baseline.json` / `v1.1-comparison.md` | v1.1 版本基线与对 v1 的等时限对战结果 |
| `v1.1-vs-v1-nnue.json` / `v1.1-experiments.json` | 双方从标准起点自行走棋的 100 局完整记录及监控配置 |
| `archive/v1.1-sources.tar.gz` / `archive/v1.1-sources.json` | 冻结的 v1.1 实测源码、Wasm 与逐文件校验值 |
| `nnue256-v1.2.otnnue` / `.pt` / `.json` | 考虑行棋方的双视角输出模型、训练权重与训练指标 |
| `v1.2-baseline.json` / `v1.2-comparison.md` | v1.2 模型实验的校验清单与对 v1.1 的对战结果 |
| `v1.2-vs-v1.1-nnue.json` / `v1.2-experiments.json` | 从标准初始棋盘自行走棋的完整记录与监控配置 |
| `archive/v1.2-sources.tar.gz` / `archive/v1.2-sources.json` | v1.2 实测源码与 Wasm 的冻结快照 |
| `v1.3-baseline.json` / `v1.3-comparison.md` | 衰减开局随机选步策略的基线、多样性和棋力记录 |
| `v1.3-vs-v1.1-nnue.json` / `v1.3-experiments.json` | 新策略对原版 v1.1 的 100 局逐步记录和监控配置 |
| `archive/v1.3-sources.tar.gz` / `archive/v1.3-sources.json` | 冻结的 v1.3 参考实现及逐文件校验 |
| `cnn8x8-v1.4.otcnn` / `.pt` / `.json` | 小卷积网络实验权重、训练状态与指标 |
| `v1.4-baseline.json` / `v1.4-comparison.md` | 8×8 CNN 对 v1.3 的等时限实验与推荐结论 |
| `v1.4-cnn-vs-v1.3.json` / `v1.4-inference-benchmark.json` | 100 局完整记录与同局面直接推理耗时 |
| `v1.4-fixed-depth-diagnostic.json` | 双方深度 4、20 局诊断；CNN 6 胜 14 负 |
| `archive/v1.4-sources.tar.gz` / `archive/v1.4-sources.json` | 冻结的小 CNN 实验源码及逐文件校验 |
| `resnet32x2-v1.5.otres` / `.pt` / `.json` | 32 通道、2 个残差块的价值网络及训练指标 |
| `v1.5-baseline.json` / `v1.5-comparison.md` | 残差网络实验的冻结基线与结论 |
| `v1.5-resnet-vs-v1.3.json` / `v1.5-3s-pilot.json` | 100 局等时限对战；3 秒试验记录未完成并有停止说明 |
| `v1.5-inference-benchmark.json` / `v1.5-experiments.json` | 推理耗时与对战监控配置 |
| `archive/v1.5-sources.tar.gz` / `archive/v1.5-sources.json` | 残差模型正式对战对应源码快照 |
| `v1.6-baseline.json` / `v1.6-comparison.md` | 模型 × 搜索四组合的冻结基线与结果表 |
| `v1.6-search-matrix.json` / `v1.6-experiments.json` | 每步 1 秒、四组合两两对战的逐步记录和监控配置 |
| `archive/v1.6-sources.tar.gz` / `archive/v1.6-sources.json` | 1 秒四组合实验对应源码快照 |
| `v1.6-100ms-baseline.json` / `v1.6-100ms-comparison.md` | 同一 v1.3 NNUE、MCTS 对 AB、100ms 的 100 局基线与结果 |
| `v1.6-nnue-mcts-vs-ab-100ms.json` / `v1.6-100ms-experiments.json` | 100ms 两搜索对战的逐步记录与监控配置 |

可读结果见 [comparison.md](comparison.md)。实际训练数据与开局库的压缩包位于 [data/](../data/README.md)，恢复命令为 `python3 scripts/data-assets.py restore`。

`*.live.json` 是运行时监控状态，不提交；监控脚本在没有实时文件时读取已完成的报告。两组从标准初始局面交替颜色、每步 50ms，固定起点的重复对局不当作独立开局样本或 Elo 评估。

当前默认发布为 v1.7：沿用 v1.2 NNUE 权重与 v1.3 开局策略，默认搜索为 MCTS，原生与网页均内嵌模型。v1/v1.1/v1.2 模型与历史报告保持冻结。`npm run test:baseline` 分别校验历史源码归档、旧结果及当前源码与模型；v1.3 还独立复核随机种子、加权抽样、分差限制和对称去重。重现实测历史版本时，将对应源码归档解压到独立目录，再对该目录运行 CMake。

v1.2 只改变 NNUE 输出头对行棋方的表达：将当前行棋方和对方的 256 维特征按顺序拼接，使用独立输出权重和偏置。共享特征层仍为 128→256，参数为 33,537，模型文件 67,612 字节。使用与 v1 相同的百万局面缓存、冻结验证/测试集和 30 轮训练设置，按验证 MSE 选择第 26 轮权重；模型格式为 `OTNNUE2`，读取器同时兼容原 `OTNNUE1`。v1.1 的源码与 Wasm 已归档，历史记录继续校验，不覆盖旧权重。

v1.1 的棋力评测只比较 NNUE 搜索，启发式引擎保留供较弱的玩家对手使用。双方加载同一 v1 模型，每步各 50ms，不查开局库，不预设开局走法，交替执黑白。使用 `npm run training:watch:v1.1` 查看该组进度；重跑时使用新的 `--output`，以保留版本基线。

`v1.7-default.json` 记录默认交付配置、内嵌模型校验和与当前源码/产物指纹。`archive/v1.6-100ms-sources.*` 保留最后一组正式对战对应的源码；默认入口的接线不会改写历史战绩。

`archive/v1.7-sources.*` 保存默认基线的原始源码、浏览器产物和独立原生构建所需的测试源码，后续实验通过独立参考库比较，不改写 v1.7。

`nnue256-policy-v1.8.otnnue` / `.pt` / `.json` 是新增策略头的候选产物，默认仍为 v1.7。`.pt` 保存策略分支及冻结特征层，价值输出权重复用父模型二进制。候选实验结果与复现说明见 [v1.8 策略头实验](v1.8-policy-comparison.md)，完整记录为 `v1.8-policy-vs-v1.7.json`，校验清单为 `v1.8-policy-experiment.json`。`archive/v1.8-policy-candidate-sources.*` 保存实测源码；它是候选实验快照，未将候选提升为默认基线。

同一模型的 [1 秒、20 局补充试验](v1.8-policy-1s-comparison.md) 得到 11 胜、2 平、7 负（60%），平均子差 −0.5，单独记录，不与 100ms 组混算。`v1.8-policy-1s-experiment.json` 为校验清单，`archive/v1.8-policy-1s-sources.*` 保存逐步监控与续跑运行器的实测源码。运行 `sh scripts/watch-policy.sh` 查看累计战绩及进行中的单局黑白比分；默认仍是 v1.7。

2026-10-27 用户决定：保留策略头实现和可选候选模型，继续作为 MCTS 的优化方向，是否采用仍看有无负收益。目前 1 秒组的小样本结果有潜力，100ms 组已观察到退步，不能概括为所有时限均无负收益。这次保留不等于提升默认；已批准默认基线仍为 v1.7。

用户随后明确要求“作为默认模型”，因此策略版已提升为 v1.8 默认基线，原生和浏览器均嵌入相同权重。历史候选记录保持不变；后续比较默认从 `default-model.json` 解析 v1.8。价值目标仍为子差，未在此次发布中改为胜负。

随后训练的 v1.9 单标量胜负候选仅更新 513 个价值参数，特征和策略字节冻结；每步 1 秒、20 局对 v1.8 取得 18 胜、2 平、0 负（95%），平均子差 +2.8。详见 [胜负价值实验](v1.9-outcome-comparison.md)，校验清单为 `v1.9-outcome-experiment.json`，完整报告为 `v1.9-outcome-vs-v1.8-1s.json`。默认仍为 v1.8，未自动提升候选。运行 `sh scripts/watch-outcome.sh` 显示单行战绩和带坐标的棋盘，`--details` 展开统计；结束后保留最终棋盘。

用户随后批准“换基线”：v1.9 胜负价值候选已提升为默认，原生与浏览器均嵌入同一模型。后续候选自动以 `v1.9-default.json` 为参考；`v1.9-outcome-experiment.json` 的候选状态与当时未提升的字段保留为历史记录。

[v1.10 更深主干实验](v1.10-deep-comparison.md) 按用户要求提前停止：1 秒预算下完成 9 局，0 胜 4 平 5 负。候选保留，默认仍为 v1.9，后续转向工程开销分析。统一监控为 `sh scripts/watch-match.sh`。

[v1.11 混合数据预测实验](v1.11-mixed-5m-comparison.md) 完整遍历固定 500 万局中的训练分片及原百万局面，沿用 v1.9 特征和结构，联合训练两头。完整轮次和验证最佳检查点在原 6,000 局面测试集上均低于 v1.9；两套模型与逐条预测分别保留，默认仍为 v1.9。

[v1.12 两层 CNN 原始数据离线组](v1.12-cnn2-offline-comparison.md) 与 [v1.13 学习率／损失权重对照](v1.13-cnn2-tuning-comparison.md) 保留各版 PyTorch 和 TorchScript 权重。v1.14 将验证集选出的 `cosine-joint` 与 `cosine-v4-value` 原权重导出为胜负／策略双头 `OTCNN2`，接入原生 PUCT MCTS，分别安排对 v1.9 每步各 2 秒、20 局。统一监控 `sh scripts/watch-match.sh`；接线与一致性记录为 `v1.14-cnn-policy-integration.json`，默认仍为 v1.9。

[v1.18 双方共同批处理诊断](v1.18-common-batch-comparison.md) 为 CNN 和 v1.9 同时提供 C++ 批次调度与单线程矩阵后端，分别保留原实现、batch 1/8/16/32/64 和相同 2 秒预算的固定局面记录。两边都有工程收益；不能只优化 CNN 后与未优化 NNUE 比较。当前仍是独立实验，v1.14 正式对战和默认发布保持原样。

## Pattern 在 v1.9 基础上的重新训练

当前 Pattern 实验以 v1.9 为对照，使用独立 `OTPAT2` 胜负价值模型，
实验标识为 v1.25。`pattern32-v1.8.otpat`、`v1.8-pattern-matches.json`
及 `v1.8-baseline.json` 仅保留其历史子差实验含义，不能替代新的对照。
训练配置和复现命令见 [训练说明](../docs/TRAINING.md#v125在-v19-基础上的-pattern-胜负价值实验)。
默认继续为 `default-model.json` 指定的已批准 v1.9。

v1.25 重训及 100ms／20 局对照已完成：候选 18 胜、0 平、2 负，
得分率 90%，平均子差 +15.5；默认仍为 v1.9。
见 [实测报告](v1.25-pattern-outcome-comparison.md) 和
[候选校验清单](v1.25-pattern-outcome-experiment.json)。

同一 v1.25 候选补充了每步 2 秒、20 局对冻结 v1.9 的对照：
**12 胜、5 平、3 负，得分率 72.5%，平均子差 +13.4**。
该组与 100ms 组独立保存；见 [2 秒报告](v1.25-pattern-outcome-2s-20-run1-comparison.md)
及 [校验清单](v1.25-pattern-outcome-2s-20-run1-experiment.json)。
默认继续为 v1.9。

## v1.25 用户批准的默认基线

2026-10-29 用户批准替换基线；`default-model.json` 现指向
`pattern32-outcome-v1.25.otpat` 和 `v1.25-default.json`。
原生及浏览器均内嵌该 OTPAT2 模型，后续比较以它为默认对手。
v1.9 通过 `archive/v1.9-default-sources.*` 保留原源码与 Wasm，旧模型文件不变。
100ms 和 2 秒的候选报告保持冻结；其中“默认仍为 v1.9”等文字描述的是候选时期。
候选验证工具通过 `archive/v1.25-candidate-era-verification.*` 检查当时的默认选择器、
浏览器文件和原验证工具，模型及对局报告仍逐项检查原文件。

v1.29 完成了 H32/H64 单因素训练对照与融合推理实验。
八种随机对称增强已在两组同步启用；H64 未超越原 v1.25 的测试胜负 MSE，
且推理更慢。本轮保留 H32，联合推理优化作为独立原生候选实现保存，
测得同一模型的 C ABI 联合推理吞吐约提升 20%，默认与浏览器 Wasm 保持冻结。
见 [v1.29 报告](v1.29-pattern-width-comparison.md) 和
[源码／模型校验清单](v1.29-pattern-width-experiment.json)。

## v1.32 INT8 / 整数 SIMD 候选

基于批准的 v1.25，将棋形表压为 INT8，保留 INT16 主干与输出层。按验证胜负 MSE 从步长 1/2/4 选择步长 1，文件 1,792,796 字节；没有重训或自动提升。原始数据见 `v1.32-pattern-int8.json`，完整精度、速度及 Linux 验证边界见 `v1.32-pattern-int8-comparison.md`。NEON/SSE2 与各自普通路径数值一致；INT8 与原模型存在量化差异。复核：`python3 scripts/finalize-pattern-int8.py`。

## v1.33 Pattern 推理热点候选

对整数舍入、位压缩棋形编码及四行 SIMD 做独立消融。整数舍入本机变慢，保留实验开关并排除在选中组合外；选中 `FAST_CODES + BLOCK4 + SIMD`，没有修改模型或提升默认。原始消融为 `v1.33-pattern-hotpath.json`，组合复测为 `v1.33-pattern-selected.json`，说明见 `v1.33-pattern-hotpath-comparison.md`。复核：`python3 scripts/finalize-pattern-hotpath.py`。

## v1.34 合法着策略投影候选

原 softmax 已仅处理合法着并缓存树先验；本轮跳过不触及合法落点的棋形策略投影，并仅汇总合法 logits，价值池化和完整诊断接口保持原样。100ms 搜索模拟数本机初测较 v1.33 增约 3.5%，保留逐轮波动。开关 `OTELLO_PATTERN_LEGAL_POLICY` 默认 OFF，未提升模型或替换 Wasm。见 `v1.34-pattern-legal-comparison.md`，复核 `python3 scripts/finalize-pattern-legal.py`。

## v1.35 提速策略集与 2s/20 局

组合融合推理、整数 SIMD、位压缩棋形编码、四行 SIMD、合法着相关策略投影；排除较慢的整数舍入和增量缓存。同局面两轮 2s 采样测得 INT16 约 493,551 模拟/秒（批准 v1.25 约 228,678，2.158×），网络评估吞吐约 1.868×。INT8 未超过预定 5% 门槛，因此用原批准 INT16 权重对战。

双方 2s/步、20 局配对执色，候选 5W/8D/7L，得分率 45%，平均子差 -3.7；1211 个行动完整回放。候选 106/602 次触及原 2,097,152 节点上限，触顶平均约 957ms，基线 14/598 次；不能把采样加速等同于棋力提升。默认与 Wasm 未提升。策略集配置见 `v1.35-pattern-speed-suite.json`，报告见 `v1.35-pattern-speed-suite-comparison.md`；复核 `python3 scripts/finalize-pattern-speed-suite.py`。

## v1.36 节点扩容实验

在 v1.35 原 INT16 提速策略集上增加显式容量 C ABI，旧入口与默认仍为 2,097,152 节点。连续局面预检比较 2M/4M/8M，选择满足预算利用且较小的 4,194,304：24 次搜索触顶由 10 次降为 0 次，未证明搜索全部用满 2s，新叶评估总量增约 18.5%；预检峰值 RSS 约 2.60 GiB。

本轮双方 2s/步、20 局对冻结 v1.25，候选 7W/7D/6L，得分率 52.5%，平均子差 +2.4。候选 603 次思考无触顶（上轮 106/602），本轮候选峰值 RSS 2493.9 MiB。1213 个行动完整回放；预算问题解决，胜负改善仍需更大样本确认。保持候选、未提升默认或 Wasm。配置 `v1.36-pattern-capacity.json`，报告 `v1.36-pattern-capacity-comparison.md`，复核 `python3 scripts/finalize-pattern-capacity.py`。


## User-approved v1.36 engineering baseline (2026-10-30)

The active baseline is `models/v1.36-default.json`, selected by
`models/default-model.json`. It retains the exact v1.25 INT16 Pattern outcome
weights and enables fused current-side inference, NEON/SSE2 with portable fallback,
packed shape encoding, four-row reuse and legal-only policy projection. The default
node budget is 4,194,304; incremental cache and integer rounding remain disabled.
The previous v1.25 baseline and candidate-era reports stay frozen.

Controlled 2s sampling measured 2.158x simulations/s and 1.868x evaluations/s.
The 4M capacity match against frozen v1.25 at 2s/20 games finished 7W/7D/6L
(52.5%, mean discs +2.4), with 0/603 candidate decisions hitting the node cap.
This is engineering parity, not evidence of statistically significant strength gain.
Candidate process peak RSS was 2493.9 MiB; provision memory accordingly on Linux.
Native kernels select AArch64 NEON or x86-64 SSE2; actual Linux throughput remains
unmeasured. Browser budgets stay 100ms/1s and the verified opening book stays enabled.
See `models/v1.36-pattern-capacity-comparison.md` and
`models/v1.35-pattern-speed-suite-comparison.md` for separate measured results.


## v1.37：两个独立 Pattern 结构候选

在批准的 v1.36 上分别评估对称表项共享和十组几何池化，另有同期重训原结构对照；
三组使用相同百万局面、采样与 D4 增强顺序、15 轮及原胜负/策略监督。
两项结构改动不组合，不自动替换默认。协议见 `models/v1.37-pattern-structure-protocol.json`，
最终离线、吞吐与分预算对战结果见 `models/v1.37-pattern-structure-comparison.md`。

候选使用 OTPAT4：32 字节头后接 uint32 标记（1=对称共享、2=十组池化），
维度分别为 28,503×32 表/97→32 主干和 55,728×32 表/321→32 主干。
文件和原生内存均保留压缩对称表；321 维点积按最多 128 项分段汇总到 INT64，
遵守原 SIMD 内核的范围约束。原结构输出与冻结 v1.36 独立参考库逐项核验。

复现脚本依次为 `scripts/train-pattern-structure.py`、
`scripts/benchmark-pattern-structure.py`、`scripts/run-pattern-structure.py`；
审计为 `python3 scripts/finalize-pattern-structure.py`。已存在的模型和报告拒绝覆盖。
监控入口仍为 `sh scripts/watch-match.sh`，显示候选视角胜平负、当前棋盘并保留终盘。

本轮实测（候选视角，预算不合并）：对称共享 100ms 为 9W/5D/6L，2s 为 10W/7D/3L；
十组池化 100ms 为 7W/1D/12L，2s 为 9W/5D/6L。80 局共 4,865 个行动已独立回放。
结果来自单训练种子及每组 20 局，不作统计显著性结论。运行期间的用户请求制图活动、
监控配置修正记录在 `models/v1.37-pattern-structure-runtime-notes.json`；吞吐基准在制图前完成。

当前共享版的结构图为 `output/pdf/pattern-symmetry-v1.37.pdf`（另有 SVG/PNG）。
`scripts/pattern_policy_mlp.py` 是随后讨论的直连共享策略头原型：
每个落点的局部 32 维、全局 32 维及 D4 位置桶 one-hot 10 维直接拼接，
使用同一个 74→8→1 MLP。它通过梯度、换色及 D4 等变测试，现已作为 v1.38 独立候选训练并导出原生模型，
随后已完成独立的 200ms/20 和 2s/20 对照，没有混入上述 v1.37 结果。示意图为 `output/pdf/policy-head-direct-mlp8-proposal.pdf`。


## User-approved v1.37 symmetry baseline (2026-10-30)

The active default is `models/v1.37-default.json`, selected by `models/default-model.json`.
User approval: “把这个作为基线：对称共享”. The exact measured weights are
`models/pattern32-symmetry-v1.37.otpat` (OTPAT4 variant 1), SHA-256
`762b51297726ac0618ffe41485ad097f13dfe7a809361c052ac795c61bc61235`.
It ties reversed lines and transposed corners: 28,503 × 32 INT16 table entries,
916,750 parameters and 1,833,692 bytes. Keep the 38 pattern instances, three
mean pools, 97→32 trunk and original conditional policy head.

Against frozen v1.36, preserve separate measured results: 100ms/20 games
9W/5D/6L (57.5%); 2s/20 games 10W/7D/3L (67.5%). These are screening
samples, not a statistically significant strength claim. No retraining or new
strength match is required for this user-approved promotion.

Retain v1.36 search settings: 4,194,304 nodes, learned-prior PUCT, finite FPU
(parent mean minus 0.15), subtree reuse and outcome terminal/proof semantics.
The verified Egaroucid book stays mandatory; browser budgets remain 100ms/1s.
Ten-pool and direct-policy MLP candidates are excluded. Keep candidate-era
reports and all previous baselines immutable. Future comparisons use v1.37.


## v1.38：直连共享策略头候选

已按用户请求训练固定共享 74→8→1 MLP 策略头，独立于已批准 v1.37 基线。
保持棋形、对称共享、三组池化和价值主干不变，局部 32 维、全局 32 维及
D4 位置桶 one-hot 10 维直接拼接打分。与基线使用相同初始化的表/主干/价值输出，
相同百万局面、样本/增强顺序和 15 轮配置，从随机参数联合重训；第 14 轮按验证价值 MSE 选中。

测试胜负 MSE：v1.37 的 0.690203 → 0.685687；记录走法命中率 41.03% → 44.08%；
策略 CE 1.611810 → 1.515972。原生 100ms/1s 模拟吞吐分别约为基线 0.866×/0.895×，
网络评估吞吐为 0.853×/0.831×；只汇总同局面所有轮次用满预算且未证明、未触顶的共同样本。
上述离线阶段只衡量质量与计算成本。随后独立对战结果见下文，不据此自动提升默认。

模型为 `models/pattern32-direct-mlp8-v1.38.otpat`（OTPAT5，915,973 参数、1,832,076 字节），
使用独立 `.build/native-pattern-direct-policy-v1.38/` 原生库。标量/SIMD 和 PyTorch/C++
在验证/测试 12,000 局面通过输出一致性；固定模拟、D4、损坏文件与 ASan/UBSan 均检查。
报告见 `models/v1.38-direct-policy-comparison.md`，历史离线审计为 `python3 scripts/verify-direct-policy-history.py`（使用冻结源码快照）。
训练入口为 `python3 scripts/train-pattern-direct-policy.py`（已存在模型时拒绝覆盖）；
监控仍为 `sh scripts/watch-match.sh`。默认、内嵌模型和浏览器 Wasm 保持冻结 v1.37。


## v1.38 实战与 v1.39 策略宽度候选

MLP8 对冻结 v1.37 的候选视角实测保持分预算记录：200ms/20 局为 9W/10D/1L，
2s/20 局为 4W/11D/5L（47.5%）。后者共 1,216 个行动均已独立回放；同机并行进行
用户要求的 MLP 加宽训练与候选代码构建，没有 CPU 隔离。结果按实际测量保留，
不能从短预算优势推出长预算优势。默认仍是 v1.37，没有提升 MLP。
分别见 `models/v1.38-direct-policy-200ms-20-comparison.md` 和
`models/v1.38-direct-policy-2s-20-comparison.md`。

v1.39 只将直连共享策略头 74→8→1 的隐藏宽度分别扩为 16、32，其他棋形、
价值主干、初始化、数据/增强顺序、损失与 15 轮训练条件保持一致；全网随机重训，
两个模型均按验证胜负 MSE 选中第 14 轮后统一评估测试集。

| 直连策略头 | 测试胜负 MSE | 测试走法命中率 | 策略 CE | 2s 评估吞吐 / MLP8 |
|---|---:|---:|---:|---:|
| MLP8 | 0.685687 | 44.08% | 1.515972 | 1.000× |
| MLP16 | 0.686852 | 44.80% | 1.485150 | 0.896× |
| MLP32 | 0.687189 | 45.40% | 1.470035 | 0.792× |

加宽改善了走法拟合，但价值 MSE 略差且新叶评估变慢。模拟吞吐受搜索路径影响，
2s 相对 MLP8 分别为 1.012× / 0.832×，不要与网络评估吞吐混用。
计时在训练与对局结束后进行。MLP16/32 尚未做棋力对战，均作为候选保存。
新宽度各通过 12,000 个局面的 Torch/C++、SIMD/portable 输出对照；独立标量参考、
固定模拟搜索、D4 与 ASan/UBSan 检查通过，原 MLP8 的二进制重新导出逐字节一致。

模型为 `models/pattern32-direct-mlp16-v1.39.otpat` 与
`models/pattern32-direct-mlp32-v1.39.otpat`；训练入口是
`python3 scripts/train-pattern-policy-width.py`，报告与审计分别为
`models/v1.39-policy-width-comparison.md`、`python3 scripts/finalize-policy-width.py`。
监控继续使用 `sh scripts/watch-match.sh`。默认权重和浏览器 Wasm 的哈希保持不变。


## 训练数据：棋手 ID 与走法质量审计（2026-10-30）

原始 WTHOR 有双方棋手索引，但旧转换器未保留；已给全部 137,507 盘转换后 WTHOR
生成独立棋手溯源 sidecar。4,225 个出现过的索引包含未知编号 0；名册不含 Elo，
历史评级尚未可靠关联，未应用棋力阈值或改写冻结训练数据。
百万缓存没有跨集合对称重复或非法策略标签，但 19,829 个位置有多走法记录，
27,154 个有多种实战终局胜平负。应先恢复逐条落子者信息，区分策略模仿质量与价值标签，
再单独比较评级权重、引擎解着和残局理论值；不按赢家身份判断招法优劣。
完整统计与建议见 `models/training-data-audit-2026-10-30.md`；复核脚本
`scripts/audit-training-data.py`。原始数据归档、默认模型和浏览器 Wasm 保持不变。


## v1.40：多标签数据与内部 Elo（未训练）

已给已知参与者统一 1500 初值，按 K=32、400 尺度的 logistic Elo 更新，完整语料
重放 136,100 盘；另算 train-only 池，避免使用 holdout 赛果。1/0.5/0 是实际得分 S，
真实更新为 K*(S-E)，不是直接加分。
多标签候选保留 2,060,308 次出现、1,255,303 个局面的走法与结果分布，提供原百万行
匹配视图。全部冻结数据和默认模型未变。
用户要求先讨论 Elo 用法；未启动新模型训练，附带权重只作试算，不是选定的训练方案。
细节见 `models/v1.40-training-consensus-elo.md`。


## WTHOR-only 数据与同结构训练（v1.41–v1.43）

训练来源限定 WTHOR，独立 Logistello 语料排除；聚合按原始出现次数等权，验证/测试
各 6,000 个局面及标签保持原值。阈值比较推荐 1600（双方都低于才剔除）：
51,854 盘、664,495 个局面；不筛 Elo 则为 80,561 盘、1,054,132 个局面。
阈值只是数据候选判断，详见 `models/v1.41-wthor-elo-filter.md`。

v1.42 使用对称共享基线原结构、WTHOR 全量和等权经验软标签重新随机训练，
15 轮后按冻结验证 MSE 选中第 12 轮；测试 MSE 0.700043、策略命中率 40.58%。
训练后独立与冻结 v1.37 做 2s/20 对照，双方使用同一冻结原生库；1600 过滤版
v1.43 按用户要求随后训练，结构与训练配置相同，不使用 MLP8。默认保持 v1.37。
训练记录与协议见 `models/v1.42-pattern-wthor*.json`、
`models/v1.43-pattern-wthor1600*.json`；v1.42 离线报告为
`models/v1.42-pattern-wthor-comparison.md`。监控为 `sh scripts/watch-match.sh`。


### v1.42 用户停止对局；v1.43 训练完成

用户要求停止 2s 对照。v1.42 实际完成 12/20 局：2W/4D/6L（33.3%），
第 13 局未完成部分不计入结果；已完成 729 个行动独立回放。停止记录保留
`complete=false`、`userStopped=true`，没有写成完成 20 局。原生进程退出后才训练
v1.43，没有并行训练干扰已完成的对局。

v1.43 的 1600 版按验证 MSE 选中第 8 轮，冻结测试 MSE 0.709274、走法命中率
39.30%、策略 CE 1.658965；没有安排新对局。当前两版离线指标均未优于 v1.37，
默认保持原基线。完整对照见 `models/v1.43-wthor-data-comparison.md`，停止审计为
`python3 scripts/finalize-pattern-wthor-stopped.py`。

WTHOR 全量有 1,054,132 个唯一局面，数量比旧百万缓存略多；来源覆盖与软标签
聚合同时变化，不能单独归因。下一步宜固定同一 WTHOR 局面集，对比首条硬标签
与聚合软标签；本轮未额外启动该消融。训练数据覆盖诊断见
`models/v1.43-wthor-data-diagnosis.json`。验证/测试及默认权重保持冻结。


## v1.44–v1.45：固定局面的硬/软标签对照

用户明确主对照要保留 WTHOR + Logistello。v1.45 固定原始百万训练棋盘、行棋方和
行顺序，硬标签直接使用冻结 v1.37 的结果和走法，软标签汇总两库原始 train 分区
中的同局面记录，再把规范化策略分布反向映射至原棋盘坐标。没有 Elo 筛选或权重，
验证/测试各 6,000 个局面与标签保持不变。

原硬标签版已逐字节复现 v1.37，15 轮全部损失与验证指标也精确相同。两版的完整
初值、样本/增强序列和输入哈希相同，各自选第 14 轮后才统一评估测试集：
硬标签 MSE 0.690203、走法命中率 41.03%、策略 CE 1.611810；
软标签 MSE 0.690274、走法命中率 40.97%、策略 CE 1.611682。离线差异很小。
每版通过验证/测试合计 12,000 个局面的 Torch/冻结原生库导出对照；还核验了
768 个局面的标量坐标映射，原硬标签在软分布中均有支持。

完整主对照：`models/v1.45-mixed-label-ablation-comparison.md`。随后用户授权 soft
与基线对战，单独沿用 2s/步、20 局；协议与现场监控为
`models/v1.45-soft-label-2s-20-protocol.json`、`sh scripts/watch-match.sh`。
默认保持 v1.37，离线分数和候选对局都不自动授权提升。

v1.44 是先前完成的 WTHOR-only 旁证，不作为这次混合语料主对照。其硬/软结果
分别为测试 MSE 0.700929 / 0.700043、走法命中率 40.67% / 40.58%；
软标签版逐字节复现 v1.42。另见 `models/v1.44-wthor-label-ablation-comparison.md`。


## v1.45 soft 对局停止；v1.46 分阶段价值头

用户停止 soft 2s/20 局对照，实际完成 17 局：soft 4 胜 / 8 平 / 5 负，
得分率 47.1%，平均子差 −2.0；第 18 局未完成，不计入统计。已独立回放
1,036 个完整行动，稳定监控保留第 17 局终盘。原始运行快照与开赛协议不改写。
详见 `models/v1.45-soft-label-2s-20-comparison.md`。

用户授权的 v1.46 在该对局停止、工作进程退出后训练。共享对称棋形表、原三组池化、
97→32 主干及条件策略结构不变，只增加 32→4 开局价值/WLD 读出，与原四行中后盘
读出按空格数平滑混合：已落子 ≤12 手使用开局头，≥24 手使用中后盘头，过渡区
线性混合量化 logits。两阶段都预测相同行棋方胜/平/负结果；策略权重仍联合训练。

原百万 WTHOR + Logistello 硬标签，验证/测试各 6,000 局面不变；没有 Elo 筛选、
加权或聚合。原共享初值、初始预测及样本/增强序列与 v1.37 一致。15 轮选第
14 轮：验证 MSE 0.726839（基线 0.722156），
测试 MSE 0.691479（基线 0.690203），策略命中率
41.02%（基线 41.03%）。这版离线总体未改善；中盘价值
MSE 略降，开局和后盘略升。单种子结果不能推断棋力。

候选 `models/pattern32-symmetry-phase-value-v1.46.otpat` 使用新 OTPAT6 格式：
916,882 参数、1,833,972 字节，分别增加 132 和 280。支持新格式的原生库在隔离
构建目录生成，默认库和浏览器嵌入权重不变。Release/关闭 SIMD 两组 CTest、
两组各 2 项 Python 测试、候选 12,000 局面导出对照和旧基线 12,000 局面回归对照通过。

未启动候选棋力对战，默认仍为 v1.37。完整报告、协议与复核入口：
`models/v1.46-phase-value-comparison.md`、`models/v1.46-phase-value-protocol.json`、
`python3 scripts/finalize-phase-value.py`。训练入口 `python3 scripts/train-pattern-phase-value.py`
拒绝覆盖已完成实验，复训应使用新的版本和输出路径。


## 分阶段头方向停止（用户决定）

用户判断“指标差不多那就不需要做这么花哨的操作了”，因此停止 v1.46 分阶段价值头、
v1.47 分阶段策略头的后续迭代，不做组合、不启动额外棋力对局，也不提升默认。
主线保留用户已批准的 v1.37 对称共享、单价值/WLD 读出和单条件策略读出。
候选权重、训练报告与源码快照保留追溯，监控条目移入历史。

v1.47 单头和分阶段策略两组各训练 15 轮，均按固定中盘验证策略 CE 选择第 15 轮。
中盘测试 CE 为 1.813585 / 1.815451（分阶段略差），命中率为 35.23% / 35.35%；
未呈现稳定的策略收益。完整结果保留在 models/v1.47-phase-policy-comparison.md，
用户决策及冻结默认哈希为 models/v1.47-phase-head-decision.json。

## v1.49：全局 8×8 策略头，全网重训候选

保留 38 个棋形投影的空间顺序，以 D4 共享的 `38→16→64` 网络一次输出
64 个策略修正分数，叠加原条件策略。新增 237 个独立策略参数；本次实际从随机
初始化训练全部 916,987 个参数，未冻结编码器或价值分支，未加载已训练权重。
原共享参数初值、百万硬标签局面、15 轮配置和样本/D4 顺序保持一致；联合损失
权重沿用 v1.37，选模改为最低验证合法策略 CE，选中第 15 轮。

测试策略 CE 为 1.546291（v1.37：1.611810），命中率 42.78%（41.03%），
胜负 MSE 为 0.686413（0.690203）。验证胜负 MSE 则为 0.724789，略高于
基线 0.722156。离线结果、单种子与不同选模标准不构成棋力提升证据。

完整训练权重为 `pattern32-global-policy16-full-v1.49.pt`；配套 `.base.otpat`
只是棋形/主干/原读出组件，必须结合 `.head.json` 才是完整策略模型，不能单独
当作完整候选引擎。12,000 局面的全局头整数参考及基础组件 Torch/C++ 导出核验
通过；D4、全网梯度和实时指标检查通过。尚未接入完整原生策略推理、测速或对战，
默认仍为冻结 v1.37，浏览器权重未改。

结果：`v1.49-pattern-global-policy-full.json`；比较：
`v1.49-global-policy-full-comparison.json`；协议与源码归档分别为
`v1.49-global-policy-full-protocol.json`、`archive/v1.49-global-policy-full-sources.json`。
训练入口 `python3 scripts/train-pattern-global-policy-full.py` 拒绝覆盖已有实验；
核验入口 `python3 scripts/finalize-global-policy-full.py`。

### v1.49 完整原生模型与 2s/20 局对照

后续按用户要求接入完整原生全局策略头，并与冻结 v1.37 比较。完整候选为
`pattern32-global-policy16-full-v1.49.otpat`（OTPAT9，1,837,288 字节），包含
选中权重的基础组件及完整 38→16→64 策略修正，不是只加载 `.base.otpat`。
部署元数据单独记录为 `.native.json`；此前训练/组件报告不改写。

双方各自独立进程，每步 2 秒、4,194,304 节点、相同 PUCT/FPU、子树复用、
v1.3 开局多样性及胜负终局证明。从初始棋盘开始，无开局库，10 对种子交换执黑白。
候选 **10 胜 / 6 平 / 4 负，得分率 65%，平均子差 +7.5**；执黑 9/1/0，
执白 1/5/4。20 局筛选样本不作 Elo 或显著优势推断，默认仍为 v1.37。

1,214 个行动已回放、逐决策选着已审计，双方均无节点容量触顶。候选/基线
平均新叶评估数为 493,055 / 592,411，峰值 RSS 为 1,732.75 / 1,627.25 MiB；
两者实际遇到的局面不同，这些是对战描述统计，不能解释为纯内核速度倍率。

完整模型的 Torch/C++ 验证/测试共 12,000 局面对照、标量/SIMD 全部 68 项输出
及固定搜索对照通过；旧基线回归、零修正等价、合法策略、D4、状态接口、坏文件
拒绝及权重变更后的缓存失效均通过。原生源码和库均独立留档，未重训、未更新
浏览器或默认权重。稳定监控 `sh scripts/watch-match.sh` 保留第 20 局终盘。

原始对战：`v1.49-global-policy-vs-v1.37-2s-20.json`；汇总：
`v1.49-global-policy-2s-20-comparison.json`；审计：
`python3 scripts/finalize-global-policy-match-2s.py`；源码归档：
`archive/v1.49-global-policy-native-sources.json`。

### v1.49 对冻结 v1.9：200ms/步、50 局

用户另行要求历史版本对照。v1.9 权重仍为
`nnue256-outcome-v1.9.otnnue`，哈希与冻结清单一致；参考引擎从校验通过的
`archive/v1.9-default-sources.tar.gz` 原样恢复，重建出的库哈希也与当年冻结库
`6f593a25a0d52f1846e29db35b10fa5423505fd820eb124d4a474489f412c188` 完全一致。
最新候选使用上一轮的同一完整 v1.49 权重和原生库，没有重训或改权重。

从初始棋盘开始，无开局库，每步 200ms，25 对种子交换执黑白。候选
**48 胜 / 1 平 / 1 负，得分率 97%，平均子差 +24.56**；执黑 24/1/0，
执白 24/0/1。3,052 个行动和全部决策均通过回放审计，双方均未触及节点容量。
稳定监控保留第 50 局终盘。该预算/对手与此前 2s/20 记录分开，不改默认 v1.37。

这是保留各版本整体设置的比较：v1.9 用原 NNUE 策略、未访问分支逐一试探及
2M 节点上限；v1.49 用 Pattern/全局策略头、有限 FPU 及 4M 上限。结果不能
单独归因于新策略头，也不是只更换权重的结构消融。实战平均模拟/步为
139,169 / 24,713，新叶评估/步为 87,309 / 17,635（候选/旧版）；实际遇到
的局面不同，不能视为纯内核速度倍率。

原始记录：`v1.49-global-policy-vs-v1.9-200ms-50.json`；汇总：
`v1.49-v1.9-200ms-50-comparison.json`；审计：
`python3 scripts/finalize-global-policy-vs-v1.9.py`；协议和源码：
`v1.49-v1.9-200ms-50-protocol.json`、`archive/v1.49-v1.9-200ms-50-sources.json`。

## User-approved v2 global policy baseline (2026-10-30)

The active default is `models/v2-default.json`, selected by `models/default-model.json`.
User approval: “感觉可以作为v2了”. The complete measured v1.49 model is preserved
byte for byte as `models/pattern32-global-policy-v2.otpat` (OTPAT9), SHA-256
`df11f7b9eed358fff33a426e2f0cb1a290083d2bc8e0781a2fbd1efdaf4270c3`.
All 916,987 independent parameters were trained together from random initialization;
237 are the additional D4-tied global policy coefficients. The 38→16→64 head runs
once per board and adds all-square corrections to the original conditional policy.
The native engine and browser Wasm embed the same complete model.

Preserve the separate measured candidate-era results: against frozen v1.37,
2s/20 games gave 10W/6D/4L (65%, mean discs +7.5); against frozen v1.9,
200ms/50 games gave 48W/1D/1L (97%, mean discs +24.56).
These use model-only search without the opening book. They are different opponents
and budgets; no Elo or statistical significance claim follows from these samples.
No retraining or new strength match was required for this user-approved promotion.

Retain the v1.37 search settings: 4,194,304 nodes, learned-prior PUCT,
finite FPU (parent mean minus 0.15), subtree reuse and outcome terminal/proof semantics.
The verified Egaroucid opening book stays mandatory, followed by v2 on a verified
miss; browser budgets stay 100ms/1s. Earlier baselines and candidate-era records
remain immutable. Future comparisons use v2. Historical preservation verification:
`python3 scripts/verify-v2-history.py`; current default verification:
`python3 scripts/verify-v2-default.py` and `npm run test:baseline`.

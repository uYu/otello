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

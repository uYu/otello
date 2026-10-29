# 模型与实验记录

补充棋形 NNUE 实验（本地标识 v1.8，独立于其他机器的版本编号）使用 `pattern32-v1.8.otpat`（独立的 `OTPAT1` 格式），同时训练策略、胜负和、子差三个输出；默认对手仍为 v1.3 NNUE＋MCTS。结构、验证、速度及策略先验消融对战见 [v1.8 实验报告](v1.8-comparison.md)。

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

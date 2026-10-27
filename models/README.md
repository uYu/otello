# 模型与实验记录

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

可读结果见 [comparison.md](comparison.md)。实际训练数据与开局库的压缩包位于 [data/](../data/README.md)，恢复命令为 `python3 scripts/data-assets.py restore`。

`*.live.json` 是运行时监控状态，不提交；监控脚本在没有实时文件时读取已完成的报告。两组从标准初始局面交替颜色、每步 50ms，固定起点的重复对局不当作独立开局样本或 Elo 评估。

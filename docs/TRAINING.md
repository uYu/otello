# 对局数据与 256 维 NNUE v1

当前交付的是一个可离线加载的 **128 → 256 → 1** 局面价值模型。它参考 NNUE 的稀疏特征、共享双视角、量化和增量更新方法；具体黑白棋特征和网络尺寸是本项目的实验选择。不是 Stockfish 原版网络，也不兼容 Stockfish `.nnue` 权重。

## 数据来源与实际体积

2026-10-27 从两个作者/维护者站点获取。原始棋谱固定打包于 [data/training/otello-v1.tar.gz](../data/training/otello-v1.tar.gz)，校验和恢复到 `.build/training/raw/` 后，用脚本重建 JSONL 与缓存。

| 来源 | 实际记录数 | 下载大小 | 原始棋谱大小 |
| --- | ---: | ---: | ---: |
| [FFO WTHOR](https://www.ffothello.org/informatique/la-base-wthor/) 1977–2025，49 个年度文件 | 137,548 | 9,354,048 字节 | 同左 |
| [Michael Buro 的 Logistello](https://skatgame.net/mburo/log.html) | 121,123 | gzip 3,268,866 字节 | 22,887,603 字节 |
| 合计 | 258,671 | 12,622,914 字节 | 32,241,651 字节 |

Logistello 页面仍写约 37K；这里是实际下载文件的非空行数。它是带最终黑方结果的开局库骨架对局，不能把页面对旧文件的质量描述直接推广到新增的所有记录。WTHOR 是真实比赛，不能把每一步当成最优策略。

两处来源均明确邀请用对局研究开局、训练评估参数。没有找到独立的 MIT/CC 数据许可；本项目按用户要求归档实际使用的数据，保留来源和 SHA-256，不为外部数据另加许可。Logistello 源代码的 GPL 与对局数据的使用说明分别记录。

## 清洗与划分

`otello_dataset` 用同一个 C++ 规则核心逐手重放，解析 [WTHOR 官方格式](https://www.ffothello.org/wthor/Format_WThor.pdf)和 Logistello 的带颜色坐标串，插入合法强制停着，拒绝非法落点、错方、终局后继续落子和损坏记录。满盘时复核实际子差。未走完的对局保留 `complete:false`，训练时排除。

本次全量结果：258,671 条输入，253 条对称重复，258,418 局保留，0 条非法记录，0 条满盘分数冲突。其中 413 局未走完；保留棋谱含 15,481,995 次落子。JSONL 约 89.25 MB，包含来源、分组、分数和显式停着；体积大于紧凑原始格式。

同一前 8 步的对称开局分配到同一集合（哈希桶 80/10/10，并非精确局数比例）。再对采样局面做八种棋盘对称与颜色相对视角去重，测试、验证集合优先占有跨集合转置局面。按整组划分后才进行训练集随机对称增强。

## v1 训练

训练使用开局分组属于 train 的全部 141,336 局完整棋谱：WTHOR 80,561 局，Logistello 60,775 局。每局在第 8–56 次落子之间每 3 步取一个局面，按对称与相对颜色去重后，固定种子抽取 **1,000,000 个训练局面**。

验证集与测试集各 6,000 个局面，逐数组保持与 48,000 局面原型版一致。训练采样前将这些局面的八种对称/相对颜色键加入排除集合，避免跨集合转置泄漏。原型的权重和元数据保留在 `models/archive/nnue256-prototype48k.*`。

- 输入：黑/白占位各 64 个二值特征。空格隐式表示，不输入卷积或合法落点通道。
- 共享特征层：128 → 256，黑、白两个视角各维护一个累加器。
- 激活：截断到 `[0,1]`；共享线性输出头对两个视角结果作差，再乘行棋方、经过 tanh。
- 训练目标：对局记录的最终黑方子差 / 64，转换到当前行棋方视角。这是对局结果监督，不是每个中盘局面的完美搜索分数。
- 33,280 个可训练参数；模型文件 67,096 字节。特征权重、输出头为 int16，偏置与累加器为 int32，输出汇总用 int64；尺度 256。
- PyTorch CPU，固定种子 20261027，AdamW，30 轮，batch 1024，学习率 0.001。按验证 MSE 选取第 30 轮权重；对战前固定模型 SHA-256。

同一测试集上，量化模型 MSE 从原型的 0.05869 降至 **0.05255**，平均绝对子差误差从 **11.24 子降至 10.58 子**。全预测零子差的基准为 11.89 子。误差是相对实战最终结果，不等于棋力；本次同时增加了训练数据和训练轮数。

1,024 个测试局面的 Python 量化计算与 C++ 导出模型评分一致（仅整数取整误差，最大 < 0.5 / 1000）。数据快照见 [v1-dataset.json](../models/v1-dataset.json)，训练指标与权重哈希见 [nnue256-v1.json](../models/nnue256-v1.json)，两组对战见 [v1 基线](../models/comparison.md)。

## 重现

有仓库内快照时，先执行 `python3 scripts/data-assets.py restore`。构建原生核心、安装 `requirements-training.txt` 后，执行 `npm run training:rebuild`：它只从原始棋谱生成清洗 JSONL、固定验证/测试缓存与百万局面缓存，不训练模型，并核对三个 v1 文件哈希。已在新目录实测重建，三个哈希均与原版一致。最后用 `npm run training:run` 训练。下面的全量下载/重建流程用于独立复核或未来更新数据。

```sh
cmake -S . -B .build/native -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native -j
python3 scripts/fetch-training-data.py
python3 scripts/prepare-training-data.py

# 全新目录先重建原型缓存，提供固定的验证/测试局面。
# 训练依赖 numpy、torch；native 推理无需 Python 或 PyTorch。
python3 scripts/train-nnue.py --hidden 256 --prepare

# v1 正式训练；输出到 .build，交付模型保留在 models/。
python3 scripts/train-nnue.py --hidden 256 --prepare \
  --all-training-games --train-positions 1000000 \
  --holdout-cache .build/training/positions.npz \
  --cache .build/training/v1/positions.npz \
  --output .build/training/v1/nnue256-v1.otnnue \
  --epochs 30 --batch-size 1024 --parity-positions 1024
npm run test:training

# 两组基线实验需要先构建 Egaroucid 原生依赖。
npm run setup:egaroucid
npm run training:baseline
# 在另一个终端监控：
npm run training:watch
```

下载脚本从官方页面发现年度链接；清洗脚本先检查 manifest 的下载校验和。缓存的 `positions.json`、训练结果 JSON、原始文件 manifest 和对战逐手记录均保存在 `.build/training/`。本次原始数据的 URL/大小/哈希清单同时保存在 [training-data-manifest.json](../models/training-data-manifest.json)。更换数据或采样参数时用 `--prepare` 重建局面缓存。重新下载会重新记录来源时间；网站更新可能改变数据快照。

macOS 默认动态库为 `.dylib`；Linux 训练/对战脚本须用 `--library .build/native/libothello.so`。支持任意 C++17 原生调用环境；没有 WebAssembly 前置依赖。

## 使用交付模型

```sh
.build/native/otello evaluate --network models/nnue256-v1.otnnue
.build/native/otello analyze --network models/nnue256-v1.otnnue --time 1000
python3 examples/nnue_ffi.py
```

C++ 使用 `nnue::Network::load`，将模型地址放入 `SearchOptions::network`，并保持模型活到搜索结束。C 使用 `ot_nnue_load/evaluate/search/destroy`，输出布局与 `ot_search` 相同；错误通过 `ot_nnue_error()` 获取。模型只读，可供多个独立搜索共享。没有指定模型时仍使用启发式；Egaroucid 使用上游自己的评估与开局库。

模型是实验版，目前通过原生 CLI/C ABI 使用，网页尚无此模型的选择项。文件头有版本、特征数、隐藏宽度和量化尺度校验；截断、尾随字节或不支持的宽度会拒绝加载。`OTNNUE1` 是本项目格式。

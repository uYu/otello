# 对局数据与 256 维 NNUE v1

初版 v1 交付的是一个可离线加载的 **128 → 256 → 1** 局面价值模型。它参考 NNUE 的稀疏特征、共享双视角、量化和增量更新方法；具体黑白棋特征和网络尺寸是本项目的实验选择。不是 Stockfish 原版网络，也不兼容 Stockfish `.nnue` 权重。

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

v1.1 只更新搜索实现，不重新训练模型。后续版本棋力评测只比较 NNUE，启发式引擎保留作为较弱的玩家对手。v1.1 对 v1 的对战从标准初始棋盘自行走棋，不预设开局走法、不使用开局库；参数与结果见 [v1.1 对比](../models/v1.1-comparison.md)。

## v1.2 行棋方输出头

v1.2 的共享特征变换器仍是 128→256 和 clipped ReLU。保留黑/白两个增量累加器，评估时按行棋方选择 `own`、`opponent`，拼接后经 512→1 线性输出（带偏置）及 tanh。这样保留棋子颜色互换与行棋方同时互换时的评分一致性，同时解除“同一棋盘仅换行棋方，评分必然反号”的限制。不增加隐藏层、阶段头、局部棋形特征或新标签。

训练复用 `.build/training/v1/positions.npz`，缓存 SHA-256 与 v1 相同；仍为 100 万训练局面、验证/测试各 6,000 个局面。训练种子 20261027、30 轮、batch 1024、AdamW 学习率 0.001、weight decay 0.001，按最小验证 MSE 选择权重。运行 `npm run training:run:v1.2` 前需已构建 `.build/native-v1.2`，Linux 将命令的 `.dylib` 换为 `.so`。训练产物先写 `.build/training/v1.2`，发布权重单独保存为 `models/nnue256-v1.2.*`，不覆盖 v1 模型。

二进制格式 `OTNNUE2\0` 沿用 24 字节头部，随后按顺序存储：128×hidden 个 int16 特征权重、hidden 个 int32 特征偏置、hidden 个 int16 行棋方输出权重、hidden 个 int16 对方输出权重、一个 int32 输出偏置。权重和特征偏置按 256 缩放，输出偏置按 256² 缩放；累加器是 int32，输出内积使用 int64。加载器兼容原 `OTNNUE1`；测试覆盖两种格式、32/256 宽度、损坏文件、换色一致性、行棋方影响、增量更新和固定深度穷举。

v1.2 对 v1.1 仍以每步各 50ms、100 局交换黑白、标准初始棋盘自行走棋、不使用开局库进行比较。见 [v1.2 对战结果](../models/v1.2-comparison.md)。

## v1.4：8×8 小卷积网络实验

该实验将输入保留为当前行棋方、对方两张 8×8 占位图：`2×8×8 → Conv3×3(8) → ReLU → Conv3×3(8) → ReLU → Flatten512 → Linear32 → ReLU → Linear1 → tanh`。两层卷积均使用一格零填充，空间尺寸始终为 8×8；全连接层保留绝对格子位置信息。按当前行棋方排列输入，所以延续了对行动权的表达和颜色重命名一致性。

共 17,185 个参数。卷积权重在棋盘上共享，参数量比 NNUE 小，但同一卷积核要反复作用于多个格子，不能从文件大小推断计算量更低。本实验原生推理使用 FP32，文件共 68,764 字节。没有批归一化、残差块、新特征、新标签或额外数据；与 v1.3 的训练数据缓存、验证/测试集、种子、30 轮、batch 1024、AdamW 学习率 0.001、weight decay 0.001、验证 MSE 选权重协议相同。

训练命令为 `npm run training:run:cnn`，默认读取冻结的 `.build/training/v1/positions.npz` 并检查其 SHA-256，输出到 `.build/training/v1.4/`。训练在 CPU、2 线程运行。原生库需先构建到 `.build/native-v1.4`。正式对战使用同样的 v1.3 开局随机策略，双方从初始棋盘自行走棋、各 50ms，交换颜色，共 100 局；见 [v1.4 实验结果](../models/v1.4-comparison.md)。

实验格式为 `OTCNN1\0\0`、四个 little-endian uint32（边长 8、输入平面 2、卷积通道 8、全连接宽度 32），随后依次写入 Conv1 权重/偏置、Conv2 权重/偏置、全连接权重/偏置、输出权重/偏置，均为 little-endian FP32。卷积权重顺序为 `[输出通道][输入通道][核行][核列]`，激活为 NCHW；文件读取拒绝不支持的尺寸、NaN/Inf、过大权重、截断和尾随数据。

CNN 每次估分进行完整前向推理，未声称支持 NNUE 式增量计算。搜索在 CNN 分支直接估分，经典 NNUE 分支继续维护原增量累加器。缓存快照深拷贝 CNN 权重并放在堆上，避免修改模型后串用旧分数及浏览器栈占用过大。测试包含独立标量卷积对照、Python/原生导出一致性、固定深度穷举、换色、模型切换和损坏模型。

结果为 4 胜 96 负，未替换 v1.3；补充同深度 4 层对照为 6 胜 14 负。该实验是两层 8 通道、只有价值输出的小 CNN，不是 AlphaGo 的策略/价值网络与搜索组合。与 2016 年 AlphaGo、2017 年 AlphaGo Zero 原论文的结构对照及后续候选设计见上述实验报告。

## v1.5：32 通道、2 个残差块

输入继续为行棋方/对方两张 8×8 占位图。网络为 `Conv3×3(2→32)/BN/ReLU → 2×[Conv3×3(32)/BN/ReLU/Conv3×3(32)/BN/残差相加/ReLU] → Conv1×1(32→1)/BN/ReLU → Flatten64 → Linear64/ReLU → Linear1/tanh`。训练参数 42,019 个；导出时将 BN 的运行均值、方差、缩放和偏置折入相邻卷积，推理参数 41,858 个。2 块和 4 块的加载、导出、跳连和推理对照都有测试，本轮正式训练选择 2 块。

训练命令为 `npm run training:run:resnet`，复用冻结的百万局面缓存、验证/测试集和原来的子差回归目标，30 轮、batch 1024、AdamW、相同学习率/weight decay 与种子。可用时通过本机 Metal GPU 训练，原生对战仍在 CPU 上进行。默认输出为 `.build/training/v1.5/resnet32x2-v1.5.otres`，原生库为 `.build/native-v1.5`。`--device cpu` 可强制 CPU；`--blocks 4` 时需自行指定独立输出路径。

格式 `OTRES1\0\0`，随后五个 little-endian uint32（8、2、32、残差块数 2 或 4、全连接宽度 64），再依次保存折叠后的起始卷积、每个残差块的两层卷积、1×1 价值投影和两层全连接的权重/偏置，均为 FP32、卷积权重为 OIHW。2 块文件为 167,460 字节。格式不存储训练时 BN 状态；`.pt` 单独保留完整训练状态字典。

搜索和 v1.3 使用同一套 alpha-beta/PVS、置换表和开局随机策略，仍无策略头或 MCTS。残差模型每 16 个节点检查一次时间，以适配较贵的估分；经典 NNUE/CNN 继续使用原检查间隔。正式测试为初始棋盘自行走棋、各 50ms、交换颜色 100 局，结果 2 胜 98 负。详见 [v1.5 实验报告](../models/v1.5-comparison.md)。

## v1.6：模型 × 搜索四组合

在同一原生库上对比 v1.3 NNUE、v1.5 ResNet32×2，分别搭配原 alpha-beta/PVS 或价值网络 MCTS，不重新训练权重。MCTS 使用均匀先验 PUCT，探索系数 1.5，未访问的边优先采样，叶节点估分为当前行棋方的预期子差/64，终局为实际子差/64；不是胜率，也不是随机走到底的 rollout。两种模型没有策略头，不能把此实验当作完整 AlphaGo Zero 复现。

树按实际走法保留达到的子树；模型内容改变会清空树。节点按需创建，最多 2,097,152 个已访问节点，报告记录是否触及内存上限。终局值在路径上逐层反号，包括强制停着；只有所有子分支都求解时才将父节点标为精确求解。正常落子选访问次数最多的分支，平手再比较均值和种子；精确求解时按精确值选步。

双方前 12 手都沿用原来的近分加权开局策略；之后 AB 选最高分，MCTS 选最多访问。全部对战从标准初始棋盘自行走棋，没有开局库。每个配对 4 局，同种子交换黑白，共 24 局，每步 1 秒。使用 `npm run training:baseline:v1.6` 和 `npm run training:watch:v1.6` 运行与监控，结果见 [四组合报告](../models/v1.6-comparison.md)。

另有同一 v1.3 NNUE 的 AB 对 MCTS 补充试验：每步 100ms、100 局、各执黑白 50 局，仍使用同一库、同一权重和同一开局随机策略。用 `npm run training:baseline:nnue-search` 运行，`npm run training:watch:nnue-search` 监控；见 [100ms 结果](../models/v1.6-100ms-comparison.md)。两种预算的结果独立记录，不混算胜率。

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

v1.7 默认使用冻结的 v1.2 行棋方 NNUE 权重、v1.3 开局策略和 MCTS，原生与网页均直接内嵌；CNN/残差网络继续保留为实验模型。文件头有版本、特征数、隐藏宽度和量化尺度校验；截断、尾随字节或不支持的宽度会拒绝加载。`OTNNUE1` 是本项目格式。

## v1.7：默认交付

不重新训练。将已验证的 v1.3 NNUE＋MCTS 接入浏览器及原生默认入口；熟练档 100ms，高级档 1 秒，入门保留弱启发式。权重与 v1.6 对战完全相同，通过内嵌字节和 SHA-256 校验保证一致。`ot_search` 等显式 AB 接口保留，历史报告按原始源码归档校验。发布配置见 `models/v1.7-default.json`。

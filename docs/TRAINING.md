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

## v1.8 策略头候选

本实验保留 v1.7 的量化特征变换和价值输出字节，冻结原参数，只训练新增的 `512→32→65` 策略分支（32 维 clipped ReLU，65 维为棋盘落点和 pass）。共享当前行棋方/对方的 512 维特征；新增 18,561 个参数，总计 52,098 个参数、104,936 字节。没有阶段头，没有修改子差价值目标。

`prepare-policy-data.py` 给原百万训练局面和各 6,000 个验证/测试局面补齐棋谱中的下一步落点；棋盘、行棋方、最终结果和原划分都必须匹配。没有移动局面到其他划分。训练对棋盘、合法走法掩码和落点标签同步做 D4 变换，在合法走法上计算交叉熵。记录落点是棋谱监督，不是最优落点证明，也不是 MCTS 访问分布。原缓存排除了 pass 样本，推理时强制停着直接返回 pass 概率 1，终局返回全零。

格式 `OTNNUE3\0` 先保存完整的 `OTNNUE2` 价值模型（替换 magic），再保存两个 uint32 维度 32、65，以及 row-major 的 32×512 int16 权重、32 个 int32 偏置、65×32 int16 权重和 65 个 int32 偏置。权重缩放 256，新增偏置缩放 65,536。策略输出在合法走法上做稳定 softmax。模型加载仍兼容旧格式。

MCTS 对带策略头的模型使用学到的先验，价值模型继续使用原均匀先验。保留原规则：所有未访问分支先各访问一次；策略版按先验排序，同先验再按种子决胜。随后使用 `Q + 1.5 × P × sqrt(N)/(1+n)`，根节点仍按访问数选择。强制停着、精确求解、树复用及原开局选步策略保持原语义。原 v1.7 模型在 30 个局面、每局面固定 512 次模拟下，与旧库的落点、评分和根访问次数逐项一致。

```sh
.build/venv/bin/python scripts/prepare-policy-data.py
.build/venv/bin/python scripts/train-policy.py
# 先从 archive/v1.7-sources.tar.gz 恢复独立源码并编译参考库：
mkdir -p .build/frozen-v1.7
tar -xzf models/archive/v1.7-sources.tar.gz -C .build/frozen-v1.7
cmake -S .build/frozen-v1.7 -B .build/frozen-v1.7/build -DCMAKE_BUILD_TYPE=Release
cmake --build .build/frozen-v1.7/build -j 2
.build/venv/bin/python scripts/compare-nnue-candidate.py \
  --model .build/training/policy/nnue-policy-v1.8.otnnue \
  --baseline-manifest models/v1.7-default.json \
  --baseline-library .build/frozen-v1.7/build/libothello.dylib \
  --games 100 --milliseconds 100 \
  --output .build/training/policy/new-match.json
```

Linux 将动态库后缀改为 `.so`。标签缓存、训练和对战输出都要求新路径，避免覆盖实验记录；重跑标签生成使用 `--output`，训练使用 `--labels`/`--output`。实验模型仅在显式加载时使用；默认和已批准基线仍是 v1.7。只有用户明确要求“作为默认模型”时，才更新默认并将该版本冻结为后续比较的新基线。

### 1 秒补充试验与实时监控

同一候选对原 v1.7，20 局、每步各 1 秒，独立记录，未重新训练：

```sh
.build/venv/bin/python scripts/compare-nnue-candidate.py \
  --model models/nnue256-policy-v1.8.otnnue --label v1.8-policy-candidate-1s \
  --baseline-manifest models/v1.7-default.json \
  --baseline-library .build/frozen-v1.7/build/libothello.dylib \
  --games 20 --milliseconds 1000 --output .build/training/policy/new-1s-match.json

# 当前这轮 1 秒试验的监控，默认每秒刷新：
sh scripts/watch-policy.sh
sh scripts/watch-policy.sh --once
```

监控显示累计胜平负、得分率、平均子差和搜索量，以及当前局的步数、执子版本、黑白棋子数和行棋方。运行时优先读取 `.build` 中的实时文件，没有运行时文件则读取已发布的完整报告。Ctrl+C 只关闭监控。棋子数是当前局面比分，不是最终胜负预测。

对局每步写入小型实时状态，每局完成后才写完整可重放报告。进程中断后，可在原命令加 `--resume`：它核对模型、原生库、预算、种子、源码与已完成棋谱，保留完成的对局，当前未完成局从标准初始棋盘重开；已完成报告拒绝续跑。不同预算不能用续跑混入同一报告。

## v1.8：用户批准的默认基线

用户明确要求将策略模型作为默认，现由 `models/default-model.json` 选择 `models/nnue256-policy-v1.8.otnnue`，冻结记录为 `models/v1.8-default.json`。提升仅更新交付入口和版本记录，不重新训练、不改写两组候选实验。原生和 Wasm 使用相同内嵌策略／价值权重，100ms／1 秒预算保持原设置；候选比较脚本不指定 `--baseline-manifest` 时自动选择该新基线。

子差价值目标保持不变。后续若改为胜／平／负分类，训练标签可由原完整棋谱结果转换；MCTS 终局值也必须相应改为胜 1、平 0、负 −1，不能与原子差/64 混用。三分类可给出 `P(胜)`、`P(平)`、`P(负)`，搜索用 `P(胜)−P(负)`；`(v+1)/2` 是含平局半分的期望得分，不一定等于胜率。该方案尚未训练和部署，需另建版本并与 v1.8 比较。

原论文核对与下一项单标量胜负价值实验设计见 [AlphaGo 价值头借鉴](ALPHAGO_VALUE.md)。三分类保留为可选扩展，先按原论文的单 tanh 输出做控制变量实验；尚未训练或部署。

## v1.9：单标量胜负候选

随后按用户要求训练上述单 tanh 方案，冻结 v1.8 特征和策略，只训练 `512→1` 的 513 个参数。标签由原百万局面缓存的最终子差取符号得到，仍使用原验证／测试集。模型为 `models/nnue256-outcome-v1.9.otnnue`，格式 `OTNNUE4`；搜索的终局与精确证明同时适配胜负语义，旧子差模型保持原分支。它尚未提升为默认。

```sh
cmake -S . -B .build/native-outcome -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native-outcome -j 2
.build/venv/bin/python scripts/train-outcome.py \
  --parent models/nnue256-policy-v1.8.otnnue --baseline-manifest models/v1.8-default.json \
  --output .build/training/outcome/retrained.otnnue \
  --library .build/native-outcome/libothello.dylib

# 参考库来自冻结的 v1.8 源码；使用独立目录构建。
mkdir -p .build/frozen-v1.8
tar -xzf models/archive/v1.8-default-sources.tar.gz -C .build/frozen-v1.8
cmake -S .build/frozen-v1.8 -B .build/frozen-v1.8/build -DCMAKE_BUILD_TYPE=Release
cmake --build .build/frozen-v1.8/build -j 2
.build/venv/bin/python scripts/compare-nnue-candidate.py \
  --model models/nnue256-outcome-v1.9.otnnue --label v1.9-outcome-candidate \
  --baseline-manifest models/v1.8-default.json \
  --library .build/native-outcome/libothello.dylib \
  --baseline-library .build/frozen-v1.8/build/libothello.dylib \
  --games 20 --milliseconds 1000 --output .build/training/outcome/new-match.json

sh scripts/watch-outcome.sh
sh scripts/watch-outcome.sh --details
```

Linux 改用 `.so`。重跑使用新输出路径。监控默认只显示候选视角的一行胜平负与当前局黑白棋子数；`--details` 展开耗时、平均子差和搜索量。不同价值目标的模拟数不能直接比较棋力；胜负奖励、终局证明停止和同一探索系数下的搜索行为也属于本次实验的变化。

## v1.9：用户批准的默认基线

用户在 1 秒、20 局对照得到 18 胜、2 平、0 负后明确要求换基线。因此默认选择器现指向 `models/v1.9-default.json` 和 `models/nnue256-outcome-v1.9.otnnue`，原生及 Wasm 嵌入同一胜负模型。提升不重训、不新增对战、不改写候选时期记录；后续比较默认使用 v1.9。v1.8 及其源码、模型与实测结果继续冻结。

## v1.10：更深共享主干候选

在冻结 v1.9 第一层特征与策略头权重后，给两个颜色视角共用的 256 维激活增加 `256→32→256` 残差块；下投影后 clipped ReLU，上投影与原激活相加后再 clipped ReLU。上投影从零开始，使 epoch 0 精确保留原模型行为。新增 16,672 个主干参数，与 513 个价值头参数一起训练；总计 68,770 参数、138,864 字节。策略头权重字节不变，但其输入经过新主干，因此策略输出会变化。

训练使用相同百万局面、冻结验证／测试划分和 D4 对称，损失为胜负 MSE 加合法落点交叉熵，权重各 1。棋谱落点标签仍来自记录走法，不是 MCTS 分布。本次因此同时涉及更深特征与价值／策略联合监督，不是单纯增加空层或完全相同训练目标的消融。30 轮中按验证总损失选第 23 轮。新格式 `OTNNUE5` 保留 v1.9 载荷布局，再附加 uint32 维度 256、32、下投影 int16 权重及 int32 偏置、上投影 int16 权重及 int32 偏置；权重／偏置缩放分别为 256／65,536，中间激活不再取整。胜负回传、PUCT、开局规则与时限沿用 v1.9。

```sh
cmake -S . -B .build/native-deep -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native-deep -j 2
.build/venv/bin/python scripts/train-deep.py \
  --parent models/nnue256-outcome-v1.9.otnnue --baseline-manifest models/v1.9-default.json \
  --output .build/training/deep/retrained.otnnue --library .build/native-deep/libothello.dylib

mkdir -p .build/frozen-v1.9
tar -xzf models/archive/v1.9-default-sources.tar.gz -C .build/frozen-v1.9
cmake -S .build/frozen-v1.9 -B .build/frozen-v1.9/build -DCMAKE_BUILD_TYPE=Release
cmake --build .build/frozen-v1.9/build -j 2
.build/venv/bin/python scripts/compare-nnue-candidate.py \
  --model models/nnue256-deep-v1.10.otnnue --label v1.10-deep-candidate \
  --baseline-manifest models/v1.9-default.json \
  --library .build/native-deep/libothello.dylib \
  --baseline-library .build/frozen-v1.9/build/libothello.dylib \
  --games 20 --milliseconds 1000 --output .build/training/deep/new-match.json

sh scripts/watch-match.sh
```

Linux 改用 `.so`。使用新输出路径重跑；原始实测代码在 `models/archive/v1.10-deep-sources.*`。统一监控入口读取 `models/current-experiment.json`，每轮对比都更新该指向并提供同一命令；显示单行候选战绩、ASCII `*` 黑棋／`o` 白棋／`.` 空位和终局棋盘。默认依然是 v1.9，只有用户批准才提升。

本轮后按用户要求停止继续对战：实际完成 9 局，更深候选 0 胜 4 平 5 负。未完成的第 10 局不计入，完整记录见 `models/v1.10-deep-comparison.md`，默认保持 v1.9。

## 独立扩展数据：Egaroucid 5,200 万局

新增官方 v0002 棋谱的本地下载、SHA-256 校验、流式合法性检查和分阶段抽样流程。
随机前缀不生成训练标签；新缓存保留经验胜负和走法分布，冻结验证／测试数组原样保留。
另提供固定 500 万局的全部非随机局面展开、精确去重和稀疏标签聚合流程，不做局面抽样或阶段限额。
数据及派生结果禁止再分发，仅保存在 `.build/training/egaroucid-v0002/`，不加入数据快照。
命令、格式、软标签使用方式和限制见 [预处理说明](EGAROUCID_DATA.md)。此次仅准备数据，默认模型仍为 v1.9。

## 500 万局与原训练集混合训练

`scripts/train-mixed-5m.py` 在 v1.9 结构上冻结共享特征层，联合拟合价值头和策略头。
完整一轮遍历 104,627,926 个 `split == 0 && !frozen` 的新增聚合局面，
并将原有 1,000,000 个训练局面分散穿插到 64 个分片后，各使用一次。
每个聚合局面等权；新增价值使用 `(win-loss)/count`，策略使用经验走法概率，
原数据使用胜负硬标签与记录走法。各来源之间不额外去重或重加权。
棋盘、合法掩码和策略标签同步作随机 D4 变换。

```sh
.build/venv/bin/python scripts/train-mixed-5m.py \
  --output .build/training/mixed-5m-new-run \
  --library .build/native-engineering/libothello.dylib
sh scripts/watch-match.sh
.build/venv/bin/python -m unittest discover -s tests -p 'mixed_5m_training_test.py' -v
```

默认完整训练 1 轮，batch 8192，AdamW 学习率 0.0003、weight decay 0.001。
每 8 个分片评估原有冻结验证集，以胜负 MSE＋记录走法交叉熵选择已训练检查点；
初始 v1.9 单独记录，不纳入候选选择。原有 6,000 个测试局面只在训练及选择结束后评估。
训练脚本核对每个分片、原缓存和冻结评估集的 SHA-256，以及原评估数组的一致性。
最终分别报告浮点候选和原生量化推理结果，并在验证和测试全集检查 Python／C++ 一致性。
输出目录保留逐分片进度、最新／最佳检查点、模型、报告和测试逐条 CSV／NPZ 预测。
完整一轮结束时的权重另行导出与评估，避免把较早的最佳检查点误认为已见过全部数据：

```sh
.build/venv/bin/python scripts/evaluate-mixed-5m-final.py --run .build/training/mixed-5m-new-run
```

它保存 `nnue256-mixed-5m-full-epoch.otnnue`、`full-epoch-report.json` 和
`full-epoch-test-predictions.csv`／`.npz`；该最终检查点不按测试指标重新挑选。
这是固定测试集预测实验；不自动启动搜索对局，也不提升默认模型。

## v1.12：两层 CNN 原始数据离线模型组

`scripts/train-cnn-policy-suite.py` 使用原始 v1 百万局面缓存及原策略标签，
核对 SHA-256，并逐项核对原有 6,000 验证／6,000 测试局面。
不使用 Egaroucid 新增训练分片。输入为行棋方／对手两张棋盘图，
两层 3×3 卷积的通道数分别训练 8、16、32 三版，保持 8×8 空间大小；
展平后共享 Linear64/ReLU，再接 Linear1/tanh 胜负价值头和 Linear65 策略头。
所有参数从头训练，没有冻结、BN、池化或残差块。

训练目标为原棋谱胜负 MSE＋合法走法掩码下的记录走法交叉熵，两项等权。
数据排列／D4 增强使用独立且相同的随机数流；所有模型同步变换棋盘、合法掩码和走法坐标。
默认 batch 1024、AdamW 学习率 0.001、weight decay 0.001，最多 30 轮，至少 10 轮；
连续 5 轮验证总损失无改善时早停，按验证总损失选择各模型检查点和整体候选。
全部模型训练结束后才评估测试集，不按测试集指标调整训练。

```sh
.build/venv/bin/python scripts/train-cnn-policy-suite.py --device mps --publish
sh scripts/watch-match.sh
.build/venv/bin/python -m unittest discover -s tests -p 'cnn_policy_training_test.py' -v
```

`--device cpu` 可在无 MPS 的机器运行。重跑请指定新的 `--output` 并省略 `--publish`，
避免覆盖原实测版本。输出包括 PyTorch 权重、CPU TorchScript、逐轮日志、验证／测试指标和
包含完整走法概率的逐条测试预测 NPZ；版本化产物发布到 `models/cnn2-c{8,16,32}-v1.12.*`。
导出模型重新加载后在两个评估全集检查一致性。当前为离线模型，尚未接入 C++／Wasm 推理或搜索；
不启动对战，默认继续为 v1.9。历史 v1.4 是子差价值单头 CNN，不能与本轮胜负／策略联合目标混算。

## v1.13：固定 32 通道调参

在 v1.12 宽度对照后，`scripts/tune-cnn-policy.py` 固定原始训练／验证／测试划分、
32 通道两层 CNN、batch 1024、weight decay 0.001 和随机序列，从头训练四组：

- `low-lr`：学习率 0.0003 恒定，价值／策略损失权重 1:1。
- `cosine`：学习率从 0.001 余弦衰减至 0.00001，损失权重 1:1。
- `cosine-v2`：相同衰减，损失权重 2:1。
- `cosine-v4`：相同衰减，损失权重 4:1。

四组均完整训练 30 轮；每组同时保存验证 MSE+CE 最低和价值 MSE 最低的检查点，
主候选按相同的未加权验证 MSE+CE 比较，价值标准候选另行标注。
全部训练结束后才统一评估原测试集，不按测试结果选择权重或调整配置。
本轮未加入开局库标签或新增棋谱，也未接入搜索。

```sh
.build/venv/bin/python scripts/tune-cnn-policy.py --device mps
sh scripts/watch-match.sh
```

默认等待 v1.12 汇总报告存在后串行使用 GPU。输出为独立的
`.build/training/cnn2-v1.13-tuning-run1/`；完成后自动发布唯一 v1.13 候选权重、
TorchScript、评估报告和源码指纹，默认 v1.9 不变。

## v1.14：两套 CNN 的原生对战接线

从 v1.13 验证集选择 `cosine-joint` 和 `cosine-v4-value`，分别代表联合损失最低和价值 MSE 最低。
原 PyTorch FP32 权重直接导出为 `OTCNN2`：magic 8 字节，随后五个 little-endian uint32
（8、2、通道数、64、65），再保存两层卷积、共享全连接、价值头、策略头的权重／偏置。
与历史 `OTCNN1` 子差单头格式隔离，`OTCNN2` 固定采用胜负价值语义。
原生加载器只接受 8/16/32 通道，拒绝尺寸异常、非有限／过大权重、截断和尾随字节。
双头共享一次卷积主干，策略仅在合法落点上归一化；强制停着为索引 64，终局策略为空。
MCTS 使用学习先验与一致的胜／和／负终局备份；模型快照独立持有权重，模型变化使搜索缓存失效。

原生及 ASan／UBSan 测试通过；每套模型在验证／测试各 6,000 局面核对 Python 与原生输出。
新库加载 v1.9，与原工程优化库在 24 个冷搜索局面和连续 16 步复用搜索中，固定 512 次模拟结果一致。
两组对战按相同引擎、每步双方各 2 秒、各 20 局、交替颜色和配对种子顺序运行，不使用开局库。
统一监控仍为 `sh scripts/watch-match.sh`，结束后保留终盘。

```sh
cmake -S . -B .build/native-cnnpolicy -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native-cnnpolicy -j 2
ctest --test-dir .build/native-cnnpolicy --output-on-failure
.build/venv/bin/python scripts/verify-cnn-policy-native.py
```

本次运行器为 `scripts/run-cnn-policy-matches.py`，唯一实验前缀 `v1.14`，拒绝覆盖已有结果。
运行数据位于 `.build/training/cnn-policy-matches-v1.14/`，每完成一组自动重放审计、发布对战报告。
源码和模型校验记录在 `models/v1.14-cnn-policy-integration.json` 及对应 source archive。
此次不改变默认模型、内嵌权重或网页发布产物。

## v1.15–v1.18：搜索采样与双方共同 batch 诊断

实验实现分别隔离在 `experiments/`，不替换 v1.14 正在运行的库和默认搜索。
逐步扩展原型只作用于 CNN 内部节点，根节点和精确证明保留全部合法分支。
GPU 原型由 C++ 收集不同待评估树节点，临时增加访问次数防止重复选叶，
Python/MPS 批量前向后统一回传；每次返回前排空队列，失败时撤销预留并重置搜索树。
该小网络的 Python/MPS 调度／传输成本超过收益，实测保存在 v1.16 诊断报告。

v1.17 使用纯 C++ 与单线程 Apple Accelerate SGEMM，避免 Python/GPU 往返。
v1.18 进一步让 CNN 与 v1.9 NNUE 共用该后端、批次调度、线程数和采样设置。
NNUE 特征矩阵保留整数精确性，策略隐藏层检查精确运算界限，超界回退 int64；
价值头用 int64，策略输出用 double。v1.9 的 batch 1 固定模拟搜索与原算法一致。
原生接口 `ot_cnn_set_native_batch_evaluator(size)` 在 v1.18 中同时支持两种模型；
`size` 为 1..128，必须在相同搜索线程配置；默认不启用，原 C ABI 仍可使用。
该历史命名接口是实验接口，不是已发布的通用 API。

```sh
cmake -S experiments/common-native-batch-v1.18 -B .build/native-common-batch-v1.18 -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native-common-batch-v1.18 -j 2
VECLIB_MAXIMUM_THREADS=1 ctest --test-dir .build/native-common-batch-v1.18 --output-on-failure
```

`benchmark-common-native-batch.py` 对双方使用同一验证局面，比较固定 8,192 模拟及 2 秒预算；
保存统一 batch 32 和每方吞吐最佳的设置。当前短诊断中两边最佳均为 32。
原生与 ASan／UBSan 的规则和 batch 合同测试通过，包括各 CNN 宽度、NNUE 精确输出、
模拟预算边界、回调失败清理及精确残局证明。基准源码与报告保持版本化，重跑应使用独立输出版本。
记录和限制见 `models/v1.18-common-batch-comparison.md`；这些吞吐诊断不是正式胜率或 Elo 结果。

## v1.25：在 v1.9 基础上的 Pattern 胜负价值实验

2026-10-29 将合并进来的补充棋形实验迁移到当前批准的 v1.9 基础上。
历史 `pattern32-v1.8.otpat` 和原 60 局报告保持原样，它们仍是子差价值实验，
其旧对手为 v1.3，不能作为对当前 v1.9 的胜率结论。

新候选为 `pattern32-outcome-v1.25.otpat`，使用独立的 `OTPAT2` 格式。
沿用 38 个三值棋形、32 维嵌入及多头结构，从随机参数重训全部网络，
不继承 v1.9 的 NNUE 权重。沿用冻结百万局面和各 6,000 个验证／测试局面，
将记录终局子差取符号作为行棋方胜 +1、平 0、负 −1 的标量标签。
损失为胜负 MSE + 0.1 WDL 交叉熵 + 0.1 合法落点策略交叉熵，
15 轮，以最低验证胜负 MSE 选权重，不按测试集或对战结果选择。
WDL 分类头仍仅作为辅助；搜索使用重新训练的标量头。

C++ 按文件格式区分子差与胜负目标。`OTPAT1` 保留原子差终局回传，
`OTPAT2` 使用 v1.9 的胜负终局回传与证明停止。
Pattern 策略先验与有限 FPU（父均值减 0.15）保留，
因此对战比较的是完整候选系统，不能单独归因于棋形特征。
v1.9 对手使用冻结参考库及其已批准权重；双方不使用开局库，
每步 100ms、20 局配对颜色，保留 v1.3 前 12 次落子的开局多样性策略。
这是独立的 100ms 实验，不与历史 1 秒组混算。

```sh
# 训练环境需 requirements-training.txt；本机可用 .build/venv/bin/python。
cmake -S . -B .build/native -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native -j 2
python3 scripts/train-pattern-outcome.py
# 参考库须由 models/archive/v1.9-default-sources.tar.gz 的冻结源码构建。
# macOS；Linux 将两处 .dylib 改为 .so。
python3 scripts/compare-nnue-candidate.py \
  --model models/pattern32-outcome-v1.25.otpat --label v1.25-pattern-outcome \
  --baseline-manifest models/v1.9-default.json \
  --baseline-library .build/native-v1.9-reference/libothello.dylib \
  --library .build/native/libothello.dylib --games 20 --milliseconds 100 \
  --seed 20261029 --output .build/training/pattern-outcome-v1.25/match.json
sh scripts/watch-match.sh
```

原训练／历史验证入口继续保留；新训练入口为
`npm run training:run:pattern:outcome`。重跑必须指定新的模型和报告路径，
不得覆盖冻结产物。默认选择器和已冻结的 v1.9 浏览器引擎保持不变；
即使候选对战更好，也需用户另行批准才可提升为默认。

本轮已完成：100ms／20 局，候选 **18 胜、0 平、2 负**，得分率 90%，
平均子差 +15.5；1,230 个动作已独立重放。
测试胜负 MSE 从 v1.9 的 0.712699 降至 0.687821，实战下一手命中率从
35.87% 提升至 40.57%。完整结果、限制和源码校验见
[Pattern outcome 实验报告](../models/v1.25-pattern-outcome-comparison.md)。

### Pattern v1.25：每步 2 秒的补充对照

沿用相同模型、候选库及冻结 v1.9 参考库，每步 2,000ms、20 局、
10 组种子交换黑白，未重训、未使用开局库。
候选取得 **12 胜、5 平、3 负，得分率 72.5%，平均子差 +13.4**。
全部 1,215 个动作通过独立逐手重放和选择审计。
候选有 11 次搜索触及既有节点上限，相关统计单独保存在报告中。
此组与原 100ms／20 局结果分开记录，默认继续为 v1.9。
见 [2 秒实验报告](../models/v1.25-pattern-outcome-2s-20-run1-comparison.md)；
复核命令为 `python3 scripts/finalize-pattern-outcome-2s.py --verify`。

## v1.25：已批准的新默认基线

2026-10-29 用户批准将已测量的 Pattern outcome 模型替换为默认，
冻结记录为 `models/v1.25-default.json`，上一个基线 `models/v1.9-default.json` 保留。
本次提升仅更新交付与基线记录，不重新训练，不追加对战；100ms／20 局
18W/0D/2L、2s／20 局 12W/5D/3L 的原始候选报告均保持不变。
网页开局库继续默认启用，出库后使用同一内嵌 Pattern 模型。
未来比较通过 `models/default-model.json` 默认选择 v1.25；它的策略先验、
有限 FPU、胜负终局回传和证明停止都属于新基线。
历史重训复现脚本中的 v1.9 引用仍表示当时的训练／比较起点。

运行 `python3 scripts/verify-baseline.py` 校验现默认和历史基线，
原候选验证命令继续可用：可变交付文件改从提升前的校验快照读取，
历史模型、对战记录及当时的源码哈希仍按原值核对。

## v1.29：简单主干宽度与推理实现对照

已确认 Pattern 原训练具备 4 种旋转 × 镜像／不镜像的八种随机增强，
棋盘、下一手和合法着掩码同步变换。本轮继续使用该增强，H32/H64
的样本顺序及对称序列完全一致，分别训练 15 轮。
H64 的测试胜负 MSE 为 0.689270，较同轮 H32 对照的 0.691565 略低，
但未超过已批准 v1.25 的 0.687821；融合推理耗时约 3.016µs，
慢于 H32 对照的 2.121µs，因而本轮不提升 H64。

同时实现价值／策略联合推理，复用棋形编码与查表结果，只汇总行棋方，
并预计算格点计数与位置桶。同一批准模型的原生联合推理从约 2.546µs
降至 2.127µs（九轮交错测量，包含 Python/C ABI 开销）；64 个验证局面
全输出及固定模拟次数 MCTS 结果完全一致，原生测试与内存检查通过。
新实现和模型存于独立原生候选库，默认 v1.25 及浏览器 Wasm 保持冻结。
详见 [完整报告](../models/v1.29-pattern-width-comparison.md)，
复核命令为 `python3 scripts/finalize-pattern-width.py`。

## Pattern INT8 / 整数 SIMD（v1.32）

本轮只做训练后棋形表量化，未重训；见 `models/v1.32-pattern-int8-comparison.md`。`OTELLO_PATTERN_SIMD=ON` 启用编译期整数内核选择：ARM64 NEON、x86-64 SSE2，其他目标普通回退；默认 OFF，不要求 `-march=native`。模型格式 OTPAT3 保留压缩 INT8 表，加载后仅按需解码行，累加器为 INT32。v1.25 及浏览器 Wasm 保持冻结。Linux CI 新增 GCC/Clang × ON/OFF 任务，Linux 实机性能尚未测量。Docker 网页服务仍运行浏览器 Wasm，不受服务端 NEON/SSE2 影响。

## Pattern 精确热点优化（v1.33）

棋形编码通过行/列/对角线位压缩、镜像及 512 项二进制到三进制小表生成同样的 38 个编号；四行整数 SIMD 共用输入加载并保留 INT64 等价结果。试验整数舍入亦通过一致性，但测得较慢，选中组合不启用它。保持 v1.25 默认、权重及 Wasm 不变。Linux 原生候选可用 `-DOTELLO_PATTERN_SIMD=ON -DOTELLO_PATTERN_FAST_CODES=ON -DOTELLO_PATTERN_BLOCK4=ON -DOTELLO_PATTERN_INTEGER_ROUNDING=OFF` 构建，详见 `models/v1.33-pattern-hotpath-comparison.md`。新增 GCC/Clang 的普通/SIMD 与整数/原舍入 CI 组合；尚无 Linux 实机性能数据。

## Pattern 合法着策略投影稀疏计算（v1.34）

已有合法着 softmax、树先验复用保持不变。原生实验开关 `-DOTELLO_PATTERN_LEGAL_POLICY=ON` 在 v1.33 选中优化基础上，仅为触及合法落点的棋形做策略投影；仍读取全部价值特征，保持值与合法策略精确相等。完整刷新、增量缓存、强制停着和终局均检查；具体计时、波动和平台边界见 `models/v1.34-pattern-legal-comparison.md`。默认与浏览器 Wasm 保持冻结。

## v1.35 提速策略集汇总

启用 `OTELLO_PATTERN_SIMD / FAST_CODES / BLOCK4 / LEGAL_POLICY`，关闭 `INTEGER_ROUNDING`，增量状态缓存为 0。同局面 2s 吞吐直接比较批准 v1.25 原库，INT16 模拟吞吐 2.158×、网络评估 1.868×；按预先定义的 >5% 门槛未选 INT8。20 局 2s 对战为 5W/8D/7L，保持原模型、开局多样性、PUCT 与节点上限，未提升默认。节点触顶明显影响候选对 2s 预算的利用。所有模型、库、源码及脚本哈希、逐步记录和终盘均保留，详见 `models/v1.35-pattern-speed-suite-comparison.md`。监控默认聚焦当前实验，`sh scripts/watch-match.sh --all` 显示历史。

## v1.36 MCTS 节点扩容

新增 `ot_nnue_mcts_play_budget`，显式传入 65..16,777,216 的节点上限，仍按需分配；所有旧 C ABI/默认配置继续使用 2,097,152。实验预检选中 4,194,304，并保留原模型、推理优化、时间与搜索规则。对战记录包含每步 maxNodes、treeNodes、memoryLimited 和每个工作进程的峰值 RSS（macOS 字节、Linux ru_maxrss KiB 转字节）。

2s/20 局结果 7W/7D/6L、52.5%，相对上一轮 45% 有小幅改善，但不是确定的容量因果结论。候选触顶降至 0/603，未证明/未触顶思考平均约 2000ms；峰值内存约 2.44 GiB（预检约 2.60 GiB），部署时应另验目标平台内存。详见 `models/v1.36-pattern-capacity-comparison.md`；未替换默认或网页 Wasm。


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

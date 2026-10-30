# Egaroucid 5,200 万局本地预处理

数据为 Takuto Yamana 发布的 [Egaroucid Free Training Data](https://www.egaroucid.nyanyan.dev/en/technology/train-data/)，
固定到 [training_data_v0002](https://github.com/Nyanyan/Egaroucid/releases/tag/training_data_v0002)。
对局双方为 Egaroucid 7.8.0 lv.11 与 Edax 4.5.5 lv.11；目录 8～59 分别表示开头随机走了多少次落子，各 100 万局。

作者允许自己的 AI 开发，禁止再分发数据。原始 ZIP、二进制棋谱、训练缓存都只放在被 Git 忽略的 `.build/`。
不要把这份数据加入 `data-assets.py pack` 或仓库中的可分发快照。来源、官方 SHA-256 和原文说明随本地结果保存。

## 执行

```sh
python3 scripts/fetch-egaroucid-data.py
mkdir -p .build/training/egaroucid-v0002
c++ -std=c++17 -O3 -Icpp scripts/prepare-egaroucid.cpp \
  -o .build/training/egaroucid-v0002/prepare-egaroucid
.build/venv/bin/python scripts/prepare-egaroucid-data.py
```

后一个命令需要 NumPy，使用已安装训练依赖的 Python 即可。默认 4 个本地处理进程，
可用 `--workers` 调整。每组处理完成后写入带输入、工具和输出哈希的检查点；
重跑会验证并复用完成的分组。输入或处理程序改变时应选择新的 `--output`，不能覆盖旧版本。
下载支持断点续传，最终同时核对固定文件大小和官方 SHA-256。

进度和结果：

```sh
cat .build/training/egaroucid-v0002/clean-v1/progress.json
cat .build/training/egaroucid-v0002/clean-v1/report.json
```

## 清洗与隔离

- 不展开 ZIP 到磁盘，流式读取每个文本成员，读取时检查 ZIP CRC。
- 原生规则逐局重放，推断合法停着；剔除格式错误、非法走子和未结束的棋谱。
- 第 N 组仅在已经落下至少 N 子后生成标签。随机前缀仍需重放，以恢复正确局面，但不充当训练走法。
- 根据实际终局棋盘计算原始黑方子差，胜负用其符号表示。未满盘终局不擅自把空格当成棋子。
- 使用原先前 8 次落子的 D4 归一化、FNV 开局组划分；只有 train 桶进入新训练缓存。
- 额外排除与现有验证／测试集相同的 D4、行棋方相对局面；原有各 6,000 条验证／测试数组与策略标签原样复制。
- 默认以确定性的棋谱指纹和阶段抽取候选，再按每个落子阶段最多 20,000 个唯一局面取样。
  唯一局面不足时保留全部，不重复填充。缓存记录局面出现次数、胜负次数和走法分布。
- 候选中相同完整棋谱指纹在同一局面仅计一次，避免跨随机前缀目录的同棋谱重复计数。
  棋谱指纹为 64 位 FNV；局面键则比较完整的两张 64 位位棋盘。

这是有界抽样：聚合标签仅覆盖抽中的观察，不是对全部 5,200 万局中每个局面的全量统计。
完整二进制棋谱保留重复对局，以便之后按需重采样；报告不会将其声称为全量去重语料。

## 输出格式与训练目标

`groups/NN.games.bin` 每条 64 字节：60 个 uint8 落子坐标（0～63，尾部 255 填充），
随后依次为 uint8 随机前缀长度、uint8 落子总数、int8 原始黑方终局子差、uint8 开局切分（0/1/2 为 train/validation/test）。
停着不占落子坐标，重放时按规则推断。

`cache/positions.npz` 中：

- `train_board`：D4 归一化、行棋方为 +1 的 int8 棋盘；`train_turn` 为 +1。
- `train_target`：抽中同局面的平均原始终局子差 /64。
- `train_outcome`：抽中同局面的平均胜负（胜 +1、和 0、负 -1）。
- `train_outcome_counts`：负／和／胜次数；`train_count` 为参与聚合的次数；`train_ply` 为已落子数。
- `validation_*`、`test_*` 的 board/turn/target 数组保持冻结数据不变；另提供 outcome=sign(target)。

`cache/labels.npz` 中：`train_policy` 为经验走法概率，`train_legal` 为合法走法掩码，
`train_move` 为频率最高的走法（供硬标签对照）；验证与测试的 move/legal 原样复制。
策略坐标随棋盘一起变换；若棋盘存在对称自同构，将等价走法归到最小坐标。

**后续 outcome 训练应直接读 `train_outcome`，策略训练应读 `train_policy`。**
不能把 `sign(train_target)` 当成平均 outcome，也不能直接使用当前硬标签训练脚本声称已经训练了软标签。
例如一次赢 2 子和一次输 30 子，平均 outcome 为 0，平均子差的符号却为负。
这里的经验结果与走法频率也不是精确最优值或搜索访问概率。

此次只准备独立数据，不训练、不改变 `models/default-model.json` 或 v1.9 冻结基线。

## 检查

```sh
.build/venv/bin/python -m unittest discover -s tests -p 'egaroucid_data_test.py' -v
```

测试使用独立 Python 规则重放，覆盖隐式停着、随机阶段边界、标签坐标变换、非法／未结束棋谱、
冻结局面排除、重复棋谱观察去重以及软标签聚合。

## 2026-10-27 实际处理结果

- 两个 ZIP 均匹配官方 SHA-256；完整重放 52,000,000 局，全部通过，无剔除。
- 1,768,568 局未填满棋盘但已合法结束；停着按规则推断。
- 抽中 2,495,810 条局面观察；另排除 42 条冻结评估局面的候选观察。
- 抽样内去掉 17,280 条同局面、同棋谱指纹的重复观察；得到 2,438,147 个不同局面。
- 最终按第 8～59 次落子后各 20,000 个不同局面，生成 1,040,000 条训练样本。
- 保留的样本里，4,050 个局面有不同胜负观察，2,351 个局面有不同走法观察，均用分布保留。
- 生成缓存的检查显示随机前缀违规为 0，与冻结评估局面的对称重叠为 0。
- 原始 ZIP 约 2.644 GB，完整二进制棋谱 3.328 GB；训练位置和标签两个压缩缓存合计约 27.68 MB。
  加上候选样本分片、元数据和日志，本次本地总占用约 6.10 GB（十进制）。

本地完整报告在 `.build/training/egaroucid-v0002/clean-v1/report.json`，
分组来源和输出哈希在其 `groups/NN.json`；原始 ZIP 与完整二进制棋谱都保留，
训练缓存去重不影响将来重新抽样或调整标签。

## 固定 500 万局的全部局面处理

前述 104 万局面缓存属于抽样产物，不是全部 5,200 万局的全量局面统计。
`process-egaroucid-games.py` 提供另一条流程：先固定选出 500 万局，再完整展开这些棋谱的所有非随机阶段，
不做局面抽样、不跳步，也不按阶段限制数量。

```sh
c++ -std=c++17 -O3 -Icpp scripts/expand-egaroucid.cpp \
  -o .build/training/egaroucid-v0002/expand-egaroucid
.build/venv/bin/python scripts/process-egaroucid-games.py
cat .build/training/egaroucid-v0002/full-5m-v1/progress.json
```

从 52 个随机前缀组各选 96,153 或 96,154 局，总计精确 5,000,000 局。
组内用 `default_rng(seed + group)` 不放回抽取，种子 20261027；实际原始行号保存在
`selection/source-indices.npz`。每个输入分组都验证原有棋谱 SHA-256。

对这 500 万局按完整 60 字节填充走法序列精确去重；没有使用指纹代替相等比较，也不做棋谱 D4 去重。
同一棋谱若在不同随机前缀目录出现，取最短随机前缀，保留这些副本可提供的全部合格阶段。
重放所有保留棋谱，从随机前缀结束后到最后一次落子前逐手提取局面。
强制停着由规则推断，不作为需要学习的策略动作。

中间局面通过 D4 和行棋方视角归一化，按完整两张位棋盘分桶，再精确聚合。
训练、验证、测试开局桶分别累计胜负和走法次数，不混合它们的标签。
一个局面若出现在多个桶，会存在多条“局面＋切分”记录；报告同时给出跨桶去重的实际不同局面数，
不能把“局面＋切分”记录总数当作不同棋盘总数。

所有原始开局切分的局面都会处理并保留。用于训练时仅选择 `split == 0` 且 `frozen == false` 的记录。
另外两个新数据开局桶用于来源分析，不能替换原有冻结评估集。原有验证／测试数组和策略标签原样保存在
`evaluation/frozen.npz`。

输出位于 `.build/training/egaroucid-v0002/full-5m-v1/`：

- `selection/`：500 万局的来源行号、精确去重棋谱和选择报告。
- `observations/`：全部非随机局面观察，32 字节定长；按原始分组与局面哈希分桶。
- `positions/000.npz` 等：全部唯一“局面＋切分”记录，采用稀疏走法计数，避免展开 65 维稠密策略数组。
- `report.json`：输入、工具、分组校验值，抽取／去重／聚合计数和训练过滤规则。
- `progress.json`：当前阶段与已完成数量。各阶段保存检查点，可用相同命令续跑。

每个聚合分片包含 `own`、`opponent`、`legal` 三张 uint64 位棋盘，`count` 出现次数，
`outcome_counts` 的负／和／胜次数，`disc_sum` 行棋方原始终局子差总和，`ply` 已落子数，
`split` 和 `frozen` 标记。策略用 CSR 保存：`policy_offsets` 长 N+1，
第 i 个局面的走法与次数为 `[policy_offsets[i]:policy_offsets[i+1]]` 对应的
`policy_move` 和 `policy_count`。

软价值标签为 `(win-loss)/count`，子差目标为 `disc_sum/(64*count)`，
软策略为各走法的 `policy_count/count`；三者不能混用。
使用时逐分片读取并一次加载所需数组，按需生成棋盘／策略张量；不必把全数据同时展开到内存。

扩大处理局数时，传入新的 `--games` 和独立 `--output` 即可，不需要重新下载：

```sh
.build/venv/bin/python scripts/process-egaroucid-games.py \
  --games 10000000 --output .build/training/egaroucid-v0002/full-10m-v1
```

本流程仍只准备数据，不自动训练或更换默认模型。检查命令：

```sh
.build/venv/bin/python -m unittest discover -s tests -p 'egaroucid*test.py' -v
```

全量分片结果还可用以下命令校验：

```sh
.build/venv/bin/python scripts/verify-egaroucid-games.py
```

它逐分片检查 SHA-256、全部出现次数／策略计数守恒、全部棋盘占用与落子阶段、
全部策略走法与合法掩码、冻结局面标记和评估数组，并另用原生规则抽查 1,024 个局面。

### 500 万局实际结果（2026-10-27）

固定输入 5,000,000 局；精确去除 1,031 个重复棋谱副本，保留 4,998,969 条不同完整棋谱。
去重前有 132,057,202 条合格局面观察，棋谱去重后全部展开得到 132,005,755 条观察。
聚合没有局面抽样、跳步或阶段数量上限。

跨开局切分去重的不同 D4／行棋方相对局面为 **130,406,923** 个。
因相同局面可出现在多个开局桶，分别保存标签后共有 130,665,612 条“局面＋切分”记录。
其中 `split == 0 && !frozen` 的训练记录为 **104,627,926** 条，累计 105,797,871 次观察。
另外的开局桶和冻结局面标记也完整保留，不将其标签混入训练计数。

64 个聚合 NPZ 分片合计 2,636,799,850 字节（约 2.64 GB）；
本批次包括选择棋谱、全部中间观察和元数据共约 **7.19 GB**，不含此前已保留的官方 ZIP 与 5,200 万局完整二进制语料。
预处理实测 319 秒。全分片校验通过，另有 1,024 个原生合法走法抽查通过；原有评估数组保持相同。
检查结果在 `full-5m-v1/verification.json`。此次未训练，默认模型保持 v1.9。

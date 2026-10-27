# 原生离线接口

## 构建与数据格式

```sh
cmake -S . -B .build/native -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native -j
```

产物为 `.build/native/otello` 和 `.build/native/libothello.dylib`（macOS）或 `libothello.so`（Linux）。当前验证平台为 macOS；Linux 可用相同 CMake 流程。网页只是额外调用方，不是原生接口的依赖。

- 格子按从上到下、从左到右排列，A1=0，H8=63；坐标采用 A1–H8。
- 棋盘字符串：64 字符，`B` 黑、`W` 白、`.` 空。也接受小写、`X/O/-`。
- C 数组：`int cells[64]`，1 黑、-1 白、0 空；行棋方 `turn` 同为 1/-1。
- 初始局面：D4、E5 白，E4、D5 黑，黑先行。
- 原生接口显式处理停着，落子后返回对手回合，即使对手需要停着；通过 `mustPass` 判定后再执行 `pass`。网页绑定自动处理它。
- 任意棋盘均可用于局面研究；接口验证编码和当前行动，不证明局面历史可达性。

## 命令行 / 子进程

```sh
.build/native/otello legal
.build/native/otello play --move D3
.build/native/otello analyze --time 1500 --depth 64 --seed 7
.build/native/otello analyze --board ...........................WB......BW........................... --turn b --time 500
.build/native/otello selfplay --time 20 --depth 5
```

支持 `--time 1..60000`（毫秒）、`--depth 1..64`、`--level 0|1|2`、`--seed 0..2147483647`。默认 level 2 使用内嵌 NNUE＋MCTS，时间上限 800ms；显式 `--depth` 会选择固定深度 AB，或用 `--search ab` 选择 AB。level 0/1 未指定模型时保留弱启发式 AB。显式 `--search mcts --depth ...` 会拒绝，以免混淆两种深度。

输出为单行 JSON。非法参数/落子返回 `{"error":"..."}` 和退出码 1；成功退出码 0。程序信息不混入标准输出。

`legal` 与 `play` 返回棋盘、行棋方、子数、64 位哈希的十六进制字符串、`over`、`mustPass`、合法坐标列表；`play` 额外返回翻子数。`analyze` 额外返回最佳坐标、行棋方视角分数、已完成深度、节点数、置换表命中数、耗时、是否精确终局和根候选。

终局分数为胜 `100000 + 子差×100`、负 `-100000 + 子差×100`、和 0；中盘分数为启发式单位，不是胜率或预计子差。无路可走的非终局返回 `move:"pass"`，并继续搜索对手回复；终局同样返回 `pass`，但此时 `over:true`，不能再执行落子。

## 可选 NNUE 模型

`analyze`、`evaluate` 和 `selfplay` 支持 `--network models/nnue256-v1.otnnue`。模型加载失败会返回错误；未指定时使用内嵌的 v1.3 NNUE；`--network heuristic` 才使用原有启发式。`evaluate` 返回静态局面分数，NNUE 的范围为 -1000..1000；`analyze` 的实际终局分数仍使用上述胜负/子差规则。

原生库在 `api.h` 提供 `ot_nnue_load`、`ot_nnue_evaluate`、`ot_nnue_search`、`ot_nnue_destroy` 和 `ot_nnue_error`。模型可一次加载、重复离线调用，搜索输出布局与 `ot_search` 相同；示例见 [nnue_ffi.py](../examples/nnue_ffi.py)。评分、搜索成功返回 1，失败返回 -1；加载失败返回空指针。保持模型存活直至并发调用结束。

v1.1 自研搜索在每个调用线程内复用一张约 8 MiB 的置换表，四项一桶，按深度、准确分数和代数选择替换项。同一线程切换启发式/NNUE 或修改模型参数时自动清表；不同线程不共享表。只有搜索深度相同的缓存分数可以直接截断，较深记录只用于走法排序，因此固定深度的候选评分含义不变。C++ 的 `clearSearchCache()`、C ABI 的 `ot_search_clear_cache()` 可显式清表（例如开始新的分析会话或测量冷缓存）。浏览器 Wasm 也导出 `_ot_search_clear_cache()`。输出缓冲区布局不变。

v1.2 的加载器兼容原模型及考虑行棋方的 `OTNNUE2` 模型，C ABI 不变。CLI 可用 `--network models/nnue256-v1.2.otnnue`，返回的 evaluator 为 `othello-nnue-v2`（表示模型格式）。置换表的模型隔离同时比较架构、双方输出权重及输出偏置，防止同一地址更换模型后串用旧分数。

v1.3 增加 NNUE 开局选步策略，模型权重沿用 v1.2。前 12 次落子按权重抽选近分候选，初始温度 32、最大分差 64，二者按剩余开局落子数线性下降；第 13 次落子起恢复最高分走法。评分单位为引擎估分，不是确定子差或胜率。已精确求解或候选包含终局分数时不随机；候选只有一个时也不抽样。

`selfplay --network ...` 默认启用该策略，未提供 `--seed` 时生成并输出一个新种子。`analyze` 默认输出所选搜索的确定走法（AB 最高分、MCTS 最多访问），可显式增加 `--variety on --opening-plies 12 --opening-temperature 32 --opening-max-loss 64 --seed 123` 检查抽样结果；`--variety off`、`--opening-plies 0` 或温度 0 可禁用。分析 JSON 的 `selection` 包含近分候选数、当前温度、分差限制、最佳分数、所选分差和抽样标记；原候选评分列表保持完整。

原 C ABI `ot_nnue_search` 继续提供最高分分析。对弈使用新增的 `ot_nnue_play`，额外传入开局落子数、温度、最大分差，以及 6 个 int 的 `policy_output` 缓冲区；其中依次是候选数、温度×1000、当前分差限制、最佳分数、所选分差、是否抽样。原 136 项输出布局不变，其 move/score 对应所选走法。每局由调用方提供新种子；Python 示例 [nnue_ffi.py](../examples/nnue_ffi.py)演示了新接口。固定候选评分和相同种子可复核抽样；限时搜索的候选评分可能随负载变化，重放完整对局请保存棋谱。

v1.4 实验版额外支持 `OTCNN1` 小卷积模型：`--network models/cnn8x8-v1.4.otcnn`。原有 `ot_nnue_load/evaluate/search/play` ABI 名称保留，加载器会识别模型族；CLI evaluator 为 `othello-cnn8x8-v1`。CNN 使用完整 FP32 前向推理，NNUE 仍使用量化增量评估。现用 NNUE 权重文件不覆盖，是否推荐 CNN 以 [等时限实验](../models/v1.4-comparison.md)为准。

训练数据、模型结构和实测对战见 [训练说明](TRAINING.md)。Egaroucid 仍使用其自身的评估数据，不能叠加 `--network`。

可选 Egaroucid 构建还提供 `ot_egaroucid_book`，可与 `ot_nnue_search` 组合为“模型搜索＋开局库”：返回 1 时直接采用库内走法，返回 0 时调用同一 NNUE 搜索，返回 -1 时读取适配错误。查库本身不执行 Egaroucid 搜索；输出布局见 `egaroucid_api.h`。`Egaroucid::probeBook` 提供对应的 C++ 接口。

## 常驻行协议

运行 `.build/native/otello protocol`。输入是空格分隔的 CLI 命令，输出是 JSON；不是 JSON 请求协议。

```text
legal
play --move D3
analyze --board ...........................WB......BW........................... --turn b --time 300 --depth 8
quit
```

每行请求独立，不保存上一行的局面。调用方在每次请求传入当前 `--board` 和 `--turn`；若省略则使用初始局面。单个请求出错不会终止协议进程。`selfplay` 是生成完整测试棋谱的工具，耗时按“每手预算 × 手数”计算。

## C++ / C ABI

C++ 可包含 `cpp/othello.hpp`，链接 `othello` 库：

```cpp
#include "othello.hpp"
auto position = othello::parse("...........................WB......BW...........................", 1);
auto result = othello::search(position, {1500, 64, 2, 1});
auto next = othello::play(position, result.move);
```

`Position` 含增量哈希；通过 `parse`、默认构造和 `play` 创建局面。直接修改位棋盘/行棋方后，需要调用 `hashBoard` 重算 `hash`。

其他语言使用 `cpp/api.h`：

```c
int cells[64] = {0}, moves[64], output[136];
cells[27] = cells[36] = -1;
cells[28] = cells[35] = 1;
int n = ot_legal(cells, 1, moves);
int ok = ot_search(cells, 1, 1500, 64, 2, 1, output);
int best_move = output[0];
```

各函数的返回值、缓冲区大小和字段顺序见头文件。`ot_search` 的毫秒参数允许 0，表示仅按深度限制；使用时请自行限制最大深度。C ABI 不抛出异常跨越语言边界。

调用者持有输入和输出缓冲区；各次搜索拥有独立置换表，可在不同线程使用独立缓冲区并行调用。浏览器额外使用的 `ot_input/ot_output` 是单线程共享暂存区，不属于原生调用推荐接口。

Python ctypes 示例：

```sh
python3 examples/ffi.py .build/native/libothello.dylib
```


## Egaroucid 原生选步

可选构建现已提供 `--engine egaroucid --eg-level 5`，默认开启 `use_book`，以及独立的 `ot_egaroucid_*` C ABI。上游的毫秒预算尚未实现，使用等级控制；完整启动、资源、输出和 Python 示例见 [Egaroucid 接入说明](EGAROUCID.md)。

## 残差模型实验

v1.5 的 `OTRES1` 模型也可通过现有 `ot_nnue_load/evaluate/search/play/destroy` 接口调用，无需 PyTorch。原生 CLI 用 `--network models/resnet32x2-v1.5.otres` 指定模型，输出 evaluator 为 `othello-resnet32-v1`。输入/评分均按当前行棋方解释；开局随机策略与 v1.3 相同。模型训练与实验记录见 [训练说明](TRAINING.md) 和 [v1.5 实验报告](../models/v1.5-comparison.md)。

## MCTS 对照

v1.6 原生 CLI 用 `analyze --search mcts --time 1000 --network MODEL` 选择价值网络 MCTS；v1.7 已将内嵌 NNUE＋MCTS 作为默认；`--search ab` 显式选择 alpha-beta。MCTS 不接受 `--depth`，报告的 `depth` 是访问到的最深路径，`depthKind=maximumVisited`，不能与 AB 的完整搜索深度混用。`selfplay` 也支持 `--search`，原生 API 不依赖浏览器。

`ot_nnue_mcts_play` 使用同一个模型句柄，输出沿用 136 个整数及 6 个选步策略字段，另需 64 个整数的访问次数缓冲区与 72 个 double 的统计缓冲区；候选均值、访问次数、模拟数、估分次数、平均访问深度、子树复用和内存上限标记见 `cpp/api.h`。预算毫秒数为 0 时必须指定正的模拟数；实验使用 1,000ms、模拟数上限 0。`ot_mcts_clear_cache()` 只清空当前线程的 MCTS 树，`ot_search_clear_cache()` 清空原 AB 置换表。

## 内嵌默认模型

`ot_default_search(cells, turn, milliseconds, seed, output)` 直接使用内嵌 v1.3 NNUE、MCTS 和前 12 手近分策略，时间参数为 1..60000ms；输出仍是 136 个整数，depth 为最大访问路径，节点字段为模拟次数。`ot_default_evaluate` 估分，`ot_default_model_sha256` 返回模型校验和。无需文件系统、模型下载或模型句柄。原 `ot_search`、`ot_nnue_search/play` 的 AB 行为保持兼容。浏览器熟练/高级调用默认接口，入门调用原启发式接口。

`cpp/model/default_nnue.inc` 由 `scripts/embed-default-model.py` 从冻结的 `nnue256-v1.2.otnnue` 生成；`npm run model:check` 检查字节和 SHA-256。Wasm 使用 SIMD128、1MiB 栈和可增长内存，支持当前主流浏览器。

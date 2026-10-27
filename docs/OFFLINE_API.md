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

支持 `--time 1..60000`（毫秒）、`--depth 1..64`、`--level 0|1|2`、`--seed 0..2147483647`。默认 800ms、深度 64、level 2；level 0 是探索式浅层策略，1/2 都使用 α–β 搜索，具体难度由调用方的时间/深度预算决定。

输出为单行 JSON。非法参数/落子返回 `{"error":"..."}` 和退出码 1；成功退出码 0。程序信息不混入标准输出。

`legal` 与 `play` 返回棋盘、行棋方、子数、64 位哈希的十六进制字符串、`over`、`mustPass`、合法坐标列表；`play` 额外返回翻子数。`analyze` 额外返回最佳坐标、行棋方视角分数、已完成深度、节点数、置换表命中数、耗时、是否精确终局和根候选。

终局分数为胜 `100000 + 子差×100`、负 `-100000 + 子差×100`、和 0；中盘分数为启发式单位，不是胜率或预计子差。无路可走的非终局返回 `move:"pass"`，并继续搜索对手回复；终局同样返回 `pass`，但此时 `over:true`，不能再执行落子。

## 可选 NNUE 模型

`analyze`、`evaluate` 和 `selfplay` 支持 `--network models/nnue256-v1.otnnue`。模型加载失败会返回错误；不传此参数使用原有启发式。`evaluate` 返回静态局面分数，NNUE 的范围为 -1000..1000；`analyze` 的实际终局分数仍使用上述胜负/子差规则。

原生库在 `api.h` 提供 `ot_nnue_load`、`ot_nnue_evaluate`、`ot_nnue_search`、`ot_nnue_destroy` 和 `ot_nnue_error`。模型可一次加载、重复离线调用，搜索输出布局与 `ot_search` 相同；示例见 [nnue_ffi.py](../examples/nnue_ffi.py)。评分、搜索成功返回 1，失败返回 -1；加载失败返回空指针。保持模型存活直至并发调用结束。

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

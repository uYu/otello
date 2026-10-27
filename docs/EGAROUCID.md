# Egaroucid 原生库接入

## 已实现

直接链接上游的 `egaroucid_search_array` C ABI，默认 `use_book=1`、`book_accuracy_level=1`。原生调用不经过浏览器或 Wasm。

- 固定源码版本：`2a65c87860be999373088ad833aa00b0844883b6`，来自 [官方仓库](https://github.com/Nyanyan/Egaroucid/tree/2a65c87860be999373088ad833aa00b0844883b6)。该提交 C ABI 的版本字符串为 `7.8`。
- 使用该版本随附的 `book.egbk3`、评估数据和哈希数据，存放在 `.build/deps/Egaroucid/bin/resources`。
- 初始化检查资源文件、book 的签名/版本/记录长度，并用初始局面做真实选步验证；没有命中 book 时初始化明确失败，不静默退回无库搜索。
- 本项目的 `0 空 / 1 黑 / -1 白` 转为上游 `-1 空 / 0 黑 / 1 白`；格子索引统一为 A1=0、H8=63。
- 上游有进程级共享状态，适配器通过互斥锁串行调用；HTTP 服务也只保留一个原生进程和一个请求队列。
- 根选择结果 `depth == -1` 时记录 `bookHit=true`。严格模式（variety 0）还会搜索库外候选；随机模式使用上游的库内加权选择策略。若上游在库内候选之外额外执行了搜索，C ABI 未必报告 `-1`，界面会保守地按搜索显示。

## 一次性构建

需要 Git、CMake 3.25+ 和支持 C++20 的编译器。首次构建需要联网下载上游源码/资源，之后使用完全离线。

```sh
npm run setup:egaroucid
```

脚本只在项目 `.build` 下获取固定版本并构建，不安装到系统目录。默认使用 Generic 版本，ARM 机器额外开启 ARM 选项。普通内置引擎仍可独立用原来的 `.build/native` 构建。

## 原生离线调用

```sh
.build/native-egaroucid/otello analyze --engine egaroucid --eg-level 5
.build/native-egaroucid/otello analyze --engine egaroucid --eg-level 1 --book off
.build/native-egaroucid/otello selfplay --engine egaroucid --eg-level 1
.build/native-egaroucid/otello protocol
```

默认 book 开启，可通过 `--book off` 做对照。原生 `--eg-level` 范围 1..60。

库内选步默认有受控随机性：`--book-variety 1`，在与库内最高评分相差最多 2 个评分单位的候选中加权随机，高分走法概率更高。不是只在同分时随机。`--book-variety 0` 恢复严格最高分（同分仍随机），2/3 分别放宽到 4/6 个评分单位。参数不改变出库后的搜索算法；这些是单步库内评分容差，不是整局子差保证。

初始局面实测输出包含：

```json
{"engine":"egaroucid","engineVersion":"7.8","score":0,"scoreUnit":"discs","depth":-1,"nodes":0,"useBook":true,"bookHit":true}
```

同分和接近最优的库内候选会按权重随机选择，因此不把某一个坐标写死。Egaroucid 的 `score` 是行棋方视角的子差估值，和内置 AI 的启发式单位不同；`ttHits:null` 表示上游 C ABI 没有提供该统计。

上游当前接受但忽略 `time_limit_ms`。适配器不伪装成支持毫秒预算；Egaroucid 命令若同时传入 `--time/--depth` 会报错，改用 `--eg-level`。内置引擎的时间/深度设置不变。

资源目录可用 `--resources PATH` 或环境变量 `OTELLO_EGAROUCID_RESOURCES` 指定。原生库默认资源路径来自构建时源码位置；移动二进制或安装后，应显式指定资源目录。常驻行协议用空格分词，带空格的资源路径请通过环境变量提供。初始化后不能在同一进程更换资源目录。

## C++ / C / Python

```cpp
#include "egaroucid.hpp"
othello::Egaroucid engine; // 或传入资源目录
auto result = engine.analyze(othello::Position{}, 5, true);
// result.bookHit 指示此次根选择是否报告开局库命中。
```

链接 `.build/native-egaroucid/libothello.dylib`（Linux 为 `.so`）。C 语言头文件为 `cpp/egaroucid_api.h`，提供持有引擎句柄的 create/search/destroy；失败时用 `ot_egaroucid_last_error` 读取线程局部错误。各输出字段见头文件。

```sh
python3 examples/egaroucid_ffi.py .build/native-egaroucid/libothello.dylib
```

## 网页使用

```sh
npm run dev:native
```

同时启动 Vite 和仅监听 `127.0.0.1:8207` 的原生桥接服务。新局菜单自动检查本机服务，通过验证后默认选中 Egaroucid。旧存档仍保留原引擎，可在暂停菜单切换，棋谱不变。

界面难度对应 Egaroucid 等级 1/5/10；所有网页原生请求都固定启用 book。入门使用 book variety 2（容差 4），熟练/高级使用 variety 1（容差 2）。命中时，仅在页脚引擎状态旁显示一个很淡的 `·`，悬停显示“开局库命中”；棋盘旁不展示显眼的命中文字。该标记指最近一次分析/AI 选步，不是对当前玩家局面的持续预测；回放中隐藏。

内置 AI 继续在浏览器 Worker 中运行；Egaroucid 使用本机原生进程，不会在网页里重新编译或下载引擎。服务不可用时明确报错，不悄悄改用另一引擎。

也可单独运行生产版：

```sh
npm run build
npm run native:serve
# http://127.0.0.1:8207/，同时提供 dist 和 /api/native/*
```

服务入口：

- `GET /api/native/health`：实际探测初始局面 book 命中，返回 `bookReady/useBook`。
- `POST /api/native/move`：`{version:1, board:"64 字符 B/W/.", turn:1|-1, difficulty:"easy"|"normal"|"hard"}`。
- 响应含 0..63 的 `move`、`bookHit`、`useBook`、搜索信息。原生停着为 -1；网页通常已由规则层自动处理停着。

请求有 4 KiB 大小上限和有限队列；服务拒绝外部站点 Origin 和非本机 Host。单次搜索超过 20 秒或客户端取消时终止原生进程，下次请求重新初始化；不使用上游未实现的毫秒预算。浏览器的请求超时为 22 秒。

Vite 默认代理到 8207；可通过 `OTELLO_API_PORT` 同时配置服务与开发代理。原静态 Docker 部署只包含内置 AI；如需原生 Egaroucid，使用这里的原生服务启动方式。

## 验证

```sh
ctest --test-dir .build/native-egaroucid --output-on-failure
npm run test:egaroucid
npm test
npm run build
```

已核对同一实际局面的 4096 次加权随机采样、严格/随机模式的评分容差、开局库开关对照、双方坐标/合法点/翻子、强制停着、终局、两局完整 AI 对战、C ABI、常驻协议和真实本机 HTTP 请求。接口测试还覆盖并发、取消后恢复、非法输入和外部 Origin 拒绝。

上游为 GPL-3.0-or-later，来源和许可证见 `third_party/egaroucid`。本项目未修改上游源码。当前随附数据的校验和记录在该目录的 README 中。

## NNUE 与开局库组合实验

`ot_egaroucid_book` / `Egaroucid::probeBook` 仅查询上游 Book，不进行 Egaroucid AI 搜索。命中返回合法走法、库内子差评分、depth=-1、nodes=0；未命中返回 0，交给调用方的 NNUE 搜索处理。停着/终局视为查库未命中。

随机度 0..3 与既有加权选步含义一致；seed=0 保持应用正常随机，非零 seed 固定当前查询的随机抽样。实验使用逐局/逐步种子，随机度 1。命中时直接用库内走法，不将库内子差评分与 NNUE 的评分单位混合。未命中时查库耗时计入 50ms 单步预算，剩余预算按原生接口的整数毫秒向下取整。

两组实验由 `npm run training:baseline` 顺序执行：纯搜索 vs NNUE、NNUE vs NNUE＋开局库。用 `npm run training:watch` 或 `python3 scripts/watch-matches.py` 实时监控。结果见 [v1 基线](../models/comparison.md)。

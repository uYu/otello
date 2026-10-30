# 黑白之间 · Othello Atelier

米白、墨绿风格的完整黑白棋程序。规则和 AI 均由 **C++17** 实现，同一核心同时提供原生库、命令行和网页适配。原生程序无需 Node、浏览器、网络或 WebAssembly。

## 数据恢复

仓库包含原始训练棋谱和开局库的两个版本化压缩包；克隆后可离线恢复，JSONL 和缓存由脚本重建：

```sh
git clone https://github.com/uYu/otello.git
cd otello
python3 scripts/data-assets.py restore
```

原始训练快照约 9.19 MB，开局库快照约 20.54 MB，均附文件级 SHA-256。[data/README.md](data/README.md)列出内容、来源和恢复目录；[AGENTS.md](AGENTS.md)说明后续代理应如何恢复原始数据、重建缓存、构建和复测。

## 原生离线程序

需要 CMake 3.16+ 与 C++17 编译器（已在本机 macOS Apple Clang 上验证）。

```sh
cmake -S . -B .build/native -DCMAKE_BUILD_TYPE=Release
cmake --build .build/native -j
ctest --test-dir .build/native --output-on-failure

.build/native/otello legal
.build/native/otello play --move D3
.build/native/otello analyze --time 1500 --depth 64
.build/native/otello selfplay --time 30 --depth 5
.build/native/otello protocol
```

命令输出 JSON；`protocol` 支持常驻进程，一行命令对应一行 JSON，输入 `quit` 退出。`--board` 接受按行排列的 64 字符 `B/W/.`，`--turn b|w` 指定行棋方；停着用 `--move pass`，仅在规则允许时接受。详见 [离线接口](docs/OFFLINE_API.md)。

CMake 同时生成 `libothello.dylib`（Linux 为 `.so`）；[cpp/api.h](cpp/api.h) 是独立 C ABI，可供 C/C++、Python ctypes、Rust FFI 等使用。原生构建完全不涉及前端或 Emscripten。

## Egaroucid 原生引擎与开局库

已接入官方 Egaroucid C ABI，默认开启 `use_book`，并在初始化时验证真实开局库命中。库内选步使用偏向高分的加权随机，熟练/高级容许最多 2 个评分单位的差距，入门容许 4 个，增加开局变化。

```sh
npm run setup:egaroucid          # 首次下载固定版本并编译，需 CMake 3.25+ / C++20
npm run dev:native               # 网页 + 本机原生服务
# 或直接离线调用：
.build/native-egaroucid/otello analyze --engine egaroucid --eg-level 5
```

开局库使用仓库中压缩保存的固定快照；引擎代码和其余评估资源由安装脚本获取，安装后可离线运行。

新局可选择 Egaroucid；已有棋局可在暂停菜单切换引擎。命中提示只在页脚引擎状态旁显示一个很淡的小点，悬停可看说明。原生 C/C++/Python 调用、资源路径、测试和启动方式见 [Egaroucid 接入说明](docs/EGAROUCID.md)。

## 网页界面

需要 Node.js 22.13+。

```sh
npm ci
npm run dev
# 打开 http://127.0.0.1:5173
```

功能：

- 中文规则、人机对战、双人同屏、AI 观战；选择黑白执子、三档 AI。
- 完整八方向翻子、强制停着、双方均无法落子时提前终局、平局与结算。
- 合法落点、悬停翻子预览、最后落子标记、后台 AI 提示、搜索和缓存命中统计。
- 人机悔棋退回到上一次玩家决策之前；观战支持暂停和逐手回看。
- 自动保存、JSON 棋谱导入/导出，导入时从初始局面重放验证每一手。
- 回放支持逐步、进度拖动和自动播放，回放不修改棋谱；打开菜单会停止搜索。
- 响应式手机布局、方向键与 Enter 落子、原生模态框、减少动态效果设置。
- CSS 棋盘及棋子，无远程字体或图片资源。页面由 HTTP 本地加载后，对战无需外部服务。

浏览器只保存一份当前棋谱；开始新局会替换它，界面提供导出备份。双人模式是同屏轮流操作。静态网页需要 HTTP 服务，不支持双击 HTML 或首次断网加载；没有安装式 PWA/Service Worker。

```sh
npm test                  # Wasm、存档、重放、接口校验与原生一致性
npm run test:native        # ASan + UBSan，规则/搜索/哈希测试
npm run build             # TypeScript 检查与生产构建
npm run preview           # 预览 dist
npm run build:wasm        # 修改 C++ 后重新编译网页适配层
```

已生成的 `src/game/wasm/othello.mjs` 包含 Wasm，纳入源码；普通网页构建不需要 Emscripten。修改 C++ 后需要 Emscripten 4.0.18 或兼容版本重建此文件。脚本会优先查找 PATH，其次查找项目内 `.build/emsdk`。

## AI 与哈希

默认对弈为 **v1.36 Pattern 策略／胜负价值模型＋MCTS**，权重直接嵌入原生库和浏览器 Wasm，无需额外模型文件或下载。MCTS 使用学习到的合法落点先验与神经网络胜负估值，并复用实际走到的子树。当前模型与冻结基线由 `models/default-model.json` 和 `models/v1.36-default.json` 记录。仍可用 `--search ab` 选择原 PVS/α–β、aspiration window、迭代加深和 NNUE 增量评估；`--network heuristic` 显式使用较弱的启发式。

**Zobrist 哈希**包含棋子颜色、格子和行棋方，落子/翻子/停着时增量异或更新。固定 SplitMix64 种子保证跨进程、原生和 Wasm 结果稳定。v1.1 的 262,144 项置换表分为四项一桶，在同一线程内跨请求复用，按深度、准确性和代数替换；命中后比较完整双方位棋盘，切换评估器或修改模型参数时自动清表。递归搜索使用固定数组排序，NNUE 累加器延迟到需要评分时更新。[v1.1 性能记录](models/v1.1-search.md)包含对比结果和复现命令。

网页入门档保留较弱的启发式；熟练档为 NNUE＋MCTS、每手最多 100ms，高级档每手最多 1 秒。正常连续对弈保留 worker 和 MCTS 子树；取消思考或重开棋局会清理。检测到本机 Egaroucid 时也不自动切换默认选择，仍可手动选用其原生引擎与开局库（等级 1/5/10）。MCTS 的最大访问路径与 AB 的完整搜索深度不同，界面分别显示模拟次数或搜索节点。

## 训练模型与对战验证

已下载并清洗 WTHOR / Logistello，共 258,418 局有效去重棋谱；v1 从其中 141,336 局训练分组的完整对局抽取 **100 万个训练局面**，冻结验证/测试集各 6,000 个局面。初版 v1 模型为 `128→256→1` NNUE，33,280 参数、约 67KB，支持 C++/C ABI 与 CLI 离线使用：

```sh
.build/native/otello analyze --time 100
# 显式保留旧搜索：--search ab；弱对手：--network heuristic
python3 examples/nnue_ffi.py
```

v1 测试集平均子差误差为 10.58 子，原型为 11.24 子。基线记录两组实验：纯搜索 vs 模型搜索、模型搜索 vs 模型搜索＋开局库，双方每步 50ms，从初始局面交换黑白。网页默认在 v1.7 切换到 NNUE＋MCTS，并在 v1.8 按用户决定启用新增策略头；Egaroucid 保留为可选项。详见 [训练与数据](docs/TRAINING.md)、[Stockfish 方法适配](docs/STOCKFISH_FRAMEWORK.md)和[v1 基线](models/comparison.md)。

**v1.3** 在 v1.2 NNUE 上增加开局近分加权选步：前 12 次落子逐渐收紧随机范围，之后恢复最高分走法。原生 NNUE 自对弈默认启用，分析接口可显式开关，每局记录种子。[v1.3 对战与多样性记录](models/v1.3-comparison.md)比较该策略对原版 v1.1 的表现。模型权重不变，启发式引擎保留为较弱的玩家对手。

v1.4 包含独立的 8×8 小 CNN 实验及加载支持，推荐算法仍为 v1.3：CNN 在每步 50ms 的 100 局中 4 胜 96 负，未替换默认权重。[实验报告](models/v1.4-comparison.md)包含同深度诊断及 AlphaGo 原论文结构对照。

v1.5 的 32 通道、2 个残差块模型将测试集误差降到 9.83 子，但在 50ms 的 alpha-beta 对战中为 2 胜 98 负。v1.6 增加价值网络 MCTS（均匀先验 PUCT），对 NNUE/CNN × AB/MCTS 四组合进行每步 1 秒的两两对战；见[结果表与完整协议](models/v1.6-comparison.md)。v1.7 将 NNUE＋MCTS 接为默认，原生 CLI 可用 `--search ab` 回到 AB。

同一 v1.3 NNUE、每步各 100ms 的 100 局补充对战中，MCTS 对 AB 为 63 胜、11 平、26 负，得分率 68.5%，平均多 4.65 子。[完整记录与限制](models/v1.6-100ms-comparison.md)；全部从初始棋盘自行走棋，各执黑白 50 局。

v1.2 新增考虑行棋方的 NNUE 模型，使用同一百万局面缓存重新训练，保留 v1.1 的搜索优化。共享特征层是 128→256，当前行棋方与对方的特征按顺序拼接后输出评分，参数为 33,537，模型约 67.6KB。[v1.2 对战结果](models/v1.2-comparison.md)记录原最高分选步对 v1.1 的 100 局测试；双方从标准初始棋盘自行走棋，不使用开局库，每步各 50ms，交替执黑白。

v1.1 的工程优化与原模型保持冻结，见 [v1.1 对战结果](models/v1.1-comparison.md)及[性能记录](models/v1.1-search.md)。新旧模型使用独立文件，原生 CLI 可通过 `--network models/nnue256-v1.2.otnnue` 选择新模型；浏览器内置对手默认使用 v1.36 Pattern 策略／胜负价值模型＋MCTS，入门档保留较弱启发式，可选原生 Egaroucid。

```sh
npm run training:baseline    # 仅供原始 v1 源码；当前 v1.1 会拒绝覆盖冻结报告
npm run training:watch       # 实时查看当前对局和胜/平/负
npm run training:watch:v1.1  # v1.1 对 v1 的 NNUE 对战进度
npm run training:watch:v1.2  # v1.2 对 v1.1 的 NNUE 对战进度
npm run training:watch:v1.3  # 开局随机选步对原版 v1.1 的对战进度
```

新增的 **补充棋形 NNUE 实验（本地标识 v1.8）**将边线、内部线与角部三值棋形映射为 32 维向量，结合局面阶段，同时学习子差、胜负和及落子策略。原生加载支持独立的 `OTPAT1` 文件，MCTS 可使用策略先验；现有默认模型保留。训练、推理一致性、速度及三组等时限对战见 [v1.8 实验报告](models/v1.8-comparison.md)。

## 公开 AI 接口调研

已找到在线 REST 示例，以及可离线接入的 Edax、Egaroucid 和带 REST 的 MCTS 开源项目。优先考虑 **Egaroucid C ABI 或 Edax 原生进程**，更贴合离线调用需求；网页以后可通过本地 HTTP 桥接。来源、协议差异和验证边界见 [AI 接入调研](docs/AI_INTEGRATION.md)。

`src/game/remote.ts` 已预留受时间限制且校验合法落点的 HTTP 客户端。当前另有 `/api/native/*` 本机桥接，网页可调用已经接入的 Egaroucid 原生库；不向第三方托管对战服务发请求。

公开开局数据与宽松许可的核查见 [开局库调研](docs/OPENING_BOOKS.md)。已区分带评分的 book、实战数据库和仅含名称的开局列表；Egaroucid 可选原生构建已使用固定版本随附 book，默认启用；纯浏览器内置引擎仍不加载第三方开局数据。

## 部署

```sh
docker compose up --build
# http://localhost:8207
```

默认端口为 8207，可用 `OTELLO_PORT` 覆盖。开局库默认固定启用，每步先查库，出库后继续 v1.36 Pattern＋MCTS，不提供关闭选项。Docker 构建会验证已批准的 v1.36 基线、冻结 Wasm 和开局库；部署版本可由 `/release.json` 核对，健康检查为 `/healthz`。已有本地构建可使用 `docker compose -f compose.yaml -f compose.prebuilt.yaml up -d --build`，先运行 `npm run build` 即会自动准备并校验发布资产。服务器部署说明见 [部署记录](docs/DEPLOYMENT.md)。

或者将 `dist/` 放到静态服务器；Vite 使用相对路径，支持子目录。Docker 只构建网页并提供内置 AI。Egaroucid 原生对战可用 `npm run build && npm run native:serve`，由同一个本机服务提供网页与原生接口。

## 结构

```text
cpp/othello.hpp          位棋盘、规则、Zobrist、C++ 接口
cpp/search.cpp           搜索和置换表
cpp/api.h / api.cpp      调用方持有缓冲区的 C ABI；Wasm 薄适配
cpp/main.cpp             JSON 输出命令行、常驻行协议
CMakeLists.txt           原生动态库、可执行程序、CTest
src/game/engine.ts       C ABI 的网页绑定，无重复 TS 规则实现
src/game/ai.worker.ts    后台搜索
src/game/storage.ts      棋谱验证和重放
src/game/remote.ts       未来 HTTP 引擎适配契约
src/components/Board.tsx 键盘与鼠标棋盘
src/App.tsx              菜单、对局、提示、回放、存档
tests/native.cpp         独立逐格规则 oracle、perft、残局、哈希、C ABI
tests/game.test.ts       Wasm/存档/原生差分/HTTP 测试
examples/ffi.py          Python 原生库调用示例
```

## v1.8 默认基线

用户于 2026-10-27 批准将策略头模型作为默认，并作为后续优化的新基线。共享 NNUE 特征和子差价值头保持原权重，新增 `512→32→65` 合法落点策略分支，共 52,098 参数、104,936 字节，原生与网页均离线内嵌。熟练／高级档仍为 100ms／1 秒，入门保留弱启发式。

1 秒组为 11 胜、2 平、7 负（20 局，60%）；100ms 组为 29 胜、2 平、69 负（100 局，30%）。提升为默认是用户的版本选择，不改写这些结果，也不表示各时限都优于 v1.7。见 [默认基线](models/v1.8-default.json)、[1 秒实验](models/v1.8-policy-1s-comparison.md)及 [100ms 实验](models/v1.8-policy-comparison.md)。

当时价值输出仍预测最终子差；后续 v1.9 已独立训练并获批准提升为默认。

## v1.9 默认基线

用户于 2026-10-27 批准将胜负价值候选作为默认，并冻结为后续优化的新基线。沿用 v1.8 共享特征和策略分支，只重训 513 个价值头参数；单 tanh 输出拟合行棋方终局胜 +1、平 0、负 −1，MCTS 终局回传和精确证明采用相同语义。`(v+1)/2` 表示含和棋半分的期望得分。格式为 `OTNNUE4`，原生和网页均离线内嵌。

对冻结 v1.8，每步各 1 秒、20 局取得 **18 胜、2 平、0 负，得分率 95%，平均子差 +2.8**。保持原有 100ms／1 秒难度预算；本次没有新增 100ms 对照。见 [新默认基线](models/v1.9-default.json)及[实测结果](models/v1.9-outcome-comparison.md)。之前的 v1.8 基线和候选时期报告保持冻结。

补充棋形实验现已另建以 **v1.9 为基线的胜负价值重训**（v1.25），
使用 `OTPAT2` 区分旧子差模型，训练和对战独立记录。
运行 `sh scripts/watch-match.sh` 查看训练进度、候选视角战绩和逐手棋盘。
复现方式见 [训练说明](docs/TRAINING.md#v125在-v19-基础上的-pattern-胜负价值实验)。

## v1.25：用户批准的 Pattern 默认基线

用户于 2026-10-29 明确批准“可以替换基线了”，因此将
`pattern32-outcome-v1.25.otpat` 提升为默认，冻结记录为
[models/v1.25-default.json](models/v1.25-default.json)。原生和 Wasm 内嵌相同
OTPAT2 权重，使用学习先验及有限 FPU 的 PUCT MCTS，终局回传胜 +1、平 0、负 −1。
网页保留默认启用的开局库，出库后使用新模型；熟练／高级档仍为 100ms／1 秒。

对冻结 v1.9 的两组模型对照独立保存：100ms／20 局为 **18 胜、0 平、2 负**；
2 秒／20 局为 **12 胜、5 平、3 负**。本次提升不重训、不增加对战、不改写原报告。
v1.9 及所有候选时期记录继续保留，后续候选默认对照 v1.25。


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

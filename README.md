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

双 `uint64_t` 位棋盘；PVS + α–β 剪枝、aspiration window、迭代加深、历史/continuation history 走法排序；默认用行动力/角位/前沿子/位置权重估值。可选 256 维量化 NNUE 增量评估。时间耗尽时返回最后一轮完整搜索的结果；深度按实际落子递减，停着不消耗深度。

**Zobrist 哈希**包含棋子颜色、格子和行棋方，落子/翻子/停着时增量异或更新。固定 SplitMix64 种子保证跨进程、原生和 Wasm 结果稳定。每次搜索的 262,144 项置换表跨迭代深度复用，保存搜索深度、分数上下界和最佳落点；命中后还比较完整双方位棋盘，防止哈希碰撞误用。当前不跨不同搜索请求保留置换表。

网页入门档从位置评估靠前的三种走法中探索选择；熟练档最多 5 层、450ms；高级档迭代加深、1500ms，时间允许时精确搜索至终局。原生接口可自行设置时间和深度。以上描述的是本项目的内置启发式 AI；Egaroucid 是单独的可选原生引擎，使用其随附评估数据与开局库，网页难度映射到上游等级 1/5/10。

## 训练模型与对战验证

已下载并清洗 WTHOR / Logistello，共 258,418 局有效去重棋谱；v1 从其中 141,336 局训练分组的完整对局抽取 **100 万个训练局面**，冻结验证/测试集各 6,000 个局面。交付模型为 `128→256→1` NNUE，33,280 参数、约 67KB，支持 C++/C ABI 与 CLI 离线使用：

```sh
.build/native/otello analyze --network models/nnue256-v1.otnnue --time 1000
python3 examples/nnue_ffi.py
```

v1 测试集平均子差误差为 10.58 子，原型为 11.24 子。基线记录两组实验：纯搜索 vs 模型搜索、模型搜索 vs 模型搜索＋开局库，双方每步 50ms，从初始局面交换黑白。网页使用原有内置评估或 Egaroucid。详见 [训练与数据](docs/TRAINING.md)、[Stockfish 方法适配](docs/STOCKFISH_FRAMEWORK.md)和[v1 基线](models/comparison.md)。

```sh
npm run training:baseline    # 顺序运行两组实验
npm run training:watch       # 实时查看当前对局和胜/平/负
```

## 公开 AI 接口调研

已找到在线 REST 示例，以及可离线接入的 Edax、Egaroucid 和带 REST 的 MCTS 开源项目。优先考虑 **Egaroucid C ABI 或 Edax 原生进程**，更贴合离线调用需求；网页以后可通过本地 HTTP 桥接。来源、协议差异和验证边界见 [AI 接入调研](docs/AI_INTEGRATION.md)。

`src/game/remote.ts` 已预留受时间限制且校验合法落点的 HTTP 客户端。当前另有 `/api/native/*` 本机桥接，网页可调用已经接入的 Egaroucid 原生库；不向第三方托管对战服务发请求。

公开开局数据与宽松许可的核查见 [开局库调研](docs/OPENING_BOOKS.md)。已区分带评分的 book、实战数据库和仅含名称的开局列表；Egaroucid 可选原生构建已使用固定版本随附 book，默认启用；纯浏览器内置引擎仍不加载第三方开局数据。

## 部署

```sh
docker compose up --build
# http://localhost:8206
```

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

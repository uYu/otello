# 公开黑白棋 AI 接口调研

调研日期：2026-10-27。目标同时包含原生离线对战和未来网页接入。已进一步完成 Egaroucid 原生库接入、真实开局库开关对照、完整 AI 对战和本机 HTTP 测试；其他第三方托管服务仍未验证长期可用性。

| 方案 | 接口 | 部署方式 | 适合的用途 |
| --- | --- | --- | --- |
| Egaroucid | 实验性 C ABI；Console 支持 GTP | 本地原生库或子进程 | 优先试验强引擎离线对战 |
| Edax | 原生命令行、多协议、GGS | 本地程序；网页需桥接 | 对照引擎、局面分析、引擎竞技 |
| royhzq/reversi-monte-carlo-tree-search | `POST /search_move` | 可自行运行 Go 服务 | 快速验证 REST 对接 |
| Othello World / flowtwo | 作者公开了无状态 REST API 示例 | 作者网站演示服务 | 在线接口候选，需进一步确认完整 AI 路由与访问约定 |

## Egaroucid：与原生接口最匹配

[作者的库文档](https://www.egaroucid.nyanyan.dev/en/library/)公开了独立于 GUI/Console 的 C ABI：初始化资源、创建引擎、选步、合法点、翻子和销毁接口。它是实验性 API，后续可能变化。也可使用 [Console GTP](https://www.egaroucid.nyanyan.dev/en/console/)。

接入需要转换棋子编码：本项目 `0 空 / 1 黑 / -1 白`，Egaroucid `-1 空 / 0 黑 / 1 白`；双方格子索引均为 A1=0 到 H8=63。Egaroucid 还需要加载评估和开局等资源，不是只链接一个函数即可运行。文档样例当前将 `time_limit_ms` 标为忽略，不能直接假定它能遵守本项目的时间预算。

现已实现原生适配器和本机 HTTP 桥接，默认 `use_book=1`，初始化时检查初始局面的真实 book 命中。原生、C ABI 和网页均可调用；坐标、停着及完整对战已测试。固定版本、资源与运行方式见 [Egaroucid 接入说明](EGAROUCID.md)。

## Edax：成熟的原生对照方案

[官方仓库](https://github.com/abulmo/edax-reversi)提供位棋盘、多线程搜索、文本界面及多协议支持，也可连接 GGS。安装还需评估数据。它提供的是开源引擎和进程协议，没有在该文档中提供可直接调用的官方托管 HTTP 选步服务。

后续可以由本地进程适配器发送局面、搜索命令并解析返回结果。比起访问他人演示站，自行运行可控制版本、搜索预算和可用性。具体协议以固定版本源码/帮助为准。本轮没有声称已连通 Edax。

## 已有 REST 方案

[royhzq 的 Go MCTS 项目](https://github.com/royhzq/reversi-monte-carlo-tree-search)文档说明了 `POST /search_move`：请求为黑白棋子二维坐标数组及行棋方，返回二维落点。源码可自行部署，适合对接实验；文档中示例的中央棋子配色与本项目开局相反，不能直接复制示例作为本项目初始请求，坐标方向也应通过测试核对。

[Othello World 作者文章](https://flowtwo.io/post/othello-world)说明引擎通过无状态 REST 提供访问，并展示 `POST https://flowtwo.io/othelloworld/api/player/move` 的 JSON 请求/响应。该示例是执行玩家落子，不能据此把它当作已经确认的“AI 选步接口”。完整选步路由、CORS、限流、授权和长期稳定性仍需核实。该服务使用 64 位位棋盘数，JavaScript 传输不能把任意 64 位整数直接当普通 Number，后续适配需要确认精确序列化方案。

结论：有公开的接口实现和在线示例；本轮没有确认一个适合直接依赖、具备稳定访问约定的免费托管 AI 选步 API。离线需求优先采用 Egaroucid 或 Edax，HTTP 实验可先自行部署上述 MCTS 服务。

## 本项目预留的 HTTP 契约

`src/game/remote.ts` 中 `requestRemoteMove` 接受由调用方明确提供的 endpoint，发出：

```json
{
  "version": 1,
  "board": "...........................WB......BW...........................",
  "turn": 1,
  "legalMoves": [19, 26, 37, 44]
}
```

响应：`{"move":19}`。索引 A1=0，H8=63；-1 为停着。客户端提供 10 秒超时、取消、HTTP 状态和落子合法性校验。不自动将失败响应当作有效落子。

这是一份本项目的统一桥接契约，不宣称上述第三方原生协议与它兼容。需要适配器转换，当前界面可选择已经实现的本机 Egaroucid 服务；任意第三方远程引擎仍未开放，也不发出第三方托管请求。

项目许可证分别见各上游仓库；Egaroucid 仓库标注 GPL-3.0-or-later，Edax 以其固定版本 LICENSE 为准，Go MCTS 仓库标注 MIT。本项目可选原生构建会获取并链接 Egaroucid，保留上游源码与许可证；其他引擎尚未获取或链接。

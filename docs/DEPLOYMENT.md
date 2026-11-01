# v2 Docker 部署

目标服务器：`root@162.14.123.217`。应用位于 `/root/app/otello`，与其他桌游项目同级；对外端口 8207，Homepage 位于 80 端口的“桌游”分类。

此部署使用冻结的 v2 全局策略浏览器 Wasm 和 `pattern32-global-policy-v2.otpat`，模型 SHA-256 为 `df11f7b9eed358fff33a426e2f0cb1a290083d2bc8e0781a2fbd1efdaf4270c3`。中级、高级和提示固定启用开局库，不提供关闭选项；每步先查库，命中直接走库，未命中继续 v2 搜索。入门固定使用嵌入前端的 beginner-error-policy-80-e2-lr0003-v1 策略模型，按学到的概率选步，不使用开局库或 MCTS。AI 在客户端 Web Worker 中运行；容器提供静态页面和按需查询的开局库分片，初级复用高级版 v2 权重，直接按策略概率选步，不搜索、不使用开局库；中级搜索200ms，高级搜索2s。

## 构建和运行

源码构建：

```sh
docker compose up -d --build
```

也可在本地验证、构建网页，再在服务器复用已有 Caddy 基础镜像构建运行容器：

```sh
npm ci
npm run book:prepare
npm test
npm run build
docker compose -f compose.yaml -f compose.prebuilt.yaml up -d --build
```

`prepare-web-book.py` 校验原书 SHA-256，生成 4096 个查询分片，保留原始书、许可及源记录。`check-web-release.mjs` 验证默认选择器、模型字节、冻结 Wasm、实际内嵌模型、胜负终局值及全部开局库资产；`npm run build` 的前置步骤自动完成二者。`dist/` 与生成文件保持忽略。Docker 的 `prebuilt` 阶段复用已验证的 `dist/`，`production` 阶段在容器内生成开局库并完成源码构建。

开局库来源记录为 `opening-book/0dd454d84aa6-v1/SOURCE.json`，原始书 SHA-256 为 `0dd454d84aa656a64329930bad796c4a48a4322b7693d50b7040006383db5012`。包含 1,302,322 个有效局面，派生查询索引提供 938,502 个带已登记后续走法的局面。只加载当前查询分片；库加载失败会提示重试，不会跳过库。出库搜索不调用 Egaroucid 搜索。模型原基线报告不因网页的开局库包装而改写。

访问与核对：

- 游戏：`http://162.14.123.217:8207/`
- 版本与模型校验值：`http://162.14.123.217:8207/release.json`
- 健康检查：`http://162.14.123.217:8207/healthz`
- 数据来源说明：`http://162.14.123.217:8207/model-data-notices.md`

## Homepage

在 `/root/web/homepage/config/services.yaml` 的“桌游”分类中增加下列条目，并在修改前保留带时间戳的备份：

```yaml
    - 黑白棋:
        icon: mdi-circle-half-full
        href: http://{{HOMEPAGE_VAR_HOST}}:8207
        description: Othello · v2 + 开局库
        server: local-docker
        container: otello-otello-1
```

镜像名为 `otello:v2`，Compose 项目名为 `otello`，服务为 `otello`。使用 `docker compose ps` 查看运行及健康状态，`docker compose logs --tail 50` 查看日志。

## 2026-10-31 实际发布记录

服务器从 v1.9 更新到已批准的 v2，镜像 `otello:v2`，使用本地测试通过的 prebuilt 资产。公网 `/release.json` 的模型、Wasm 和开局库元数据与本地发布包一致，`/healthz` 返回 `ok`，容器状态 healthy。Homepage 版本说明已更新。

旧服务文件、解析后的 Compose 配置和容器信息保留在 `/root/app/otello/.deployment-backups/v1.9-before-v2-20261031T045902Z`；发布暂存目录为 `/root/app/otello/releases/v2-20261031`。未发布逐点特征等实验候选，正式模型仍为批准的 v2。

验证：19 项网页测试、2 项原生测试；12,000 个局面的嵌入/显式推理一致，32 个固定模拟根局面一致。历史 v1.49 动态库在本机已不存在，因此原生默认检查使用已保留并校验源码的 v2 reference rebuild，原始基线记录不变。完整记录见 `models/v2-server-deployment-20261031.json`。

## 2026-11-01 四档正式发布

已提交并推送代码 `a2259da2cdc8e3ea35b10fae77bb7079f5cd8d9a`，部署到 [线上游戏](http://162.14.123.217:8207/)。四档为入门独立模型、初级 v2 策略概率采样、中级查库后搜索200ms、高级查库后搜索2s；提示使用高级档。冻结v2权重和原搜索Wasm未变。

发布目录：`/root/app/otello/releases/v2-four-levels-20261101T123359Z`。回滚配置和旧静态文件：`/root/app/otello/.deployment-backups/before-v2-four-levels-20261101T123359Z`，旧镜像保留为 `otello:before-v2-four-levels-20261101T123359Z`。当前容器健康；公网首页、全部JS/CSS/Worker、release.json及healthz已核对。29项网页测试、2项原生测试通过；干净提交快照另通过构建和27项网页测试（2项依赖本机原生可执行文件的测试跳过）。可选独立Egaroucid集成测试因缺少专用可执行文件未运行，网页四档不调用它。

公网首页和`release.json`返回`Cache-Control: no-cache`，每次重新访问或刷新会重新校验；JS/CSS/Worker使用内容哈希文件名和一年immutable缓存。新版本生成新文件名，因此普通刷新即可更新。已打开的页面保留当前执行代码，需要用户刷新一次；同一站点的localStorage存档保留。详细发布记录见 `models/four-level-server-deployment-20261101.json`。

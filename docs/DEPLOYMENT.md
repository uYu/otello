# v2 Docker 部署

目标服务器：`root@162.14.123.217`。应用位于 `/root/app/otello`，与其他桌游项目同级；对外端口 8207，Homepage 位于 80 端口的“桌游”分类。

此部署使用冻结的 v2 全局策略浏览器 Wasm 和 `pattern32-global-policy-v2.otpat`，模型 SHA-256 为 `df11f7b9eed358fff33a426e2f0cb1a290083d2bc8e0781a2fbd1efdaf4270c3`。开局库固定默认启用，不提供关闭选项；每步先查库，命中直接走库，未命中继续 v2 搜索。AI 在客户端 Web Worker 中运行；容器提供静态页面和按需查询的开局库分片，正常／高级档搜索仍为 100ms／1 秒。

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

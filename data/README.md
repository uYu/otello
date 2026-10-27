# 固定数据快照

仓库随附两个实际的压缩包，普通 Git 克隆即可取得，无需 Git LFS。每个包及包内每个文件的大小、SHA-256、恢复目录均记录在 [manifest.json](manifest.json)。

| 快照 | 压缩包大小 | 内容 | 恢复目录 |
| --- | ---: | --- | --- |
| [原始训练数据 v1](training/otello-v1.tar.gz) | 9,188,280 字节（9.19 MB） | 49 个 WTHOR 年度文件、Logistello 原始 gzip、来源清单 | `.build/training/raw/` |
| [Egaroucid 开局库](opening-book/egaroucid-2a65c87860be.tar.gz) | 20,541,299 字节（20.54 MB） | 32,558,064 字节的 `book.egbk3`、GPL 许可、固定来源记录 | `.build/data/egaroucid/` |

## 恢复与校验

```sh
python3 scripts/data-assets.py restore
# 只恢复一个包：
python3 scripts/data-assets.py restore --only training-v1
python3 scripts/data-assets.py restore --only egaroucid-book
# 只验证，不替换已有数据：
python3 scripts/data-assets.py verify
```

只依赖 Python 标准库，不需要网络或 PyTorch。脚本校验压缩包和全部成员，拒绝不在清单内的文件、路径越界和链接。相同文件会跳过；已有文件内容不同时默认报错，明确要恢复快照时才用 `--force`。

压缩包只保存原始文件；解压文本、JSONL、训练缓存和固定评测缓存均不打包。构建原生核心、安装 `requirements-training.txt` 后，运行 `npm run training:rebuild` 生成这些派生文件。脚本不训练模型，并核对重建 JSONL、原型固定缓存和百万局面缓存的 v1 SHA-256；已实际验证三个哈希均与原版一致。随后可运行 `npm run training:run`。完整步骤见 [AGENTS.md](../AGENTS.md) 和[训练说明](../docs/TRAINING.md)。

`npm run setup:egaroucid` 会恢复开局库、获取固定版本的引擎代码和其余评估资源，再把快照中的同字节开局库安装到引擎资源目录。单独恢复开局库不等于已安装完整 Egaroucid 引擎。

## 来源和使用说明

- 开局库来自 Egaroucid 提交 `2a65c87860be999373088ad833aa00b0844883b6` 的 `bin/resources/book.egbk3`；保留上游 GPL-3.0-or-later 仓库许可，见 [依赖说明](../third_party/egaroucid/README.md)。未另加 MIT/BSD 等许可。
- 训练棋谱来自 [FFO WTHOR](https://www.ffothello.org/informatique/la-base-wthor/) 和 [Michael Buro 的 Logistello 页面](https://skatgame.net/mburo/log.html)。两处页面均说明可用于评估训练、开局研究；未找到独立 MIT/CC 数据许可。本项目按用户要求固定归档实际使用的数据，保留来源说明，不赋予数据新的许可。逐文件来源见 [来源清单](../models/training-data-manifest.json)。
- 本仓库未使用或打包 Egaroucid 的 52 百万局训练数据。

`.build/` 是恢复位置和可再生成工作目录，不纳入 Git。正式快照只有这里的两个包；需要有意更新版本时才执行 `npm run data:pack`。模型、训练权重和实验报告位于 [models/](../models/README.md)。

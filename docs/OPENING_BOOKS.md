# 公开开局库与使用条件

调研日期：2026-10-27；使用场景为个人、离线、非商用。下面分别记录“能下载”“作者明确允许的用途”“是否找到宽松的数据许可”，不将项目代码许可自动套到外部数据。

## 实际可用的数据源

| 来源 | 内容/格式 | 使用条件核查 | 建议 |
| --- | --- | --- | --- |
| [Egaroucid 默认 book](https://github.com/Nyanyan/Egaroucid/blob/main/bin/resources/book.egbk3) | 仓库内约 31 MB 的 `.egbk3`，保存局面评分等信息 | 仓库整体为 GPL-3.0-or-later；不是 MIT/BSD 式许可，未找到该 book 单独的宽松授权 | 个人离线实验优先候选，可直接让 Egaroucid 原生引擎读库 |
| [Logistello book skeleton](https://skatgame.net/mburo/log.html) | 约 37,000 行完整棋谱与黑方终局结果，ASCII/gzip，也有 WTHOR | 作者明确提供给开局研究、评估训练和改进机器人开局库；页面没有单独标注 MIT/CC0 | 很适合为本项目自建局面库；先做本地使用，保留来源 |
| [FFO WTHOR](https://www.ffothello.org/informatique/la-base-wthor/) | 多年高水平实战棋谱，年度 `.wtb`/ZIP；有公开格式文档 | 页面邀请免费下载，并明确提到可供程序员制作开局库；未找到明确 MIT/CC 数据许可 | 做实战频率/胜率统计，不能将样本胜率当作最优走法证明 |

Egaroucid 的[使用说明](https://www.egaroucid.nyanyan.dev/en/usage/)支持选择、合并、学习 book，也支持导入/导出 Edax book。其[原生 C ABI](https://www.egaroucid.nyanyan.dev/en/library/)提供 `use_book` 搜索选项，适合以后与本项目的 C++ 接口衔接。格式转换不能假定完全兼容。

## 找到的宽松许可候选与实际限制

- [Othello Sensei](https://github.com/borassi/othello-sensei)：C++ 引擎，根许可为 **Apache-2.0**。进一步检查固定提交 `07818a00a30b86eab610d4265984e7ff43de77c9` 的仓库树，确认 [assets/book](https://github.com/borassi/othello-sensei/tree/07818a00a30b86eab610d4265984e7ff43de77c9/assets/book) 实际包含 257 个文件、14,745,959 字节（约 14.7 MB），包括 `index.sen`、`roots.val` 和一组 `.val`。树中未发现另列的 book LICENSE/NOTICE，因而它是目前最值得优先评估的宽松许可候选。原生读取代码位于 `engine/book`，格式与 Edax/Egaroucid 不同。README 另提供下载更新 book 的功能，独立下载的更新包仍需按其具体条款核查，不自动由仓库许可推定。
- [eigenfoo/otto-othello](https://github.com/eigenfoo/otto-othello)：仓库为 **MIT**，包含 464 行 `openings.dat`。实际内容是“命名开局序列”，不含评估分数，且 README 标注其开局来源于外部网站；因此不能直接认定它是独立 MIT 授权的高质量评分库。
- [precursor-othello](https://github.com/tbcolby/precursor-othello)：仓库为 **Apache-2.0**。已下载检查其 `libs/othello-core/src/opening.rs`，实际 `BOOK` 只有一条初始响应和待扩充注释，且对称局面查找只返回固定坐标。它不构成值得接入的实战开局库，排除。

若优先宽松许可，先评估 Sensei 仓库内随附数据和 C++ 读取器；实际覆盖深度和评分质量仍要通过局面测试确认。本轮未找到单独明确标为 MIT/BSD/CC0/CC BY、同时经过质量核查的完整评分库。个人离线使用也可将 Egaroucid 默认 book 和作者明确提供用于改进机器人的 Logistello skeleton 作为备选。

另外，Egaroucid 的[52 百万局训练数据](https://github.com/Nyanyan/Egaroucid/releases/tag/training_data_v0002)有独立条款：允许用于自己的 AI 开发，但禁止再分发。它与仓库内默认 `book.egbk3` 是两项不同资源，不能混用许可判断。

## 本项目当前状态与接入次序

现已按用户选择接入 Egaroucid 原生 C ABI 并默认开启 `use_book`。固定版本 book 同时压缩保存于 `data/opening-book/`；用 `python3 scripts/data-assets.py restore --only egaroucid-book` 恢复，再由安装脚本放入 `.build/deps/Egaroucid` 的资源目录。原生初始化验证真实命中；不嵌入纯浏览器 Wasm 产物。详见 [数据快照](../data/README.md) 和 [实际接入说明](EGAROUCID.md)。前文关于 Sensei 等候选的比较仍供后续参考。

按宽松许可偏好，建议优先研究 Sensei 的仓库内 book 读取器；若更重视现成引擎能力，也可接入 Egaroucid 原生库并开启其 `use_book`。若保留完全自主的 C++ 搜索，则可从 Logistello skeleton 制作本地库：逐手合法性验证 → 转为双方位棋盘 → 对八种几何对称归一化 → 用 Zobrist 索引并同时校验完整局面 → 候选走法和评分入库。查询命中时还必须把坐标逆变换回当前棋盘；出库后回到现有迭代加深搜索。

这样可以复用已经实现的哈希，同时避免把“开局名/流行程度”误当作精确评估。若以后需要发布或随程序分发数据，再按具体固定版本及数据条款处理。

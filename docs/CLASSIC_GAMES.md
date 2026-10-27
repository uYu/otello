# 首页名局

首页棋盘来自 **1977 年首届世界黑白棋锦标赛决赛**：日本的井上博执黑，对挪威的 Thomas Heiberg，最终 **34∶30**。页面展示的是第 27 次落子后的中盘局面。

比赛背景与结果来源：[世界黑白棋联合会的 1977 年赛事记载](https://www.worldothello.org/about/tournaments/world-othello-championship/woc/1977/story)。完整落子取自 [FFO WTHOR](https://www.ffothello.org/informatique/la-base-wthor/) 的固定版本 `WTH_1977.wtb` 第 0 条记录；该记录与两人在循环赛中的 45∶19 对局分开。

实现位于 `src/game/classic-games.ts`。棋盘从标准初始局面通过同一 C++ 规则核心逐手重放，没有手工摆子；测试重放完整 60 手并核对 34∶30 的终局。小型棋谱是应用使用的原始名局记录，不依赖恢复大型训练数据即可展示。

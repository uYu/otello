# Stockfish 方法与黑白棋适配

参考官方 Stockfish 主线固定提交 [`ff3a6c508ed731f0dcf253f4ddf646df19a62f84`](https://github.com/official-stockfish/Stockfish/tree/ff3a6c508ed731f0dcf253f4ddf646df19a62f84)（2026-10-25），重点读取 `search.cpp`、`movepick.cpp` 和 NNUE 结构/特征变换器。代码采用独立的 Othello 实现；没有复制国际象棋规则、参数表或预训练权重。

## 已适配

| 方法 | 本项目行为与验证 |
| --- | --- |
| PVS | 首着完整窗口，后续走法零窗口探测，改善 alpha 时重搜；与独立穷举 minimax 对照 |
| Aspiration window | 深度 3 起按上一轮各根走法评分开窄窗口，失败后扩大；根候选均完成重搜再发布 |
| 历史与 continuation history | 按执子方、落点和前一落点记录 beta 截断，有限幅更新及衰减，辅助 TT/角位排序 |
| Zobrist 与 TT 界限 | 保存 exact/lower/upper，仅在界限证明可截断时返回；比较完整双方位棋盘避免碰撞误用 |
| NNUE | 量化的棋子位置特征，双颜色共享权重，搜索栈逐层维护累加器，仅增减落子/翻子对应的特征 |
| 迭代加深 | 超时返回最后一轮全部根候选完成的结果，保留搜索计数与重搜统计 |

停着是黑白棋真实规则动作；不减少落子搜索深度。双方无合法落点时用真实子差确定胜负，神经网络不替代终局结算。没有移植棋类空着剪枝、吃子静态搜索和未校准的选择性剪枝；它们的假设不能直接用于黑白棋。

## 鳕鱼网络并非棋盘卷积

该固定提交的[结构头文件](https://github.com/official-stockfish/Stockfish/blob/ff3a6c508ed731f0dcf253f4ddf646df19a62f84/src/nnue/nnue_architecture.h)定义：`K32Q2` 棋子位置特征、`FullThreats` 威胁特征、`PP_3Wide` 兵对特征；`L1=1024`、`L2=32`、`L3=32`、`LayerStacks=32`。特征变换器维护各方累加器，组合视角后送入小型网络；网络同时组合 clipped/squared clipped 激活，有跨层输出与跳连。因此简单画成 `1024→32→32→1` 只是概览。

Stockfish 不同版本的 NNUE 结构会改变，上述尺寸只针对这个提交。方法说明见[官方 NNUE 文档](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)。当前本项目的 `128→256→1` 是用于训练、推理及对战验证的小型适配，不等于完整复刻 Stockfish 网络。

`cpp/model/tiny_nn.h` 原样复用用户提供的 Patchwork SIMD 全连接计算代码，用作 FP32 对照。生产搜索采用整数增量核；测试覆盖整数增量与全量重算、FP32 对照、固定深度穷举、停着和精确残局。

Stockfish 源码采用 GPLv3+；本项目只将其作为这些通用算法的参考。可选 Egaroucid 库仍按其 GPLv3 和资源说明单独集成。

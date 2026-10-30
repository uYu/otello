# Tiny network inference helpers

`tiny_nn.h` is reused, unmodified, from the user-provided sibling project
`patchwork/cpp/model/tiny_nn.h`.

SHA-256: `5900a0125246045fa4536df2e4a47f09690c944a017c7f4a9ce7f05eba8f3d0e`.

The FP32 NNUE reference uses its SIMD linear/dot helpers. Production search
uses quantized integer accumulators; both implementations are compared in tests.
The experimental 8x8 CNN also uses these FP32 dot/linear helpers in its native
inference path, with a separate scalar convolution reference for verification.
No Patchwork model weights are reused.

`default_nnue.inc` 是用户批准的 v2 Pattern 全局策略／胜负价值默认模型（OTPAT9）的生成字节，由 `scripts/embed-default-model.py` 产生和校验。数据来源与使用约束仍见 `data/README.md` 和模型清单。

Pattern 的 v1.32 候选使用独立整数内核 `../pattern_kernels.hpp`，未调用或修改 FP32 `tiny_nn.h`。实验开关 `OTELLO_PATTERN_SIMD` 默认 OFF，ARM64 使用 NEON、x86-64 使用 SSE2；OTPAT3 只压缩棋形表，其余层保留原整数精度。格式和验证边界见 `models/v1.32-pattern-int8-comparison.md`。

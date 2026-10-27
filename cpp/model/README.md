# Tiny network inference helpers

`tiny_nn.h` is reused, unmodified, from the user-provided sibling project
`patchwork/cpp/model/tiny_nn.h`.

SHA-256: `5900a0125246045fa4536df2e4a47f09690c944a017c7f4a9ce7f05eba8f3d0e`.

The FP32 NNUE reference uses its SIMD linear/dot helpers. Production search
uses quantized integer accumulators; both implementations are compared in tests.
No Patchwork model weights are reused.

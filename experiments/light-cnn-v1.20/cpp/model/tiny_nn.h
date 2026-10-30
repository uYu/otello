#ifndef TINY_NN_H_INCLUDED
#define TINY_NN_H_INCLUDED

#include <cstddef>

// Intrinsic headers must be included outside user namespaces.
#if defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#elif defined(__AVX__)
#  include <immintrin.h>
#elif defined(__wasm_simd128__)
#  include <wasm_simd128.h>
#endif

/*
 * tiny_nn.h
 *
 * Minimal header-only FP32 MLP inference helpers.
 *
 * Requires:
 *
 *   C++17 or later.
 *
 *
 * Supported backends:
 *
 *   AArch64 / Apple Silicon:
 *       ARM NEON (128-bit, 4 floats)
 *
 *   x86 / x86_64 with AVX enabled:
 *       AVX (256-bit, 8 floats)
 *       FMA is used when enabled by the compiler
 *
 *   Other targets:
 *       Scalar fallback
 *
 *   WebAssembly with SIMD128 enabled:
 *       SIMD128 (4 floats)
 *
 *
 * Public API:
 *
 *   tiny_nn::dot(...)
 *   tiny_nn::matvec(...)
 *   tiny_nn::linear(...)
 *   tiny_nn::relu(...)
 *   tiny_nn::linear_relu(...)
 *
 *   tiny_nn::backend_name
 *   tiny_nn::simd_width
 *
 *
 * Matrix layout:
 *
 *   Row-major:
 *
 *       weights[output][input]
 *
 *   For a linear layer:
 *
 *       output = weights * input + bias
 *
 *
 * General preconditions:
 *
 * 1. Every pointer must refer to a sufficiently large readable or
 *    writable range as required by the corresponding function.
 *
 * 2. Matrix dimensions and their products must be representable by
 *    std::size_t. The complete matrix must fit in one valid C++ array
 *    object.
 *
 * 3. Except for relu(), output buffers must not overlap input,
 *    weights or bias buffers.
 *
 *
 * Floating-point notes:
 *
 * 1. SIMD loads and stores access exactly `simd_width` floats.
 *    Internal loops only perform SIMD accesses when a complete vector
 *    remains.
 *
 * 2. madd(a, b, c) computes:
 *
 *        a * b + c
 *
 *    It may or may not use fused multiply-add, depending on the
 *    backend and compiler options.
 *
 * 3. SIMD maximum uses these selection semantics:
 *
 *        a > b ? a : b
 *
 *    Consequently, maximum(NaN, b) returns b.
 *
 * 4. ReLU uses:
 *
 *        value > 0.0f ? value : 0.0f
 *
 *    Therefore NaN and -0.0f are converted to +0.0f.
 *
 * 5. Floating-point reduction order differs between backends and
 *    build configurations. Results are not guaranteed to be
 *    bit-identical.
 *
 * 6. Fast floating-point modes such as -ffast-math or /fp:fast may
 *    change NaN and floating-point contraction behavior.
 *
 *
 * Backend notes:
 *
 * 1. SIMD implementation types are private implementation details.
 *
 * 2. Backend selection happens at compile time.
 *
 * 3. There is no runtime CPU feature detection. A binary compiled
 *    with AVX must only run on a CPU and operating system that support
 *    AVX.
 */

namespace tiny_nn {
namespace detail {

// ============================================================
// ARM64 / Apple Silicon / AArch64 NEON
// ============================================================

#if defined(__aarch64__) || defined(_M_ARM64)

using vec = float32x4_t;

inline constexpr std::size_t width = 4;
inline constexpr const char* backend_name = "NEON";

inline vec zero() noexcept
{
    return vdupq_n_f32(0.0f);
}
inline vec splat(float value) noexcept { return vdupq_n_f32(value); }

// Requires `width` readable floats at p.
inline vec load(const float* p) noexcept
{
    return vld1q_f32(p);
}

// Requires `width` writable floats at p.
inline void store(float* p, vec value) noexcept
{
    vst1q_f32(p, value);
}

inline vec add(vec a, vec b) noexcept
{
    return vaddq_f32(a, b);
}

// Per lane:
//
//     a > b ? a : b
inline vec maximum(vec a, vec b) noexcept
{
    const uint32x4_t mask = vcgtq_f32(a, b);
    return vbslq_f32(mask, a, b);
}

// Computes a * b + c.
//
// AArch64 uses fused multiply-add here.
inline vec madd(vec a, vec b, vec c) noexcept
{
    return vfmaq_f32(c, a, b);
}

inline float hsum(vec value) noexcept
{
    return vaddvq_f32(value);
}

// ============================================================
// x86 / x86_64 AVX
// ============================================================

#elif defined(__AVX__)

using vec = __m256;

inline constexpr std::size_t width = 8;

#if defined(__FMA__)
inline constexpr const char* backend_name = "AVX+FMA";
#else
inline constexpr const char* backend_name = "AVX";
#endif

inline vec zero() noexcept
{
    return _mm256_setzero_ps();
}
inline vec splat(float value) noexcept { return _mm256_set1_ps(value); }

// Requires `width` readable floats at p.
inline vec load(const float* p) noexcept
{
    return _mm256_loadu_ps(p);
}

// Requires `width` writable floats at p.
inline void store(float* p, vec value) noexcept
{
    _mm256_storeu_ps(p, value);
}

inline vec add(vec a, vec b) noexcept
{
    return _mm256_add_ps(a, b);
}

// Per lane:
//
//     a > b ? a : b
inline vec maximum(vec a, vec b) noexcept
{
    // Ordered comparison:
    // NaN comparisons produce false, causing b to be selected.
    const __m256 mask =
        _mm256_cmp_ps(a, b, _CMP_GT_OQ);

    return _mm256_blendv_ps(b, a, mask);
}

// Computes a * b + c.
inline vec madd(vec a, vec b, vec c) noexcept
{
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(
        _mm256_mul_ps(a, b),
        c
    );
#endif
}

inline float hsum(vec value) noexcept
{
    // value:
    //
    //     [a b c d e f g h]

    const __m128 low =
        _mm256_castps256_ps128(value);

    const __m128 high =
        _mm256_extractf128_ps(value, 1);

    // [a+e, b+f, c+g, d+h]
    __m128 sum = _mm_add_ps(low, high);

    // [c+g, d+h, c+g, d+h]
    const __m128 high_pair =
        _mm_movehl_ps(sum, sum);

    // lane 0: a+e+c+g
    // lane 1: b+f+d+h
    sum = _mm_add_ps(sum, high_pair);

    // Broadcast lane 1.
    const __m128 lane1 =
        _mm_shuffle_ps(sum, sum, 0x55);

    // Add lane 1 into lane 0.
    sum = _mm_add_ss(sum, lane1);

    return _mm_cvtss_f32(sum);
}

// ============================================================
// WebAssembly SIMD128
// ============================================================

#elif defined(__wasm_simd128__)

using vec = v128_t;

inline constexpr std::size_t width = 4;
inline constexpr const char* backend_name = "Wasm SIMD128";

inline vec zero() noexcept { return wasm_f32x4_splat(0.0f); }
inline vec splat(float value) noexcept { return wasm_f32x4_splat(value); }
inline vec load(const float* p) noexcept { return wasm_v128_load(p); }
inline void store(float* p, vec value) noexcept { wasm_v128_store(p, value); }
inline vec add(vec a, vec b) noexcept { return wasm_f32x4_add(a, b); }
inline vec maximum(vec a, vec b) noexcept {
    return wasm_v128_bitselect(a, b, wasm_f32x4_gt(a, b));
}
inline vec madd(vec a, vec b, vec c) noexcept {
    return wasm_f32x4_add(wasm_f32x4_mul(a, b), c);
}
inline float hsum(vec value) noexcept {
    return wasm_f32x4_extract_lane(value, 0)
         + wasm_f32x4_extract_lane(value, 1)
         + wasm_f32x4_extract_lane(value, 2)
         + wasm_f32x4_extract_lane(value, 3);
}

// ============================================================
// Scalar fallback
// ============================================================

#else

struct vec {
    float value;
};

inline constexpr std::size_t width = 1;
inline constexpr const char* backend_name = "scalar";

inline vec zero() noexcept
{
    return {0.0f};
}
inline vec splat(float value) noexcept { return {value}; }

inline vec load(const float* p) noexcept
{
    return {*p};
}

inline void store(float* p, vec value) noexcept
{
    *p = value.value;
}

inline vec add(vec a, vec b) noexcept
{
    return {
        a.value + b.value
    };
}

// Same selection semantics as SIMD:
//
//     a > b ? a : b
inline vec maximum(vec a, vec b) noexcept
{
    return {
        a.value > b.value
            ? a.value
            : b.value
    };
}

inline vec madd(vec a, vec b, vec c) noexcept
{
    return {
        a.value * b.value + c.value
    };
}

inline float hsum(vec value) noexcept
{
    return value.value;
}

#endif

} // namespace detail

// ============================================================
// Backend information
// ============================================================

inline constexpr const char* backend_name =
    detail::backend_name;

inline constexpr std::size_t simd_width =
    detail::width;

// In-place vector update: output[i] += weights[i] * scale.
// Output and weights must not overlap.
inline void axpy(float* output, const float* weights,
                 float scale, std::size_t n) noexcept {
    using namespace detail;
    const vec multiplier = splat(scale);
    std::size_t i = 0;
    for (; n - i >= width; i += width)
        store(output + i, madd(load(weights + i), multiplier,
                               load(output + i)));
    for (; i < n; ++i) output[i] += weights[i] * scale;
}

// ============================================================
// dot
//
// Computes:
//
//     sum(a[i] * b[i])
//
// Preconditions:
//
//   - a points to at least n readable floats.
//   - b points to at least n readable floats.
//   - a and b may overlap because both are read-only.
//
// Four independent accumulators are used to reduce the dependency
// chain on SIMD hardware.
// ============================================================

inline float dot(
    const float* a,
    const float* b,
    std::size_t n) noexcept
{
    using namespace detail;

    vec acc0 = zero();
    vec acc1 = zero();
    vec acc2 = zero();
    vec acc3 = zero();

    std::size_t i = 0;

    constexpr std::size_t accumulator_count = 4;
    constexpr std::size_t block =
        width * accumulator_count;

    // Four-way unrolled SIMD loop.
    //
    // The invariant i <= n ensures that n - i cannot underflow.
    for (; n - i >= block; i += block) {
        acc0 = madd(
            load(a + i),
            load(b + i),
            acc0
        );

        acc1 = madd(
            load(a + i + width),
            load(b + i + width),
            acc1
        );

        acc2 = madd(
            load(a + i + width * 2),
            load(b + i + width * 2),
            acc2
        );

        acc3 = madd(
            load(a + i + width * 3),
            load(b + i + width * 3),
            acc3
        );
    }

    // Combine independent accumulators.
    vec acc = add(
        add(acc0, acc1),
        add(acc2, acc3)
    );

    // Remaining complete SIMD vectors.
    for (; n - i >= width; i += width) {
        acc = madd(
            load(a + i),
            load(b + i),
            acc
        );
    }

    float sum = hsum(acc);

    // Scalar tail.
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// ============================================================
// matvec
//
// Computes:
//
//     output = weights * input
//
// Weight layout:
//
//     weights[rows][cols]
//
// Preconditions:
//
//   - weights points to at least rows * cols readable floats.
//   - input points to at least cols readable floats.
//   - output points to at least rows writable floats.
//   - output must not overlap weights or input.
//   - rows * cols must be representable by std::size_t.
// ============================================================

inline void matvec(
    const float* weights,
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols) noexcept
{
    const float* weights_row = weights;

    for (std::size_t row = 0; row < rows; ++row) {
        output[row] = dot(
            weights_row,
            input,
            cols
        );

        weights_row += cols;
    }
}

// ============================================================
// linear
//
// Fully connected neural-network layer:
//
//     output = weights * input + bias
//
// Weight layout:
//
//     weights[output_size][input_size]
//
// Preconditions:
//
//   - weights points to at least
//     output_size * input_size readable floats.
//   - bias points to at least output_size readable floats.
//   - input points to at least input_size readable floats.
//   - output points to at least output_size writable floats.
//   - output must not overlap weights, bias or input.
//   - output_size * input_size must be representable by
//     std::size_t.
// ============================================================

inline void linear(
    const float* weights,
    const float* bias,
    const float* input,
    float* output,
    std::size_t input_size,
    std::size_t output_size) noexcept
{
    const float* weights_row = weights;

    for (std::size_t row = 0;
         row < output_size;
         ++row) {
        output[row] =
            dot(
                weights_row,
                input,
                input_size
            )
            + bias[row];

        weights_row += input_size;
    }
}

// ============================================================
// relu
//
// In-place operation:
//
//     data[i] = data[i] > 0.0f ? data[i] : 0.0f
//
// Preconditions:
//
//   - data points to at least n writable floats.
//
// NaN and -0.0f are converted to +0.0f under normal IEEE
// floating-point compiler settings.
// ============================================================

inline void relu(
    float* data,
    std::size_t n) noexcept
{
    using namespace detail;

    const vec zero_value = zero();

    std::size_t i = 0;

    // Complete SIMD vectors.
    for (; n - i >= width; i += width) {
        const vec value =
            load(data + i);

        store(
            data + i,
            maximum(value, zero_value)
        );
    }

    // Scalar tail.
    for (; i < n; ++i) {
        data[i] =
            data[i] > 0.0f
                ? data[i]
                : 0.0f;
    }
}

// ============================================================
// linear_relu
//
// Logically computes:
//
//     output = ReLU(weights * input + bias)
//
// ReLU is applied immediately after computing each output neuron,
// avoiding an additional pass over the output buffer.
//
// Preconditions:
//
//   - weights points to at least
//     output_size * input_size readable floats.
//   - bias points to at least output_size readable floats.
//   - input points to at least input_size readable floats.
//   - output points to at least output_size writable floats.
//   - output must not overlap weights, bias or input.
//   - output_size * input_size must be representable by
//     std::size_t.
// ============================================================

inline void linear_relu(
    const float* weights,
    const float* bias,
    const float* input,
    float* output,
    std::size_t input_size,
    std::size_t output_size) noexcept
{
    const float* weights_row = weights;

    for (std::size_t row = 0;
         row < output_size;
         ++row) {
        const float value =
            dot(
                weights_row,
                input,
                input_size
            )
            + bias[row];

        // Same semantics as maximum(value, 0):
        //
        //     value > 0 ? value : 0
        //
        // NaN and -0.0f are converted to +0.0f.
        output[row] =
            value > 0.0f
                ? value
                : 0.0f;

        weights_row += input_size;
    }
}

} // namespace tiny_nn

#endif // TINY_NN_H_INCLUDED

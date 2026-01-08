// NEON special functions.

#include "simd_arm64.h"

#include <arm_neon.h>
#include <cmath>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

extern "C" float vek_neon_sqrt_f32(float* x, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vsqrtq_f32(vld1q_f32(&x[i])));
    }
    for (; i < len; i++) x[i] = std::sqrt(x[i]);
    return 0.0f;
}

extern "C" double vek_neon_sqrt_f64(double* x, int len) {
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vsqrtq_f64(vld1q_f64(&x[i])));
    }
    for (; i < len; i++) x[i] = std::sqrt(x[i]);
    return 0.0;
}

extern "C" float vek_neon_round_f32(float* x, int len) {
    int i = 0;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        uint32x4_t negMask = vcltq_f32(v, zero);
        float32x4_t pos = vrndmq_f32(vaddq_f32(v, half));
        float32x4_t neg = vrndpq_f32(vsubq_f32(v, half));
        vst1q_f32(&x[i], vbslq_f32(negMask, neg, pos));
    }
    for (; i < len; i++) x[i] = std::round(x[i]);
    return 0.0f;
}

extern "C" double vek_neon_round_f64(double* x, int len) {
    int i = 0;
    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t half = vdupq_n_f64(0.5);
    for (; i + 2 <= len; i += 2) {
        float64x2_t v = vld1q_f64(&x[i]);
        uint64x2_t negMask = vcltq_f64(v, zero);
        float64x2_t pos = vrndmq_f64(vaddq_f64(v, half));
        float64x2_t neg = vrndpq_f64(vsubq_f64(v, half));
        vst1q_f64(&x[i], vbslq_f64(negMask, neg, pos));
    }
    for (; i < len; i++) x[i] = std::round(x[i]);
    return 0.0;
}

extern "C" float vek_neon_floor_f32(float* x, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vrndmq_f32(vld1q_f32(&x[i])));
    }
    for (; i < len; i++) x[i] = std::floor(x[i]);
    return 0.0f;
}

extern "C" double vek_neon_floor_f64(double* x, int len) {
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vrndmq_f64(vld1q_f64(&x[i])));
    }
    for (; i < len; i++) x[i] = std::floor(x[i]);
    return 0.0;
}

extern "C" float vek_neon_ceil_f32(float* x, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vrndpq_f32(vld1q_f32(&x[i])));
    }
    for (; i < len; i++) x[i] = std::ceil(x[i]);
    return 0.0f;
}

extern "C" double vek_neon_ceil_f64(double* x, int len) {
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vrndpq_f64(vld1q_f64(&x[i])));
    }
    for (; i < len; i++) x[i] = std::ceil(x[i]);
    return 0.0;
}

extern "C" void vek_neon_pow_f32(float* x, const float* y, int len) {
#if defined(__APPLE__)
    int n = len;
    vvpowf(x, y, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::pow(x[i], y[i]);
#endif
}

extern "C" void vek_neon_pow_f64(double* x, const double* y, int len) {
#if defined(__APPLE__)
    int n = len;
    vvpow(x, y, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::pow(x[i], y[i]);
#endif
}

extern "C" void vek_neon_sin_f32(float* x, int len) {
#if defined(__APPLE__)
    int n = len;
    vvsinf(x, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::sin(x[i]);
#endif
}

extern "C" void vek_neon_cos_f32(float* x, int len) {
#if defined(__APPLE__)
    int n = len;
    vvcosf(x, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::cos(x[i]);
#endif
}

extern "C" void vek_neon_sincos_f32(float* dst_sin, float* dst_cos, const float* x, int len) {
#if defined(__APPLE__)
    int n = len;
    vvcosf(dst_cos, const_cast<float*>(x), &n);
    vvsinf(dst_sin, const_cast<float*>(x), &n);
#else
    for (int i = 0; i < len; i++) {
        dst_sin[i] = std::sin(x[i]);
        dst_cos[i] = std::cos(x[i]);
    }
#endif
}

extern "C" void vek_neon_exp_f32(float* x, int len) {
#if defined(__APPLE__)
    int n = len;
    vvexpf(x, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::exp(x[i]);
#endif
}

extern "C" void vek_neon_log_f32(float* x, int len) {
#if defined(__APPLE__)
    int n = len;
    vvlogf(x, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::log(x[i]);
#endif
}

extern "C" void vek_neon_log2_f32(float* x, int len) {
#if defined(__APPLE__)
    int n = len;
    vvlog2f(x, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::log2(x[i]);
#endif
}

extern "C" void vek_neon_log10_f32(float* x, int len) {
#if defined(__APPLE__)
    int n = len;
    vvlog10f(x, x, &n);
#else
    for (int i = 0; i < len; i++) x[i] = std::log10(x[i]);
#endif
}


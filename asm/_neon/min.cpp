// NEON min operations.

#include "simd_arm64.h"

#include <arm_neon.h>
#include <algorithm>

extern "C" float vek_neon_min_f32(const float* x, int len) {
    if (len <= 0) return 0.0f;
    float32x4_t m0 = vdupq_n_f32(x[0]);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        m0 = vminq_f32(m0, vld1q_f32(&x[i]));
    }
    float minv = vminvq_f32(m0);
    for (; i < len; i++) minv = std::min(minv, x[i]);
    return minv;
}

extern "C" double vek_neon_min_f64(const double* x, int len) {
    if (len <= 0) return 0.0;
    float64x2_t m0 = vdupq_n_f64(x[0]);
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        m0 = vminq_f64(m0, vld1q_f64(&x[i]));
    }
    double minv = vminvq_f64(m0);
    for (; i < len; i++) minv = std::min(minv, x[i]);
    return minv;
}

extern "C" void vek_neon_minimum_f32(float* x, const float* y, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vminq_f32(vld1q_f32(&x[i]), vld1q_f32(&y[i])));
    }
    for (; i < len; i++) x[i] = std::min(x[i], y[i]);
}

extern "C" void vek_neon_minimum_f64(double* x, const double* y, int len) {
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vminq_f64(vld1q_f64(&x[i]), vld1q_f64(&y[i])));
    }
    for (; i < len; i++) x[i] = std::min(x[i], y[i]);
}

extern "C" void vek_neon_minimum_number_f32(float* x, float a, int len) {
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vminq_f32(vld1q_f32(&x[i]), va));
    }
    for (; i < len; i++) x[i] = std::min(x[i], a);
}

extern "C" void vek_neon_minimum_number_f64(double* x, double a, int len) {
    float64x2_t va = vdupq_n_f64(a);
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vminq_f64(vld1q_f64(&x[i]), va));
    }
    for (; i < len; i++) x[i] = std::min(x[i], a);
}


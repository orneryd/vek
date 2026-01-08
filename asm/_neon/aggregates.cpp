// NEON aggregate operations.

#include "simd_arm64.h"

#include <arm_neon.h>

extern "C" float vek_neon_sum_f32(const float* x, int len) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        sum0 = vaddq_f32(sum0, vld1q_f32(&x[i]));
        sum1 = vaddq_f32(sum1, vld1q_f32(&x[i + 4]));
    }
    for (; i + 4 <= len; i += 4) {
        sum0 = vaddq_f32(sum0, vld1q_f32(&x[i]));
    }
    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < len; i++) sum += x[i];
    return sum;
}

extern "C" double vek_neon_sum_f64(const double* x, int len) {
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        sum0 = vaddq_f64(sum0, vld1q_f64(&x[i]));
        sum1 = vaddq_f64(sum1, vld1q_f64(&x[i + 2]));
    }
    for (; i + 2 <= len; i += 2) {
        sum0 = vaddq_f64(sum0, vld1q_f64(&x[i]));
    }
    double sum = vaddvq_f64(vaddq_f64(sum0, sum1));
    for (; i < len; i++) sum += x[i];
    return sum;
}

extern "C" float vek_neon_prod_f32(const float* x, int len) {
    float32x4_t prod0 = vdupq_n_f32(1.0f);
    float32x4_t prod1 = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        prod0 = vmulq_f32(prod0, vld1q_f32(&x[i]));
        prod1 = vmulq_f32(prod1, vld1q_f32(&x[i + 4]));
    }
    for (; i + 4 <= len; i += 4) {
        prod0 = vmulq_f32(prod0, vld1q_f32(&x[i]));
    }
    float32x4_t prod = vmulq_f32(prod0, prod1);
    float out = vgetq_lane_f32(prod, 0) * vgetq_lane_f32(prod, 1) * vgetq_lane_f32(prod, 2) * vgetq_lane_f32(prod, 3);
    for (; i < len; i++) out *= x[i];
    return out;
}

extern "C" double vek_neon_prod_f64(const double* x, int len) {
    float64x2_t prod0 = vdupq_n_f64(1.0);
    float64x2_t prod1 = vdupq_n_f64(1.0);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        prod0 = vmulq_f64(prod0, vld1q_f64(&x[i]));
        prod1 = vmulq_f64(prod1, vld1q_f64(&x[i + 2]));
    }
    for (; i + 2 <= len; i += 2) {
        prod0 = vmulq_f64(prod0, vld1q_f64(&x[i]));
    }
    float64x2_t prod = vmulq_f64(prod0, prod1);
    double out = vgetq_lane_f64(prod, 0) * vgetq_lane_f64(prod, 1);
    for (; i < len; i++) out *= x[i];
    return out;
}


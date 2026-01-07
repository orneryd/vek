// ARM64 NEON SIMD implementation for float32 vector operations

#include "neon_simd_arm64.h"

#include <arm_neon.h>
#include <cmath>

extern "C" float vek_neon_dot_product_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 8 <= len; i += 8) {
        float32x4_t va0 = vld1q_f32(&a[i]);
        float32x4_t vb0 = vld1q_f32(&b[i]);
        sum0 = vfmaq_f32(sum0, va0, vb0);

        float32x4_t va1 = vld1q_f32(&a[i + 4]);
        float32x4_t vb1 = vld1q_f32(&b[i + 4]);
        sum1 = vfmaq_f32(sum1, va1, vb1);
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        sum0 = vfmaq_f32(sum0, va, vb);
    }

    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

extern "C" float vek_neon_norm_f32(const float* VEK_RESTRICT v, int len) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 8 <= len; i += 8) {
        float32x4_t vv0 = vld1q_f32(&v[i]);
        sum0 = vfmaq_f32(sum0, vv0, vv0);

        float32x4_t vv1 = vld1q_f32(&v[i + 4]);
        sum1 = vfmaq_f32(sum1, vv1, vv1);
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t vv = vld1q_f32(&v[i]);
        sum0 = vfmaq_f32(sum0, vv, vv);
    }

    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < len; i++) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

extern "C" float vek_neon_distance_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 8 <= len; i += 8) {
        float32x4_t va0 = vld1q_f32(&a[i]);
        float32x4_t vb0 = vld1q_f32(&b[i]);
        float32x4_t d0 = vsubq_f32(va0, vb0);
        sum0 = vfmaq_f32(sum0, d0, d0);

        float32x4_t va1 = vld1q_f32(&a[i + 4]);
        float32x4_t vb1 = vld1q_f32(&b[i + 4]);
        float32x4_t d1 = vsubq_f32(va1, vb1);
        sum1 = vfmaq_f32(sum1, d1, d1);
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t d = vsubq_f32(va, vb);
        sum0 = vfmaq_f32(sum0, d, d);
    }

    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < len; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

extern "C" float vek_neon_cosine_similarity_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len) {
    float32x4_t dot0 = vdupq_n_f32(0.0f);
    float32x4_t dot1 = vdupq_n_f32(0.0f);
    float32x4_t aa0 = vdupq_n_f32(0.0f);
    float32x4_t aa1 = vdupq_n_f32(0.0f);
    float32x4_t bb0 = vdupq_n_f32(0.0f);
    float32x4_t bb1 = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 8 <= len; i += 8) {
        float32x4_t va0 = vld1q_f32(&a[i]);
        float32x4_t vb0 = vld1q_f32(&b[i]);
        dot0 = vfmaq_f32(dot0, va0, vb0);
        aa0 = vfmaq_f32(aa0, va0, va0);
        bb0 = vfmaq_f32(bb0, vb0, vb0);

        float32x4_t va1 = vld1q_f32(&a[i + 4]);
        float32x4_t vb1 = vld1q_f32(&b[i + 4]);
        dot1 = vfmaq_f32(dot1, va1, vb1);
        aa1 = vfmaq_f32(aa1, va1, va1);
        bb1 = vfmaq_f32(bb1, vb1, vb1);
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        dot0 = vfmaq_f32(dot0, va, vb);
        aa0 = vfmaq_f32(aa0, va, va);
        bb0 = vfmaq_f32(bb0, vb, vb);
    }

    float dot = vaddvq_f32(vaddq_f32(dot0, dot1));
    float sum_a = vaddvq_f32(vaddq_f32(aa0, aa1));
    float sum_b = vaddvq_f32(vaddq_f32(bb0, bb1));
    for (; i < len; i++) {
        dot += a[i] * b[i];
        sum_a += a[i] * a[i];
        sum_b += b[i] * b[i];
    }

    return dot / std::sqrt(sum_a * sum_b);
}

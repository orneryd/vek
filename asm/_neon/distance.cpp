// NEON distance operations.

#include "simd_arm64.h"

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

extern "C" double vek_neon_dot_product_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len) {
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        float64x2_t va0 = vld1q_f64(&a[i]);
        float64x2_t vb0 = vld1q_f64(&b[i]);
        sum0 = vfmaq_f64(sum0, va0, vb0);

        float64x2_t va1 = vld1q_f64(&a[i + 2]);
        float64x2_t vb1 = vld1q_f64(&b[i + 2]);
        sum1 = vfmaq_f64(sum1, va1, vb1);
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        sum0 = vfmaq_f64(sum0, va, vb);
    }

    double sum = vaddvq_f64(vaddq_f64(sum0, sum1));
    for (; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

extern "C" double vek_neon_norm_f64(const double* VEK_RESTRICT v, int len) {
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        float64x2_t vv0 = vld1q_f64(&v[i]);
        sum0 = vfmaq_f64(sum0, vv0, vv0);

        float64x2_t vv1 = vld1q_f64(&v[i + 2]);
        sum1 = vfmaq_f64(sum1, vv1, vv1);
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t vv = vld1q_f64(&v[i]);
        sum0 = vfmaq_f64(sum0, vv, vv);
    }

    double sum = vaddvq_f64(vaddq_f64(sum0, sum1));
    for (; i < len; i++) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

extern "C" double vek_neon_distance_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len) {
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        float64x2_t va0 = vld1q_f64(&a[i]);
        float64x2_t vb0 = vld1q_f64(&b[i]);
        float64x2_t d0 = vsubq_f64(va0, vb0);
        sum0 = vfmaq_f64(sum0, d0, d0);

        float64x2_t va1 = vld1q_f64(&a[i + 2]);
        float64x2_t vb1 = vld1q_f64(&b[i + 2]);
        float64x2_t d1 = vsubq_f64(va1, vb1);
        sum1 = vfmaq_f64(sum1, d1, d1);
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t d = vsubq_f64(va, vb);
        sum0 = vfmaq_f64(sum0, d, d);
    }

    double sum = vaddvq_f64(vaddq_f64(sum0, sum1));
    for (; i < len; i++) {
        const double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

extern "C" double vek_neon_manhattan_norm_f64(const double* VEK_RESTRICT x, int len) {
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        float64x2_t v0 = vabsq_f64(vld1q_f64(&x[i]));
        sum0 = vaddq_f64(sum0, v0);
        float64x2_t v1 = vabsq_f64(vld1q_f64(&x[i + 2]));
        sum1 = vaddq_f64(sum1, v1);
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t v0 = vabsq_f64(vld1q_f64(&x[i]));
        sum0 = vaddq_f64(sum0, v0);
    }

    double sum = vaddvq_f64(vaddq_f64(sum0, sum1));
    for (; i < len; i++) {
        sum += std::abs(x[i]);
    }
    return sum;
}

extern "C" float vek_neon_manhattan_norm_f32(const float* VEK_RESTRICT x, int len) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 8 <= len; i += 8) {
        float32x4_t v0 = vabsq_f32(vld1q_f32(&x[i]));
        sum0 = vaddq_f32(sum0, v0);
        float32x4_t v1 = vabsq_f32(vld1q_f32(&x[i + 4]));
        sum1 = vaddq_f32(sum1, v1);
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t v0 = vabsq_f32(vld1q_f32(&x[i]));
        sum0 = vaddq_f32(sum0, v0);
    }

    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < len; i++) {
        sum += std::abs(x[i]);
    }
    return sum;
}

extern "C" double vek_neon_manhattan_distance_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len) {
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        float64x2_t d0 = vabsq_f64(vsubq_f64(vld1q_f64(&a[i]), vld1q_f64(&b[i])));
        sum0 = vaddq_f64(sum0, d0);
        float64x2_t d1 = vabsq_f64(vsubq_f64(vld1q_f64(&a[i + 2]), vld1q_f64(&b[i + 2])));
        sum1 = vaddq_f64(sum1, d1);
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t d0 = vabsq_f64(vsubq_f64(vld1q_f64(&a[i]), vld1q_f64(&b[i])));
        sum0 = vaddq_f64(sum0, d0);
    }

    double sum = vaddvq_f64(vaddq_f64(sum0, sum1));
    for (; i < len; i++) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

extern "C" float vek_neon_manhattan_distance_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 8 <= len; i += 8) {
        float32x4_t d0 = vabsq_f32(vsubq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
        sum0 = vaddq_f32(sum0, d0);
        float32x4_t d1 = vabsq_f32(vsubq_f32(vld1q_f32(&a[i + 4]), vld1q_f32(&b[i + 4])));
        sum1 = vaddq_f32(sum1, d1);
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t d0 = vabsq_f32(vsubq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
        sum0 = vaddq_f32(sum0, d0);
    }

    float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < len; i++) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

extern "C" double vek_neon_cosine_similarity_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len) {
    float64x2_t dot0 = vdupq_n_f64(0.0);
    float64x2_t dot1 = vdupq_n_f64(0.0);
    float64x2_t aa0 = vdupq_n_f64(0.0);
    float64x2_t aa1 = vdupq_n_f64(0.0);
    float64x2_t bb0 = vdupq_n_f64(0.0);
    float64x2_t bb1 = vdupq_n_f64(0.0);
    int i = 0;

    for (; i + 4 <= len; i += 4) {
        float64x2_t va0 = vld1q_f64(&a[i]);
        float64x2_t vb0 = vld1q_f64(&b[i]);
        dot0 = vfmaq_f64(dot0, va0, vb0);
        aa0 = vfmaq_f64(aa0, va0, va0);
        bb0 = vfmaq_f64(bb0, vb0, vb0);

        float64x2_t va1 = vld1q_f64(&a[i + 2]);
        float64x2_t vb1 = vld1q_f64(&b[i + 2]);
        dot1 = vfmaq_f64(dot1, va1, vb1);
        aa1 = vfmaq_f64(aa1, va1, va1);
        bb1 = vfmaq_f64(bb1, vb1, vb1);
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        dot0 = vfmaq_f64(dot0, va, vb);
        aa0 = vfmaq_f64(aa0, va, va);
        bb0 = vfmaq_f64(bb0, vb, vb);
    }

    double dot = vaddvq_f64(vaddq_f64(dot0, dot1));
    double sum_a = vaddvq_f64(vaddq_f64(aa0, aa1));
    double sum_b = vaddvq_f64(vaddq_f64(bb0, bb1));
    for (; i < len; i++) {
        dot += a[i] * b[i];
        sum_a += a[i] * a[i];
        sum_b += b[i] * b[i];
    }

    return dot / std::sqrt(sum_a * sum_b);
}


// NEON arithmetic operations (in-place).

#include "simd_arm64.h"

#include <arm_neon.h>
#include <cmath>

extern "C" void vek_neon_add_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t x0 = vld1q_f32(&x[i]);
        float32x4_t y0 = vld1q_f32(&y[i]);
        vst1q_f32(&x[i], vaddq_f32(x0, y0));

        float32x4_t x1 = vld1q_f32(&x[i + 4]);
        float32x4_t y1 = vld1q_f32(&y[i + 4]);
        vst1q_f32(&x[i + 4], vaddq_f32(x1, y1));
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t x0 = vld1q_f32(&x[i]);
        float32x4_t y0 = vld1q_f32(&y[i]);
        vst1q_f32(&x[i], vaddq_f32(x0, y0));
    }
    for (; i < len; i++) {
        x[i] += y[i];
    }
}

extern "C" void vek_neon_add_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float64x2_t x0 = vld1q_f64(&x[i]);
        float64x2_t y0 = vld1q_f64(&y[i]);
        vst1q_f64(&x[i], vaddq_f64(x0, y0));

        float64x2_t x1 = vld1q_f64(&x[i + 2]);
        float64x2_t y1 = vld1q_f64(&y[i + 2]);
        vst1q_f64(&x[i + 2], vaddq_f64(x1, y1));
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t x0 = vld1q_f64(&x[i]);
        float64x2_t y0 = vld1q_f64(&y[i]);
        vst1q_f64(&x[i], vaddq_f64(x0, y0));
    }
    for (; i < len; i++) {
        x[i] += y[i];
    }
}

extern "C" void vek_neon_sub_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t x0 = vld1q_f32(&x[i]);
        float32x4_t y0 = vld1q_f32(&y[i]);
        vst1q_f32(&x[i], vsubq_f32(x0, y0));
        float32x4_t x1 = vld1q_f32(&x[i + 4]);
        float32x4_t y1 = vld1q_f32(&y[i + 4]);
        vst1q_f32(&x[i + 4], vsubq_f32(x1, y1));
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t x0 = vld1q_f32(&x[i]);
        float32x4_t y0 = vld1q_f32(&y[i]);
        vst1q_f32(&x[i], vsubq_f32(x0, y0));
    }
    for (; i < len; i++) {
        x[i] -= y[i];
    }
}

extern "C" void vek_neon_sub_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float64x2_t x0 = vld1q_f64(&x[i]);
        float64x2_t y0 = vld1q_f64(&y[i]);
        vst1q_f64(&x[i], vsubq_f64(x0, y0));
        float64x2_t x1 = vld1q_f64(&x[i + 2]);
        float64x2_t y1 = vld1q_f64(&y[i + 2]);
        vst1q_f64(&x[i + 2], vsubq_f64(x1, y1));
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t x0 = vld1q_f64(&x[i]);
        float64x2_t y0 = vld1q_f64(&y[i]);
        vst1q_f64(&x[i], vsubq_f64(x0, y0));
    }
    for (; i < len; i++) {
        x[i] -= y[i];
    }
}

extern "C" void vek_neon_mul_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t x0 = vld1q_f32(&x[i]);
        float32x4_t y0 = vld1q_f32(&y[i]);
        vst1q_f32(&x[i], vmulq_f32(x0, y0));
        float32x4_t x1 = vld1q_f32(&x[i + 4]);
        float32x4_t y1 = vld1q_f32(&y[i + 4]);
        vst1q_f32(&x[i + 4], vmulq_f32(x1, y1));
    }
    for (; i + 4 <= len; i += 4) {
        float32x4_t x0 = vld1q_f32(&x[i]);
        float32x4_t y0 = vld1q_f32(&y[i]);
        vst1q_f32(&x[i], vmulq_f32(x0, y0));
    }
    for (; i < len; i++) {
        x[i] *= y[i];
    }
}

extern "C" void vek_neon_mul_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float64x2_t x0 = vld1q_f64(&x[i]);
        float64x2_t y0 = vld1q_f64(&y[i]);
        vst1q_f64(&x[i], vmulq_f64(x0, y0));
        float64x2_t x1 = vld1q_f64(&x[i + 2]);
        float64x2_t y1 = vld1q_f64(&y[i + 2]);
        vst1q_f64(&x[i + 2], vmulq_f64(x1, y1));
    }
    for (; i + 2 <= len; i += 2) {
        float64x2_t x0 = vld1q_f64(&x[i]);
        float64x2_t y0 = vld1q_f64(&y[i]);
        vst1q_f64(&x[i], vmulq_f64(x0, y0));
    }
    for (; i < len; i++) {
        x[i] *= y[i];
    }
}

extern "C" void vek_neon_div_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t x0 = vld1q_f32(&x[i]);
        float32x4_t y0 = vld1q_f32(&y[i]);
        vst1q_f32(&x[i], vdivq_f32(x0, y0));
    }
    for (; i < len; i++) {
        x[i] /= y[i];
    }
}

extern "C" void vek_neon_div_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len) {
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        float64x2_t x0 = vld1q_f64(&x[i]);
        float64x2_t y0 = vld1q_f64(&y[i]);
        vst1q_f64(&x[i], vdivq_f64(x0, y0));
    }
    for (; i < len; i++) {
        x[i] /= y[i];
    }
}

extern "C" void vek_neon_add_number_f32(float* x, float a, int len) {
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        vst1q_f32(&x[i], vaddq_f32(vld1q_f32(&x[i]), va));
        vst1q_f32(&x[i + 4], vaddq_f32(vld1q_f32(&x[i + 4]), va));
    }
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vaddq_f32(vld1q_f32(&x[i]), va));
    }
    for (; i < len; i++) {
        x[i] += a;
    }
}

extern "C" void vek_neon_add_number_f64(double* x, double a, int len) {
    float64x2_t va = vdupq_n_f64(a);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f64(&x[i], vaddq_f64(vld1q_f64(&x[i]), va));
        vst1q_f64(&x[i + 2], vaddq_f64(vld1q_f64(&x[i + 2]), va));
    }
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vaddq_f64(vld1q_f64(&x[i]), va));
    }
    for (; i < len; i++) {
        x[i] += a;
    }
}

extern "C" void vek_neon_sub_number_f32(float* x, float a, int len) {
    vek_neon_add_number_f32(x, -a, len);
}
extern "C" void vek_neon_sub_number_f64(double* x, double a, int len) {
    vek_neon_add_number_f64(x, -a, len);
}

extern "C" void vek_neon_mul_number_f32(float* x, float a, int len) {
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        vst1q_f32(&x[i], vmulq_f32(vld1q_f32(&x[i]), va));
        vst1q_f32(&x[i + 4], vmulq_f32(vld1q_f32(&x[i + 4]), va));
    }
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vmulq_f32(vld1q_f32(&x[i]), va));
    }
    for (; i < len; i++) {
        x[i] *= a;
    }
}

extern "C" void vek_neon_mul_number_f64(double* x, double a, int len) {
    float64x2_t va = vdupq_n_f64(a);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f64(&x[i], vmulq_f64(vld1q_f64(&x[i]), va));
        vst1q_f64(&x[i + 2], vmulq_f64(vld1q_f64(&x[i + 2]), va));
    }
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vmulq_f64(vld1q_f64(&x[i]), va));
    }
    for (; i < len; i++) {
        x[i] *= a;
    }
}

extern "C" void vek_neon_div_number_f32(float* x, float a, int len) {
    vek_neon_mul_number_f32(x, 1.0f / a, len);
}
extern "C" void vek_neon_div_number_f64(double* x, double a, int len) {
    vek_neon_mul_number_f64(x, 1.0 / a, len);
}

extern "C" void vek_neon_abs_f32(float* x, int len) {
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        vst1q_f32(&x[i], vabsq_f32(vld1q_f32(&x[i])));
        vst1q_f32(&x[i + 4], vabsq_f32(vld1q_f32(&x[i + 4])));
    }
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vabsq_f32(vld1q_f32(&x[i])));
    }
    for (; i < len; i++) {
        x[i] = std::abs(x[i]);
    }
}

extern "C" void vek_neon_abs_f64(double* x, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f64(&x[i], vabsq_f64(vld1q_f64(&x[i])));
        vst1q_f64(&x[i + 2], vabsq_f64(vld1q_f64(&x[i + 2])));
    }
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vabsq_f64(vld1q_f64(&x[i])));
    }
    for (; i < len; i++) {
        x[i] = std::abs(x[i]);
    }
}

extern "C" void vek_neon_neg_f32(float* x, int len) {
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        vst1q_f32(&x[i], vnegq_f32(vld1q_f32(&x[i])));
        vst1q_f32(&x[i + 4], vnegq_f32(vld1q_f32(&x[i + 4])));
    }
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(&x[i], vnegq_f32(vld1q_f32(&x[i])));
    }
    for (; i < len; i++) {
        x[i] = -x[i];
    }
}

extern "C" void vek_neon_neg_f64(double* x, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        vst1q_f64(&x[i], vnegq_f64(vld1q_f64(&x[i])));
        vst1q_f64(&x[i + 2], vnegq_f64(vld1q_f64(&x[i + 2])));
    }
    for (; i + 2 <= len; i += 2) {
        vst1q_f64(&x[i], vnegq_f64(vld1q_f64(&x[i])));
    }
    for (; i < len; i++) {
        x[i] = -x[i];
    }
}

extern "C" void vek_neon_inv_f32(float* x, int len) {
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v0 = vld1q_f32(&x[i]);
        vst1q_f32(&x[i], vdivq_f32(vdupq_n_f32(1.0f), v0));
    }
    for (; i < len; i++) {
        x[i] = 1.0f / x[i];
    }
}

extern "C" void vek_neon_inv_f64(double* x, int len) {
    int i = 0;
    for (; i + 2 <= len; i += 2) {
        float64x2_t v0 = vld1q_f64(&x[i]);
        vst1q_f64(&x[i], vdivq_f64(vdupq_n_f64(1.0), v0));
    }
    for (; i < len; i++) {
        x[i] = 1.0 / x[i];
    }
}


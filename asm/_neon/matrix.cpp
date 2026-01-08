// NEON matrix operations.

#include "simd_arm64.h"

#include <arm_neon.h>

extern "C" void vek_neon_mat4mul_f32(float* dst, const float* x, const float* y) {
    for (int i = 0; i < 4; i++) {
        float32x4_t xrow = vld1q_f32(&x[i * 4]);
        for (int j = 0; j < 4; j++) {
            float32x4_t ycol = {y[j], y[4 + j], y[8 + j], y[12 + j]};
            float32x4_t prod = vmulq_f32(xrow, ycol);
            dst[i * 4 + j] = vaddvq_f32(prod);
        }
    }
}

extern "C" void vek_neon_mat4mul_f64(double* dst, const double* x, const double* y) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            sum += x[i * 4 + 0] * y[0 * 4 + j];
            sum += x[i * 4 + 1] * y[1 * 4 + j];
            sum += x[i * 4 + 2] * y[2 * 4 + j];
            sum += x[i * 4 + 3] * y[3 * 4 + j];
            dst[i * 4 + j] = sum;
        }
    }
}

extern "C" void vek_neon_matmul_f32(float* dst, const float* x, const float* y, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        const float* xrow = &x[i * n];
        float* dstrow = &dst[i * p];
        for (int k = 0; k < n; k++) {
            const float a = xrow[k];
            float32x4_t va = vdupq_n_f32(a);
            const float* yrow = &y[k * p];
            int j = 0;
            for (; j + 4 <= p; j += 4) {
                float32x4_t d = vld1q_f32(&dstrow[j]);
                float32x4_t yy = vld1q_f32(&yrow[j]);
                d = vfmaq_f32(d, yy, va);
                vst1q_f32(&dstrow[j], d);
            }
            for (; j < p; j++) {
                dstrow[j] += a * yrow[j];
            }
        }
    }
}

extern "C" void vek_neon_matmul_f64(double* dst, const double* x, const double* y, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        const double* xrow = &x[i * n];
        double* dstrow = &dst[i * p];
        for (int k = 0; k < n; k++) {
            const double a = xrow[k];
            float64x2_t va = vdupq_n_f64(a);
            const double* yrow = &y[k * p];
            int j = 0;
            for (; j + 2 <= p; j += 2) {
                float64x2_t d = vld1q_f64(&dstrow[j]);
                float64x2_t yy = vld1q_f64(&yrow[j]);
                d = vfmaq_f64(d, yy, va);
                vst1q_f64(&dstrow[j], d);
            }
            for (; j < p; j++) {
                dstrow[j] += a * yrow[j];
            }
        }
    }
}

extern "C" void vek_neon_matmulvec_f32(float* dst, const float* x, const float* y, int m, int n) {
    for (int i = 0; i < m; i++) {
        const float* xrow = &x[i * n];
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        int k = 0;
        for (; k + 8 <= n; k += 8) {
            sum0 = vfmaq_f32(sum0, vld1q_f32(&xrow[k]), vld1q_f32(&y[k]));
            sum1 = vfmaq_f32(sum1, vld1q_f32(&xrow[k + 4]), vld1q_f32(&y[k + 4]));
        }
        for (; k + 4 <= n; k += 4) {
            sum0 = vfmaq_f32(sum0, vld1q_f32(&xrow[k]), vld1q_f32(&y[k]));
        }
        float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
        for (; k < n; k++) sum += xrow[k] * y[k];
        dst[i] += sum;
    }
}

extern "C" void vek_neon_matmulvec_f64(double* dst, const double* x, const double* y, int m, int n) {
    for (int i = 0; i < m; i++) {
        const double* xrow = &x[i * n];
        float64x2_t sum0 = vdupq_n_f64(0.0);
        float64x2_t sum1 = vdupq_n_f64(0.0);
        int k = 0;
        for (; k + 4 <= n; k += 4) {
            sum0 = vfmaq_f64(sum0, vld1q_f64(&xrow[k]), vld1q_f64(&y[k]));
            sum1 = vfmaq_f64(sum1, vld1q_f64(&xrow[k + 2]), vld1q_f64(&y[k + 2]));
        }
        for (; k + 2 <= n; k += 2) {
            sum0 = vfmaq_f64(sum0, vld1q_f64(&xrow[k]), vld1q_f64(&y[k]));
        }
        double sum = vaddvq_f64(vaddq_f64(sum0, sum1));
        for (; k < n; k++) sum += xrow[k] * y[k];
        dst[i] += sum;
    }
}


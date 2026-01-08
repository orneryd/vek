// ARM64 NEON SIMD C interface

#ifndef VEK_NEON_SIMD_ARM64_H
#define VEK_NEON_SIMD_ARM64_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define VEK_RESTRICT __restrict
#else
#define VEK_RESTRICT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Arithmetic (in-place on x)
void vek_neon_add_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len);
void vek_neon_add_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len);
void vek_neon_add_number_f64(double* x, double a, int len);
void vek_neon_add_number_f32(float* x, float a, int len);

void vek_neon_sub_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len);
void vek_neon_sub_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len);
void vek_neon_sub_number_f64(double* x, double a, int len);
void vek_neon_sub_number_f32(float* x, float a, int len);

void vek_neon_mul_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len);
void vek_neon_mul_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len);
void vek_neon_mul_number_f64(double* x, double a, int len);
void vek_neon_mul_number_f32(float* x, float a, int len);

void vek_neon_div_f64(double* VEK_RESTRICT x, const double* VEK_RESTRICT y, int len);
void vek_neon_div_f32(float* VEK_RESTRICT x, const float* VEK_RESTRICT y, int len);
void vek_neon_div_number_f64(double* x, double a, int len);
void vek_neon_div_number_f32(float* x, float a, int len);

void vek_neon_abs_f64(double* x, int len);
void vek_neon_abs_f32(float* x, int len);
void vek_neon_neg_f64(double* x, int len);
void vek_neon_neg_f32(float* x, int len);
void vek_neon_inv_f64(double* x, int len);
void vek_neon_inv_f32(float* x, int len);

// Aggregates
double vek_neon_sum_f64(const double* x, int len);
float vek_neon_sum_f32(const float* x, int len);
double vek_neon_prod_f64(const double* x, int len);
float vek_neon_prod_f32(const float* x, int len);

// Distance
double vek_neon_dot_product_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len);
float vek_neon_dot_product_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len);
double vek_neon_norm_f64(const double* v, int len);
float vek_neon_norm_f32(const float* v, int len);
double vek_neon_distance_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len);
float vek_neon_distance_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len);
double vek_neon_manhattan_norm_f64(const double* x, int len);
float vek_neon_manhattan_norm_f32(const float* x, int len);
double vek_neon_manhattan_distance_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len);
float vek_neon_manhattan_distance_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len);
double vek_neon_cosine_similarity_f64(const double* VEK_RESTRICT a, const double* VEK_RESTRICT b, int len);
float vek_neon_cosine_similarity_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len);

// Matrix
void vek_neon_mat4mul_f64(double* dst, const double* x, const double* y);
void vek_neon_mat4mul_f32(float* dst, const float* x, const float* y);
void vek_neon_matmul_f64(double* dst, const double* x, const double* y, int m, int n, int p);
void vek_neon_matmul_f32(float* dst, const float* x, const float* y, int m, int n, int p);
void vek_neon_matmulvec_f64(double* dst, const double* x, const double* y, int m, int n);
void vek_neon_matmulvec_f32(float* dst, const float* x, const float* y, int m, int n);

// Special (in-place)
double vek_neon_sqrt_f64(double* x, int len);
float vek_neon_sqrt_f32(float* x, int len);
double vek_neon_round_f64(double* x, int len);
float vek_neon_round_f32(float* x, int len);
double vek_neon_floor_f64(double* x, int len);
float vek_neon_floor_f32(float* x, int len);
double vek_neon_ceil_f64(double* x, int len);
float vek_neon_ceil_f32(float* x, int len);

void vek_neon_pow_f64(double* x, const double* y, int len);
void vek_neon_pow_f32(float* x, const float* y, int len);
void vek_neon_sin_f32(float* x, int len);
void vek_neon_cos_f32(float* x, int len);
void vek_neon_sincos_f32(float* dst_sin, float* dst_cos, const float* x, int len);
void vek_neon_exp_f32(float* x, int len);
void vek_neon_log_f32(float* x, int len);
void vek_neon_log2_f32(float* x, int len);
void vek_neon_log10_f32(float* x, int len);

// Min/Max
double vek_neon_min_f64(const double* x, int len);
float vek_neon_min_f32(const float* x, int len);
void vek_neon_minimum_f64(double* x, const double* y, int len);
void vek_neon_minimum_f32(float* x, const float* y, int len);
void vek_neon_minimum_number_f64(double* x, double a, int len);
void vek_neon_minimum_number_f32(float* x, float a, int len);

double vek_neon_max_f64(const double* x, int len);
float vek_neon_max_f32(const float* x, int len);
void vek_neon_maximum_f64(double* x, const double* y, int len);
void vek_neon_maximum_f32(float* x, const float* y, int len);
void vek_neon_maximum_number_f64(double* x, double a, int len);
void vek_neon_maximum_number_f32(float* x, float a, int len);

#ifdef __cplusplus
}
#endif

#endif // VEK_NEON_SIMD_ARM64_H

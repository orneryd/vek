// ARM64 NEON SIMD C interface

#ifndef VEK_NEON_SIMD_ARM64_H
#define VEK_NEON_SIMD_ARM64_H

#if defined(__GNUC__) || defined(__clang__)
#define VEK_RESTRICT __restrict
#else
#define VEK_RESTRICT
#endif

#ifdef __cplusplus
extern "C" {
#endif

float vek_neon_dot_product_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len);
float vek_neon_norm_f32(const float* VEK_RESTRICT v, int len);
float vek_neon_distance_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len);
float vek_neon_cosine_similarity_f32(const float* VEK_RESTRICT a, const float* VEK_RESTRICT b, int len);

#ifdef __cplusplus
}
#endif

#endif // VEK_NEON_SIMD_ARM64_H

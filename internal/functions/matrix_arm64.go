//go:build arm64

package functions

func MatMul_Parallel_NEON_F64(dst, x, y []float64, m, n, p int) {
	matMulParallel(dst, x, y, m, n, p, MatMulVec_NEON_F64, MatMul_NEON_F64)
}

func MatMul_Parallel_NEON_F32(dst, x, y []float32, m, n, p int) {
	matMulParallel(dst, x, y, m, n, p, MatMulVec_NEON_F32, MatMul_NEON_F32)
}

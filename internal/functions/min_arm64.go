//go:build arm64

package functions

func ArgMin_NEON_F64(x []float64) int {
	min := Min_NEON_F64(x)
	idx := Find_NEON_F64(x, min)
	if idx == len(x) {
		return -1
	}
	return idx
}

func ArgMin_NEON_F32(x []float32) int {
	min := Min_NEON_F32(x)
	idx := Find_NEON_F32(x, min)
	if idx == len(x) {
		return -1
	}
	return idx
}

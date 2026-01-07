//go:build !arm64 || !cgo || nosimd

package functions

var HasNEON bool = false
var UseNEON bool = false

func Dot_NEON_F32(x, y []float32) float32 {
	return Dot_Go(x, y)
}

func Norm_NEON_F32(x []float32) float32 {
	return Norm_Go_F32(x)
}

func Distance_NEON_F32(x, y []float32) float32 {
	return Distance_Go_F32(x, y)
}

func CosineSimilarity_NEON_F32(x, y []float32) float32 {
	return CosineSimilarity_Go_F32(x, y)
}

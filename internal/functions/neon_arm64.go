//go:build arm64 && cgo && !nosimd

package functions

import (
	"github.com/viterin/vek/internal/functions/_neon"
	"golang.org/x/sys/cpu"
)

var HasNEON bool = cpu.ARM64.HasASIMD
var UseNEON bool = HasNEON

func Dot_NEON_F32(x, y []float32) float32 {
	return neon.DotF32(x, y)
}

func Norm_NEON_F32(x []float32) float32 {
	return neon.NormF32(x)
}

func Distance_NEON_F32(x, y []float32) float32 {
	return neon.DistanceF32(x, y)
}

func CosineSimilarity_NEON_F32(x, y []float32) float32 {
	return neon.CosineSimilarityF32(x, y)
}


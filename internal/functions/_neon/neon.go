//go:build arm64 && cgo && !nosimd

package neon

/*
#cgo CXXFLAGS: -O3 -std=c++11
#include "neon_simd_arm64.h"
*/
import "C"
import "unsafe"

func float32PtrOrNil(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

func DotF32(x, y []float32) float32 {
	return float32(C.vek_neon_dot_product_f32(
		float32PtrOrNil(x),
		float32PtrOrNil(y),
		C.int(len(x)),
	))
}

func NormF32(x []float32) float32 {
	return float32(C.vek_neon_norm_f32(
		float32PtrOrNil(x),
		C.int(len(x)),
	))
}

func DistanceF32(x, y []float32) float32 {
	return float32(C.vek_neon_distance_f32(
		float32PtrOrNil(x),
		float32PtrOrNil(y),
		C.int(len(x)),
	))
}

func CosineSimilarityF32(x, y []float32) float32 {
	return float32(C.vek_neon_cosine_similarity_f32(
		float32PtrOrNil(x),
		float32PtrOrNil(y),
		C.int(len(x)),
	))
}

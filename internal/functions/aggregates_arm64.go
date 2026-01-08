//go:build arm64

package functions

import (
	"github.com/viterin/partial"
	"golang.org/x/exp/slices"
	"math"
)

func Mean_NEON_F64(x []float64) float64 {
	return Sum_NEON_F64(x) / float64(len(x))
}

func Mean_NEON_F32(x []float32) float32 {
	return Sum_NEON_F32(x) / float32(len(x))
}

func Median_NEON_F64(x []float64) float64 {
	if len(x)%2 == 1 {
		x = slices.Clone(x)
		i := len(x) / 2
		partial.TopK(x, i+1)
		return x[i]
	}
	return Quantile_NEON_F64(x, 0.5)
}

func Median_NEON_F32(x []float32) float32 {
	if len(x)%2 == 1 {
		x = slices.Clone(x)
		i := len(x) / 2
		partial.TopK(x, i+1)
		return x[i]
	}
	return Quantile_NEON_F32(x, 0.5)
}

func Quantile_NEON_F64(x []float64, q float64) float64 {
	if len(x) == 1 {
		return x[0]
	}
	if q == 0 {
		return Min_NEON_F64(x)
	}
	if q == 1 {
		return Max_NEON_F64(x)
	}
	x = slices.Clone(x)
	f := float64(len(x)-1) * q
	i := int(math.Floor(f))
	if q < 0.5 {
		partial.TopK(x, i+2)
		a := Max_NEON_F64(x[:i+1])
		b := x[i+1]
		return a + (b-a)*(f-float64(i))
	}
	partial.TopK(x, i+1)
	a := x[i]
	b := Min_NEON_F64(x[i+1:])
	return a + (b-a)*(f-float64(i))
}

func Quantile_NEON_F32(x []float32, q float32) float32 {
	if len(x) == 1 {
		return x[0]
	}
	if q == 0 {
		return Min_NEON_F32(x)
	}
	if q == 1 {
		return Max_NEON_F32(x)
	}
	x = slices.Clone(x)
	f := float32(len(x)-1) * q
	i := int(math.Floor(float64(f)))
	if q < 0.5 {
		partial.TopK(x, i+2)
		a := Max_NEON_F32(x[:i+1])
		b := x[i+1]
		return a + (b-a)*(f-float32(i))
	}
	partial.TopK(x, i+1)
	a := x[i]
	b := Min_NEON_F32(x[i+1:])
	return a + (b-a)*(f-float32(i))
}

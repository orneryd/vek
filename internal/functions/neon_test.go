//go:build arm64 && cgo && !nosimd

package functions

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/rand"
	"math"
	"testing"
)

func TestNEONDistanceOpsF32(t *testing.T) {
	if !UseNEON {
		t.Skip("NEON not enabled")
	}

	rand.Seed(2)
	for i := 0; i < 1000; i++ {
		size := 1 + (i / 5)
		x := Random[float32](size)
		y := Random[float32](size)

		require.InEpsilon(t, float64(Dot_Go(x, y)), float64(Dot_NEON_F32(x, y)), 1e-4)
		require.InEpsilon(t, float64(Norm_Go_F32(x)), float64(Norm_NEON_F32(x)), 1e-4)
		require.InEpsilon(t, float64(Distance_Go_F32(x, y)), float64(Distance_NEON_F32(x, y)), 1e-4)
		require.InDelta(t, float64(CosineSimilarity_Go_F32(x, y)), float64(CosineSimilarity_NEON_F32(x, y)), 1e-5)
	}
}

func TestNEONDistanceOpsF32_EdgeCases(t *testing.T) {
	if !UseNEON {
		t.Skip("NEON not enabled")
	}

	requireClose := func(t *testing.T, expected, actual float32, relTol, absTol float64) {
		t.Helper()
		// require.InEpsilon can't handle expected == 0 (relative error undefined),
		// so use absolute tolerance in that case.
		if expected == 0 {
			require.InDelta(t, float64(expected), float64(actual), absTol)
			return
		}
		require.InEpsilon(t, float64(expected), float64(actual), relTol)
	}

	t.Run("sizes around vector widths", func(t *testing.T) {
		rand.Seed(3)
		sizes := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33}
		for _, size := range sizes {
			x := Random[float32](size)
			y := Random[float32](size)

			requireClose(t, Dot_Go(x, y), Dot_NEON_F32(x, y), 1e-4, 1e-6)
			requireClose(t, Norm_Go_F32(x), Norm_NEON_F32(x), 1e-4, 1e-6)
			requireClose(t, Distance_Go_F32(x, y), Distance_NEON_F32(x, y), 1e-4, 1e-6)
			require.InDelta(t, float64(CosineSimilarity_Go_F32(x, y)), float64(CosineSimilarity_NEON_F32(x, y)), 1e-5)
		}
	})

	t.Run("cosine similarity known cases", func(t *testing.T) {
		x := []float32{1, 0, 0}
		y := []float32{0, 1, 0}
		require.InDelta(t, 0.0, float64(CosineSimilarity_NEON_F32(x, y)), 1e-6)

		z := []float32{1, 2, 3}
		require.InDelta(t, 1.0, float64(CosineSimilarity_NEON_F32(z, z)), 1e-6)
	})

	t.Run("zero vectors produce NaN like Go", func(t *testing.T) {
		zeros := []float32{0, 0, 0, 0}
		got := CosineSimilarity_NEON_F32(zeros, zeros)
		require.True(t, math.IsNaN(float64(got)))
	})
}

func BenchmarkNEONDistanceOpsF32(b *testing.B) {
	if !UseNEON {
		b.Skip("NEON not enabled")
	}

	sizes := []int{10, 100, 1000, 10_000}
	for _, size := range sizes {
		x := Random[float32](size)
		y := Random[float32](size)

		b.Run(fmt.Sprintf("dot_go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Dot_Go(x, y)
			}
		})
		b.Run(fmt.Sprintf("dot_neon_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Dot_NEON_F32(x, y)
			}
		})

		b.Run(fmt.Sprintf("norm_go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Norm_Go_F32(x)
			}
		})
		b.Run(fmt.Sprintf("norm_neon_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Norm_NEON_F32(x)
			}
		})

		b.Run(fmt.Sprintf("dist_go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Distance_Go_F32(x, y)
			}
		})
		b.Run(fmt.Sprintf("dist_neon_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Distance_NEON_F32(x, y)
			}
		})

		b.Run(fmt.Sprintf("cosim_go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = CosineSimilarity_Go_F32(x, y)
			}
		})
		b.Run(fmt.Sprintf("cosim_neon_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = CosineSimilarity_NEON_F32(x, y)
			}
		})
	}
}

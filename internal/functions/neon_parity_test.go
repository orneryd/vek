//go:build arm64 && cgo && !nosimd

package functions

import (
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/rand"
	"math"
	"testing"
)

func TestNEONParity_F32(t *testing.T) {
	if !UseNEON {
		t.Skip("NEON not enabled")
	}

	rand.Seed(42)
	sizes := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 127}

	for _, size := range sizes {
		x := Random[float32](size)
		y := Random[float32](size)

		t.Run("arithmetic_size_"+itoa(size), func(t *testing.T) {
			x1, x2 := append([]float32(nil), x...), append([]float32(nil), x...)
			Add_NEON_F32(x1, y)
			Add_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-6)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			Sub_NEON_F32(x1, y)
			Sub_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-6)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			Mul_NEON_F32(x1, y)
			Mul_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-6)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			Div_NEON_F32(x1, y)
			Div_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-5)
		})

		t.Run("minmax_size_"+itoa(size), func(t *testing.T) {
			if size == 0 {
				return
			}
			require.InDelta(t, float64(Min_Go(x)), float64(Min_NEON_F32(x)), 1e-6)
			require.InDelta(t, float64(Max_Go(x)), float64(Max_NEON_F32(x)), 1e-6)
			require.Equal(t, ArgMin_Go(x), ArgMin_NEON_F32(x))
			require.Equal(t, ArgMax_Go(x), ArgMax_NEON_F32(x))
		})
	}

	t.Run("round_ties_away_from_zero", func(t *testing.T) {
		x := []float32{0.5, 1.5, 2.5, -0.5, -1.5, -2.5}
		want := append([]float32(nil), x...)
		got := append([]float32(nil), x...)

		Round_Go_F32(want)
		Round_NEON_F32(got)
		require.Equal(t, want, got)
	})

	t.Run("transcendentals_close_to_go", func(t *testing.T) {
		requireClose := func(t *testing.T, expected, actual float32, relTol, absTol float64) {
			t.Helper()
			if expected == 0 {
				require.InDelta(t, float64(expected), float64(actual), absTol)
				return
			}
			require.InEpsilon(t, float64(expected), float64(actual), relTol)
		}

		x := []float32{0, 0.1, 1.2, -2.3, 10.0}
		want := append([]float32(nil), x...)
		got := append([]float32(nil), x...)
		Sin_Go_F32(want)
		Sin_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 1e-5)

		want = append([]float32(nil), x...)
		got = append([]float32(nil), x...)
		Cos_Go_F32(want)
		Cos_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 1e-5)

		want = append([]float32(nil), x...)
		got = append([]float32(nil), x...)
		Exp_Go_F32(want)
		Exp_NEON_F32(got)
		for i := range want {
			requireClose(t, want[i], got[i], 1e-6, 1e-6)
		}

		want = []float32{0.1, 1.2, 10.0}
		got = append([]float32(nil), want...)
		Log_Go_F32(want)
		Log_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 1e-5)
	})

	t.Run("cosine_similarity_nan_on_zero_vectors", func(t *testing.T) {
		zeros := []float32{0, 0, 0, 0}
		require.True(t, math.IsNaN(float64(CosineSimilarity_NEON_F32(zeros, zeros))))
	})
}

func TestNEONParity_F64(t *testing.T) {
	if !UseNEON {
		t.Skip("NEON not enabled")
	}

	rand.Seed(43)
	sizes := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 127}

	for _, size := range sizes {
		x := Random[float64](size)
		y := Random[float64](size)

		t.Run("arithmetic_size_"+itoa(size), func(t *testing.T) {
			x1, x2 := append([]float64(nil), x...), append([]float64(nil), x...)
			Add_NEON_F64(x1, y)
			Add_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			Sub_NEON_F64(x1, y)
			Sub_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			Mul_NEON_F64(x1, y)
			Mul_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			Div_NEON_F64(x1, y)
			Div_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-12)
		})

		t.Run("minmax_size_"+itoa(size), func(t *testing.T) {
			if size == 0 {
				return
			}
			require.InDelta(t, Min_Go(x), Min_NEON_F64(x), 1e-12)
			require.InDelta(t, Max_Go(x), Max_NEON_F64(x), 1e-12)
			require.Equal(t, ArgMin_Go(x), ArgMin_NEON_F64(x))
			require.Equal(t, ArgMax_Go(x), ArgMax_NEON_F64(x))
		})
	}

	t.Run("round_ties_away_from_zero", func(t *testing.T) {
		x := []float64{0.5, 1.5, 2.5, -0.5, -1.5, -2.5}
		want := append([]float64(nil), x...)
		got := append([]float64(nil), x...)

		Round_Go_F64(want)
		Round_NEON_F64(got)
		require.Equal(t, want, got)
	})
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [32]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

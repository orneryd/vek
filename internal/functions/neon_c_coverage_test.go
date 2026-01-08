//go:build arm64 && cgo && !nosimd

package functions

import (
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/rand"
	"math"
	"testing"
)

func TestNEON_CCoverage_F64(t *testing.T) {
	if !UseNEON {
		t.Skip("NEON not enabled")
	}

	rand.Seed(1001)
	sizes := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 127}

	for _, size := range sizes {
		x := RandomRange[float64](-10, 10, size)
		y := RandomRange[float64](-10, 10, size)

		t.Run("arith_size_"+itoa(size), func(t *testing.T) {
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
			require.InDeltaSlice(t, x2, x1, 1e-10)

			a := float64(1.25)
			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			AddNumber_NEON_F64(x1, a)
			AddNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			SubNumber_NEON_F64(x1, a)
			SubNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			MulNumber_NEON_F64(x1, a)
			MulNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			DivNumber_NEON_F64(x1, a)
			DivNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			Abs_NEON_F64(x1)
			Abs_Go_F64(x2)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			Neg_NEON_F64(x1)
			Neg_Go(x2)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			xPos := RandomRange[float64](0.1, 10, size)
			x1, x2 = append([]float64(nil), xPos...), append([]float64(nil), xPos...)
			Inv_NEON_F64(x1)
			Inv_Go(x2)
			require.InDeltaSlice(t, x2, x1, 1e-12)
		})

		t.Run("aggregates_size_"+itoa(size), func(t *testing.T) {
			if size == 0 {
				return
			}

			xSum := RandomRange[float64](-10, 10, size)
			xProd := RandomRange[float64](0.9, 1.1, size)
			require.InEpsilon(t, Sum_Go(xSum), Sum_NEON_F64(xSum), 1e-12)
			require.InEpsilon(t, Prod_Go(xProd), Prod_NEON_F64(xProd), 1e-12)
		})

		t.Run("distance_size_"+itoa(size), func(t *testing.T) {
			if size == 0 {
				return
			}

			require.InDelta(t, Dot_Go(x, y), Dot_NEON_F64(x, y), 1e-8)
			require.InDelta(t, Norm_Go_F64(x), Norm_NEON_F64(x), 1e-8)
			require.InDelta(t, Distance_Go_F64(x, y), Distance_NEON_F64(x, y), 1e-8)
			require.InDelta(t, ManhattanNorm_Go_F64(x), ManhattanNorm_NEON_F64(x), 1e-8)
			require.InDelta(t, ManhattanDistance_Go_F64(x, y), ManhattanDistance_NEON_F64(x, y), 1e-8)

			got := CosineSimilarity_NEON_F64(x, y)
			want := CosineSimilarity_Go_F64(x, y)
			if math.IsNaN(want) {
				require.True(t, math.IsNaN(got))
			} else {
				require.InDelta(t, want, got, 1e-8)
			}
		})

		t.Run("minmax_size_"+itoa(size), func(t *testing.T) {
			if size == 0 {
				return
			}
			require.InDelta(t, Min_Go(x), Min_NEON_F64(x), 1e-12)
			require.InDelta(t, Max_Go(x), Max_NEON_F64(x), 1e-12)

			x1, x2 := append([]float64(nil), x...), append([]float64(nil), x...)
			Minimum_NEON_F64(x1, y)
			Minimum_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			Maximum_NEON_F64(x1, y)
			Maximum_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			a := float64(1.25)
			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			MinimumNumber_NEON_F64(x1, a)
			MinimumNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-12)

			x1, x2 = append([]float64(nil), x...), append([]float64(nil), x...)
			MaximumNumber_NEON_F64(x1, a)
			MaximumNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-12)
		})
	}
}

func TestNEON_CCoverage_F32(t *testing.T) {
	if !UseNEON {
		t.Skip("NEON not enabled")
	}

	rand.Seed(1003)
	sizes := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 127}

	for _, size := range sizes {
		x := RandomRange[float32](-10, 10, size)
		y := RandomRange[float32](-10, 10, size)

		t.Run("arith_size_"+itoa(size), func(t *testing.T) {
			a := float32(1.25)

			x1, x2 := append([]float32(nil), x...), append([]float32(nil), x...)
			AddNumber_NEON_F32(x1, a)
			AddNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			SubNumber_NEON_F32(x1, a)
			SubNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			MulNumber_NEON_F32(x1, a)
			MulNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			DivNumber_NEON_F32(x1, a)
			DivNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			Abs_NEON_F32(x1)
			Abs_Go_F32(x2)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			Neg_NEON_F32(x1)
			Neg_Go(x2)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			xPos := RandomRange[float32](0.1, 10, size)
			x1, x2 = append([]float32(nil), xPos...), append([]float32(nil), xPos...)
			Inv_NEON_F32(x1)
			Inv_Go(x2)
			require.InDeltaSlice(t, x2, x1, 1e-5)
		})

		t.Run("distance_size_"+itoa(size), func(t *testing.T) {
			if size == 0 {
				return
			}
			require.InDelta(t, float64(ManhattanNorm_Go_F32(x)), float64(ManhattanNorm_NEON_F32(x)), 1e-4)
			require.InDelta(t, float64(ManhattanDistance_Go_F32(x, y)), float64(ManhattanDistance_NEON_F32(x, y)), 1e-4)
		})

		t.Run("minmax_size_"+itoa(size), func(t *testing.T) {
			if size == 0 {
				return
			}

			x1, x2 := append([]float32(nil), x...), append([]float32(nil), x...)
			Minimum_NEON_F32(x1, y)
			Minimum_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			Maximum_NEON_F32(x1, y)
			Maximum_Go(x2, y)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			a := float32(1.25)
			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			MinimumNumber_NEON_F32(x1, a)
			MinimumNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-5)

			x1, x2 = append([]float32(nil), x...), append([]float32(nil), x...)
			MaximumNumber_NEON_F32(x1, a)
			MaximumNumber_Go(x2, a)
			require.InDeltaSlice(t, x2, x1, 1e-5)
		})
	}
}

func TestNEON_CCoverage_SpecialAndMatrix(t *testing.T) {
	if !UseNEON {
		t.Skip("NEON not enabled")
	}

	rand.Seed(1002)

	t.Run("special_f64", func(t *testing.T) {
		x := RandomRange[float64](-10, 10, 257)
		xPos := RandomRange[float64](0.0, 10, 257)

		want, got := append([]float64(nil), xPos...), append([]float64(nil), xPos...)
		Sqrt_Go_F64(want)
		Sqrt_NEON_F64(got)
		require.InDeltaSlice(t, want, got, 1e-12)

		want, got = append([]float64(nil), x...), append([]float64(nil), x...)
		Round_Go_F64(want)
		Round_NEON_F64(got)
		require.Equal(t, want, got)

		want, got = append([]float64(nil), x...), append([]float64(nil), x...)
		Floor_Go_F64(want)
		Floor_NEON_F64(got)
		require.Equal(t, want, got)

		want, got = append([]float64(nil), x...), append([]float64(nil), x...)
		Ceil_Go_F64(want)
		Ceil_NEON_F64(got)
		require.Equal(t, want, got)

		base := RandomRange[float64](0.1, 10, 257)
		exp := RandomRange[float64](-3, 3, 257)
		want, got = append([]float64(nil), base...), append([]float64(nil), base...)
		Pow_Go_F64(want, exp)
		Pow_NEON_F64(got, exp)
		require.InDeltaSlice(t, want, got, 1e-10)
	})

	t.Run("special_f32", func(t *testing.T) {
		x := RandomRange[float32](-10, 10, 257)
		xPos := RandomRange[float32](0.0, 10, 257)

		want, got := append([]float32(nil), xPos...), append([]float32(nil), xPos...)
		Sqrt_Go_F32(want)
		Sqrt_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 1e-5)

		want, got = append([]float32(nil), x...), append([]float32(nil), x...)
		Round_Go_F32(want)
		Round_NEON_F32(got)
		require.Equal(t, want, got)

		want, got = append([]float32(nil), x...), append([]float32(nil), x...)
		Floor_Go_F32(want)
		Floor_NEON_F32(got)
		require.Equal(t, want, got)

		want, got = append([]float32(nil), x...), append([]float32(nil), x...)
		Ceil_Go_F32(want)
		Ceil_NEON_F32(got)
		require.Equal(t, want, got)

		base := RandomRange[float32](0.1, 10, 257)
		exp := RandomRange[float32](-3, 3, 257)
		want, got = append([]float32(nil), base...), append([]float32(nil), base...)
		Pow_Go_F32(want, exp)
		Pow_NEON_F32(got, exp)
		require.InDeltaSlice(t, want, got, 1e-4)

		trig := RandomRange[float32](-10, 10, 257)
		want, got = append([]float32(nil), trig...), append([]float32(nil), trig...)
		Sin_Go_F32(want)
		Sin_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 1e-5)

		want, got = append([]float32(nil), trig...), append([]float32(nil), trig...)
		Cos_Go_F32(want)
		Cos_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 1e-5)

		dstSinWant := make([]float32, len(trig))
		dstCosWant := make([]float32, len(trig))
		copy(dstSinWant, trig)
		copy(dstCosWant, trig)
		Sin_Go_F32(dstSinWant)
		Cos_Go_F32(dstCosWant)

		dstSinGot := make([]float32, len(trig))
		dstCosGot := make([]float32, len(trig))
		SinCos_NEON_F32(dstSinGot, dstCosGot, trig)
		require.InDeltaSlice(t, dstSinWant, dstSinGot, 1e-5)
		require.InDeltaSlice(t, dstCosWant, dstCosGot, 1e-5)

		expIn := RandomRange[float32](-3, 3, 257)
		want, got = append([]float32(nil), expIn...), append([]float32(nil), expIn...)
		Exp_Go_F32(want)
		Exp_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 2e-5)

		logIn := RandomRange[float32](0.01, 10, 257)
		want, got = append([]float32(nil), logIn...), append([]float32(nil), logIn...)
		Log_Go_F32(want)
		Log_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 2e-5)

		want, got = append([]float32(nil), logIn...), append([]float32(nil), logIn...)
		Log2_Go_F32(want)
		Log2_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 2e-5)

		want, got = append([]float32(nil), logIn...), append([]float32(nil), logIn...)
		Log10_Go_F32(want)
		Log10_NEON_F32(got)
		require.InDeltaSlice(t, want, got, 2e-5)
	})

	t.Run("aggregates_f32", func(t *testing.T) {
		xSum := RandomRange[float32](-10, 10, 257)
		xProd := RandomRange[float32](0.9, 1.1, 257)
		require.InEpsilon(t, float64(Sum_Go(xSum)), float64(Sum_NEON_F32(xSum)), 1e-5)
		require.InEpsilon(t, float64(Prod_Go(xProd)), float64(Prod_NEON_F32(xProd)), 1e-5)
	})

	t.Run("matrix_f64", func(t *testing.T) {
		{
			x := RandomRange[float64](-2, 2, 16)
			y := RandomRange[float64](-2, 2, 16)
			want := make([]float64, 16)
			got := make([]float64, 16)
			Mat4Mul_Go(want, x, y)
			Mat4Mul_NEON_F64(got, x, y)
			require.InDeltaSlice(t, want, got, 1e-10)
		}
		{
			m, n, p := 3, 5, 4
			x := RandomRange[float64](-2, 2, m*n)
			y := RandomRange[float64](-2, 2, n*p)
			want := make([]float64, m*p)
			got := make([]float64, m*p)
			MatMul_Go(want, x, y, m, n, p)
			MatMul_NEON_F64(got, x, y, m, n, p)
			require.InDeltaSlice(t, want, got, 1e-10)
		}
		{
			m, n := 7, 11
			x := RandomRange[float64](-2, 2, m*n)
			y := RandomRange[float64](-2, 2, n)
			want := make([]float64, m)
			got := make([]float64, m)
			MatMulVec_Go(want, x, y, m, n)
			MatMulVec_NEON_F64(got, x, y, m, n)
			require.InDeltaSlice(t, want, got, 1e-10)
		}
	})

	t.Run("matrix_f32", func(t *testing.T) {
		{
			x := RandomRange[float32](-2, 2, 16)
			y := RandomRange[float32](-2, 2, 16)
			want := make([]float32, 16)
			got := make([]float32, 16)
			Mat4Mul_Go(want, x, y)
			Mat4Mul_NEON_F32(got, x, y)
			require.InDeltaSlice(t, want, got, 1e-5)
		}
		{
			m, n, p := 3, 5, 4
			x := RandomRange[float32](-2, 2, m*n)
			y := RandomRange[float32](-2, 2, n*p)
			want := make([]float32, m*p)
			got := make([]float32, m*p)
			MatMul_Go(want, x, y, m, n, p)
			MatMul_NEON_F32(got, x, y, m, n, p)
			require.InDeltaSlice(t, want, got, 1e-5)
		}
		{
			m, n := 7, 11
			x := RandomRange[float32](-2, 2, m*n)
			y := RandomRange[float32](-2, 2, n)
			want := make([]float32, m)
			got := make([]float32, m)
			MatMulVec_Go(want, x, y, m, n)
			MatMulVec_NEON_F32(got, x, y, m, n)
			require.InDeltaSlice(t, want, got, 1e-5)
		}
	})
}

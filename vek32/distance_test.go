package vek32

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDistanceOps_EmptySlices(t *testing.T) {
	for _, accel := range accel {
		SetAcceleration(accel)

		require.Panics(t, func() { Dot(empty, empty) })
		require.Panics(t, func() { Distance(empty, empty) })
		require.Panics(t, func() { CosineSimilarity(empty, empty) })
		require.Panics(t, func() { Norm(empty) })
	}
}

func TestDistanceOps_SimpleVectors(t *testing.T) {
	for _, accel := range accel {
		SetAcceleration(accel)

		x := []float32{1, 0, 0}
		y := []float32{0, 1, 0}
		require.InDelta(t, 0.0, float64(Dot(x, y)), d)
		require.InDelta(t, 0.0, float64(CosineSimilarity(x, y)), d)

		z := []float32{1, 2, 3}
		require.InDelta(t, 1.0, float64(CosineSimilarity(z, z)), d)
	}
}

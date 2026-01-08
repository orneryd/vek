//go:build arm64 && cgo && !nosimd
// +build arm64,cgo,!nosimd

// This compilation unit pulls in the NEON implementation from asm/_neon so the
// internal/functions package can link it via cgo.

#include "../../asm/_neon/arithmetic.cpp"
#include "../../asm/_neon/aggregates.cpp"
#include "../../asm/_neon/distance.cpp"
#include "../../asm/_neon/max.cpp"
#include "../../asm/_neon/min.cpp"
#include "../../asm/_neon/special.cpp"
#include "../../asm/_neon/matrix.cpp"

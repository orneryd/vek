//go:build arm64 && cgo && !nosimd

package functions

import (
	"golang.org/x/sys/cpu"
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../../asm/_neon
#cgo CXXFLAGS: -O3 -std=c++11 -I${SRCDIR}/../../asm/_neon
#cgo darwin LDFLAGS: -framework Accelerate
#include "simd_arm64.h"
*/
import "C"

var HasNEON bool = cpu.ARM64.HasASIMD
var UseNEON bool = HasNEON

func f32PtrOrNil(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

func f64PtrOrNil(x []float64) *C.double {
	if len(x) == 0 {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&x[0]))
}

// Arithmetic

func Add_NEON_F64(x, y []float64) {
	C.vek_neon_add_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x)))
}

func Add_NEON_F32(x, y []float32) {
	C.vek_neon_add_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x)))
}

func AddNumber_NEON_F64(x []float64, a float64) {
	C.vek_neon_add_number_f64(f64PtrOrNil(x), C.double(a), C.int(len(x)))
}

func AddNumber_NEON_F32(x []float32, a float32) {
	C.vek_neon_add_number_f32(f32PtrOrNil(x), C.float(a), C.int(len(x)))
}

func Sub_NEON_F64(x, y []float64) {
	C.vek_neon_sub_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x)))
}

func Sub_NEON_F32(x, y []float32) {
	C.vek_neon_sub_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x)))
}

func SubNumber_NEON_F64(x []float64, a float64) {
	C.vek_neon_sub_number_f64(f64PtrOrNil(x), C.double(a), C.int(len(x)))
}

func SubNumber_NEON_F32(x []float32, a float32) {
	C.vek_neon_sub_number_f32(f32PtrOrNil(x), C.float(a), C.int(len(x)))
}

func Mul_NEON_F64(x, y []float64) {
	C.vek_neon_mul_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x)))
}

func Mul_NEON_F32(x, y []float32) {
	C.vek_neon_mul_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x)))
}

func MulNumber_NEON_F64(x []float64, a float64) {
	C.vek_neon_mul_number_f64(f64PtrOrNil(x), C.double(a), C.int(len(x)))
}

func MulNumber_NEON_F32(x []float32, a float32) {
	C.vek_neon_mul_number_f32(f32PtrOrNil(x), C.float(a), C.int(len(x)))
}

func Div_NEON_F64(x, y []float64) {
	C.vek_neon_div_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x)))
}

func Div_NEON_F32(x, y []float32) {
	C.vek_neon_div_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x)))
}

func DivNumber_NEON_F64(x []float64, a float64) {
	C.vek_neon_div_number_f64(f64PtrOrNil(x), C.double(a), C.int(len(x)))
}

func DivNumber_NEON_F32(x []float32, a float32) {
	C.vek_neon_div_number_f32(f32PtrOrNil(x), C.float(a), C.int(len(x)))
}

func Abs_NEON_F64(x []float64) {
	C.vek_neon_abs_f64(f64PtrOrNil(x), C.int(len(x)))
}

func Abs_NEON_F32(x []float32) {
	C.vek_neon_abs_f32(f32PtrOrNil(x), C.int(len(x)))
}

func Neg_NEON_F64(x []float64) {
	C.vek_neon_neg_f64(f64PtrOrNil(x), C.int(len(x)))
}

func Neg_NEON_F32(x []float32) {
	C.vek_neon_neg_f32(f32PtrOrNil(x), C.int(len(x)))
}

func Inv_NEON_F64(x []float64) {
	C.vek_neon_inv_f64(f64PtrOrNil(x), C.int(len(x)))
}

func Inv_NEON_F32(x []float32) {
	C.vek_neon_inv_f32(f32PtrOrNil(x), C.int(len(x)))
}

// Aggregates

func Sum_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_sum_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Sum_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_sum_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Prod_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_prod_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Prod_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_prod_f32(f32PtrOrNil(x), C.int(len(x))))
}

// Distance

func Dot_NEON_F64(x, y []float64) float64 {
	return float64(C.vek_neon_dot_product_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x))))
}

func Dot_NEON_F32(x, y []float32) float32 {
	return float32(C.vek_neon_dot_product_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x))))
}

func Norm_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_norm_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Norm_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_norm_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Distance_NEON_F64(x, y []float64) float64 {
	return float64(C.vek_neon_distance_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x))))
}

func Distance_NEON_F32(x, y []float32) float32 {
	return float32(C.vek_neon_distance_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x))))
}

func ManhattanNorm_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_manhattan_norm_f64(f64PtrOrNil(x), C.int(len(x))))
}

func ManhattanNorm_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_manhattan_norm_f32(f32PtrOrNil(x), C.int(len(x))))
}

func ManhattanDistance_NEON_F64(x, y []float64) float64 {
	return float64(C.vek_neon_manhattan_distance_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x))))
}

func ManhattanDistance_NEON_F32(x, y []float32) float32 {
	return float32(C.vek_neon_manhattan_distance_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x))))
}

func CosineSimilarity_NEON_F64(x, y []float64) float64 {
	return float64(C.vek_neon_cosine_similarity_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x))))
}

func CosineSimilarity_NEON_F32(x, y []float32) float32 {
	return float32(C.vek_neon_cosine_similarity_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x))))
}

// Matrix

func Mat4Mul_NEON_F64(dst, x, y []float64) {
	C.vek_neon_mat4mul_f64(f64PtrOrNil(dst), f64PtrOrNil(x), f64PtrOrNil(y))
}

func Mat4Mul_NEON_F32(dst, x, y []float32) {
	C.vek_neon_mat4mul_f32(f32PtrOrNil(dst), f32PtrOrNil(x), f32PtrOrNil(y))
}

func MatMul_NEON_F64(dst, x, y []float64, m, n, p int) {
	C.vek_neon_matmul_f64(f64PtrOrNil(dst), f64PtrOrNil(x), f64PtrOrNil(y), C.int(m), C.int(n), C.int(p))
}

func MatMul_NEON_F32(dst, x, y []float32, m, n, p int) {
	C.vek_neon_matmul_f32(f32PtrOrNil(dst), f32PtrOrNil(x), f32PtrOrNil(y), C.int(m), C.int(n), C.int(p))
}

func MatMulVec_NEON_F64(dst, x, y []float64, m, n int) {
	C.vek_neon_matmulvec_f64(f64PtrOrNil(dst), f64PtrOrNil(x), f64PtrOrNil(y), C.int(m), C.int(n))
}

func MatMulVec_NEON_F32(dst, x, y []float32, m, n int) {
	C.vek_neon_matmulvec_f32(f32PtrOrNil(dst), f32PtrOrNil(x), f32PtrOrNil(y), C.int(m), C.int(n))
}

// Special

func Sqrt_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_sqrt_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Sqrt_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_sqrt_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Round_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_round_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Round_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_round_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Floor_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_floor_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Floor_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_floor_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Ceil_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_ceil_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Ceil_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_ceil_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Pow_NEON_F64(x, y []float64) {
	C.vek_neon_pow_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x)))
}

func Pow_NEON_F32(x, y []float32) {
	C.vek_neon_pow_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x)))
}

func Sin_NEON_F32(x []float32) {
	C.vek_neon_sin_f32(f32PtrOrNil(x), C.int(len(x)))
}

func Cos_NEON_F32(x []float32) {
	C.vek_neon_cos_f32(f32PtrOrNil(x), C.int(len(x)))
}

func SinCos_NEON_F32(dstSin, dstCos, x []float32) {
	C.vek_neon_sincos_f32(f32PtrOrNil(dstSin), f32PtrOrNil(dstCos), f32PtrOrNil(x), C.int(len(x)))
}

func Exp_NEON_F32(x []float32) {
	C.vek_neon_exp_f32(f32PtrOrNil(x), C.int(len(x)))
}

func Log_NEON_F32(x []float32) {
	C.vek_neon_log_f32(f32PtrOrNil(x), C.int(len(x)))
}

func Log2_NEON_F32(x []float32) {
	C.vek_neon_log2_f32(f32PtrOrNil(x), C.int(len(x)))
}

func Log10_NEON_F32(x []float32) {
	C.vek_neon_log10_f32(f32PtrOrNil(x), C.int(len(x)))
}

// Min/Max

func Min_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_min_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Min_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_min_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Minimum_NEON_F64(x, y []float64) {
	C.vek_neon_minimum_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x)))
}

func Minimum_NEON_F32(x, y []float32) {
	C.vek_neon_minimum_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x)))
}

func MinimumNumber_NEON_F64(x []float64, a float64) {
	C.vek_neon_minimum_number_f64(f64PtrOrNil(x), C.double(a), C.int(len(x)))
}

func MinimumNumber_NEON_F32(x []float32, a float32) {
	C.vek_neon_minimum_number_f32(f32PtrOrNil(x), C.float(a), C.int(len(x)))
}

func Max_NEON_F64(x []float64) float64 {
	return float64(C.vek_neon_max_f64(f64PtrOrNil(x), C.int(len(x))))
}

func Max_NEON_F32(x []float32) float32 {
	return float32(C.vek_neon_max_f32(f32PtrOrNil(x), C.int(len(x))))
}

func Maximum_NEON_F64(x, y []float64) {
	C.vek_neon_maximum_f64(f64PtrOrNil(x), f64PtrOrNil(y), C.int(len(x)))
}

func Maximum_NEON_F32(x, y []float32) {
	C.vek_neon_maximum_f32(f32PtrOrNil(x), f32PtrOrNil(y), C.int(len(x)))
}

func MaximumNumber_NEON_F64(x []float64, a float64) {
	C.vek_neon_maximum_number_f64(f64PtrOrNil(x), C.double(a), C.int(len(x)))
}

func MaximumNumber_NEON_F32(x []float32, a float32) {
	C.vek_neon_maximum_number_f32(f32PtrOrNil(x), C.float(a), C.int(len(x)))
}

// Scan ops (not vectorized yet)

func CumSum_NEON_F64(x []float64) {
	CumSum_Go(x)
}

func CumSum_NEON_F32(x []float32) {
	CumSum_Go(x)
}

func CumProd_NEON_F64(x []float64) {
	CumProd_Go(x)
}

func CumProd_NEON_F32(x []float32) {
	CumProd_Go(x)
}

// MatMulTiled parity (delegates to MatMul_NEON).
func MatMulTiled_NEON_F64(dst, x, y []float64, m, n, p int) {
	MatMul_NEON_F64(dst, x, y, m, n, p)
}

func MatMulTiled_NEON_F32(dst, x, y []float32, m, n, p int) {
	MatMul_NEON_F32(dst, x, y, m, n, p)
}

// Find parity: AVX2 returns len(x) when not found.

func Find_NEON_F64(x []float64, a float64) int {
	for i, v := range x {
		if v == a {
			return i
		}
	}
	return len(x)
}

func Find_NEON_F32(x []float32, a float32) int {
	for i, v := range x {
		if v == a {
			return i
		}
	}
	return len(x)
}

// Comparisons

func Lt_NEON_F64(dst []bool, x, y []float64)  { Lt_Go(dst, x, y) }
func Lt_NEON_F32(dst []bool, x, y []float32)  { Lt_Go(dst, x, y) }
func Lte_NEON_F64(dst []bool, x, y []float64) { Lte_Go(dst, x, y) }
func Lte_NEON_F32(dst []bool, x, y []float32) { Lte_Go(dst, x, y) }
func Gt_NEON_F64(dst []bool, x, y []float64)  { Gt_Go(dst, x, y) }
func Gt_NEON_F32(dst []bool, x, y []float32)  { Gt_Go(dst, x, y) }
func Gte_NEON_F64(dst []bool, x, y []float64) { Gte_Go(dst, x, y) }
func Gte_NEON_F32(dst []bool, x, y []float32) { Gte_Go(dst, x, y) }
func Eq_NEON_F64(dst []bool, x, y []float64)  { Eq_Go(dst, x, y) }
func Eq_NEON_F32(dst []bool, x, y []float32)  { Eq_Go(dst, x, y) }
func Neq_NEON_F64(dst []bool, x, y []float64) { Neq_Go(dst, x, y) }
func Neq_NEON_F32(dst []bool, x, y []float32) { Neq_Go(dst, x, y) }

func LtNumber_NEON_F64(dst []bool, x []float64, a float64)  { LtNumber_Go(dst, x, a) }
func LtNumber_NEON_F32(dst []bool, x []float32, a float32)  { LtNumber_Go(dst, x, a) }
func LteNumber_NEON_F64(dst []bool, x []float64, a float64) { LteNumber_Go(dst, x, a) }
func LteNumber_NEON_F32(dst []bool, x []float32, a float32) { LteNumber_Go(dst, x, a) }
func GtNumber_NEON_F64(dst []bool, x []float64, a float64)  { GtNumber_Go(dst, x, a) }
func GtNumber_NEON_F32(dst []bool, x []float32, a float32)  { GtNumber_Go(dst, x, a) }
func GteNumber_NEON_F64(dst []bool, x []float64, a float64) { GteNumber_Go(dst, x, a) }
func GteNumber_NEON_F32(dst []bool, x []float32, a float32) { GteNumber_Go(dst, x, a) }
func EqNumber_NEON_F64(dst []bool, x []float64, a float64)  { EqNumber_Go(dst, x, a) }
func EqNumber_NEON_F32(dst []bool, x []float32, a float32)  { EqNumber_Go(dst, x, a) }
func NeqNumber_NEON_F64(dst []bool, x []float64, a float64) { NeqNumber_Go(dst, x, a) }
func NeqNumber_NEON_F32(dst []bool, x []float32, a float32) { NeqNumber_Go(dst, x, a) }

// Boolean ops

func Not_NEON(x []bool)    { Not_Go(x) }
func And_NEON(x, y []bool) { And_Go(x, y) }
func Or_NEON(x, y []bool)  { Or_Go(x, y) }
func Xor_NEON(x, y []bool) { Xor_Go(x, y) }

func All_NEON(x []bool) int {
	if All_Go(x) {
		return 1
	}
	return 0
}

func Any_NEON(x []bool) int {
	if Any_Go(x) {
		return 1
	}
	return 0
}

func None_NEON(x []bool) int {
	if None_Go(x) {
		return 1
	}
	return 0
}

func Count_NEON(x []bool) int { return Count_Go(x) }

// Construct / conversions

func Repeat_NEON_F64(dst []float64, a float64, n int) { Repeat_Go(dst, a, n) }
func Repeat_NEON_F32(dst []float32, a float32, n int) { Repeat_Go(dst, a, n) }
func Range_NEON_F64(dst []float64, a float64, n int)  { Range_Go(dst, a, n) }
func Range_NEON_F32(dst []float32, a float32, n int)  { Range_Go(dst, a, n) }

func FromBool_NEON_F64(dst []float64, x []bool)       { FromBool_Go(dst, x) }
func FromBool_NEON_F32(dst []float32, x []bool)       { FromBool_Go(dst, x) }
func FromInt32_NEON_F64(dst []float64, x []int32)     { FromNumber_Go(dst, x) }
func FromInt32_NEON_F32(dst []float32, x []int32)     { FromNumber_Go(dst, x) }
func FromInt64_NEON_F64(dst []float64, x []int64)     { FromNumber_Go(dst, x) }
func FromInt64_NEON_F32(dst []float32, x []int64)     { FromNumber_Go(dst, x) }
func FromFloat32_NEON_F64(dst []float64, x []float32) { FromNumber_Go(dst, x) }
func FromFloat64_NEON_F32(dst []float32, x []float64) { FromNumber_Go(dst, x) }

func ToBool_NEON_F64(dst []bool, x []float64)   { ToBool_Go(dst, x) }
func ToBool_NEON_F32(dst []bool, x []float32)   { ToBool_Go(dst, x) }
func ToInt32_NEON_F64(dst []int32, x []float64) { ToNumber_Go(dst, x) }
func ToInt32_NEON_F32(dst []int32, x []float32) { ToNumber_Go(dst, x) }
func ToInt64_NEON_F64(dst []int64, x []float64) { ToNumber_Go(dst, x) }
func ToInt64_NEON_F32(dst []int64, x []float32) { ToNumber_Go(dst, x) }

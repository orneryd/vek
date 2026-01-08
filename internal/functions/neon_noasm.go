//go:build !arm64 || !cgo || nosimd

package functions

var HasNEON bool = false
var UseNEON bool = false

// Arithmetic

func Add_NEON_F64(x, y []float64) { Add_Go(x, y) }
func Add_NEON_F32(x, y []float32) { Add_Go(x, y) }

func AddNumber_NEON_F64(x []float64, a float64) { AddNumber_Go(x, a) }
func AddNumber_NEON_F32(x []float32, a float32) { AddNumber_Go(x, a) }

func Sub_NEON_F64(x, y []float64) { Sub_Go(x, y) }
func Sub_NEON_F32(x, y []float32) { Sub_Go(x, y) }

func SubNumber_NEON_F64(x []float64, a float64) { SubNumber_Go(x, a) }
func SubNumber_NEON_F32(x []float32, a float32) { SubNumber_Go(x, a) }

func Mul_NEON_F64(x, y []float64) { Mul_Go(x, y) }
func Mul_NEON_F32(x, y []float32) { Mul_Go(x, y) }

func MulNumber_NEON_F64(x []float64, a float64) { MulNumber_Go(x, a) }
func MulNumber_NEON_F32(x []float32, a float32) { MulNumber_Go(x, a) }

func Div_NEON_F64(x, y []float64) { Div_Go(x, y) }
func Div_NEON_F32(x, y []float32) { Div_Go(x, y) }

func DivNumber_NEON_F64(x []float64, a float64) { DivNumber_Go(x, a) }
func DivNumber_NEON_F32(x []float32, a float32) { DivNumber_Go(x, a) }

func Abs_NEON_F64(x []float64) { Abs_Go_F64(x) }
func Abs_NEON_F32(x []float32) { Abs_Go_F32(x) }

func Neg_NEON_F64(x []float64) { Neg_Go(x) }
func Neg_NEON_F32(x []float32) { Neg_Go(x) }

func Inv_NEON_F64(x []float64) { Inv_Go(x) }
func Inv_NEON_F32(x []float32) { Inv_Go(x) }

// Aggregates

func Sum_NEON_F64(x []float64) float64 { return Sum_Go(x) }
func Sum_NEON_F32(x []float32) float32 { return Sum_Go(x) }

func CumSum_NEON_F64(x []float64) { CumSum_Go(x) }
func CumSum_NEON_F32(x []float32) { CumSum_Go(x) }

func Prod_NEON_F64(x []float64) float64 { return Prod_Go(x) }
func Prod_NEON_F32(x []float32) float32 { return Prod_Go(x) }

func CumProd_NEON_F64(x []float64) { CumProd_Go(x) }
func CumProd_NEON_F32(x []float32) { CumProd_Go(x) }

func Mean_NEON_F64(x []float64) float64 { return Mean_Go(x) }
func Mean_NEON_F32(x []float32) float32 { return Mean_Go(x) }

func Median_NEON_F64(x []float64) float64 { return Median_Go(x) }
func Median_NEON_F32(x []float32) float32 { return Median_Go(x) }

func Quantile_NEON_F64(x []float64, q float64) float64 { return Quantile_Go(x, q) }
func Quantile_NEON_F32(x []float32, q float32) float32 { return Quantile_Go(x, q) }

// Distance

func Dot_NEON_F64(x, y []float64) float64 { return Dot_Go(x, y) }
func Dot_NEON_F32(x, y []float32) float32 { return Dot_Go(x, y) }

func Norm_NEON_F64(x []float64) float64 { return Norm_Go_F64(x) }
func Norm_NEON_F32(x []float32) float32 { return Norm_Go_F32(x) }

func Distance_NEON_F64(x, y []float64) float64 { return Distance_Go_F64(x, y) }
func Distance_NEON_F32(x, y []float32) float32 { return Distance_Go_F32(x, y) }

func ManhattanNorm_NEON_F64(x []float64) float64 { return ManhattanNorm_Go_F64(x) }
func ManhattanNorm_NEON_F32(x []float32) float32 { return ManhattanNorm_Go_F32(x) }

func ManhattanDistance_NEON_F64(x, y []float64) float64 { return ManhattanDistance_Go_F64(x, y) }
func ManhattanDistance_NEON_F32(x, y []float32) float32 { return ManhattanDistance_Go_F32(x, y) }

func CosineSimilarity_NEON_F64(x, y []float64) float64 { return CosineSimilarity_Go_F64(x, y) }
func CosineSimilarity_NEON_F32(x, y []float32) float32 { return CosineSimilarity_Go_F32(x, y) }

// Matrix

func Mat4Mul_NEON_F64(dst, x, y []float64) { Mat4Mul_Go(dst, x, y) }
func Mat4Mul_NEON_F32(dst, x, y []float32) { Mat4Mul_Go(dst, x, y) }

func MatMul_NEON_F64(dst, x, y []float64, m, n, p int) { MatMul_Go(dst, x, y, m, n, p) }
func MatMul_NEON_F32(dst, x, y []float32, m, n, p int) { MatMul_Go(dst, x, y, m, n, p) }

func MatMulVec_NEON_F64(dst, x, y []float64, m, n int) { MatMulVec_Go(dst, x, y, m, n) }
func MatMulVec_NEON_F32(dst, x, y []float32, m, n int) { MatMulVec_Go(dst, x, y, m, n) }

func MatMulTiled_NEON_F64(dst, x, y []float64, m, n, p int) { MatMul_Go(dst, x, y, m, n, p) }
func MatMulTiled_NEON_F32(dst, x, y []float32, m, n, p int) { MatMul_Go(dst, x, y, m, n, p) }

func MatMul_Parallel_NEON_F64(dst, x, y []float64, m, n, p int) {
	MatMul_Parallel_Go(dst, x, y, m, n, p)
}

func MatMul_Parallel_NEON_F32(dst, x, y []float32, m, n, p int) {
	MatMul_Parallel_Go(dst, x, y, m, n, p)
}

// Special

func Sqrt_NEON_F64(x []float64) float64 {
	Sqrt_Go_F64(x)
	return 0
}

func Sqrt_NEON_F32(x []float32) float32 {
	Sqrt_Go_F32(x)
	return 0
}

func Round_NEON_F64(x []float64) float64 {
	Round_Go_F64(x)
	return 0
}

func Round_NEON_F32(x []float32) float32 {
	Round_Go_F32(x)
	return 0
}

func Floor_NEON_F64(x []float64) float64 {
	Floor_Go_F64(x)
	return 0
}

func Floor_NEON_F32(x []float32) float32 {
	Floor_Go_F32(x)
	return 0
}

func Ceil_NEON_F64(x []float64) float64 {
	Ceil_Go_F64(x)
	return 0
}

func Ceil_NEON_F32(x []float32) float32 {
	Ceil_Go_F32(x)
	return 0
}

func Pow_NEON_F64(x, y []float64) { Pow_Go_F64(x, y) }
func Pow_NEON_F32(x, y []float32) { Pow_Go_F32(x, y) }

func Sin_NEON_F32(x []float32) { Sin_Go_F32(x) }
func Cos_NEON_F32(x []float32) { Cos_Go_F32(x) }
func SinCos_NEON_F32(dstSin, dstCos, x []float32) {
	SinCos_Go_F32(dstSin, dstCos, x)
}
func Exp_NEON_F32(x []float32)   { Exp_Go_F32(x) }
func Log_NEON_F32(x []float32)   { Log_Go_F32(x) }
func Log2_NEON_F32(x []float32)  { Log2_Go_F32(x) }
func Log10_NEON_F32(x []float32) { Log10_Go_F32(x) }

// Min/Max

func Min_NEON_F64(x []float64) float64              { return Min_Go(x) }
func Min_NEON_F32(x []float32) float32              { return Min_Go(x) }
func Minimum_NEON_F64(x, y []float64)               { Minimum_Go(x, y) }
func Minimum_NEON_F32(x, y []float32)               { Minimum_Go(x, y) }
func MinimumNumber_NEON_F64(x []float64, a float64) { MinimumNumber_Go(x, a) }
func MinimumNumber_NEON_F32(x []float32, a float32) { MinimumNumber_Go(x, a) }

func Max_NEON_F64(x []float64) float64              { return Max_Go(x) }
func Max_NEON_F32(x []float32) float32              { return Max_Go(x) }
func Maximum_NEON_F64(x, y []float64)               { Maximum_Go(x, y) }
func Maximum_NEON_F32(x, y []float32)               { Maximum_Go(x, y) }
func MaximumNumber_NEON_F64(x []float64, a float64) { MaximumNumber_Go(x, a) }
func MaximumNumber_NEON_F32(x []float32, a float32) { MaximumNumber_Go(x, a) }

func ArgMin_NEON_F64(x []float64) int { return ArgMin_Go(x) }
func ArgMin_NEON_F32(x []float32) int { return ArgMin_Go(x) }
func ArgMax_NEON_F64(x []float64) int { return ArgMax_Go(x) }
func ArgMax_NEON_F32(x []float32) int { return ArgMax_Go(x) }

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

func Not_NEON(x []bool)       { Not_Go(x) }
func And_NEON(x, y []bool)    { And_Go(x, y) }
func Or_NEON(x, y []bool)     { Or_Go(x, y) }
func Xor_NEON(x, y []bool)    { Xor_Go(x, y) }
func Count_NEON(x []bool) int { return Count_Go(x) }
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

//============================================================================
// df64.h — double-single ("double-float") arithmetic for the hybrid-precision
// Metal port.  Apple GPUs have NO native fp64, but CUDA mini-ramses carries
// real(kind=dp)=fp64 in the multigrid.  Per the port decision we keep fp32 for
// the bulk and use df64 ONLY where fp32 demonstrably loses parity with CUDA:
//   - the periodic-base RHS mean / residual,
//   - the global reductions (cmp_epot, residual_norm, restrict_residual sum).
//
// A df64 value v is represented as an unevaluated sum v = hi + lo of two fp32,
// with |lo| <= 0.5 ulp(hi).  This yields ~ fp64 precision (~48-bit mantissa)
// using only fp32 ops.  Algorithms are the standard Dekker/Knuth error-free
// transforms (two-sum, two-prod via fma).  Reference: QD / dsfun90.
//
// Pure device-side header (no kernels); #included by reduce.metal / mg.metal.
//============================================================================
#ifndef RAMSES_MSL_DF64_H
#define RAMSES_MSL_DF64_H

#include <metal_stdlib>
using namespace metal;

struct df64 { float hi; float lo; };

inline df64 df64_from(float a)        { return df64{a, 0.0f}; }
inline float df64_value(df64 a)       { return a.hi + a.lo; }   // round to nearest fp32

// Error-free transforms ------------------------------------------------------
// two_sum: s = a + b exactly, s.hi = fl(a+b), s.lo = round-off (no |a|>=|b| assumption).
inline df64 two_sum(float a, float b) {
    float s  = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return df64{s, err};
}
// quick_two_sum: same, but requires |a| >= |b|.
inline df64 quick_two_sum(float a, float b) {
    float s   = a + b;
    float err = b - (s - a);
    return df64{s, err};
}
// two_prod: p = a*b exactly as (hi,lo), using fused-multiply-add for the error.
inline df64 two_prod(float a, float b) {
    float p   = a * b;
    float err = fma(a, b, -p);
    return df64{p, err};
}

// df64 arithmetic ------------------------------------------------------------
inline df64 df64_add(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    df64 t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    s = quick_two_sum(s.hi, s.lo);
    return s;
}
inline df64 df64_add_f(df64 a, float b) {   // df64 + float
    df64 s = two_sum(a.hi, b);
    s.lo += a.lo;
    s = quick_two_sum(s.hi, s.lo);
    return s;
}
inline df64 df64_neg(df64 a)          { return df64{-a.hi, -a.lo}; }
inline df64 df64_sub(df64 a, df64 b)  { return df64_add(a, df64_neg(b)); }

inline df64 df64_mul(df64 a, df64 b) {
    df64 p = two_prod(a.hi, b.hi);
    p.lo  += a.hi * b.lo + a.lo * b.hi;     // cross terms (lo*lo negligible)
    p = quick_two_sum(p.hi, p.lo);
    return p;
}
inline df64 df64_mul_f(df64 a, float b) {   // df64 * float
    df64 p = two_prod(a.hi, b);
    p.lo  += a.lo * b;
    p = quick_two_sum(p.hi, p.lo);
    return p;
}
// fused: a*b + c  (all df64)
inline df64 df64_fma(df64 a, df64 b, df64 c) { return df64_add(df64_mul(a, b), c); }
// accumulate a single fp32 product x*y into a df64 accumulator (the common
// reduction inner loop): acc += two_prod(x,y).
inline df64 df64_acc_prod(df64 acc, float x, float y) {
    return df64_add(acc, two_prod(x, y));
}

#endif // RAMSES_MSL_DF64_H

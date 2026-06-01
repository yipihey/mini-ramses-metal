//============================================================================
// reduce.h  —  Metal analog of gpu_reduce.cuf (+ the fixed-point accumulation
// primitives the deposit/reduction kernels share).
//
//   * 64-bit signed fixed-point accumulation from two 32-bit atomic words
//     (this GPU has no native 64-bit atomics; integer add is exactly
//     associative -> order-independent, bit-reproducible).
//   * atomic max of a non-negative float via monotone uint bit pattern.
//   * SIMD/threadgroup sum & max reductions (gpu_reduce.cuf analogues).
//============================================================================
#ifndef RAMSES_MSL_REDUCE_H
#define RAMSES_MSL_REDUCE_H

#include <metal_stdlib>
#include "../ramses_metal.h"
#include "df64.h"
using namespace metal;

// Two-word (lo,hi) signed fixed-point 64-bit atomic add.
inline void atomic_add_i64(device atomic_uint* lo, device atomic_uint* hi, long q) {
    uint qlo = (uint)((ulong)q & 0xffffffffUL);
    uint qhi = (uint)((ulong)q >> 32);                  // sign-extended high half
    uint old = atomic_fetch_add_explicit(lo, qlo, memory_order_relaxed);
    uint carry = (old + qlo < old) ? 1u : 0u;           // unsigned overflow == carry
    atomic_fetch_add_explicit(hi, qhi + carry, memory_order_relaxed);
}
// Convenience: quantise a float and accumulate at flat index `idx`.
inline void atomic_add_fixed(device atomic_uint* lo, device atomic_uint* hi,
                             int idx, float v, float scale) {
    long q = (long)round(v * scale);
    atomic_add_i64(&lo[idx], &hi[idx], q);
}
// Recombine a two-word fixed-point accumulator into a float.
inline float fixed_to_float(uint lo, uint hi, float inv_scale) {
    long q = (long)(((ulong)hi << 32) | (ulong)lo);
    return (float)q * inv_scale;
}

// atomic max of a NON-NEGATIVE float via monotonic uint bit pattern.
inline void atomic_max_f_nonneg(device atomic_uint* acc, float v) {
    atomic_fetch_max_explicit(acc, as_type<uint>(max(v, 0.0f)), memory_order_relaxed);
}

// Threadgroup reductions.  SIMD width is not hardcoded; callers pass the
// threadgroup's simdgroup geometry.
inline float block_reduce_sum(float v, threadgroup float* scratch,
                              uint lane, uint sg_id, uint n_sg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    if (sg_id == 0) {
        float x = (lane < n_sg) ? scratch[lane] : 0.0f;
        r = simd_sum(x);
    }
    return r;   // valid in (sg_id==0, lane==0)
}
inline float block_reduce_max(float v, threadgroup float* scratch,
                              uint lane, uint sg_id, uint n_sg) {
    v = simd_max(v);
    if (lane == 0) scratch[sg_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    if (sg_id == 0) {
        float x = (lane < n_sg) ? scratch[lane] : 0.0f;
        r = simd_max(x);
    }
    return r;
}

// df64 reductions (gpu_reduce.cuf warp_reduce_sum / block_reduce_sum, but the
// fp64 accumulation is emulated in double-single).  Used for the fp64-critical
// global sums (residual_norm, cmp_epot).  simd_sum can't reduce a struct, so we
// shuffle hi/lo and df64_add (the shuffle-down reduction CUDA's warp_reduce_sum
// does).  Requires -fno-fast-math (see df64.h).
inline df64 simd_sum_df64(df64 v) {
    for (uint off = 16; off > 0; off >>= 1)           // warpsize/2 = 16 (Apple simd=32)
        v = df64_add(v, df64{ simd_shuffle_down(v.hi, off), simd_shuffle_down(v.lo, off) });
    return v;                                          // valid in lane 0
}
inline df64 block_reduce_sum_df64(df64 v, threadgroup df64* scratch,
                                  uint lane, uint sg_id, uint n_sg) {
    v = simd_sum_df64(v);
    if (lane == 0) scratch[sg_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    df64 r = df64_from(0.0f);
    if (sg_id == 0) {
        df64 x = (lane < n_sg) ? scratch[lane] : df64_from(0.0f);
        r = simd_sum_df64(x);
    }
    return r;   // valid in (sg_id==0, lane==0)
}

#endif // RAMSES_MSL_REDUCE_H

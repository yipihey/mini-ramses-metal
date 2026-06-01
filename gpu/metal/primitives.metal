//============================================================================
// primitives.metal — test/validation kernels for the leaf primitives in
// ramses_msl.h (hilbert_key, two-word fixed-point atomics).  Exercised by
// test_harness.mm.  These are validation-only and not part of the solver.
//============================================================================
#include "ramses_msl.h"

kernel void test_hilbert(
    device const int* ckx   [[buffer(0)]],
    device const int* cky   [[buffer(1)]],
    device const int* ckz   [[buffer(2)]],
    device long*      outk  [[buffer(3)]],
    device atomic_uint* acc_lo [[buffer(4)]],
    device atomic_uint* acc_hi [[buffer(5)]],
    constant int&     level [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int3 ix = int3(ckx[gid], cky[gid], ckz[gid]);
    outk[gid] = hilbert_key(ix, level);
    // Each thread deposits exactly 1.0 into the shared fixed-point accumulator;
    // the host checks the recombined sum equals the thread count.
    atomic_add_fixed(acc_lo, acc_hi, 0, 1.0f, (float)(1L << FP_SHIFT_RHO));
}

// Report the metallib's COMPILE-TIME NDIM/TWOTONDIM so the host can verify the
// loaded .metallib matches its own NDIM.  A mismatch (e.g. a 1D binary loading the
// default ../../bin/ NDIM=3 lib) silently strides every oct by the wrong factor
// and produces garbage; the host aborts instead (see adaptive_loop mtl_init guard).
kernel void probe_ndim(device int* out [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid == 0) { out[0] = NDIM; out[1] = TWOTONDIM; }
}

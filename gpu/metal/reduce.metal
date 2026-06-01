//============================================================================
// reduce.metal  —  Metal analog of gpu_reduce.cuf reduction KERNELS.
//
//   * residual_norm : sum r^2 over unmasked cells (per-threadgroup partial;
//                        host sums the few partials) -> MG convergence norm.
//   * mg_phi_sum       : sum phi over active cells (for the zero-mean gauge pin).
//   * mg_phi_shift     : phi -= c (apply the gauge shift).
// Device-side reduction primitives (block_reduce_sum/max, fixed-point atomics)
// live in reduce.h.
//============================================================================
#include "ramses_msl.h"

// residual_norm (gpu_mg.cuf:954): sum f(:,1)^2 over UNMASKED cells (f(:,3)>0).
// CUDA reduces in fp64 -> df64 here (hybrid: a global reduction; reduce.h).
// One thread per cell (gid -> oct,cell, like cmp_epot / CUDA); the host sums the
// df64 partials.  This is the MG convergence norm (squared).
kernel void residual_norm(
    device const float* f       [[buffer(0)]],   // f(:,1)=resid, f(:,3)=mask
    device df64*        partial [[buffer(1)]],   // one df64 per threadgroup
    constant MgParams&  P       [[buffer(2)]],
    uint  gid   [[thread_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  n_sg  [[simdgroups_per_threadgroup]],
    uint  tgid  [[threadgroup_position_in_grid]])
{
    float s = 0.0f;
    int total = P.num_octs * TWOTONDIM;
    if ((int)gid < total) {
        int oct  = P.head_idx + (int)gid / TWOTONDIM;
        int cell = (int)gid % TWOTONDIM + 1;
        if (f[IDX3(cell, 3, oct)] > 0.0f) { float r = f[IDX3(cell, 1, oct)]; s = r * r; }
    }
    threadgroup df64 scratch[32];
    df64 res = block_reduce_sum_df64(df64_from(s), scratch, lane, sg_id, n_sg);
    if (sg_id == 0 && lane == 0) partial[tgid] = res;
}

// (mg_phi_sum / mg_phi_shift -- the zero-mean "gauge pin" -- are DELETED: they are a
// Metal-only invention, not in CUDA/CPU.  The periodic-base null space is handled the
// faithful way: fp64 in CUDA, df64 in the hybrid Metal port (the residual/RHS-mean and
// reductions carry df64 so the constant mode does not leak as it did in pure fp32).)

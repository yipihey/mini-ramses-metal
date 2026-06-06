//============================================================================
// sort.metal — Hilbert-key LSD radix sort of particles (gpu_part.cuf +
// gpu_refine.cuf swap-table kernels).
//
// Stable counting sort, ONE key bit per pass, over the active particle range
// [head_idx .. head_idx+num_parts-1].  Per pass:
//   init_prefix_sum_part_bit -> (device inclusive scan, scan.metal) ->
//   compute_local_swap_table -> update_global_swap_table.
// `sortp` is the running permutation (swap_global); `isp_swap` is swap_local;
// `prefix_sum` holds the per-element bit then its inclusive scan.
//
// 1-based Fortran index arithmetic preserved; arrays dereferenced 0-based.
//============================================================================
#include "ramses_msl.h"

// Cartesian key from fixed-point position (exact bit-shift) -> Hilbert key.
kernel void compute_hkey_part(
    device const long*  ipos      [[buffer(0)]],   // 64-bit fixed-point (3 cols)
    device long*        hkey_part [[buffer(1)]],
    constant ScanParams& P        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)P.n) return;
    int ipart = P.head_idx + (int)gid;
    int ix3[3] = {0,0,0};
    for (int idim = 1; idim <= NDIM; ++idim)
        ix3[idim-1] = cell_index_fix(ipos[IDXP(ipart, idim, P.npartmax)], P.ilevel);
    hkey_part[ipart - 1] = hilbert_key(int3(ix3[0], ix3[1], ix3[2]), P.ilevel);
}

kernel void init_global_swap_table(
    device int*          swap_global [[buffer(0)]],
    constant ScanParams& P           [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)P.n) return;
    int oct = P.head_idx + (int)gid;
    swap_global[oct - 1] = oct;            // identity (1-based values)
}

kernel void init_prefix_sum_part_bit(
    device const long*   hkey_part   [[buffer(0)]],
    device const int*    swap_global [[buffer(1)]],
    device int*          prefix_sum  [[buffer(2)]],
    constant ScanParams& P           [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)P.n) return;
    int idx = P.head_idx + (int)gid;
    int old_idx = swap_global[idx - 1];
    prefix_sum[idx - 1] = (int)((hkey_part[old_idx - 1] >> P.ibit) & 1L);
}

// Stable split using the INCLUSIVE-scanned bit array in prefix_sum.
kernel void compute_local_swap_table(
    device int*          swap_local  [[buffer(0)]],
    device const int*    swap_global [[buffer(1)]],
    device const int*    prefix_sum  [[buffer(2)]],
    constant ScanParams& P           [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)P.n) return;
    int oct    = P.head_idx + (int)gid;
    int nfinal = P.head_idx + P.n - 1;
    int nones  = prefix_sum[nfinal - 1];
    int nzeros = P.n - nones;
    int prev   = (oct > P.head_idx) ? prefix_sum[oct - 2] : 0;
    int bit    = prefix_sum[oct - 1] - prev;
    int old    = swap_global[oct - 1];
    int sorted = (bit == 0) ? (oct - prev) : (P.head_idx + nzeros + prev);
    swap_local[sorted - 1] = old;
}

kernel void update_global_swap_table(
    device int*          swap_global [[buffer(0)]],
    device const int*    swap_local  [[buffer(1)]],
    constant ScanParams& P           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)P.n) return;
    int oct = P.head_idx + (int)gid;
    swap_global[oct - 1] = swap_local[oct - 1];
}

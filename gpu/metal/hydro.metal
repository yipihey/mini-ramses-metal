//============================================================================
// hydro.metal  <-  gpu_hydro.cuf  (hydro kernels)
//
// The Metal hydro kernels.  Device math lives in hydro.h.  This file currently
// has the simple whole-array kernels (set_unew/set_uold/upload); the Godunov
// pipeline (subgrid load -> trace -> riemann -> flux -> update) lands next.
// State uold/unew is column-major (twotondim, nvar, noct); index via UH().
//
// Launch convention (simpler than the CUDA 2D thread block): ONE thread per oct,
// looping its TWOTONDIM cells x NHVAR vars.  oct = head_idx + gid (1-based).
//============================================================================
#include <metal_stdlib>
#include "../ramses_metal.h"
#include "hydro.h"
using namespace metal;

// unew <- uold  (set_unew_kernel, gpu_hydro.cuf:1643)
kernel void set_unew(device const float*       uold [[buffer(0)]],
                     device       float*       unew [[buffer(1)]],
                     constant     HydroParams& P    [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    for (int c = 1; c <= TWOTONDIM; ++c)
        for (int v = 1; v <= NHVAR; ++v)
            unew[UH(c, v, oct)] = uold[UH(c, v, oct)];
}

// uold <- unew  (set_uold_kernel, gpu_hydro.cuf:1613)
kernel void set_uold(device       float*       uold [[buffer(0)]],
                     device const float*       unew [[buffer(1)]],
                     constant     HydroParams& P    [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    for (int c = 1; c <= TWOTONDIM; ++c)
        for (int v = 1; v <= NHVAR; ++v)
            uold[UH(c, v, oct)] = unew[UH(c, v, oct)];
}

// Restriction: average the TWOTONDIM cells of a fine oct into its parent cell
// (upload_kernel, gpu_hydro.cuf:1567).  One fine oct -> one parent cell, so no
// race (each parent cell has exactly one child oct).  NDIM-generic cell index.
kernel void upload(device const Oct*         grid   [[buffer(0)]],
                   device const int*         father [[buffer(1)]],
                   device       float*       uold   [[buffer(2)]],
                   constant     HydroParams& P      [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    int f = father[oct - 1];

    int cell = 1 + (grid[oct - 1].ckey[0] - 2 * grid[f - 1].ckey[0]);
#if NDIM >= 2
    cell += 2 * (grid[oct - 1].ckey[1] - 2 * grid[f - 1].ckey[1]);
#endif
#if NDIM >= 3
    cell += 4 * (grid[oct - 1].ckey[2] - 2 * grid[f - 1].ckey[2]);
#endif

    const float inv = 1.0f / (float)TWOTONDIM;
    for (int v = 1; v <= NHVAR; ++v) {
        float acc = 0.0f;
        for (int ind = 1; ind <= TWOTONDIM; ++ind)
            acc += uold[UH(ind, v, oct)] * inv;
        uold[UH(cell, v, f)] = acc;
    }
}

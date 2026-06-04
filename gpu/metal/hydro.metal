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

//----------------------------------------------------------------------------
// Godunov integrator (uniform-grid path).  One thread per oct: gather the
// 6^NDIM primitive subgrid from the 3^NDIM neighbour-oct cube (nbor), apply the
// gravity half-step predictor, run the validated godunov_oct, and ADD the
// conservative update into unew (which already holds a copy of uold from
// set_unew) -- faithful to hydro_integrator_kernel for a single-level region.
// AMR coarse-fine (zero_fine_fluxes / coarse_cell_update / cache octs) is added
// next; on a uniform level there are no refined neighbours so this is exact.
//----------------------------------------------------------------------------
#if   NDIM == 1
#define SG_N 6
#elif NDIM == 2
#define SG_N 36
#else
#define SG_N 216
#endif

// Load one cell's primitives with the gravity half-step predictor (gpu_hydro.cuf:355).
inline HPrimitive load_cell_prim(device const float* uold, device const float* fgrav,
                                 int cell, int oct, float gamma, float halfdt) {
    HConserved c = { uold[UH(cell,1,oct)], uold[UH(cell,2,oct)], uold[UH(cell,3,oct)],
                     uold[UH(cell,4,oct)], uold[UH(cell,5,oct)] };
    HPrimitive p = conserved_2_primitive(c, gamma);
    p.velocity_x += fgrav[IDX3(cell,1,oct)] * halfdt;
    p.velocity_y += fgrav[IDX3(cell,2,oct)] * halfdt;
    p.velocity_z += fgrav[IDX3(cell,3,oct)] * halfdt;
    return p;
}

kernel void hydro_godunov(device const float*       uold  [[buffer(0)]],
                          device       float*       unew  [[buffer(1)]],
                          device const float*       fgrav [[buffer(2)]],
                          device const int*         nbor  [[buffer(3)]],
                          constant     HydroParams& P     [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    float gamma = P.gamma, halfdt = 0.5f * P.dt, dtdx = P.dt / P.dx;

    // Gather the 6^NDIM subgrid from the 3^NDIM neighbour cube.  Subgrid cell
    // position sp in 0..5 per axis -> cube offset sp/2 (0..2) + sub-bit sp%2.
    HPrimitive sg[SG_N];
#if NDIM == 1
    for (int sx = 0; sx < 6; ++sx) {
        int nb = nbor[(oct-1)*SUBGRIDSIZE + (sx/2)];
        if (nb <= 0) nb = oct;                                   // uniform: never hit
        sg[sx] = load_cell_prim(uold, fgrav, 1 + (sx&1), nb, gamma, halfdt);
    }
    HConserved du[2];
    godunov_oct_1d(sg, gamma, dtdx, P.slope_type, P.riemann, du);
#else
    for (int sz = 0; sz < 6; ++sz)
    for (int sy = 0; sy < 6; ++sy)
    for (int sx = 0; sx < 6; ++sx) {
        int cube = (sx/2) + 3*(sy/2) + 9*(sz/2);
        int nb = nbor[(oct-1)*SUBGRIDSIZE + cube];
        if (nb <= 0) nb = oct;
        int cell = 1 + (sx&1) + 2*(sy&1) + 4*(sz&1);
        sg[sx + 6*sy + 36*sz] = load_cell_prim(uold, fgrav, cell, nb, gamma, halfdt);
    }
    HConserved du[8];
    godunov_oct_3d(sg, gamma, dtdx, P.slope_type, P.riemann, du);
#endif

    for (int c = 1; c <= TWOTONDIM; ++c) {
        unew[UH(c,1,oct)] += du[c-1].density;
        unew[UH(c,2,oct)] += du[c-1].momentum_x;
        unew[UH(c,3,oct)] += du[c-1].momentum_y;
        unew[UH(c,4,oct)] += du[c-1].momentum_z;
        unew[UH(c,5,oct)] += du[c-1].energy;
    }
}

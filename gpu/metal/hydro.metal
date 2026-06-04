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
#include "reduce.h"     // atomic_add_fixed / fixed_to_float for the coarse-fine reflux
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
// Godunov integrator.  One thread per oct: gather the 6^NDIM primitive subgrid
// (+ refined flags) from the 3^NDIM neighbour-oct cube (nbor), apply the gravity
// half-step predictor, run the AMR-aware godunov_oct (which zeros fluxes at
// refined faces and returns the boundary-face flux sums), ADD the conservative
// update into unew, and scatter the coarse-fine reflux correction onto any
// coarser (cache-oct) neighbour's parent cell via reproducible fixed-point
// atomics.  Faithful to hydro_integrator_kernel (subgrid_conserved_2_primitive +
// trace_3d + riemann_driver + zero_fine_fluxes + conservative_update +
// coarse_cell_update).  On a uniform level (no refined cells, no cache octs)
// this reduces EXACTLY to the plain Godunov update -- the reflux is a no-op.
//----------------------------------------------------------------------------
#if   NDIM == 1
#define SG_N 6
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

// Scatter one boundary-face flux sum onto the coarse parent of a cache-oct
// neighbour (coarse_cell_update, gpu_hydro.cuf:1118).  `signed_w` already folds
// in dt/dx, 1/twotondim and the face sign (-1 low / +1 high).
inline void reflux_face(device atomic_uint* lo, device atomic_uint* hi,
                        device const Oct* grid, device const int* father,
                        int nb, HConserved b, float signed_w, float fp_scale) {
    int fa = father[nb-1];
    int cell = 1 + (grid[nb-1].ckey[0] - 2*grid[fa-1].ckey[0]);
#if NDIM >= 2
    cell += 2 * (grid[nb-1].ckey[1] - 2*grid[fa-1].ckey[1]);
#endif
#if NDIM >= 3
    cell += 4 * (grid[nb-1].ckey[2] - 2*grid[fa-1].ckey[2]);
#endif
    atomic_add_fixed(lo, hi, UH(cell,1,fa), b.density   *signed_w, fp_scale);
    atomic_add_fixed(lo, hi, UH(cell,2,fa), b.momentum_x*signed_w, fp_scale);
    atomic_add_fixed(lo, hi, UH(cell,3,fa), b.momentum_y*signed_w, fp_scale);
    atomic_add_fixed(lo, hi, UH(cell,4,fa), b.momentum_z*signed_w, fp_scale);
    atomic_add_fixed(lo, hi, UH(cell,5,fa), b.energy    *signed_w, fp_scale);
}

kernel void hydro_godunov(device const float*       uold     [[buffer(0)]],
                          device       float*       unew     [[buffer(1)]],
                          device const float*       fgrav    [[buffer(2)]],
                          device const int*         nbor     [[buffer(3)]],
                          device const Oct*         grid     [[buffer(4)]],
                          device const int*         father   [[buffer(5)]],
                          device atomic_uint*       reflux_lo[[buffer(6)]],
                          device atomic_uint*       reflux_hi[[buffer(7)]],
                          constant     HydroParams& P        [[buffer(8)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    float gamma = P.gamma, halfdt = 0.5f * P.dt, dtdx = P.dt / P.dx;

    // Gather the 6^NDIM subgrid + refined flags from the 3^NDIM neighbour cube.
    // Subgrid cell position sp in 0..5/axis -> cube offset sp/2 + sub-bit sp&1.
    HPrimitive sg[SG_N];
    bool ref[SG_N];
#if NDIM == 1
    for (int sx = 0; sx < 6; ++sx) {
        int nb = nbor[(oct-1)*SUBGRIDSIZE + (sx/2)];
        if (nb <= 0) nb = oct;
        int cell = 1 + (sx&1);
        sg[sx]  = load_cell_prim(uold, fgrav, cell, nb, gamma, halfdt);
        ref[sx] = grid[nb-1].refined[cell-1] != 0;
    }
    HConserved du[2], bnd[2];
    godunov_oct_1d_amr(sg, ref, gamma, dtdx, P.slope_type, P.riemann, du, bnd);
#else
    for (int sz = 0; sz < 6; ++sz)
    for (int sy = 0; sy < 6; ++sy)
    for (int sx = 0; sx < 6; ++sx) {
        int cube = (sx/2) + 3*(sy/2) + 9*(sz/2);
        int nb = nbor[(oct-1)*SUBGRIDSIZE + cube];
        if (nb <= 0) nb = oct;
        int cell = 1 + (sx&1) + 2*(sy&1) + 4*(sz&1);
        int idx = sx + 6*sy + 36*sz;
        sg[idx]  = load_cell_prim(uold, fgrav, cell, nb, gamma, halfdt);
        ref[idx] = grid[nb-1].refined[cell-1] != 0;
    }
    HConserved du[8], bnd[6];
    godunov_oct_3d_amr(sg, ref, gamma, dtdx, P.slope_type, P.riemann, du, bnd);
#endif

    for (int c = 1; c <= TWOTONDIM; ++c) {
        unew[UH(c,1,oct)] += du[c-1].density;
        unew[UH(c,2,oct)] += du[c-1].momentum_x;
        unew[UH(c,3,oct)] += du[c-1].momentum_y;
        unew[UH(c,4,oct)] += du[c-1].momentum_z;
        unew[UH(c,5,oct)] += du[c-1].energy;
    }

    // Coarse-fine reflux: for each outer face whose neighbour is a coarser
    // (cache) oct, add the boundary-flux correction to its coarse parent cell.
    if (P.ilevel <= P.levelmin) return;
    float w = dtdx / (float)TWOTONDIM;
    int base = (oct-1)*SUBGRIDSIZE;
#if NDIM == 1
    const int cube[2] = {0, 2};                     // -x, +x  (center cube = 1)
#else
    const int cube[6] = {12, 14, 10, 16, 4, 22};    // -x,+x,-y,+y,-z,+z (center = 13)
#endif
    const float face_sign[6] = {-1.0f,+1.0f,-1.0f,+1.0f,-1.0f,+1.0f};
    for (int fc = 0; fc < TWONDIM; ++fc) {
        int nb = nbor[base + cube[fc]];
        if (nb > P.ngridmax)   // cache oct == coarser neighbour
            reflux_face(reflux_lo, reflux_hi, grid, father, nb, bnd[fc], face_sign[fc]*w, P.fp_scale);
    }
}

// Finalize the coarse-fine reflux: add the fixed-point corrections accumulated
// by hydro_godunov into unew, for the octs of the COARSER level.  One thread per
// oct; the host zeroes the lo/hi buffers before each integrator pass.
kernel void hydro_reflux_finalize(device       float*       unew      [[buffer(0)]],
                                  device       atomic_uint* reflux_lo [[buffer(1)]],
                                  device       atomic_uint* reflux_hi [[buffer(2)]],
                                  constant     HydroParams& P         [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    float inv = 1.0f / P.fp_scale;
    for (int c = 1; c <= TWOTONDIM; ++c)
        for (int v = 1; v <= NHVAR; ++v) {
            int idx = UH(c, v, oct);
            uint lo = atomic_load_explicit(&reflux_lo[idx], memory_order_relaxed);
            uint hi = atomic_load_explicit(&reflux_hi[idx], memory_order_relaxed);
            unew[idx] += fixed_to_float(lo, hi, inv);
        }
}

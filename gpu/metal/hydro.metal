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
#include "nbor.h"       // mg_nbor (3^NDIM-neighbour oct) for the flag kernel
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

//----------------------------------------------------------------------------
// Gravity source coupling (one thread per cell: oct = head_idx + gid/twotondim,
// cell = gid%twotondim + 1).  The port always carries the force in `f` (zero for
// pure hydro), so the constant_gravity branch of the CUDA code is not needed.
//----------------------------------------------------------------------------

// sync_hydro (gpu_hydro.cuf:1673): full-step velocity kick on uold, u += f*dt.
kernel void sync_hydro(device       float*       uold  [[buffer(0)]],
                       device const float*       fgrav [[buffer(1)]],
                       constant     HydroParams& P     [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)(P.num_octs * TWOTONDIM)) return;
    int oct  = P.head_idx + (int)gid / TWOTONDIM;
    int cell = (int)gid % TWOTONDIM + 1;
    HConserved c = { uold[UH(cell,1,oct)], uold[UH(cell,2,oct)], uold[UH(cell,3,oct)],
                     uold[UH(cell,4,oct)], uold[UH(cell,5,oct)] };
    HPrimitive p = conserved_2_primitive(c, P.gamma);
    p.velocity_x += fgrav[IDX3(cell,1,oct)] * P.dt;
    p.velocity_y += fgrav[IDX3(cell,2,oct)] * P.dt;
    p.velocity_z += fgrav[IDX3(cell,3,oct)] * P.dt;
    c = primitive_2_conserved(p, P.gamma);
    uold[UH(cell,1,oct)]=c.density; uold[UH(cell,2,oct)]=c.momentum_x; uold[UH(cell,3,oct)]=c.momentum_y;
    uold[UH(cell,4,oct)]=c.momentum_z; uold[UH(cell,5,oct)]=c.energy;
}

// grav_hydro (gpu_hydro.cuf:1731): velocity kick on unew with the rho_old/rho_new
// momentum-conservation factor; rho_old=uold, rho_new=unew.
kernel void grav_hydro(device const float*       uold  [[buffer(0)]],
                       device       float*       unew  [[buffer(1)]],
                       device const float*       fgrav [[buffer(2)]],
                       constant     HydroParams& P     [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)(P.num_octs * TWOTONDIM)) return;
    int oct  = P.head_idx + (int)gid / TWOTONDIM;
    int cell = (int)gid % TWOTONDIM + 1;
    HConserved c = { unew[UH(cell,1,oct)], unew[UH(cell,2,oct)], unew[UH(cell,3,oct)],
                     unew[UH(cell,4,oct)], unew[UH(cell,5,oct)] };
    HPrimitive p = conserved_2_primitive(c, P.gamma);
    float rho_old = uold[UH(cell,1,oct)], rho_new = unew[UH(cell,1,oct)];
    float fac = P.dt * rho_old / rho_new;
    p.velocity_x += fgrav[IDX3(cell,1,oct)] * fac;
    p.velocity_y += fgrav[IDX3(cell,2,oct)] * fac;
    p.velocity_z += fgrav[IDX3(cell,3,oct)] * fac;
    c = primitive_2_conserved(p, P.gamma);
    unew[UH(cell,1,oct)]=c.density; unew[UH(cell,2,oct)]=c.momentum_x; unew[UH(cell,3,oct)]=c.momentum_y;
    unew[UH(cell,4,oct)]=c.momentum_z; unew[UH(cell,5,oct)]=c.energy;
}

//----------------------------------------------------------------------------
// CFL timestep + conservation diagnostics (cmpdt_kernel, gpu_hydro.cuf:1795).
// One thread per cell; refined cells are skipped.  Reductions are reproducible:
// dt via atomic uint-min, mass/ekin/eint/emag via two-word fixed-point atomics
// at P.fp_scale.  red[0]=dt (init to FLT_MAX bits), red[1..8]= the four sums.
//----------------------------------------------------------------------------
kernel void hydro_cmpdt(device const Oct*         grid  [[buffer(0)]],
                        device const float*       uold  [[buffer(1)]],
                        device const float*       fgrav [[buffer(2)]],
                        device       atomic_uint* red   [[buffer(3)]],
                        constant     HydroParams& P     [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)(P.num_octs * TWOTONDIM)) return;
    int oct  = P.head_idx + (int)gid / TWOTONDIM;
    int cell = (int)gid % TWOTONDIM + 1;
    if (grid[oct-1].refined[cell-1] != 0) return;     // leaf cells only

    HConserved c = { uold[UH(cell,1,oct)], uold[UH(cell,2,oct)], uold[UH(cell,3,oct)],
                     uold[UH(cell,4,oct)], uold[UH(cell,5,oct)] };
    HPrimitive p = conserved_2_primitive(c, P.gamma);

    float dx = P.dx, gamma = P.gamma;
    float vol = dx*dx*dx;                              // CUDA uses dx^3 for all NDIM
    float mass = p.density*vol;
    float ekin = c.energy*vol;
    float eint = p.pressure/(gamma-1.0f)*vol;

    float cs   = sqrt(gamma*p.pressure/p.density);
    float ctot = fabs(p.velocity_x)+fabs(p.velocity_y)+fabs(p.velocity_z)+3.0f*cs;
    float grav = fabs(fgrav[IDX3(cell,1,oct)])+fabs(fgrav[IDX3(cell,2,oct)])+fabs(fgrav[IDX3(cell,3,oct)]);
    grav = grav*dx/(ctot*ctot);
    grav = max(grav, 1.0e-4f);
    float dt_loc = dx/ctot*(sqrt(1.0f+2.0f*P.courant_factor*grav)-1.0f)/grav;

    float s = P.fp_scale;
    atomic_min_f_nonneg(&red[0], dt_loc);
    atomic_add_i64(&red[1], &red[2], (long)round(mass*s));
    atomic_add_i64(&red[3], &red[4], (long)round(ekin*s));
    atomic_add_i64(&red[5], &red[6], (long)round(eint*s));
    // emag = 0 (no MHD) -> red[7,8] untouched.
}

//----------------------------------------------------------------------------
// Refinement flag from density/pressure gradients (hydro_flag_kernel,
// gpu_hydro.cuf:1466 + hydro_crit:1437).  One thread per cell; sets flag1=1 if
// the gradient across any dimension exceeds the threshold.  Neighbour cells come
// from the FLAG_hhh/FLAG_iii tables (== the MG tables) via mg_nbor.
//----------------------------------------------------------------------------
constant int FLAG_hhh[6][8] = {
    {2,1,4,3,6,5,8,7}, {2,1,4,3,6,5,8,7},
    {3,4,1,2,7,8,5,6}, {3,4,1,2,7,8,5,6},
    {5,6,7,8,1,2,3,4}, {5,6,7,8,1,2,3,4}
};
constant int FLAG_iii[6][8] = {
    {-1, 0,-1, 0,-1, 0,-1, 0}, { 0, 1, 0, 1, 0, 1, 0, 1},
    {-1,-1, 0, 0,-1,-1, 0, 0}, { 0, 0, 1, 1, 0, 0, 1, 1},
    {-1,-1,-1,-1, 0, 0, 0, 0}, { 0, 0, 0, 0, 1, 1, 1, 1}
};

inline bool hydro_crit(HPrimitive l, HPrimitive m, HPrimitive r,
                       float egd, float egp, float fd) {
    bool ok = false;
    if (egd > 0.0f) {
        float el = fabs((m.density - l.density)/(m.density + l.density + fd));
        float er = fabs((r.density - m.density)/(r.density + m.density + fd));
        if (2.0f*max(el,er) > egd) ok = true;
    }
    if (egp > 0.0f) {
        float el = fabs((m.pressure - l.pressure)/(m.pressure + l.pressure + fd));
        float er = fabs((r.pressure - m.pressure)/(r.pressure + m.pressure + fd));
        if (2.0f*max(el,er) > egp) ok = true;
    }
    return ok;
}

inline HPrimitive load_prim(device const float* uold, int cell, int oct, float gamma) {
    HConserved c = { uold[UH(cell,1,oct)], uold[UH(cell,2,oct)], uold[UH(cell,3,oct)],
                     uold[UH(cell,4,oct)], uold[UH(cell,5,oct)] };
    return conserved_2_primitive(c, gamma);
}

kernel void hydro_flag(device       int*             flag1 [[buffer(0)]],
                       device const Oct*             grid  [[buffer(1)]],
                       device const int*             nbor  [[buffer(2)]],
                       device const float*           uold  [[buffer(3)]],
                       constant     HydroFlagParams& P     [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)(P.num_octs * TWOTONDIM)) return;
    int oct  = P.head_idx + (int)gid / TWOTONDIM;
    int cell = (int)gid % TWOTONDIM + 1;
    HPrimitive mid = load_prim(uold, cell, oct, P.gamma);

    for (int idim = 1; idim <= NDIM; ++idim) {
        int dL = 2*(idim-1), dR = dL + 1;            // 0-based table rows
        int offL = FLAG_iii[dL][cell-1], offR = FLAG_iii[dR][cell-1];
        int icL  = FLAG_hhh[dL][cell-1], icR  = FLAG_hhh[dR][cell-1];
        int inL=0,jnL=0,knL=0, inR=0,jnR=0,knR=0;
        if      (idim==1) { inL=offL; inR=offR; }
        else if (idim==2) { jnL=offL; jnR=offR; }
        else              { knL=offL; knR=offR; }
        int nbL = mg_nbor(nbor, oct, inL, jnL, knL);
        int nbR = mg_nbor(nbor, oct, inR, jnR, knR);
        HPrimitive lft = load_prim(uold, icL, nbL, P.gamma);
        HPrimitive rgt = load_prim(uold, icR, nbR, P.gamma);
        if (hydro_crit(lft, mid, rgt, P.err_grad_d, P.err_grad_p, P.floor_d))
            flag1[IDX2(cell, oct)] = 1;
    }
}

//----------------------------------------------------------------------------
// Fill a coarse-fine CACHE (ghost) oct's uold by interpol_hydro from its coarse
// parent cell + that cell's 2*NDIM coarse face neighbours (interpol_hydro.f90,
// the godfine1:534 ghost).  One thread per cache oct.  After mtl_make_cache has
// created the cache octs (correct father/ckey + patched the fine octs' nbor),
// this gives them the SAME ghost the CPU Godunov reads, so the coarse-fine flux
// (and hence the reflux) matches.  Coarse neighbour gather: FLAG_hhh/FLAG_iii +
// mg_nbor on the COARSE parent oct (== get_twondim_nbor_parent_cell).
//----------------------------------------------------------------------------
kernel void hydro_fill_cache(device       float*               uold  [[buffer(0)]],
                             device const Oct*                 grid  [[buffer(1)]],
                             device const int*                 nbor  [[buffer(2)]],
                             device const int*                 father[[buffer(3)]],
                             constant     HydroInterpolParams& P     [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)P.num_octs) return;
    int cache = P.head_idx + (int)gid;
    int fa = father[cache-1];
    if (fa <= 0) return;
    // parent cell of the cache oct within its coarse father
    int cellp = 1 + (grid[cache-1].ckey[0] - 2*grid[fa-1].ckey[0]);
#if NDIM >= 2
    cellp += 2 * (grid[cache-1].ckey[1] - 2*grid[fa-1].ckey[1]);
#endif
#if NDIM >= 3
    cellp += 4 * (grid[cache-1].ckey[2] - 2*grid[fa-1].ckey[2]);
#endif
    // gather the coarse stencil: u1[0]=parent cell, u1[2i-1]=- nbr, u1[2i]=+ nbr
    HConserved u1[1 + 2*NDIM];
    u1[0] = (HConserved){ uold[UH(cellp,1,fa)], uold[UH(cellp,2,fa)], uold[UH(cellp,3,fa)],
                          uold[UH(cellp,4,fa)], uold[UH(cellp,5,fa)] };
    for (int dir = 1; dir <= 2*NDIM; ++dir) {
        int idim = (dir-1)/2;
        int off  = FLAG_iii[dir-1][cellp-1];
        int nc   = FLAG_hhh[dir-1][cellp-1];
        int in=0,jn=0,kn=0;
        if      (idim==0) in=off; else if (idim==1) jn=off; else kn=off;
        int no = mg_nbor(nbor, fa, in, jn, kn);
        if (no <= 0) no = fa;                         // domain edge fallback (periodic resolves)
        u1[dir] = (HConserved){ uold[UH(nc,1,no)], uold[UH(nc,2,no)], uold[UH(nc,3,no)],
                                uold[UH(nc,4,no)], uold[UH(nc,5,no)] };
    }
    HConserved u2[TWOTONDIM];
    interpol_hydro_oct(u1, P.interpol_var, P.interpol_type, P.smallr, u2);
    for (int c = 1; c <= TWOTONDIM; ++c) {
        uold[UH(c,1,cache)] = u2[c-1].density;    uold[UH(c,2,cache)] = u2[c-1].momentum_x;
        uold[UH(c,3,cache)] = u2[c-1].momentum_y; uold[UH(c,4,cache)] = u2[c-1].momentum_z;
        uold[UH(c,5,cache)] = u2[c-1].energy;
    }
}

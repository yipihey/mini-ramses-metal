//============================================================================
// flag.metal — AMR refinement flagging on the GPU (gpu_flag.cuf), nsubgrid==1.
// Computes flag1 (the refinement map) from the resident nref + grid + nbor +
// father, matching the CPU flag chain so the refined mesh is identical.  No
// atomicCAS / 64-bit atomics: init_flag's parent writes are idempotent (all
// write 1) and the count is done on the host.
//============================================================================
#include "ramses_msl.h"

constant int F_hhh[6][8] = {
    {2,1,4,3,6,5,8,7}, {2,1,4,3,6,5,8,7},
    {3,4,1,2,7,8,5,6}, {3,4,1,2,7,8,5,6},
    {5,6,7,8,1,2,3,4}, {5,6,7,8,1,2,3,4}
};
constant int F_iii[6][8] = {
    {-1, 0,-1, 0,-1, 0,-1, 0}, { 0, 1, 0, 1, 0, 1, 0, 1},
    {-1,-1, 0, 0,-1,-1, 0, 0}, { 0, 0, 1, 1, 0, 0, 1, 1},
    {-1,-1,-1,-1, 0, 0, 0, 0}, { 0, 0, 0, 0, 1, 1, 1, 1}
};

kernel void flag_reset(device int* flag, constant FlagParams& P [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.y >= P.num_octs) return;
    flag[IDX2((int)gid.x+1, P.head_idx+(int)gid.y)] = 0;
}

// Over level (ilevel+1) octs: flag the parent cell at ilevel if any child cell
// is refined or already flagged (init_flag_kernel).
kernel void flag_init(device int* flag1 [[buffer(0)]], device const Oct* grid [[buffer(1)]],
                      device const int* father [[buffer(2)]], constant FlagParams& P [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= P.num_octs) return;
    int oct = P.head_idx + (int)gid, fa = father[oct-1];
    if (fa <= 0) return;
    int cell = 1;                              // which parent cell this oct refines
    for (int d=0; d<NDIM; ++d) cell += (grid[oct-1].ckey[d] - 2*grid[fa-1].ckey[d]) << d;
    bool ok = false;
    for (int ind=1; ind<=TWOTONDIM; ++ind)
        ok = ok || (grid[oct-1].refined[ind-1] != 0) || (flag1[IDX2(ind,oct)] == 1);
    if (ok) flag1[IDX2(cell, fa)] = 1;       // idempotent across the 8 children
}

// enforce_subgrid_kernel: refinement is per-oct (nsubgrid==1), so if ANY cell of
// an oct is flagged, flag the WHOLE oct.  Runs after init_flag, before the
// poisson/smooth passes (host gpu_enforce_subgrid).
kernel void flag_enforce_subgrid(device int* flag1 [[buffer(0)]],
                                 constant FlagParams& P [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    bool ok = false;
    for (int ind=1; ind<=TWOTONDIM; ++ind) ok = ok || (flag1[IDX2(ind,oct)] == 1);
    if (ok) for (int ind=1; ind<=TWOTONDIM; ++ind) flag1[IDX2(ind,oct)] = 1;
}

// flag2 = number of face-neighbours with flag1==1 (count_neighbors).
kernel void flag_count_nbor(device int* flag2 [[buffer(0)]], device const int* flag1 [[buffer(1)]],
                            device const int* nbor [[buffer(2)]], constant FlagParams& P [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.y >= P.num_octs) return;
    int oct = P.head_idx + (int)gid.y, cell = (int)gid.x + 1, n = 0;
    for (int idir=1; idir<=TWONDIM; ++idir) {
        int icell = F_hhh[idir-1][cell-1];
        int off = F_iii[idir-1][cell-1], in=0,jn=0,kn=0;
        if (idir<3) in=off; else if (idir<5) jn=off; else kn=off;
        int off3[3]={in,jn,kn}, ind=1;
        for (int d=0; d<NDIM; ++d) ind += (off3[d]+1) * POW3[d];
        int src = nbor[(oct-1)*SUBGRIDSIZE + (ind-1)];
        if (src > 0) n += flag1[IDX2(icell, src)];
    }
    flag2[IDX2(cell, oct)] = n;
}

// flag_count_kernel: flagged cells zero their count; cells with >=num_nbors
// flagged neighbours become flagged.
kernel void flag_propagate(device int* flag1 [[buffer(0)]], device int* flag2 [[buffer(1)]],
                           constant FlagParams& P [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.y >= P.num_octs) return;
    int oct = P.head_idx + (int)gid.y, cell = (int)gid.x + 1, idx = IDX2(cell, oct);
    if (flag1[idx] == 1) flag2[idx] = 0;
    if (flag2[idx] >= P.num_nbors) flag1[idx] = 1;
}

// DM refinement: flag cells whose particle count nref >= m_refine(level).
kernel void flag_poisson(device int* flag1 [[buffer(0)]], device const float* nref [[buffer(1)]],
                         constant FlagParams& P [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.y >= P.num_octs) return;
    int oct = P.head_idx + (int)gid.y, cell = (int)gid.x + 1, idx = IDX2(cell, oct);
    if (P.m_refine >= 0.0f && nref[idx] >= P.m_refine) flag1[idx] = 1;
}

// enforce_rules: if any of the 27 neighbours is missing/in-cache, unflag the oct.
kernel void flag_enforce_rules(device int* flag1 [[buffer(0)]], device const int* nbor [[buffer(1)]],
                               constant FlagParams& P [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    bool bad = false;
    for (int q=0; q<THREETONDIM; ++q) {            // 3^NDIM neighbour cube
        int nb = nbor[(oct-1)*SUBGRIDSIZE + q];
        if (nb == 0 || nb > P.ngridmax) bad = true;
    }
    if (bad) for (int c=1;c<=TWOTONDIM;++c) flag1[IDX2(c,oct)] = 0;
}

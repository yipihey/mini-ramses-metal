//============================================================================
// utils.h  —  Metal analog of gpu_utils.cuf (NDIM-generic spatial indexing).
//
// RAMSES Cartesian-key convention helpers, generalised so the kernels work for
// NDIM=1/2/3 (replacing the hardcoded 3D arithmetic):
//   * parent-cell index of an oct  = 1 + sum_d (ckey[d] & 1) * 2^d   (1+i+2j+4k)
//   * global oct hash key          = key_off[L] + sum_d ckey[d] * nx^d
//   * 3^NDIM neighbour-cube index   = 1 + sum_d (off[d]+1) * 3^d , off in {-1,0,1}
//   * CIC prolongation father index + weight (generalises MG_ccc/MG_bbb tables)
//   * fixed-point cell index / fraction from a packed position
//============================================================================
#ifndef RAMSES_MSL_UTILS_H
#define RAMSES_MSL_UTILS_H

#include <metal_stdlib>
#include "../ramses_metal.h"
using namespace metal;

// POW3[d] = 3^d for the neighbour-cube indexing (d < NDIM).
constant int POW3[4] = { 1, 3, 9, 27 };

// Floor division by 2 (round toward -inf), for signed coarse-oct offsets.
inline int floor_div2(int n) { return (n >= 0) ? (n/2) : ((n-1)/2); }

inline int mg_parent_cell(device const Oct& o) {
    int c = 0;
    for (int d = 0; d < NDIM; ++d) c += (o.ckey[d] & 1) << d;
    return c + 1;
}
// 1-based parent cell from an explicit (thread) Cartesian key.
inline int mg_parent_cell_ck(thread const int* ck) {
    int c = 0;
    for (int d = 0; d < NDIM; ++d) c += (ck[d] & 1) << d;
    return c + 1;
}
// Global oct key from a (thread) Cartesian key at level L (nx = ckey_max[L]).
inline long mg_oct_key(int nx, long off, thread const int* ck) {
    long key = off, stride = 1;
    for (int d = 0; d < NDIM; ++d) { key += (long)ck[d] * stride; stride *= (long)nx; }
    return key;
}

// CIC prolongation (interpol_phi) father-cube index + weight, generalising the
// 3D MG_ccc/MG_bbb tables (reproduces them exactly for NDIM=3):
//   fine cell `cell` (1..2^NDIM), CIC point `ia` (1..2^NDIM):
//     per dim d: a_d=1 -> NEAR (own parent cell, offset 0); a_d=0 -> FAR
//     (neighbour at -1 if the fine cell is in the lower half of the parent,
//      +1 if upper half).  father index = 1 + sum_d (off_d+1)*3^d.
//     weight = 3^popcount(ia-1) / 4^NDIM  (tensor CIC: 3/4 near, 1/4 far).
inline int mg_cic_father_index(int cell, int ia) {
    int idx = 1;
    for (int d = 0; d < NDIM; ++d) {
        int c_d = ((cell - 1) >> d) & 1;
        int a_d = ((ia   - 1) >> d) & 1;
        int off = (a_d == 1) ? 0 : (c_d ? +1 : -1);
        idx += (off + 1) * POW3[d];
    }
    return idx;
}
inline float mg_cic_weight(int ia) {
    float w = 1.0f;
    for (int d = 0; d < NDIM; ++d) w *= 0.25f;     // 1/4^NDIM
    int pc = popcount((uint)(ia - 1));
    for (int i = 0; i < pc; ++i) w *= 3.0f;        // * 3^popcount
    return w;
}

// Fixed-point position helpers (see NBITS_POS note in ramses_metal.h).
// Level-independent: cell index is an exact shift; the fraction keeps ~24 bits.
inline int cell_index_fix(long ipos, int level) {
    return (int)(ipos >> (NBITS_POS - level));
}
inline float cell_frac_fix(long ipos, int level) {
    int  shift = NBITS_POS - level;
    long low   = ipos & ((1L << shift) - 1L);
    return ldexp((float)low, -shift);     // (float)low keeps top ~24 bits; 2^-shift exact
}

#endif // RAMSES_MSL_UTILS_H

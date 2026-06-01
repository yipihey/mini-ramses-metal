//============================================================================
// nbor.h  —  Metal analog of gpu_nbor.cuf (device-side 3^NDIM father/neighbour
// gathers, nsubgrid==1).
//
//   * nbor_father_cells_mg : the THREETONDIM coarse (father-level) cells around
//     a fine oct, indexed q = sum_d (off_d+1)*3^d so it agrees with
//     mg_cic_father_index (the CIC prolongation; gpu_nbor.cuf nbor_father_cells*).
//   * mg_nbor              : the same-level 3^NDIM-neighbour oct for an offset.
// The non-MG nbor_father_cells (force-gradient ghost) is realised by mg_ghost_cell
// in mg.metal (the one-way coarse-fine interface, no ghost octs).
//============================================================================
#ifndef RAMSES_MSL_NBOR_H
#define RAMSES_MSL_NBOR_H

#include <metal_stdlib>
#include "../ramses_metal.h"
#include "utils.h"        // POW3, floor_div2
using namespace metal;

// Centre cell of the 3^NDIM cube (all offsets 0): 3D->13, 2D->4, 1D->1.
#define MG_CUBE_CENTER ((THREETONDIM - 1) / 2)

// 3^NDIM father (coarse) cells around a fine oct (nsubgrid==1). Fills THREETONDIM
// entries, indexed q = sum_d (off_d+1)*3^d so MG_ccc==mg_cic_father_index agree.
inline void nbor_father_cells_mg(device const Oct* grid, device const int* father,
        device const int* nbor_c, int oct_idx, int mg_idx,
        thread int* igrid_nbor, thread int* icell_nbor) {
    int father_idx = father[mg_idx - 1];        // coarse oct (1-based)
    int p[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) p[d] = grid[oct_idx-1].ckey[d] & 1;   // parity in parent
    for (int q=0; q<THREETONDIM; ++q) {
        int octoff = 1, cell = 1, rem = q;
        for (int d=0; d<NDIM; ++d) {
            int o_d = (rem % 3) - 1; rem /= 3;     // cube offset -1,0,1 for dim d
            int s   = p[d] + o_d;                  // sub-position
            int oo  = floor_div2(s);               // coarse-oct offset
            int cc  = s - 2*oo;                     // coarse-cell sub bit
            octoff += (oo+1) * POW3[d];
            cell   += cc << d;
        }
        igrid_nbor[q] = nbor_c[(father_idx-1)*SUBGRIDSIZE + (octoff-1)];
        icell_nbor[q] = cell;
    }
}

// 3^NDIM-neighbour oct (1-based value) for offset (in,jn,kn) in {-1,0,1} (jn/kn
// ignored for NDIM<their dim).
inline int mg_nbor(device const int* nbor, int oct, int in, int jn, int kn) {
    int off[3] = {in, jn, kn};
    int ind = 1;
    for (int d=0; d<NDIM; ++d) ind += (off[d]+1) * POW3[d];
    return nbor[(oct-1)*SUBGRIDSIZE + (ind-1)];
}

#endif // RAMSES_MSL_NBOR_H

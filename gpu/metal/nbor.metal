//============================================================================
// nbor.metal  —  Metal analog of gpu_nbor.cuf connectivity BUILD kernels
// (CUDA update_father_array / update_nbor_array, driven from gpu_runner).
//
// After the GPU hash rebuild (hash.metal), these fill, per oct, the AMR-parent
// father[] and the 3^NDIM same-level neighbour cube nbor[] by hash_get lookups
// (pure reads of the immutable table).  Device gathers nbor_father_cells_mg /
// mg_nbor live in nbor.h.
//============================================================================
#include "ramses_msl.h"

// father[oct] = AMR parent oct at level L-1 (0 if none / base).
kernel void conn_build_father(
    device const Oct*  grid     [[buffer(0)]],
    device int*        father   [[buffer(1)]],
    device const long* hkey     [[buffer(2)]],
    device const int*  hval     [[buffer(3)]],
    device const int*  ckey_max [[buffer(4)]],
    device const long* key_off  [[buffer(5)]],
    constant ConnParams& P      [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int o = P.head_idx + (int)gid, L = grid[o-1].lev;
    if (L < 1 || L > P.nlevelmax || L <= 1) { father[o-1] = 0; return; }
    int ck[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) ck[d] = grid[o-1].ckey[d] / 2;   // parent Cartesian key
    long key = mg_oct_key(ckey_max[L-1], key_off[L-1], ck);
    father[o-1] = hash_get(hkey, hval, P.hash_size, key);
}

// nbor[oct, 1..3^NDIM] = same-level neighbour octs (periodic wrap via box_min/max).
kernel void conn_build_nbor(
    device const Oct*  grid     [[buffer(0)]],
    device int*        nbor     [[buffer(1)]],
    device const long* hkey     [[buffer(2)]],
    device const int*  hval     [[buffer(3)]],
    device const int*  ckey_max [[buffer(4)]],
    device const long* key_off  [[buffer(5)]],
    device const int*  box_min  [[buffer(6)]],
    device const int*  box_max  [[buffer(7)]],
    constant ConnParams& P      [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int o = P.head_idx + (int)gid, L = grid[o-1].lev;
    if (L < 1 || L > P.nlevelmax) { for (int q=0;q<SUBGRIDSIZE;++q) nbor[(o-1)*SUBGRIDSIZE+q]=0; return; }
    int ck[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) ck[d] = grid[o-1].ckey[d];
    for (int q=0; q<THREETONDIM; ++q) {           // 3^NDIM neighbour cube
        int nck[3] = {0,0,0}, rem = q;
        for (int d=0; d<NDIM; ++d) {
            int o_d = (rem % 3) - 1; rem /= 3;
            int c = ck[d] + o_d;
            if (P.per[d]) {
                int bmn=box_min[(L-1)*3+d], bmx=box_max[(L-1)*3+d];
                if (c <  bmn) c = bmx-1;
                if (c >= bmx) c = bmn;
            }
            nck[d] = c;
        }
        long key = mg_oct_key(ckey_max[L], key_off[L], nck);
        nbor[(o-1)*SUBGRIDSIZE + q] = hash_get(hkey, hval, P.hash_size, key);
    }
}

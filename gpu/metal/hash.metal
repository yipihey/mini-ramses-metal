//============================================================================
// hash.metal  —  Metal analog of gpu_hash.cuf (the GPU-parallel hash REBUILD).
//
// The grid_dict is rebuilt on the GPU after the host has memset hkey/hval to 0
// (sentinel).  Each oct claims a bucket via 32-bit atomicCAS on the value slot
// (Apple GPUs have 32-bit atomics but NOT the 64-bit CAS the original host
// insert used), then writes its 64-bit key non-atomically — concurrent inserts
// only ever inspect the atomic value slot while probing, so the key write needs
// no atomic.  Oct keys are unique, so no two threads target the same final slot.
// (gpu_hash.cuf insert_hash_kernel.)  Device read path: hash.h.
//============================================================================
#include "ramses_msl.h"

kernel void hash_insert(
    device const Oct*  grid     [[buffer(0)]],
    device long*       hkey     [[buffer(1)]],
    device atomic_int* hval     [[buffer(2)]],
    device const int*  ckey_max [[buffer(3)]],
    device const long* key_off  [[buffer(4)]],
    constant ConnParams& P      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int o = P.head_idx + (int)gid, L = grid[o-1].lev;
    if (L < 1 || L > P.nlevelmax) return;          // free / invalid slot
    int ck[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) ck[d] = grid[o-1].ckey[d];
    long key = mg_oct_key(ckey_max[L], key_off[L], ck);
    int b = hash_bucket(key, P.hash_size);         // 1-based bucket
    for (;;) {
        int expected = 0;
        if (atomic_compare_exchange_weak_explicit(&hval[b-1], &expected, o,
                memory_order_relaxed, memory_order_relaxed)) {
            hkey[b-1] = key;                       // claimed an empty slot
            return;
        }
        if (expected != 0) b = (b % P.hash_size) + 1;   // occupied -> linear probe
        // expected==0 here means a spurious weak-CAS failure -> retry same slot
    }
}

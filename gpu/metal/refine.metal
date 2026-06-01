//============================================================================
// refine.metal — AMR mesh refinement on the GPU (gpu_refine.cuf), nsubgrid==1.
//
// data_on_device refine: the GPU owns the mesh.  Per level (top-down for create,
// then derefine), based on flag1:
//   * refine_create  : flagged + not-yet-refined cell -> allocate a child oct
//                       (atomicadd on a free-slot counter), set its ckey/lev/
//                       hkey, zero its refined/flag1, straight-inject the parent
//                       cell's grav fields, mark the parent cell refined.
//   * refine_derefine: oct whose parent cell is no longer flagged but is refined
//                       -> mark dead (lev=0); parent cell unrefined.
// Hole removal (lev==0) + reordering is a separate compaction pass; the hash /
// father / nbor are then rebuilt from the compacted grid (mtl_conn_*), so NO
// on-device hash delete is needed (the Apple-hostile part of gpu_refine).
//============================================================================
#include "ramses_msl.h"

// Create child octs for flagged, not-yet-refined cells at one level.
// 2D grid: x = cell (0..twotondim-1), y = oct offset within the level.
// `ifree` is a 1-based next-free-slot counter (atomic); the returned old value
// is the child's slot.  The caller seeds it to noct_used+1 and reads it back to
// learn how many octs were created.
kernel void refine_create(
    device Oct*        grid     [[buffer(0)]],
    device int*        flag1    [[buffer(1)]],
    device float*      f        [[buffer(2)]],
    device float*      phi      [[buffer(3)]],
    device float*      phi_old  [[buffer(4)]],
    device atomic_int* ifree    [[buffer(5)]],
    constant RefineParams& P    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct  = P.head_idx + (int)gid.y;
    int cell = (int)gid.x + 1;
    if (cell > TWOTONDIM) return;

    bool flagged = flag1[IDX2(cell,oct)] == 1;
    bool refined = grid[oct-1].refined[cell-1] != 0;
    if (!(flagged && !refined)) return;

    // Allocate a child slot (atomic returns the old next-free value).
    int child = atomic_fetch_add_explicit(ifree, 1, memory_order_relaxed);

    int lev = grid[oct-1].lev + 1;
    grid[child-1].lev = lev;
    int ck3[3] = {0,0,0};                       // child Cartesian key (NDIM-generic)
    for (int d=0; d<NDIM; ++d) {
        int bit = ((cell-1) >> d) & 1;
        int c   = 2*grid[oct-1].ckey[d] + bit;
        grid[child-1].ckey[d] = c;
        ck3[d] = c;
    }
    for (int q=0; q<TWOTONDIM; ++q) grid[child-1].refined[q] = 0;
    grid[child-1].hkey[0] = hilbert_key(int3(ck3[0],ck3[1],ck3[2]), lev-1);
    for (int q=1; q<=TWOTONDIM; ++q) flag1[IDX2(q,child)] = 0;

    // Straight injection of the parent cell's gravity fields into all 8 children
    // (make_new_oct GRAV block): f, phi, AND phi_old — the multigrid warm-start +
    // the time-extrapolation history for the next step's interpol_phi.
    for (int q=1; q<=TWOTONDIM; ++q) {
        f[IDX3(q,1,child)]     = f[IDX3(cell,1,oct)];
        f[IDX3(q,2,child)]     = f[IDX3(cell,2,oct)];
        f[IDX3(q,3,child)]     = f[IDX3(cell,3,oct)];
        phi[IDX2(q,child)]     = phi[IDX2(cell,oct)];
        phi_old[IDX2(q,child)] = phi_old[IDX2(cell,oct)];
    }

    grid[oct-1].refined[cell-1] = 1;   // mark parent cell refined
}

// Mark derefined octs dead (lev=0).  One thread per oct at the level.  Needs a
// CURRENT hash (rebuilt after create) to find the parent oct.
kernel void refine_derefine(
    device Oct*        grid     [[buffer(0)]],
    device const int*  flag1    [[buffer(1)]],
    device const long* hkey     [[buffer(2)]],
    device const int*  hval     [[buffer(3)]],
    device const int*  ckey_max [[buffer(4)]],
    device const long* key_off  [[buffer(5)]],
    constant RefineParams& P    [[buffer(6)]],
    device atomic_int* killctr  [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    int lev = grid[oct-1].lev;
    if (lev <= 1) return;

    int ck[3] = {0,0,0}, pck[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) { ck[d] = grid[oct-1].ckey[d]; pck[d] = ck[d] / 2; }
    long pkey = mg_oct_key(ckey_max[lev-1], key_off[lev-1], pck);
    int parent = hash_get(hkey, hval, P.hash_size, pkey);
    if (parent <= 0) return;

    int cell = mg_parent_cell_ck(ck);          // which parent cell this oct refines

    bool flagged = flag1[IDX2(cell,parent)] == 1;
    bool refined = grid[parent-1].refined[cell-1] != 0;
    if (!flagged && refined) {
        grid[oct-1].lev = 0;                       // mark dead (compaction drops it)
        grid[parent-1].refined[cell-1] = 0;        // unmark parent cell
        atomic_fetch_add_explicit(killctr, 1, memory_order_relaxed);
    }
}

// Compaction gather.  swap[j] (1-based, for j in [base, base+n)) holds the OLD
// oct index whose data moves to NEW position j (host-built counting-sort
// permutation: octs grouped by level, lev==0 holes dropped).  Gather into the
// scratch buffers; the bridge then blits the [base, base+n) range back over the
// live arrays.  Octs at levels <= the refined level are untouched (j < base).
kernel void refine_gather_oct(
    device Oct*        gout [[buffer(0)]],
    device const Oct*  gin  [[buffer(1)]],
    device const int*  swap [[buffer(2)]],
    constant int2&     bn   [[buffer(3)]],   // bn.x = base (1-based), bn.y = n
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= bn.y) return;
    int j = bn.x + (int)gid;
    gout[j-1] = gin[swap[j-1] - 1];
}

// Gather a per-cell field of width W (W=twotondim for phi/phi_old; the f array
// is gathered as W=twotondim*ndim since its (cell,dim) block is contiguous per
// oct).  2D dispatch: x = element-within-oct (0..W-1), y = oct offset.
kernel void refine_gather_field(
    device float*       fout [[buffer(0)]],
    device const float* fin  [[buffer(1)]],
    device const int*   swap [[buffer(2)]],
    constant int2&      bn   [[buffer(3)]],
    constant int&       W    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= bn.y || (int)gid.x >= W) return;
    int j   = bn.x + (int)gid.y;
    int old = swap[j-1];
    fout[(j-1)*W + (int)gid.x] = fin[(old-1)*W + (int)gid.x];
}

// INT version of the gather — MUST be used for flag1.  Reinterpreting an int
// field (values 0/1) through the float gather above flushes int 1 (=denormal
// float 1.4e-45) to 0 on Apple GPUs (FTZ), silently clearing flags during
// compaction -> next-step derefine mass-kills the refined region (refinement
// oscillates and never accumulates).  Integer load/store is bit-preserving.
kernel void refine_gather_int(
    device int*        fout [[buffer(0)]],
    device const int*  fin  [[buffer(1)]],
    device const int*  swap [[buffer(2)]],
    constant int2&     bn   [[buffer(3)]],
    constant int&      W    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= bn.y || (int)gid.x >= W) return;
    int j   = bn.x + (int)gid.y;
    int old = swap[j-1];
    fout[(j-1)*W + (int)gid.x] = fin[(old-1)*W + (int)gid.x];
}

// make_cache_octs (gpu_refine.cuf): materialise a boundary ghost ("cache") oct for
// the MISSING same-level neighbour of oct swap_local[gid] in direction input_ind.
// The cache oct is a REAL oct (cache region beyond ngridmax) with a correct father
// (parent_idx via hash at lev-1) and straight-injected f/phi from the parent cell;
// make_initial_phi then CIC-refines it.  This is the faithful coarse-fine boundary
// (correct father -> correct CIC), replacing the invented inline mg_ghost_cell.
// nsubgrid=1: subgrid_idx == oct_idx; box arrays indexed [(lev-1)*3 + d].
kernel void make_cache_octs(
    device Oct*         grid       [[buffer(0)]],
    device int*         flag1      [[buffer(1)]],
    device float*       f          [[buffer(2)]],
    device float*       phi        [[buffer(3)]],
    device float*       phi_old    [[buffer(4)]],
    device const int*   swap_local [[buffer(5)]],
    device int*         father     [[buffer(6)]],
    device int*         nbor       [[buffer(7)]],
    device const long*  hash_key   [[buffer(8)]],
    device const int*   hash_val   [[buffer(9)]],
    device const int*   ckey_max   [[buffer(10)]],
    device const long*  key_off    [[buffer(11)]],
    device const int*   box_min    [[buffer(12)]],
    device const int*   box_max    [[buffer(13)]],
    constant CacheParams& P        [[buffer(14)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int subgrid = swap_local[gid];                          // 1-based oct with the missing nbr
    int oct     = subgrid;                                  // nsubgrid=1
    int cache   = P.ngridmax + P.ifree_cache + (int)gid + 1;// 1-based cache-oct index
    nbor[(subgrid-1)*SUBGRIDSIZE + (P.input_ind-1)] = cache;// patch the boundary nbor

    int rem = P.input_ind - 1, off[3] = {0,0,0};            // decode 3^NDIM-cube offset
    for (int d=0; d<NDIM; ++d) { off[d] = (rem % 3) - 1; rem /= 3; }
    int lev = grid[oct-1].lev;
    int ckey[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) {
        int c = grid[oct-1].ckey[d] + off[d];
        if (P.per[d]) { int bmn = box_min[(lev-1)*3 + d], bmx = box_max[(lev-1)*3 + d];
                        if (c <  bmn) c = bmx - 1;
                        if (c >= bmx) c = bmn; }
        ckey[d] = c;
    }
    grid[cache-1].lev = lev;
    for (int d=0; d<NDIM; ++d) grid[cache-1].ckey[d] = ckey[d];
    for (int c=1; c<=TWOTONDIM; ++c) { grid[cache-1].refined[c-1] = 0; flag1[IDX2(c, cache)] = 0; }
    grid[cache-1].hkey[0] = hilbert_key(int3(ckey[0], ckey[1], ckey[2]), lev - 1);

    int pl = lev - 1;                                       // parent (coarse) level
    int pck[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) pck[d] = ckey[d] / 2;        // ckey>=0 after wrap
    long pkey   = mg_oct_key(ckey_max[pl], key_off[pl], pck);
    int  parent = hash_get(hash_key, hash_val, P.hash_size, pkey);
    father[cache-1] = parent;
    int cell = 1;                                           // parent cell of the cache oct
    for (int d=0; d<NDIM; ++d) cell += (ckey[d] - 2*pck[d]) << d;
    // CIC-refine the cache oct's phi (interpol_phi, IDENTICAL to make_initial_phi) from the
    // coarse parent's 3^NDIM neighbour cells.  The old PIECEWISE-CONSTANT straight injection
    // made the 4th-order force stencil at the coarse-fine boundary non-antisymmetric, so
    // Sum f*rho != 0 at refined levels -> spurious net momentum (1D <vx> drift) + over-
    // energization (3D).  father[cache] is set just above, so nbor_father_cells_mg(...,cache,
    // cache,...) resolves the parent's coarse cells via the parent's real connectivity (nbor).
    int igc[THREETONDIM], icc[THREETONDIM];
    nbor_father_cells_mg(grid, father, nbor, cache, cache, igc, icc);
    for (int c=1; c<=TWOTONDIM; ++c) {
        float corr = 0.0f;
        for (int ia=1; ia<=TWOTONDIM; ++ia) {
            int indf = mg_cic_father_index(c, ia);
            int igr = igc[indf-1], inr = icc[indf-1];
            if (igr <= 0 || igr > P.ngridmax) { igr = igc[MG_CUBE_CENTER]; inr = icc[MG_CUBE_CENTER]; }
            corr += mg_cic_weight(ia) * phi[IDX2(inr, igr)];
        }
        phi[IDX2(c,cache)]     = corr;                      // CIC-refined boundary phi
        f[IDX3(c,1,cache)]     = f[IDX3(cell,1,parent)];    // f/phi_old: straight inject (not read
        f[IDX3(c,2,cache)]     = f[IDX3(cell,2,parent)];    // by the solve boundary; gather rejects
        f[IDX3(c,3,cache)]     = f[IDX3(cell,3,parent)];    // cache octs)
        phi_old[IDX2(c,cache)] = phi_old[IDX2(cell,parent)];
    }
}

// init_prefix_sum_nbor (gpu_refine.cuf): cache-oct compaction predicate.  Writes 1
// into prefix_sum[oct-1] iff this oct's same-level neighbour in direction input_ind
// is TRULY MISSING (nbor==0), else 0.  Note: a nbor > ngridmax is an ALREADY-created
// cache oct (from an earlier direction's pass) and must NOT be recreated, so the test
// is ==0 only (faithful to CUDA; differs from enforce_rules' 0-or->ngridmax test).
// The host inclusive-scans prefix_sum (block_scan/uniform_add) before
// compute_cache_swap_table consumes it.  nsubgrid==1: subgrid==oct.
kernel void init_prefix_sum_nbor(
    device const int*    nbor       [[buffer(0)]],
    device int*          prefix_sum [[buffer(1)]],
    constant ScanParams& P          [[buffer(2)]],   // P.n = octs at level, P.head_idx
    constant int&        input_ind  [[buffer(3)]],   // 1..3^NDIM neighbour direction
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.n) return;
    int oct = P.head_idx + (int)gid;
    int nb  = nbor[(oct-1)*SUBGRIDSIZE + (input_ind - 1)];
    prefix_sum[oct-1] = (nb == 0) ? 1 : 0;            // prefix_sum(subgrid_idx), 1-based
}

// compute_cache_swap_table (gpu_refine.cuf): stream-compaction scatter.  Given the
// EXCLUSIVE/inclusive prefix-sum of the 0/1 predicate "this oct has a missing nbor
// in direction input_ind", write the compacted list of those octs into swap_local
// (consumed by make_cache_octs).  P.n = num octs at the level, P.head_idx = its head.
kernel void compute_cache_swap_table(
    device int*          swap_local [[buffer(0)]],
    device const int*    prefix_sum [[buffer(1)]],   // inclusive scan of the predicate (1-based)
    constant ScanParams& P          [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.n) return;
    int oct  = P.head_idx + (int)gid;
    int prev = (oct > P.head_idx) ? prefix_sum[oct - 2] : 0;  // prefix_sum(oct-1)
    int bit  = prefix_sum[oct - 1] - prev;                    // 0 or 1
    if (bit == 1) swap_local[prev] = oct;                     // swap_local(prev+1) 1-based
}

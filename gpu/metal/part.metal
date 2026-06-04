//============================================================================
// part.metal — DM particle dynamics: CIC force gather + leapfrog kick/drift
// (gpu_part.cuf: gather_cic_force_part + kick_drift_part_kernel).
//
// Positions are 64-bit fixed-point (ipos).  The force gather derives the CIC
// cell index (exact bit-shift) and sub-cell fraction (~24-bit) directly from
// ipos at the requested level — so the coarse-level fallback is just a smaller
// shift, with NO large 1/dx and no x/2 precision loss.  The drift accumulates
// the displacement into ipos as an exact 64-bit add (positions never lose their
// low bits over many steps, even though each v*dt increment is fp32-limited).
//
// 1-based Fortran index arithmetic preserved; arrays dereferenced 0-based.
//============================================================================
#include "ramses_msl.h"

// CIC trilinear gather of f at a particle, at cell_level (oct grid_level).
// Returns ff (fp32 accel) and ok (false if any of the 8 corners is OOD/miss).
inline bool gather_cic_force(
    device const float* f, device const long* hkey, device const int* hval,
    int hash_size, device const int* ckey_max, device const long* key_off,
    constant int* periodic, int ngridmax,
    long ip0, long ip1, long ip2, int cell_level, int grid_level,
    thread float* ff)
{
    ff[0]=ff[1]=ff[2]=0.0f;
    if (grid_level < 1) return false;

    long ipos[3] = { ip0, ip1, ip2 };
    int  il[3], ir[3], bmin[3], bmax[3];
    float dr[3], dl[3];
    for (int idim = 0; idim < NDIM; ++idim) {
        int  shift = NBITS_POS - cell_level;
        int  C     = (int)(ipos[idim] >> shift);                    // exact cell index
        long low   = ipos[idim] & (((long)1 << shift) - 1);         // cell fraction, fixed-point
        // EXACT stencil rounding (matches CPU floor(x+0.5)): decide ir from the
        // INTEGER fraction, not the fp32 cell_frac_fix.  fp32 rounds fractions
        // within ~6e-8 of 0.5 UP to 0.5, flipping ir at a half-cell boundary ->
        // flips the fine/coarse fallback for boundary particles (the 13% L10
        // gather remnant).  Integer compare matches the CPU to ~2^-48.
        int  add = (low >= ((long)1 << (shift - 1))) ? 1 : 0;       // fraction >= 0.5
        ir[idim] = C + add;
        float fr = ldexp((float)low, -shift);                       // weights (fp32 ok)
        dr[idim] = fr + 0.5f - (float)add;
        dl[idim] = 1.0f - dr[idim];
        il[idim] = ir[idim] - 1;
        bmin[idim] = 0;
        bmax[idim] = 1 << cell_level;                               // periodic full box
        if (periodic[idim]) {
            if (il[idim] <  bmin[idim]) il[idim] = bmax[idim] - 1;
            if (ir[idim] >= bmax[idim]) ir[idim] = bmin[idim];
        }
    }

    long nx  = (long)ckey_max[grid_level];
    long off = key_off[grid_level];
    bool ok = true;
    for (int corner = 0; corner < TWOTONDIM; ++corner) {   // 2^NDIM CIC corners
        int tgt[3] = {0,0,0}; float w = 1.0f;
        for (int d = 0; d < NDIM; ++d) {
            int b = (corner >> d) & 1;
            tgt[d] = b ? ir[d] : il[d];
            w     *= b ? dr[d] : dl[d];
        }
        bool in_domain = true;
        for (int idim = 0; idim < NDIM; ++idim)
            if (!periodic[idim] && (tgt[idim] < bmin[idim] || tgt[idim] >= bmax[idim])) in_domain = false;
        if (!in_domain) { ok = false; continue; }

        int father[3] = {0,0,0}, icell = 1;
        for (int idim = 0; idim < NDIM; ++idim) {
            father[idim] = tgt[idim] >> 1;
            icell += (tgt[idim] - 2*father[idim]) * (1 << idim);
        }
        long key = mg_oct_key((int)nx, off, father);
        int igrid = hash_get(hkey, hval, hash_size, key);
        if (igrid == 0 || igrid > ngridmax) { ok = false; continue; }

        for (int idim = 0; idim < NDIM; ++idim)
            ff[idim] += f[IDX3(icell, idim+1, igrid)] * w;
    }
    if (!ok) { ff[0]=ff[1]=ff[2]=0.0f; }
    return ok;
}

// ---- #6 particle residency: split (level bucket) + reorder on the GPU --------
// bucket_part: for a particle at level ilevel, hash to its oct and test whether
// the cell it sits in is refined.  bucket=1 -> the particle descends to ilevel+1
// (mirrors gpu_part.cuf bucket_part_kernel); =0 -> it stays at ilevel.
kernel void bucket_part(
    device const long* ipos     [[buffer(0)]],
    device int*        bucket   [[buffer(1)]],
    device const Oct*  grid     [[buffer(2)]],
    device const long* hkey     [[buffer(3)]],
    device const int*  hval     [[buffer(4)]],
    device const int*  ckey_max [[buffer(5)]],
    device const long* key_off  [[buffer(6)]],
    constant ScanParams& S      [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= S.n) return;
    int ip = S.head_idx + (int)gid;
    int L  = S.ilevel;
    int cc[3] = {0,0,0}, oc[3] = {0,0,0};
    for (int d = 0; d < NDIM; ++d) {
        cc[d] = cell_index_fix(ipos[IDXP(ip, d+1, S.npartmax)], L);
        oc[d] = cc[d] >> 1;
    }
    long key = mg_oct_key(ckey_max[L], key_off[L], oc);
    int ig = hash_get(hkey, hval, S.hash_size, key);
    if (ig == 0) { bucket[ip-1] = 0; return; }
    int ic = 1;
    for (int d = 0; d < NDIM; ++d) ic += (cc[d] - 2*oc[d]) << d;
    bucket[ip-1] = (grid[ig-1].refined[ic-1] != 0) ? 1 : 0;
}

// Gather one column of a particle array by a permutation perm (1-based old idx):
//   dst[off + (head-1+i)] = src[off + (perm[head-1+i]-1)],  off=(dim-1)*npartmax.
kernel void part_gather_long(
    device long* dst [[buffer(0)]], device const long* src [[buffer(1)]],
    device const int* perm [[buffer(2)]], constant ScanParams& S [[buffer(3)]],
    constant int& dim [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= S.n) return; int i = S.head_idx + (int)gid;
    int off = (dim-1)*S.npartmax;
    dst[off + i-1] = src[off + perm[i-1]-1];
}
kernel void part_gather_float(
    device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
    device const int* perm [[buffer(2)]], constant ScanParams& S [[buffer(3)]],
    constant int& dim [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= S.n) return; int i = S.head_idx + (int)gid;
    int off = (dim-1)*S.npartmax;
    dst[off + i-1] = src[off + perm[i-1]-1];
}
kernel void part_gather_int(
    device int* dst [[buffer(0)]], device const int* src [[buffer(1)]],
    device const int* perm [[buffer(2)]], constant ScanParams& S [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= S.n) return; int i = S.head_idx + (int)gid;
    dst[i-1] = src[perm[i-1]-1];
}

kernel void kick_drift_part(
    device long*        ipos     [[buffer(0)]],
    device float*       vp       [[buffer(1)]],
    device int*         levelp   [[buffer(2)]],
    device const float* f        [[buffer(3)]],
    device const long*  hkey     [[buffer(4)]],
    device const int*   hval     [[buffer(5)]],
    device const int*   ckey_max [[buffer(6)]],
    device const long*  key_off  [[buffer(7)]],
    constant int*       periodic [[buffer(8)]],
    constant PartParams& P       [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)P.num_parts) return;
    int ipart = P.head_idx + (int)gid;

    long ip0 = ipos[IDXP(ipart,1,P.npartmax)];
    long ip1 = ipos[IDXP(ipart,2,P.npartmax)];
    long ip2 = ipos[IDXP(ipart,3,P.npartmax)];

    // Gather the force at the particle's own level (cell_level==grid_level==ilevel),
    // matching the deposit / connectivity convention ckey_max[L]=2^(L-1); fall back
    // to the coarser level if the fine CIC cells aren't all present (boundary).
    float ff[3];
    bool ok = gather_cic_force(f, hkey, hval, P.hash_size, ckey_max, key_off,
                               periodic, P.ngridmax, ip0, ip1, ip2,
                               P.ilevel, P.ilevel, ff);
    if (!ok)
        gather_cic_force(f, hkey, hval, P.hash_size, ckey_max, key_off,
                         periodic, P.ngridmax, ip0, ip1, ip2,
                         P.ilevel-1, P.ilevel-1, ff);

    if (P.action_part == 2) {                 // kick + drift
        for (int idim = 1; idim <= NDIM; ++idim) {
            int vi = IDXP(ipart, idim, P.npartmax);
            float v = vp[vi] + ff[idim-1] * 0.5f * P.dtnew;
            vp[vi] = v;
            // drift in fixed-point: ipos += round(v*dt / box_size * 2^NBITS), wrap.
            float disp = (v * P.dtnew) / P.box_size[idim-1];
            long  d = (long)round(disp * ldexp(1.0f, NBITS_POS));
            long  M = 1L << NBITS_POS;
            long  pnew = (idim==1?ip0:(idim==2?ip1:ip2)) + d;
            if (periodic[idim-1]) { pnew %= M; if (pnew < 0) pnew += M; }
            ipos[IDXP(ipart, idim, P.npartmax)] = pnew;
        }
    } else if (P.action_part == 1) {          // level-transition half-kick
        int lp = levelp[ipart-1];
        // Use the PARTICLE's own level dt (CPU move_fine.f90: dteff =
        // levelp>=ilevel ? dtnew(levelp) : dtold(levelp)).  Was P.dtnew/P.dtold
        // (=dt at ilevel) — wrong by the subcycle ratio for levelp != ilevel,
        // the over-energization seed for the ~1% level-transition particles.
        // (fall back to the scalar P.dtnew/dtold if the per-level array is unset,
        //  so standalone test harnesses that fill only P.dtnew still work.)
        float dteff = (lp >= P.ilevel)
            ? (P.dtnew_lv[lp] != 0.0f ? P.dtnew_lv[lp] : P.dtnew)
            : (P.dtold_lv[lp] != 0.0f ? P.dtold_lv[lp] : P.dtold);
        levelp[ipart-1] = P.ilevel;
        for (int idim = 1; idim <= NDIM; ++idim) {
            int vi = IDXP(ipart, idim, P.npartmax);
            vp[vi] = vp[vi] + ff[idim-1] * 0.5f * dteff;
        }
    }
}

// =========================================================================
// newdt_part_reduce — per-level particle reduction for the CFL timestep:
//   vmax  = max over particles of max_dim |vp|         (atomic_uint max bit-trick)
//   ekin  = sum  over particles of 0.5*mp*|vp|^2        (two-word fixed-point)
// Replaces the CPU newdt_part loop so the host vp/xp need not be marshalled
// back every step.  red[0]=vmax bits, red[1]=ekin_lo, red[2]=ekin_hi
// (host zeroes them, dispatches, then reads vmax=as_float(red[0]) and
// ekin=fixed_to_float(red[1],red[2], 2^-FP_SHIFT_RHO)).
// =========================================================================
kernel void newdt_part_reduce(
    device const float* vp   [[buffer(0)]],
    device const float* mp   [[buffer(1)]],
    device atomic_uint* red  [[buffer(2)]],          // [vmax, ekin_lo, ekin_hi]
    constant ScanParams& P   [[buffer(3)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_id[[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]])
{
    threadgroup float scr_m[32], scr_e[32];
    int i = (int)gid;
    float vm = 0.0f, ek = 0.0f;
    if (i < P.n) {
        int ip = P.head_idx + i;                     // 1-based particle index
        float v2 = 0.0f;
        for (int idim = 1; idim <= NDIM; ++idim) {   // NDIM-generic (CUDA: do idim=1,ndim)
            float v = vp[IDXP(ip, idim, P.npartmax)];
            vm = max(vm, fabs(v));
            v2 += v * v;
        }
        ek = 0.5f * mp[ip-1] * v2;
    }
    float bvm = block_reduce_max(vm, scr_m, lane, sg_id, n_sg);
    float bek = block_reduce_sum(ek, scr_e, lane, sg_id, n_sg);
    if (sg_id == 0 && lane == 0) {
        atomic_max_f_nonneg(&red[0], bvm);
        long q = (long)round(bek * (float)(1L << FP_SHIFT_RHO));
        atomic_add_i64(&red[1], &red[2], q);
    }
}

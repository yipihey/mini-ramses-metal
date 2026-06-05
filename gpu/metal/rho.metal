//============================================================================
// rho.metal — CIC density deposit (gpu_part.cuf: build_src_part_kernel +
// cic_part_warp_kernel) and the fixed-point -> float finalize pass.
//
// Deposit accuracy strategy (selective extended precision):
//   * Within a SIMD-group segment (particles sharing a source cell, made
//     contiguous by the Hilbert sort) the partial sum is float and computed by
//     a deterministic segmented Hillis-Steele scan (<= 32 terms).
//   * The per-segment tail emits ONE accumulate into a 64-bit two-word
//     fixed-point cell accumulator (rho_lo/rho_hi), which is exactly
//     associative -> order-independent and bit-reproducible across warps/blocks.
//   * rho_finalize recombines (hi<<32|lo)/scale -> float rho once afterwards.
//
// 1-based Fortran index arithmetic preserved; arrays dereferenced 0-based.
//============================================================================
#include "ramses_msl.h"

// Per-particle source map: parent oct (hashed) + in-oct cell, packed as
// (igrid<<5 | icell).  Miss -> 0 (out-of-domain sentinel).
kernel void build_src_part(
    device const long*  ipos      [[buffer(0)]],   // 64-bit fixed-point positions
    device const int*   sortp     [[buffer(1)]],
    device int*         isp_swap  [[buffer(2)]],
    device const long*  hash_key  [[buffer(3)]],
    device const int*   hash_val  [[buffer(4)]],
    constant CicParams& P         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)P.num_parts) return;
    int ipart = sortp[P.head_idx + (int)gid - 1];

    int host_ckey[3] = {0,0,0};
    for (int idim = 1; idim <= NDIM; ++idim)
        host_ckey[idim-1] = cell_index_fix(ipos[IDXP(ipart, idim, P.npartmax)], P.ilevel);
    int father[3], ii[3], icell = 1;
    for (int idim = 1; idim <= NDIM; ++idim) {
        father[idim-1] = host_ckey[idim-1] >> 1;          // /2 (host_ckey >= 0)
        ii[idim-1]     = host_ckey[idim-1] - 2*father[idim-1];
        icell += ii[idim-1] * (1 << (idim-1));
    }
    long key = mg_oct_key(P.ckey_max, P.key_off, father);
    int igrid = hash_get(hash_key, hash_val, P.hash_size, key);
    isp_swap[ipart] = (igrid == 0) ? 0 : ((igrid << 5) | icell);
}

// Warp-segmented CIC deposit.  rho_lo/hi and nref_lo/hi are 64-bit two-word
// fixed-point accumulators laid out like Fortran rho(twotondim, ncell):
// flat index IDX2(cell, oct).
kernel void cic_part_warp(
    device const int*   sortp    [[buffer(0)]],
    device const int*   isp_swap [[buffer(1)]],
    device const Oct*   grid     [[buffer(2)]],
    device const long*  hash_key [[buffer(3)]],
    device const int*   hash_val [[buffer(4)]],
    device const long*  ipos     [[buffer(5)]],   // 64-bit fixed-point positions
    device const float* mp       [[buffer(6)]],
    device atomic_uint* rho_lo   [[buffer(7)]],
    device atomic_uint* rho_hi   [[buffer(8)]],
    device atomic_uint* nref_lo  [[buffer(9)]],
    device atomic_uint* nref_hi  [[buffer(10)]],
    constant CicParams& P        [[buffer(11)]],
    constant BoxParams& BX       [[buffer(12)]],
    uint  tg_pos [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sg_id [[simdgroup_index_in_threadgroup]],
    ushort n_sg  [[simdgroups_per_threadgroup]],
    ushort width [[threads_per_simdgroup]])
{
    uint warp_global = tg_pos * n_sg + sg_id;
    int  slot_global = (int)(warp_global * width + lane) + 1;   // 1-based
    bool valid = (slot_global <= P.num_parts);

    int ipart = 0, combined = 0, icell_src = 0, igrid_src = 0;
    if (valid) {
        // Fortran sortp_d(head_idx + slot_global - 1) is 1-based -> deref [.-1].
        ipart    = sortp[P.head_idx + slot_global - 2];
        combined = isp_swap[ipart];
        if (combined == 0) valid = false;
        else { icell_src = combined & 31; igrid_src = combined >> 5; }
    }

    float frac[3] = {0,0,0};
    float mp_i = 0.0f;
    if (valid) {
        mp_i = mp[ipart - 1];
        for (int idim = 1; idim <= NDIM; ++idim)
            frac[idim-1] = cell_frac_fix(ipos[IDXP(ipart, idim, P.npartmax)], P.ilevel);
    }
    // 1D CIC weights for offsets -1,0,+1 (central slot = 1 keeps 1D/2D correct).
    float wx[3], wy[3] = {0,1,0}, wz[3] = {0,1,0};
    wx[0] = max(0.0f, 0.5f - frac[0]);
    wx[1] = 1.0f - fabs(frac[0] - 0.5f);
    wx[2] = max(0.0f, frac[0] - 0.5f);
#if NDIM >= 2
    wy[0] = max(0.0f, 0.5f - frac[1]);
    wy[1] = 1.0f - fabs(frac[1] - 0.5f);
    wy[2] = max(0.0f, frac[1] - 0.5f);
#endif
#if NDIM >= 3
    wz[0] = max(0.0f, 0.5f - frac[2]);
    wz[1] = 1.0f - fabs(frac[2] - 0.5f);
    wz[2] = max(0.0f, frac[2] - 0.5f);
#endif

    int box_min[3] = { BX.box_min[0], BX.box_min[1], BX.box_min[2] };
    int box_max[3] = { BX.box_max[0], BX.box_max[1], BX.box_max[2] };

    int src_full[3] = {0,0,0};
    if (valid) {
        int ii_src[3];
        for (int idim = 1; idim <= NDIM; ++idim)
            ii_src[idim-1] = ((icell_src - 1) / (1 << (idim-1))) % 2;
        for (int idim = 1; idim <= NDIM; ++idim)
            src_full[idim-1] = 2 * grid[igrid_src - 1].ckey[idim-1] + ii_src[idim-1];
    }

    int prev = simd_shuffle_up(combined, 1);
    int next = simd_shuffle_down(combined, 1);
    bool is_head = (lane == 0)         || (combined != prev);
    bool is_tail = (lane == width - 1) || (combined != next);

    for (int k = 1; k <= THREETONDIM; ++k) {
        // Decode the 3^NDIM stencil offset; unused dims stay 0 (weight 1) so the
        // deposit spreads only over the NDIM active axes.
        int o3[3] = {0,0,0}, krem = k - 1;
        for (int d = 0; d < NDIM; ++d) { o3[d] = (krem % 3) - 1; krem /= 3; }
        int ox = o3[0], oy = o3[1], oz = o3[2];

        int dst_igrid = 0, dst_icell = 0;
        float my_rho = 0.0f, my_nref = 0.0f;
        if (valid) {
            int tgt[3] = {0,0,0};
            tgt[0] = src_full[0] + ox;
#if NDIM >= 2
            tgt[1] = src_full[1] + oy;
#endif
#if NDIM >= 3
            tgt[2] = src_full[2] + oz;
#endif
            bool in_domain = true;
            for (int idim = 1; idim <= NDIM; ++idim) {
                if (BX.periodic[idim-1]) {
                    if (tgt[idim-1] <  box_min[idim-1]) tgt[idim-1] = box_max[idim-1] - 1;
                    if (tgt[idim-1] >= box_max[idim-1]) tgt[idim-1] = box_min[idim-1];
                } else {
                    if (tgt[idim-1] <  box_min[idim-1]) in_domain = false;
                    if (tgt[idim-1] >= box_max[idim-1]) in_domain = false;
                }
            }
            if (in_domain) {
                int father[3], ii[3];
                dst_icell = 1;
                for (int idim = 1; idim <= NDIM; ++idim) {
                    father[idim-1] = tgt[idim-1] >> 1;
                    ii[idim-1]     = tgt[idim-1] - 2*father[idim-1];
                    dst_icell += ii[idim-1] * (1 << (idim-1));
                }
                long key = mg_oct_key(P.ckey_max, P.key_off, father);
                dst_igrid = hash_get(hash_key, hash_val, P.hash_size, key);
            }
            if (dst_igrid != 0 && dst_igrid <= P.ngridmax) {   // skip cache (ghost) octs: CUDA deposit_rho (gpu_rho.cuf:438) does `if(tgt_oct_idx>ngridmax) cycle`
                float w = wx[ox+1] * wy[oy+1] * wz[oz+1];
                if (w > 0.0f) {
                    // Deposit the MONOPOLE (mass per cell), NOT mass/vol: avoids
                    // the 2^(3L) blow-up and fixed-point overflow at deep levels.
                    // The multigrid applies the grouped single 1/dx factor.
                    my_rho = mp_i * w;
                    if (P.m_refine >= 0.0f) {
                        if (P.mass_cut > 0.0f) { if (mp_i < P.mass_cut) my_nref = w; }
                        else                   { my_nref = w; }
                    }
                }
            }
        }

        // Segmented inclusive Hillis-Steele scan (offsets 1,2,4,8,16).
        int head_acc = is_head ? 1 : 0;
        for (int s = 1; s <= 16; s <<= 1) {
            float o_rho  = simd_shuffle_up(my_rho,  (ushort)s);
            float o_nref = simd_shuffle_up(my_nref, (ushort)s);
            int   o_head = simd_shuffle_up(head_acc,(ushort)s);
            if (lane >= (ushort)s) {
                if (head_acc == 0) { my_rho += o_rho; my_nref += o_nref; }
                head_acc |= o_head;
            }
        }

        if (is_tail && valid && dst_igrid != 0) {
            int idx = IDX2(dst_icell, dst_igrid);
            if (my_rho != 0.0f)
                atomic_add_fixed(rho_lo, rho_hi, idx, my_rho, P.fp_scale_rho);
            if (P.m_refine >= 0.0f && my_nref != 0.0f)
                atomic_add_fixed(nref_lo, nref_hi, idx, my_nref, P.fp_scale_rho);
        }
    }
}

// Recombine two-word fixed-point accumulators into float fields after deposit.
// Launched over (twotondim * ncell) elements.
kernel void rho_finalize(
    device const uint* rho_lo  [[buffer(0)]],
    device const uint* rho_hi  [[buffer(1)]],
    device const uint* nref_lo [[buffer(2)]],
    device const uint* nref_hi [[buffer(3)]],
    device float*      rho     [[buffer(4)]],
    device float*      nref    [[buffer(5)]],
    constant float&    inv_scale [[buffer(6)]],
    constant int&      n        [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)n) return;
    rho[gid]  = fixed_to_float(rho_lo[gid],  rho_hi[gid],  inv_scale);
    nref[gid] = fixed_to_float(nref_lo[gid], nref_hi[gid], inv_scale);
}

//============================================================================
// gas_deposit — add the GAS mass to the Poisson source rho for self-gravitating
// DM+gas runs.  One thread per cell; only LEAF cells (not refined) deposit their
// gas mass (uold[rho]*vol_loc) into the cell's own fixed-point rho accumulator
// (rho_lo/hi at IDX2(cell,oct)), the SAME accumulator + scale the particle CIC
// uses.  The gas is a grid quantity already, so a monopole (cell-local) deposit
// is the natural, self-consistent source -> the gas now self-gravitates (the CPU
// gas multipole was never uploaded to the GPU -> econs was badly broken).
//============================================================================
kernel void gas_deposit(device const float*       uold    [[buffer(0)]],
                        device const Oct*         grid    [[buffer(1)]],
                        device atomic_uint*       rho_lo  [[buffer(2)]],
                        device atomic_uint*       rho_hi  [[buffer(3)]],
                        constant     GasDepParams& P      [[buffer(4)]],
                        device atomic_uint*       nref_lo [[buffer(5)]],
                        device atomic_uint*       nref_hi [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)(P.num_octs * TWOTONDIM)) return;
    int oct  = P.head_idx + (int)gid / TWOTONDIM;
    int cell = (int)gid % TWOTONDIM + 1;
    if (grid[oct-1].refined[cell-1] != 0) return;          // leaf cells only
    float gas_mass = uold[UH(cell,1,oct)] * P.vol_loc;
    atomic_add_fixed(rho_lo, rho_hi, IDX2(cell, oct), gas_mass, P.fp_scale);
    // Gas also counts toward the refinement criterion nref (poisson_flag uses
    // nref >= m_refine for GRAV builds, and rho_fine adds gas as mmm/mass_sph).
    // Monopole (cell-local) -> same per-cell total as the CPU CIC of the gas
    // multipole; without this the GPU refines on particles only -> under-refines.
    if (P.refine_on)
        atomic_add_fixed(nref_lo, nref_hi, IDX2(cell, oct), gas_mass*P.inv_mass_sph, P.fp_scale);
}

//============================================================================
// epot_reduce — sum f^2 over leaf cells (all dims) into a single two-word
// fixed-point accumulator (acc_lo/hi[0]).  The host multiplies the result by
// fact = -dx^ndim/(4pi)/2 to get the level's potential energy (mirrors
// force_fine.f90 compute_epot), but reads the RESIDENT GPU force B.f instead of
// a host-downloaded m%f — avoiding a per-step host sweep over millions of cells.
//============================================================================
kernel void epot_reduce(device const float*       f      [[buffer(0)]],
                        device const Oct*         grid   [[buffer(1)]],
                        device atomic_uint*       acc_lo [[buffer(2)]],
                        device atomic_uint*       acc_hi [[buffer(3)]],
                        constant     EpotParams&  P      [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)(P.num_octs * TWOTONDIM)) return;
    int oct  = P.head_idx + (int)gid / TWOTONDIM;
    int cell = (int)gid % TWOTONDIM + 1;
    if (grid[oct-1].refined[cell-1] != 0) return;          // leaf cells only
    float s = 0.0f;
    for (int d = 1; d <= NDIM; ++d) { float fv = f[IDX3(cell, d, oct)]; s += fv*fv; }
    if (s != 0.0f) atomic_add_fixed(acc_lo, acc_hi, 0, s, P.fp_scale);
}

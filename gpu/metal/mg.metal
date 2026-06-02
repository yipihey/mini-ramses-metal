//============================================================================
// mg.metal — multigrid Poisson smoother/residual (gpu_mg.cuf), in GROUPED form.
//
// To avoid the deep-AMR fp32 dynamic-range trap (forming 1/dx^2 then x dx^2),
// we carry the GROUPED source  S = dx^2 * RHS = 4*pi*G*(m_cell - mbar) * 2^L
// directly in f(:,2).  Then NO dx^2/oneoverdx2 appears in the solver:
//   Gauss-Seidel : phi = (sum phi_nb - S) / (twondim - weight)
//   residual     : f(:,1) = S - (sum phi_nb - twondim*phi_c)
//   boundary     : the original  f(:,2) -= 2*oneoverdx2*phi_b  becomes (x dx^2)
//                  a correction of -2*phi_b  -> handled by reset_rhs in S.
//
// nsubgrid==1: each oct is its own subgrid; nbor(1:27, oct) is the 3x3x3 oct
// stencil (center = self at ind 14).  Arrays are Fortran column-major; 1-based
// index arithmetic preserved, deref 0-based.
//============================================================================
#include "ramses_msl.h"

// Cell-index / oct-offset stencil tables (gpu_mg.cuf), stored [dir-1][cell-1].
constant int MG_hhh[6][8] = {
    {2,1,4,3,6,5,8,7}, {2,1,4,3,6,5,8,7},
    {3,4,1,2,7,8,5,6}, {3,4,1,2,7,8,5,6},
    {5,6,7,8,1,2,3,4}, {5,6,7,8,1,2,3,4}
};
constant int MG_iii[6][8] = {
    {-1, 0,-1, 0,-1, 0,-1, 0}, { 0, 1, 0, 1, 0, 1, 0, 1},
    {-1,-1, 0, 0,-1,-1, 0, 0}, { 0, 0, 1, 1, 0, 0, 1, 1},
    {-1,-1,-1,-1, 0, 0, 0, 0}, { 0, 0, 0, 0, 1, 1, 1, 1}
};
constant int MG_ired[4]   = {1,4,6,7};
constant int MG_iblack[4] = {2,3,5,8};

// 4th-order gradient stencil tables (gpu_mg.cuf), stored [idim-1][cell-1].
constant int MG_gg1[3][8] = {{1,0,1,0,1,0,1,0},{3,3,0,0,3,3,0,0},{5,5,5,5,0,0,0,0}};
constant int MG_gg2[3][8] = {{0,2,0,2,0,2,0,2},{0,0,4,4,0,0,4,4},{0,0,0,0,6,6,6,6}};
constant int MG_gg3[3][8] = {{1,1,1,1,1,1,1,1},{3,3,3,3,3,3,3,3},{5,5,5,5,5,5,5,5}};
constant int MG_gg4[3][8] = {{2,2,2,2,2,2,2,2},{4,4,4,4,4,4,4,4},{6,6,6,6,6,6,6,6}};
constant int MG_hh1[3][8] = {{2,1,4,3,6,5,8,7},{3,4,1,2,7,8,5,6},{5,6,7,8,1,2,3,4}};
constant int MG_hh3[3][8] = {{1,2,3,4,5,6,7,8},{1,2,3,4,5,6,7,8},{1,2,3,4,5,6,7,8}};

// The CIC father-cell tables (ccc/bbb) are realised by mg_cic_father_index /
// mg_cic_weight in utils.h (NDIM-generic), matching gpu_mg.cuf's ccc/bbb.
// floor_div2, MG_CUBE_CENTER, nbor_father_cells_mg, mg_nbor are in nbor.h
// (gpu_nbor.cuf analog); hash_insert in hash.metal; conn_build_* in nbor.metal.
//
// The AMR mask is built faithfully (gpu_mg.cuf) by reset_mask_kernel (fine=1) +
// restrict_mask + volume_to_mask -- there is NO bespoke "make_mask_amr".  The
// coarse-fine boundary is the materialised cache oct (gpu_refine.cuf make_cache_octs),
// read directly by reset_rhs_kernel/gradient_phi -- NOT an inline interpol ghost.

// Coarse-fine boundary value: SET the fine phi by CIC-interpolating the coarse
// (ilevel-1) phi via the AMR father + 3x3x3 father-cell stencil (gpu_mg
// make_initial_phi, time-extrapolation omitted -> tfrac=0).  father maps each
// fine oct to its coarse oct; nbor_c is the (all-level) nbor table used to reach
// the 27 father cells; phi holds coarse values (read) and fine values (write).
kernel void make_initial_phi(
    device const Oct*   grid    [[buffer(0)]],
    device const int*   father  [[buffer(1)]],
    device const int*   nbor_c  [[buffer(2)]],
    device float*       phi     [[buffer(3)]],
    constant MgParams&  P       [[buffer(4)]],  // head_idx=fine head, head_father=fine father base
    device const float* phi_old [[buffer(5)]],  // for time-extrapolation (P.tfrac)
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct  = P.head_idx + (int)gid.y;
    int mg   = P.head_father + (int)gid.y;
    int cell = (int)gid.x + 1;
    int ig[27], ic[27];
    nbor_father_cells_mg(grid, father, nbor_c, oct, mg, ig, ic);
    float corr = 0.0f;
    for (int ia=1; ia<=TWOTONDIM; ++ia) {
        int indf = mg_cic_father_index(cell, ia);   // NDIM-generic ccc
        int igr = ig[indf-1], inr = ic[indf-1];
        // Missing father neighbour -> fall back to the CENTRE father cell (CUDA
        // gpu_mg make_initial_phi), so the CIC weights still sum to 1.  SKIPPING it
        // (sum<1) leaves the boundary phi systematically too low -> over-energization
        // wherever the coarse level is itself partially refined (deep levels).
        if (igr <= 0 || igr > P.ngridmax) { igr = ig[MG_CUBE_CENTER]; inr = ic[MG_CUBE_CENTER]; }
        // Same interpol_phi as the gradient ghost: 3rd-order CIC of the coarse phi,
        // linearly extrapolated in time so the held boundary matches the gradient.
        float pv = phi[IDX2(inr, igr)];
        if (P.tfrac != 0.0f) pv += (pv - phi_old[IDX2(inr, igr)]) * P.tfrac;
        corr += mg_cic_weight(ia) * pv;             // NDIM-generic bbb
    }
    phi[IDX2(cell, oct)] = corr;
}

// Set the mask f(:,3)=mask_val over a level's octs (gpu_mg reset_mask_kernel):
// mask_val=1 for the fine interior, =0 to zero a coarse mask region before
// restrict_mask accumulates into it.
kernel void reset_mask_kernel(
    device float*      f        [[buffer(0)]],
    constant MgParams& P        [[buffer(1)]],
    constant float&    mask_val [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct = P.head_idx + (int)gid.y, cell = (int)gid.x + 1;
    f[IDX3(cell, 3, oct)] = mask_val;
}

// Build coarse mask from fine: f_mg(:,3) += sum_children (1+f(:,3))/2/twotondim
// (gpu_mg restrict_mask).  Caller must zero the coarse mask region first.
kernel void restrict_mask(
    device const Oct*   grid   [[buffer(0)]],
    device const int*   father [[buffer(1)]],
    device const float* f      [[buffer(2)]],
    device float*       f_mg   [[buffer(3)]],
    constant MgParams&  P      [[buffer(4)]],   // head_idx=fine head, head_father=fine-in-coarse base
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    int mg  = P.head_father + (int)gid;
    int father_idx = father[mg - 1];
    if (father_idx <= 0) return;
    int cell = mg_parent_cell(grid[oct-1]);     // coarse cell this fine oct maps to
    float acc = 0.0f;
    for (int ind=1; ind<=TWOTONDIM; ++ind)
        acc += (1.0f + f[IDX3(ind, 3, oct)]) / 2.0f / (float)TWOTONDIM;
    // single fine oct maps to one coarse cell; no contention across fine octs of
    // the same parent only if dispatched serially per parent -- but distinct fine
    // octs CAN share a parent cell? No: 8 fine octs -> 8 distinct coarse cells.
    f_mg[IDX3(cell, 3, father_idx)] += acc;
}

// Finalise coarse mask: f(:,3) = 2*f(:,3) - 1  (gpu_mg volume_to_mask; the
// atomicmax "allmasked" reduction is done on the host by reading f back).
kernel void volume_to_mask(
    device float*      f  [[buffer(0)]],
    constant MgParams& P  [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct = P.head_idx + (int)gid.y, cell = (int)gid.x + 1;
    f[IDX3(cell, 3, oct)] = 2.0f * f[IDX3(cell, 3, oct)] - 1.0f;
}

// (The inline mg_ghost_cell coarse-fine ghost is GONE: CUDA materialises the boundary
// as a cache oct (gpu_refine.cuf make_cache_octs) filled by make_initial_phi, read
// directly by reset_rhs_kernel / gradient_phi.  That is the faithful boundary.)

// Grouped RHS.  S = coef*(rho_mono - offset_mono) (grouped: dx^2 absorbed).
// use_ghost=1 (fine AMR level): no boundary reconstruction term -- the missing
// same-level neighbours are supplied as interpol_phi ghosts in the smoother/residual
// (every cell is interior).  use_ghost=0 (coarse MG levels): Dirichlet-0 linear
// reconstruction for the correction (S -= 2*phi_b), the masked-multigrid boundary.
//   coef = fourpi*2^ilevel, offset_mono = mean mass/cell (charge neutrality).
// rho holds the MONOPOLE (mass/cell); f(:,2)=S, f(:,3)=mask.
// reset_rhs_kernel (gpu_mg.cuf:237): RHS f(:,2)=4piG*(rho_density - rho_bar), then
// the masked-boundary correction.  rho holds the MONOPOLE -> density = monopole/vol_loc
// (CUDA stores density directly; the /vol_loc is the only convention difference).  The
// boundary phi is read from the MATERIALISED neighbour (a cache oct > ngridmax, filled by
// make_initial_phi) -- exactly like CUDA's phi(cell_nbr,oct_nbr); dis_nbr=-1 beyond ngridmax.
// No inline ghost.  f(:,2)=S, f(:,3)=mask.
kernel void reset_rhs_kernel(
    device const float* phi  [[buffer(0)]],
    device const float* rho  [[buffer(1)]],
    device float*       f    [[buffer(2)]],
    device const int*   nbor [[buffer(3)]],
    constant MgParams&  P    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct  = P.head_idx + (int)gid.y;
    int cell = (int)gid.x + 1;

    float oneoverdx2 = 1.0f / (P.dx * P.dx);
    float rho_d = rho[IDX2(cell, oct)] / P.vol_loc;
    float S     = P.fourpi * (rho_d - P.offset);
    float dis_c = f[IDX3(cell, 3, oct)];
    float phi_c = phi[IDX2(cell, oct)];
    for (int idim = 1; idim <= NDIM; ++idim)
        for (int inbor = 1; inbor <= 2; ++inbor) {
            int dir = 2*(idim-1) + inbor;
            int off = MG_iii[dir-1][cell-1];
            int in=0, jn=0, kn=0;
            if (idim==1) in=off; else if (idim==2) jn=off; else kn=off;
            int onb = mg_nbor(nbor, oct, in, jn, kn);
            int cnb = MG_hhh[dir-1][cell-1];
            // CUDA reads phi(cell_nbr,oct_nbr) for the materialised neighbour (incl cache
            // octs > ngridmax); dis_nbr only valid for a real oct (<= ngridmax), else -1.
            float phinb = (onb >= 1) ? phi[IDX2(cnb, onb)] : 0.0f;
            float disnb = (onb >= 1 && onb <= P.ngridmax) ? f[IDX3(cnb, 3, onb)] : -1.0f;
            if (disnb <= 0.0f && dis_c > 0.0f) {
                float w    = disnb / (disnb - dis_c);
                float phib = (1.0f - w) * phinb + w * phi_c;
                S -= 2.0f * oneoverdx2 * phib;
            }
        }
    f[IDX3(cell, 2, oct)] = S;
}

// One red or black Gauss-Seidel sweep (grouped: S = f(:,2), no dx^2).
kernel void gauss_seidel(
    device float*       phi   [[buffer(0)]],
    device const float* f     [[buffer(1)]],   // f(:,2)=S(grouped), f(:,3)=mask
    device const int*   nbor  [[buffer(2)]],
    constant MgParams&  P     [[buffer(3)]],
    constant int&       safe  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct  = P.head_idx + (int)gid.y;                 // 1-based
    int cell = P.redstep ? MG_ired[gid.x] : MG_iblack[gid.x];

    float dis_c = f[IDX3(cell, 3, oct)];
    if (!(dis_c > 0.0f && (safe == 0 || dis_c >= 1.0f))) return;

    // Masked multigrid (matches CPU): a masked/missing neighbour (disnb<=0) adds a
    // distance weight to the DIAGONAL (2ndim - weight, weight<0 -> stronger diagonal),
    // and its boundary phi is already in S via reset_rhs.  This is the correct
    // Dirichlet treatment -- adding the ghost as a full neighbour (denom 2ndim) would
    // UNDER-constrain the boundary cells and over-deepen phi.
    float nb = 0.0f, weight = 0.0f;
    for (int idim = 1; idim <= NDIM; ++idim)
        for (int inbor = 1; inbor <= 2; ++inbor) {
            int dir = 2*(idim-1) + inbor;
            int off = MG_iii[dir-1][cell-1];
            int in=0, jn=0, kn=0;
            if (idim==1) in=off; else if (idim==2) jn=off; else kn=off;
            int onb = mg_nbor(nbor, oct, in, jn, kn);
            int cnb = MG_hhh[dir-1][cell-1];
            float disnb = (onb > 0 && onb <= P.ngridmax) ? f[IDX3(cnb, 3, onb)] : -1.0f;
            if (disnb <= 0.0f) weight += disnb / dis_c;
            else               nb += phi[IDX2(cnb, onb)];
        }
    // NON-GROUPED (CUDA gauss_seidel): phi = (sum phi_nb - dx2*RHS) / (twondim - weight).
    float dx2 = P.dx * P.dx;
    phi[IDX2(cell, oct)] = (nb - dx2 * f[IDX3(cell, 2, oct)]) / ((float)TWONDIM - weight);
}

// Grouped residual into f(:,1): r = S - (sum phi_nb - twondim*phi_c).
kernel void cmp_residual(
    device const float* phi  [[buffer(0)]],
    device float*       f    [[buffer(1)]],
    device const int*   nbor [[buffer(2)]],
    constant MgParams&  P    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct  = P.head_idx + (int)gid.y;
    int cell = (int)gid.x + 1;                           // 1..8

    float phi_c = phi[IDX2(cell, oct)];
    float dis_c = f[IDX3(cell, 3, oct)];
    if (!(dis_c > 0.0f)) { f[IDX3(cell, 1, oct)] = 0.0f; return; }

    float nb = 0.0f;
    for (int idim = 1; idim <= NDIM; ++idim)
        for (int inbor = 1; inbor <= 2; ++inbor) {
            int dir = 2*(idim-1) + inbor;
            int off = MG_iii[dir-1][cell-1];
            int in=0, jn=0, kn=0;
            if (idim==1) in=off; else if (idim==2) jn=off; else kn=off;
            int onb = mg_nbor(nbor, oct, in, jn, kn);
            int cnb = MG_hhh[dir-1][cell-1];
            float disnb = (onb > 0 && onb <= P.ngridmax) ? f[IDX3(cnb, 3, onb)] : -1.0f;
            if (disnb <= 0.0f) nb += phi_c * disnb / dis_c;  // masked/missing: stronger diagonal
            else               nb += phi[IDX2(cnb, onb)];    // present interior
        }
    // NON-GROUPED (CUDA cmp_residual): r = -oneoverdx2*(sum phi_nb - twondim*phi_c) + RHS.
    float oneoverdx2 = 1.0f / (P.dx * P.dx);
    f[IDX3(cell, 1, oct)] = -oneoverdx2 * (nb - (float)TWONDIM * phi_c) + f[IDX3(cell, 2, oct)];
}




// 4th-order force f(:,idim) = a(phi1-phi2) - b(phi3-phi4), a=(2/3)/dx, b=(1/12)/dx.
// NOTE: this genuinely needs 1/dx (= 2^L); at deep levels the phi differences are
// O(dx) so it is cancellation-prone in fp32 (the dual of the dynamic-range trap).
//
// H6 (coarse-fine boundary): when a face-neighbour oct is MISSING (refined-region
// edge), its 8 cell potentials are the ghost/Dirichlet values RAMSES holds there:
// the 3rd-order CIC interpolation of the 27 coarse (ilevel-1) parent cells of the
// ghost oct, with optional time-extrapolation (tfrac via phi_old).  This is an
// exact transliteration of force_fine.f90 gradient_phi + interpol_phi.f90 — NOT an
// approximation.  Per-oct (one thread builds phi_nbor[0..6][1..8] once, like the CPU
// loop over the 6 face octs, then runs the 4-node stencil for all 8 cells).
// INLINE interpol_phi ghost (the RESTORED faithful method, no cache octs): compute the
// TWOTONDIM cell potentials of a MISSING coarse-fine boundary neighbour of `oct` in face
// direction (idim1, sgn) by 3rd-order CIC of its 27 coarse (lev-1) parent cells +
// time-extrapolation -- IDENTICAL to make_cache_octs / the CPU interpol_phi.  Used by
// gradient_phi / gauss_seidel / cmp_residual when a neighbour oct index is 0 (cache OFF).
inline void mg_interpol_ghost(
    device const Oct* grid, device const int* nbor, device const long* hkey,
    device const int* hval, device const int* ckey_max, device const long* key_off,
    device const int* box_min, device const int* box_max, device const float* phi,
    device const float* phi_old, int hash_size, int ngridmax, float tfrac,
    int oct, int idim1, int sgn, thread float* out /* TWOTONDIM */)
{
    int lev = grid[oct-1].lev;
    int nck[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) nck[d] = grid[oct-1].ckey[d];
    nck[idim1-1] += sgn;                                  // step to the missing neighbour
    for (int d=0; d<NDIM; ++d) {                          // periodic wrap at this level
        int bmn = box_min[(lev-1)*3+d], bmx = box_max[(lev-1)*3+d];
        if (nck[d] <  bmn) nck[d] = bmx-1;
        if (nck[d] >= bmx) nck[d] = bmn;
    }
    int pl = lev-1, pck[3] = {0,0,0}, p[3] = {0,0,0};
    for (int d=0; d<NDIM; ++d) { pck[d] = floor_div2(nck[d]); p[d] = nck[d] & 1; }
    long pkey  = mg_oct_key(ckey_max[pl], key_off[pl], pck);
    int father = hash_get(hkey, hval, hash_size, pkey);   // coarse parent of the neighbour
    int igc[THREETONDIM], icc[THREETONDIM];
    if (father > 0) nbor_father_cells_at(father, p, nbor, igc, icc);
    for (int c=1; c<=TWOTONDIM; ++c) {
        float corr = 0.0f, corr_old = 0.0f;
        if (father > 0) {
            for (int ia=1; ia<=TWOTONDIM; ++ia) {
                int indf = mg_cic_father_index(c, ia);
                int igr = igc[indf-1], inr = icc[indf-1];
                if (igr <= 0 || igr > ngridmax) { igr = igc[MG_CUBE_CENTER]; inr = icc[MG_CUBE_CENTER]; }
                float w = mg_cic_weight(ia);
                corr     += w * phi[IDX2(inr, igr)];
                corr_old += w * phi_old[IDX2(inr, igr)];
            }
        }
        out[c-1] = corr + (corr - corr_old) * tfrac;
    }
}

// gradient_phi (gpu_mg.cuf:1094): force = 4th-order central difference of phi.  With cache
// octs ON the boundary neighbour is a materialised cache oct (index>0, read directly);
// with cache OFF it is 0 -> reconstruct it inline via mg_interpol_ghost (the faithful CPU
// method).  Same kernel serves both boundary paths.  f(:,idim)=a*(p1-p2)-b*(p3-p4).
kernel void gradient_phi(
    device const float* phi  [[buffer(0)]],
    device float*       f    [[buffer(1)]],
    device const int*   nbor [[buffer(2)]],
    constant MgParams&  P    [[buffer(3)]],
    device const Oct*   grid     [[buffer(4)]],
    device const long*  hkey     [[buffer(5)]],
    device const int*   hval     [[buffer(6)]],
    device const int*   ckey_max [[buffer(7)]],
    device const long*  key_off  [[buffer(8)]],
    device const int*   box_min  [[buffer(9)]],
    device const int*   box_max  [[buffer(10)]],
    device const float* phi_old  [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int oct = P.head_idx + (int)gid;
    float phinb[7][TWOTONDIM];                            // self + 6 face neighbours' cells
    for (int c=1; c<=TWOTONDIM; ++c) phinb[0][c-1] = phi[IDX2(c,oct)];
    for (int idim=1; idim<=NDIM; ++idim)
        for (int inbor=1; inbor<=2; ++inbor) {
            int s = -1 + 2*(inbor-1);
            int in=0,jn=0,kn=0;
            if (idim==1) in=s; else if (idim==2) jn=s; else kn=s;
            int k = 2*(idim-1)+inbor;
            int o = mg_nbor(nbor, oct, in, jn, kn);
            if (o > 0) { for (int c=1; c<=TWOTONDIM; ++c) phinb[k][c-1] = phi[IDX2(c,o)]; }
            else mg_interpol_ghost(grid, nbor, hkey, hval, ckey_max, key_off, box_min, box_max,
                                   phi, phi_old, P.hash_size, P.ngridmax, P.tfrac, oct, idim, s, phinb[k]);
        }
    float a = (0.5f*4.0f/3.0f)/P.dx, b = (0.25f*1.0f/3.0f)/P.dx;
    for (int cell=1; cell<=TWOTONDIM; ++cell)
        for (int idim=1; idim<=NDIM; ++idim) {
            int g1=MG_gg1[idim-1][cell-1], g2=MG_gg2[idim-1][cell-1];
            int g3=MG_gg3[idim-1][cell-1], g4=MG_gg4[idim-1][cell-1];
            int h1=MG_hh1[idim-1][cell-1], h3=MG_hh3[idim-1][cell-1];
            float p1=phinb[g1][h1-1], p2=phinb[g2][h1-1];
            float p3=phinb[g3][h3-1], p4=phinb[g4][h3-1];
            f[IDX3(cell, idim, oct)] = a*(p1-p2) - b*(p3-p4);
        }
}

// V-cycle restriction: average 8 fine-cell residuals f(:,1) into the coarse cell
// f_mg(:,2) (the coarse RHS).  oct_idx fine, mg_idx fine-in-coarse-order.
kernel void restrict_residual(
    device const Oct*   grid   [[buffer(0)]],
    device const int*   father [[buffer(1)]],
    device const float* f      [[buffer(2)]],
    device float*       f_mg   [[buffer(3)]],
    constant MgParams&  P      [[buffer(4)]],   // head_idx=fine head, ngridmax=head_father
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= P.num_octs) return;
    int oct = P.head_idx + (int)gid;            // fine oct (1-based)
    int mg  = P.head_father + (int)gid;       // fine-in-coarse base + gid
    int father_idx = father[mg - 1];
    int cell = mg_parent_cell(grid[oct-1]);     // coarse cell this fine oct maps to
    float dis_father = f_mg[IDX3(cell, 3, father_idx)];
    float s = 0.0f;
    if (dis_father > 0.0f)
        for (int ind=1; ind<=TWOTONDIM; ++ind)
            if (f[IDX3(ind, 3, oct)] > 0.0f) s += f[IDX3(ind, 1, oct)];
    // NON-GROUPED (CUDA restrict_residual): coarse RHS = mean of fine residual.
    // (The dx_c^2 scaling now lives in the coarse GS/residual via P.dx per level.)
    f_mg[IDX3(cell, 2, father_idx)] = s / (float)TWOTONDIM;
}

// V-cycle prolongation: phi += CIC-interpolated coarse correction phi_mg.
kernel void interpolate_correct(
    device const Oct*   grid   [[buffer(0)]],
    device const int*   father [[buffer(1)]],
    device const int*   nbor_c [[buffer(2)]],   // COARSE nbor table
    device float*       phi    [[buffer(3)]],
    device const float* phi_mg [[buffer(4)]],
    device const float* f      [[buffer(5)]],
    constant MgParams&  P      [[buffer(6)]],   // head_idx=fine, ngridmax=head_father
    uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.y >= P.num_octs) return;
    int oct  = P.head_idx + (int)gid.y;
    int mg   = P.head_father + (int)gid.y;
    int cell = (int)gid.x + 1;
    int ig[27], ic[27];
    nbor_father_cells_mg(grid, father, nbor_c, oct, mg, ig, ic);
    float corr = 0.0f;
    if (f[IDX3(cell, 3, oct)] > 0.0f)
        for (int ia=1; ia<=TWOTONDIM; ++ia) {
            int indf = mg_cic_father_index(cell, ia);   // NDIM-generic ccc
            int igr = ig[indf-1], inr = ic[indf-1];
            if (igr > 0) corr += mg_cic_weight(ia) * phi_mg[IDX2(inr, igr)];
        }
    phi[IDX2(cell, oct)] += corr;
}

// cmp_epot (gpu_mg.cuf): potential energy = sum over UNREFINED leaf cells of
// |f|^2 (sum over the NDIM force components), reduced per threadgroup.  The host
// sums the df64 partials and applies fact = -dx^ndim/(4*pi*G)/2.  This is a
// global fp64 reduction in CUDA -> df64 in the hybrid Metal port (reduce.h).
// One thread per cell: gid -> oct = head + gid/TWOTONDIM, cell = gid%TWOTONDIM+1.
kernel void cmp_epot(
    device const Oct*   grid    [[buffer(0)]],
    device const float* f       [[buffer(1)]],
    device df64*        partial [[buffer(2)]],   // one df64 per threadgroup
    constant MgParams&  P       [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint n_sg  [[simdgroups_per_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]])
{
    float e = 0.0f;
    int total = P.num_octs * TWOTONDIM;
    if ((int)gid < total) {
        int oct  = P.head_idx + (int)gid / TWOTONDIM;
        int cell = (int)gid % TWOTONDIM + 1;
        if (grid[oct-1].refined[cell-1] == 0)               // CPU compute_epot skips refined cells
            for (int d = 1; d <= NDIM; ++d) { float fc = f[IDX3(cell, d, oct)]; e += fc*fc; }
    }
    threadgroup df64 scratch[32];
    df64 r = block_reduce_sum_df64(df64_from(e), scratch, lane, sg_id, n_sg);
    if (sg_id == 0 && lane == 0) partial[tgid] = r;
}

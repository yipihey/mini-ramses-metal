//============================================================================
// ramses_metal.h
//
// Shared POD layouts and conventions for the Metal port of the mini-ramses
// DM-gravity GPU kernels.  Included by BOTH the Obj-C++ host bridge
// (metal_bridge.mm) and every Metal kernel source (gpu/metal/*.metal).
//
// CONVENTIONS (read before touching any kernel)
// ---------------------------------------------------------------------------
//  * The original CUDA-Fortran code is 1-based and column-major.  We preserve
//    the 1-based index *arithmetic* verbatim from the .cuf sources and subtract
//    1 only at the final pointer dereference.  Helper macros below do this.
//
//  * Column-major flattening of Fortran arrays (leading dim is contiguous):
//       rho(1:twotondim, 1:ncell)            -> RHO(rho, cell, oct)
//       f  (1:twotondim, 1:ndim, 1:ncell)    -> F3 (f,   cell, dim, oct)
//       xp (1:npartmax,  1:ndim)             -> XP (xp,  ipart, dim, npartmax)
//    i.e. element (a,b) of a Fortran (NA,NB) array is at index (b-1)*NA+(a-1).
//
//  * The `oct` derived type is copied byte-for-byte from the Fortran host into a
//    MTLStorageModeShared buffer (no repacking), so `Oct` here MUST match
//    gfortran's native layout of `type oct` in amr/oct_commons.f90:
//        integer(kind=8), dimension(1:nhilbert) :: hkey   ! 8 B  (nhilbert=1)
//        integer(kind=4), dimension(1:ndim)     :: ckey   ! 12 B (ndim=3)
//        logical,         dimension(1:twotondim):: refined! 32 B (8 x logical(4))
//        integer(kind=4)                        :: lev    ! 4 B
//        integer(kind=4)                        :: superoct! 4 B
//    gfortran default LOGICAL is 4 bytes; .true. encodes as 1, .false. as 0,
//    but we always test `!= 0` to be encoding-agnostic.  Total 60 B, padded to
//    64 B (8-byte alignment from the leading int64).  A static_assert in the
//    bridge enforces sizeof(Oct)==64.
//============================================================================
#ifndef RAMSES_METAL_H
#define RAMSES_METAL_H

//----------------------------------------------------------------------------
// Compile-time geometry.  Must match the Fortran build's -DNDIM=... .
// Pass -DNDIM=3 to both xcrun metal and clang++; defaults to 3 here.
//----------------------------------------------------------------------------
#ifndef NDIM
#define NDIM 3
#endif

#ifndef NHILBERT
#define NHILBERT 1
#endif

#define TWOTONDIM (1 << NDIM)        // cells per oct: 2/4/8 for NDIM=1/2/3
#define TWONDIM (2*NDIM)             // face neighbours: 2/4/6
#define NSUBGRID 1
// THREETONDIM = 3^NDIM = (NSUBGRID+2)^NDIM : the 3x..x3 neighbour-oct cube.
#if   NDIM == 1
#define THREETONDIM 3
#elif NDIM == 2
#define THREETONDIM 9
#else
#define THREETONDIM 27
#endif
#define SUBGRIDSIZE THREETONDIM
// f field-column count.  The gravity reuses f(:,1)=residual, f(:,2)=RHS, f(:,3)=mask
// AND f(:,1:ndim)=force, so it always needs 3 columns regardless of NDIM -- exactly
// like the CPU `allocate(m%f(1:twotondim,1:3,...))`.  Using NDIM here (the old code)
// undersizes/aliases f for NDIM<3 (RHS/mask overwrite neighbouring octs).
#define NF 3

//----------------------------------------------------------------------------
// Oct struct: byte-for-byte mirror of gfortran `type oct`.  See header note.
//----------------------------------------------------------------------------
// NOTE: use `long` (64-bit in both macOS clang LP64 and Metal Shading Language;
// MSL has no `long long`).  static_assert in the bridge pins this to 8 bytes.
typedef struct {
    long      hkey[NHILBERT];   // offset 0,  8 B
    int       ckey[NDIM];       // offset 8,  12 B
    int       refined[TWOTONDIM];// offset 20, 32 B  (logical(4) each, !=0 == true)
    int       lev;              // offset 52, 4 B
    int       superoct;         // offset 56, 4 B
} Oct;                          // sizeof == 64 (60 + 4 tail pad)

//----------------------------------------------------------------------------
// Column-major flat-index helpers (1-based logical indices in, 0-based out).
//----------------------------------------------------------------------------
// rho/phi/nref: Fortran (twotondim, ncell)
#define IDX2(cell, oct)            (((oct)-1)*TWOTONDIM + ((cell)-1))
// f: Fortran (twotondim, ndim, ncell)
#define IDX3(cell, dim, oct)       ((((oct)-1)*NF + ((dim)-1))*TWOTONDIM + ((cell)-1))
// particle columns: Fortran (npartmax, ndim) -> element (ipart, dim)
#define IDXP(ipart, dim, npartmax) (((dim)-1)*(npartmax) + ((ipart)-1))

//----------------------------------------------------------------------------
// Fixed-point accumulation (selective extended precision).
//
// rho/nref CIC deposit and global energy/multipole/residual reductions are
// accumulated as signed 64-bit fixed-point in atomic_ulong (two's complement),
// which is exactly associative -> order-independent & bit-reproducible.
//
// A deposit value v (already in float) is converted with:  (long)llround(v * S)
// where S = 2^FP_SHIFT.  The host chooses/validates FP_SHIFT per field so the
// worst-case cell sum stays < 2^62.  Conversion back to float divides by S.
//----------------------------------------------------------------------------
#ifndef FP_SHIFT_RHO
// Monopole (mass/cell) & nref deposit scale exponent.  Cosmological ICs have
// tiny per-particle mass mp ~ 1/Npart (e.g. 5e-7 for 2M particles); a CIC
// sub-deposit mp*w must quantise to many fixed-point units to conserve mass, so
// the scale must be large.  2^40 gives per-particle ~5e5 units (fractions ~1e-6
// accurate) while total mass ~1.0 stays at 2^40 << 2^62 (no accumulator
// overflow); for O(1) masses it is still safe up to ~4e9 total.
#define FP_SHIFT_RHO 40
#endif

//----------------------------------------------------------------------------
// Fixed-point particle positions (Enzo-style, level-independent precision).
//
// A box-normalised coordinate u = (xp+skip)/boxlen in [0,1) is stored as the
// 64-bit integer  ipos = llround(u * 2^NBITS_POS).  At any level L:
//   cell index  = ipos >> (NBITS_POS - L)      (exact bit-shift, no 1/dx)
//   sub-cell fraction = (ipos & ((1<<(NBITS_POS-L))-1)) / 2^(NBITS_POS-L)
// The fraction's top ~24 bits survive fp32, so CIC weights stay ~24-bit
// accurate at EVERY level — instead of losing one bit per level the way the
// old frac = xp/dx - floor(xp/dx) did.  NBITS_POS=48 supports levels up to 48
// while leaving >= 18 fractional bits at level 30.
//----------------------------------------------------------------------------
#ifndef NBITS_POS
#define NBITS_POS 48
#endif
#ifndef FP_SHIFT_RED
#define FP_SHIFT_RED 20        // global reduction (energy/multipole) scale exponent
#endif

//----------------------------------------------------------------------------
// Per-dispatch scalar parameter blocks (passed by value via setBytes).
// One struct per kernel family; mirrors the `value` args of the CUDA kernels.
// Kept POD and 8-byte-field-ordered (doubles first) to avoid padding surprises.
//----------------------------------------------------------------------------

// NOTE: Metal has no `double`.  These structs are shared with the kernels, so
// all reals are `float`; the Obj-C++ bridge narrows the Fortran doubles when
// filling them.  This is the fp32-storage path; extended precision lives inside
// the kernels (two-word fixed-point atomics), not in these scalars.

// CIC density deposit (build_src_part + cic_part_warp)
typedef struct {
    float skip[3];         // sim%m%skip(1:3)
    float dx_loc;          // boxlen / 2^ilevel
    float vol_loc;         // dx_loc^ndim
    float m_refine;        // sim%r%m_refine(ilevel)
    float mass_cut;        // sim%r%mass_cut_refine
    float fp_scale_rho;    // 2^FP_SHIFT_RHO (exact in fp32 for shifts <= 24)
    long  key_off;         // hash key offset for this level (int64, non-atomic)
    int  ckey_max;         // 2^ilevel (cartesian key wrap)
    int  hash_size;
    int  ilevel;
    int  head_idx;         // 1-based first particle of this level
    int  num_parts;
    int  npartmax;
    int  refine_on;        // deposit nref? (rtype controls)
    int  ngridmax;         // real-oct bound; cache (ghost) octs are > ngridmax -> skip in deposit (CUDA deposit_rho:438)
} CicParams;

// Multigrid relaxation / residual / gradient (per-level)
typedef struct {
    float dx;              // cell size AT THIS MG LEVEL (ifine): boxlen/2^ifine
    float fourpi;          // 4*pi*G factor for RHS (NOT the grouped coef)
    float offset;          // mean DENSITY (rho_tot) subtracted in reset_rhs
    float vol_loc;         // cell volume at the FINE level = (boxlen/2^ilevel)^ndim (monopole->density)
    int  hash_size;
    int  head_idx;
    int  num_octs;
    int  redstep;          // 0/1 red-black phase for gauss_seidel
    int  ilevel;
    int  ngridmax;         // neighbor-validity bound (octs beyond are cache/ghost)
    int  head_father;      // fine->coarse base index for restrict/interpolate
    float tfrac;           // time-extrapolation fraction for interpol_phi (0 = none)
    int  use_ghost;        // 1 = fine AMR level: smooth all cells, missing nbor -> interpol_phi ghost
} MgParams;

// Particle kick/drift + force gather
#define PART_MAXLEVEL 32
typedef struct {
    float dtnew;           // dtnew(ilevel)   — action 2 (kick+drift) uses this
    float dtold;           // dtold(ilevel)
    // Per-level timesteps so the action-1 level-transition half-kick can use the
    // PARTICLE's own level dt (matches CPU move_fine: dteff = levelp>=ilevel ?
    // dtnew(levelp) : dtold(levelp)).  Indexed by 1-based level (entry [0] unused).
    float dtnew_lv[PART_MAXLEVEL];
    float dtold_lv[PART_MAXLEVEL];
    float box_size[3];     // periodic box extent (normalised units); drift wraps here
    int  hash_size;
    int  ilevel;
    int  head_idx;
    int  num_parts;
    int  npartmax;
    int  ngridmax;
    int  action_part;      // 1 = level-transition kick, 2 = kick+drift
} PartParams;

// Periodic-box bounds at the deposit level (cic_part_warp stencil wrap).
typedef struct {
    int box_min[3];
    int box_max[3];
    int periodic[3];   // 0/1 per axis
    int _pad[3];
} BoxParams;

// Refinement flagging launch params
typedef struct {
    int head_idx;
    int num_octs;
    int ngridmax;
    int num_nbors;     // smoothing threshold for this pass
    float m_refine;    // particle-count refine threshold (DM)
    int _pad[3];
} FlagParams;

// Connectivity build (father + 27-neighbour) launch params
typedef struct {
    int num_octs;
    int hash_size;
    int nlevelmax;
    int head_idx;      // 1-based first oct of the range (for per-level rebuilds)
    int per[3];
    int _pad[2];
} ConnParams;

// Refinement (oct create / derefine) launch params
typedef struct {
    int head_idx;      // 1-based first oct of the level being processed
    int num_octs;      // octs at that level
    int nlevelmax;
    int hash_size;
    int _pad[4];
} RefineParams;

// Cache-oct (boundary ghost) creation launch params (gpu_refine.cuf make_cache_octs).
// One launch per neighbour direction input_ind; num_octs = #missing nbors in that dir.
typedef struct {
    int num_octs;      // number of cache octs to create this launch (= missing nbors)
    int ngridmax;      // real-oct bound; cache octs live at ngridmax+ifree_cache+...
    int ifree_cache;   // current free offset in the cache region
    int input_ind;     // 1..3^NDIM neighbour-cube direction being filled
    int hash_size;
    int nlevelmax;
    int per[3];        // periodic(1:3)
    float tfrac;       // time-extrapolation factor: phi_b = corr + (corr - corr_old)*tfrac
    int _pad[2];
} CacheParams;

// Generic per-bit radix-sort / scan / elementwise launch params
typedef struct {
    int  n;                // element count
    int  head_idx;
    int  ibit;             // current radix bit
    int  npartmax;
    int  ilevel;
    int  hash_size;        // for bucket_part (particle level assignment)
    int  ckey_max;         // 2^(ilevel-1) at ilevel
    long key_off;          // hash key offset at ilevel
} ScanParams;

//----------------------------------------------------------------------------
// HYDRO (gpu_hydro.cuf port).  Conservative state uold/unew is Fortran
// (twotondim, nvar, noct); the variable order is [rho, rho*u_x, rho*u_y,
// rho*u_z, E_tot].  nener / passive scalars are deferred -> NHVAR = 5 for now
// (= 1 density + 3 momenta + 1 energy, the conserved_t / primitive_t of the CUDA
// port).  The 3 momentum components are always carried even for NDIM<3, exactly
// like the CPU `nvar=5+nener` (hydro_parameters.f90).
//----------------------------------------------------------------------------
#ifndef NHVAR
#define NHVAR 5
#endif
// uold/unew(1:twotondim, 1:nvar, oct) -> element (cell, ivar, oct), 0-based out.
#define UH(cell, ivar, oct)  ((((oct)-1)*NHVAR + ((ivar)-1))*TWOTONDIM + ((cell)-1))

// Riemann solver ids (must match hydro_parameters.f90 solver_*).
#define SOLVER_LLF  1
#define SOLVER_HLL  2
#define SOLVER_HLLC 3

// Per-level hydro launch params (mirrors the `value` args of the CUDA kernels).
// All reals fp32 (Metal has no double); the bridge narrows the Fortran doubles.
typedef struct {
    float gamma;           // adiabatic index (default 1.4)
    float dt;              // dtnew(ilevel)
    float dx;              // boxlen / 2^ilevel
    float smallr;          // density floor (1e-10)
    float smallc;          // sound-speed floor (1e-10)
    float courant_factor;  // CFL number (0.5)
    float fp_scale;        // 2^FP_SHIFT for the reproducible coarse-fine reflux atomics
    int   slope_type;      // 0=1st order, 1=minmod, 2=moncen
    int   riemann;         // SOLVER_LLF / SOLVER_HLL / SOLVER_HLLC
    int   head_idx;        // 1-based first oct of this level
    int   num_octs;        // octs at this level
    int   ngridmax;        // real-oct bound (cache/ghost octs are beyond)
    int   ilevel;
    int   levelmin;        // zero_fine_fluxes only if ilevel<levelmax; reflux only if ilevel>levelmin
    int   levelmax;
} HydroParams;

// hydro_flag_kernel params (density/pressure-gradient refinement criterion).
typedef struct {
    float gamma;
    float err_grad_d;      // density gradient threshold (<=0 disables)
    float err_grad_p;      // pressure gradient threshold (<=0 disables)
    float floor_d;         // density/pressure denominator floor
    float floor_p;         // (unused by the criterion -- CUDA uses floor_d for both)
    int   head_idx;
    int   num_octs;
} HydroFlagParams;

// cmpdt reduction buffer (atomic_uint[9]): [0]=dt min (FLT_MAX-bits init),
// [1,2]=mass lo/hi, [3,4]=ekin, [5,6]=eint, [7,8]=emag (all two-word fixed-point
// at HydroParams.fp_scale).  Host inits red[0]=0x7F7FFFFF, rest 0.
#define HYDRO_RED_N 9

#endif // RAMSES_METAL_H

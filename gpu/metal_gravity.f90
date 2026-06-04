!============================================================================
! metal_gravity.f90 — HYBRID in-loop gravity offload (H5 + H6).
!
! Per AMR level, replace the CPU Poisson+force with the validated Metal
! pipeline.  Mirrors the CUDA layout: the whole mesh is mirrored into unified
! buffers (grid + flat hash + AMR-father + same-level nbor) whenever the mesh
! changes; then each level's gravity runs on the GPU:
!   * consume the CPU-deposited density rho (m%rho)  -> grouped multigrid V-cycle
!   * coarse-fine boundary: interpolate the coarser level's phi into the fine
!     level and hold the refined-region edge cells there (one-way interface)
!   * 4th-order force gradient -> m%f
! No particle data crosses to the GPU (the CPU deposit already gave us rho), so
! there is no per-step particle marshalling.  AMR mesh management (refine/flag/
! move) stays on the CPU.  Host gravity fields are fp64; copy-in narrows rho,
! copy-out widens phi/f.
!============================================================================
module metal_gravity_module
  use metal_bridge_iface
  use iso_c_binding
  implicit none

  logical :: metal_enabled = .false.     ! set true by adaptive_loop after mtl_init
  ! GPU refinement flagging (flag.metal + m_metal_flag), DEFAULT ON.  Connectivity
  ! is fast + verified correct (GPU parallel atomicCAS hash rebuild, cross-checked
  ! 0 lookup failures; per-level nbor/father).  m_metal_flag forces a full grid+
  ! hash re-sync each call and uploads the host's current-layout m%flag1/m%nref,
  ! which fixed the staleness that caused "FATAL: no parent" orphans (the CPU
  ! compacts the mesh between per-level flag passes).  VALIDATED: total oct count
  ! matches the CPU flag to ~0.2% at every step (50/80/100/120/140), mass + energy
  ! conserved (econs=1.00), bit-identical ekin for ~50 steps; thereafter the
  ! trajectory diverges as Lyapunov chaos (same refinement amount, different
  ! specific octs — expected for AMR N-body, like CUDA vs CPU, NOT a bias).  It is
  ! also ~4x FASTER than the CPU flag during active refinement (parallel kernels
  ! vs per-oct hash-dictionary + cache ops).  Set RAMSES_GPU_FLAG=0 for the CPU
  ! flag (bit-reproducible vs the historical CPU runs).
  logical :: metal_flag_on = .true.      ! route refinement flagging to the GPU
  ! GPU AMR refine (data_on_device): create/derefine/compact octs on the GPU
  ! (refine.metal: refine_create/refine_derefine + counting-sort compaction gather
  ! + GPU hash/nbor rebuild), then sync the mesh to the host.  DEFAULT ON.
  ! VALIDATED: total oct count matches the CPU refine EXACTLY through step 50, then
  ! stays statistically equivalent (chaotic divergence, like the flag); mass +
  ! energy conserved (mcons=0, econs=1.00) over 490+ steps, no orphans/crash.
  ! AMR refine ALWAYS runs on the GPU when metal is enabled.  The former
  ! RAMSES_GPU_REFINE=0 path ("CPU refine inside a GPU run") was REMOVED: it handed
  ! the GPU solve a host-marshalled mesh/connectivity that produced a hot
  ! high-velocity tail (refined-level over-energization) -- a known-wrong hybrid we
  ! will not use.  Pure CPU runs (metal disabled) still use r_refine_fine.
  ! grid_dict rebuild: only needed if a CPU consumer reads it.  All per-step
  ! physics is GPU-routed (uses B.hash) so the full-GPU default does not need it;
  ! RAMSES_REFINE_REHASH=1 forces it (e.g. CPU-flag + GPU-refine).
  logical :: refine_rehash   = .false.   ! rebuild CPU grid_dict after GPU refine
  logical :: refine_hostflag1 = .false.  ! EXPERIMENT: force host-mediated flag1 round-trip in both-GPU refine
  logical :: refine_hostmed   = .false.  ! both-GPU path: fully host-mediate flag<->refine (sync B.flag1<->host)
  ! MG V-cycle counts (RAMSES_NCYC_FINE/BASE).  Default 5/4: validated to give a
  ! bit-identical trajectory to 10 cycles for 40 steps (the V-cycle converges to
  ! the fp32 force floor by ~3 cycles given the warm/interpolated initial guess),
  ! ~10% faster overall.
  ! Fixed MG V-cycle counts (Stage 2): the solve reaches its fp32 residual floor then
  ! plateaus dead-flat, so a fixed count replaces the per-iteration convergence readback
  ! (which serialised the async kernel pipeline).  Measured optima: refined levels hit
  ! eps=1e-4 at ~4 cycles (matching the original loop's exit point); the periodic base
  ! plateaus at its floor by ~6 (verified: base ncyc=6 and ncyc=12 give an identical
  ! residual + ekin).  Tune via RAMSES_NCYC_FINE / RAMSES_NCYC_BASE.
  integer :: g_ncyc_fine = 4, g_ncyc_base = 6
  ! RAMSES_MG_CHECK_EVERY=N: re-enable a residual-norm readback at the END of each level
  ! solve every N coarse steps as a convergence monitor (0 = off, the production default).
  integer :: g_mg_check_every = 0
  ! CUDA-style multigrid: run the Fortran multigrid()/recursive_multigrid mirror
  ! (m_metal_multigrid) calling per-leaf mtl_mg_* kernels, instead of the
  ! monolithic hand-coded V-cycle in mtl_poisson_level.  Set RAMSES_METAL_MG=1.
  logical :: metal_mg_driver_on = .false.
  logical :: metal_mg_all_levels = .false.          ! RAMSES_METAL_MG=2: driver for ALL levels (else base only)
  ! Materialised coarse-fine CACHE (ghost) octs (CUDA-faithful boundary) instead of
  ! the inline use_ghost path.  RAMSES_METAL_CACHE=1 -> allocate a cache region
  ! (g_ncell = 2*ngridmax) and run mtl_make_cache before each level's solve.
  ! EXPERIMENTAL: forces a fresh hash/nbor rebuild per solve so the cache region is
  ! clean (single refined level per solve); validate end-to-end with the 1D pancake
  ! <vx> momentum check.  Default off = the prior (inert-cache) behaviour.
  logical :: metal_cache_on = .false.
  logical, private :: g_mg_safe_f(0:63) = .false.   ! per-level MG safe-mode flag
  integer, private :: g_mg_probe_done = 0           ! one-shot V-cycle probe guard
  integer :: g_sort_every = 1            ! Hilbert re-sort cadence (RAMSES_SORT_EVERY)
  logical, private :: alloc_done = .false.
  logical, private :: b_grid_seeded = .false.    ! B.grid seeded from the initial CPU mesh
  logical, private :: part_resident = .false.   ! particles uploaded & owned by the GPU
  integer, private :: g_synced_ifree = -1        ! ifree the GPU hash is current for
  integer, private :: g_nbor_synced  = -1        ! ifree the FULL nbor/father is current for
  ! Mesh-mutation version counter (bumped by m_metal_refine on any create/derefine,
  ! INCLUDING net-zero compaction that leaves ifree unchanged).  The GPU flag uses it
  ! to rebuild connectivity only when the mesh actually changed since the last flag,
  ! instead of forcing a full hash+nbor rebuild on EVERY call (~19% of wall-clock).
  integer, private :: g_mesh_version        = 0
  integer, private :: g_flag_synced_version = -1
  integer, private :: g_ncell, g_hash, g_npm, g_ngridmax
  real(8), private :: g_tpois = 0.0d0, g_tsync = 0.0d0   ! cumulative GPU-gravity / sync wall (s)
  ! Fine-grained profiling accumulators (printed by m_metal_prof_report).
  real(8) :: gt_conn=0, gt_sort=0, gt_split=0, gt_dep=0, gt_mg=0, gt_grad=0, &
             gt_pmar=0, gt_kick=0, gt_kmar=0, gt_refine=0
  real(8) :: gt_mg_lvl(0:30) = 0      ! per-level MG wall (DIAG: where the fine-level cost lives)
  integer(8) :: gt_mg_cnt(0:30) = 0   ! per-level solve count
  real(8) :: gt_norm = 0              ! DIAG: time in mtl_mg_residual_norm2 (the per-iter readback drain)
  real(8) :: gt_setup = 0             ! DIAG: MG prologue (build + make_mask/rhs/initial_phi + restrict_mask drains)
  integer(8), private :: tk0, tkr
  integer, allocatable, private :: g_box_min(:), g_box_max(:)

contains

  ! Mirror the host mesh into the unified buffers and keep the GPU hash current.
  ! First call: full build (alloc + ckey/key_off/box + full hash).  Later calls
  ! refresh the grid (refined flags) and INCREMENTALLY insert newly-created octs
  ! into the hash — the mesh only grows during refine, so we never re-insert the
  ! whole table.  Connectivity (nbor/father) is rebuilt separately by the caller
  ! (full for gravity, per-level for the flag) so the flag stays cheap.
  subroutine metal_ensure_hash(pst)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, twotondim
    type(pst_t), target :: pst
    integer :: L, ddim, num_octs
    integer(c_int),     pointer :: m_ckmax(:)
    integer(c_int64_t), pointer :: m_koff(:)

    associate(r=>pst%s%r, m=>pst%s%m)
    if (.not. alloc_done) then
       block
         character(len=8) :: ce
         call get_environment_variable("RAMSES_METAL_CACHE", ce)
         metal_cache_on = (len_trim(ce) > 0 .and. ce(1:1) /= '0')
       end block
       g_ngridmax = m%ngridmax              ! real-oct bound
       g_ncell = m%ngridmax                 ! main region only...
       ! main region (ngridmax) + coarse-fine GHOST cache region.  The cache holds one
       ! ghost oct per coarse-fine boundary; at a~1 with deep refinement a 1x-ngridmax
       ! cache (g_ncell=2x) OVERFLOWED (need ~3.08M ghosts, cap 3M -> 169 fallbacks).
       ! 2.5x gives a 1.5x-ngridmax cache region (~46% headroom).  Tunable via
       ! RAMSES_METAL_CACHE_MULT (numerator over 2; default 5 = 2.5x).
       if (metal_cache_on) then
          block
            integer :: cmul; character(len=8) :: cm
            cmul = 5
            call get_environment_variable("RAMSES_METAL_CACHE_MULT", cm)
            if (len_trim(cm) > 0) read(cm,*) cmul
            g_ncell = (cmul * m%ngridmax) / 2
          end block
       end if
       g_hash  = 2*g_ncell + 3              ! m%hash_size is only set under _CUDA; size it here
       g_npm   = r%npartmax
       call mtl_alloc_buffers(g_ncell, g_npm, g_hash, r%nlevelmax)
       if (metal_cache_on) call mtl_set_cache_region(g_ngridmax)  ! octs >ngridmax = cache
       ! per-level ckey_max / key_off (1-based-padded: C slot L <-> Fortran L+1)
       call c_f_pointer(mtl_ptr_ckey_max(), m_ckmax, [r%nlevelmax+2])
       call c_f_pointer(mtl_ptr_key_off(),  m_koff,  [r%nlevelmax+2])
       m_ckmax = 0; m_koff = 0
       do L = 1, r%nlevelmax
          m_ckmax(L+1) = m%ckey_max(L)
       end do
       m_koff(2) = 1_c_int64_t
       do L = 2, r%nlevelmax
          m_koff(L+1) = m_koff(L) + m%hkey_max(1, L-1)
       end do
       allocate(g_box_min(3*r%nlevelmax), g_box_max(3*r%nlevelmax))
       do L = 1, r%nlevelmax
          do ddim = 1, ndim
             g_box_min((L-1)*3 + ddim) = m%box_ckey_min(ddim, L)
             g_box_max((L-1)*3 + ddim) = m%box_ckey_max(ddim, L)
          end do
       end do
       alloc_done = .true.
    end if

    if (m%ifree /= g_synced_ifree) then         ! mesh changed since last sync
       ! RAMSES stores octs per level contiguously and COMPACTS on every refine,
       ! so all oct indices shift and the whole hash must be rebuilt (incremental
       ! insertion is invalid).  The rebuild runs on the GPU (parallel atomicCAS)
       ! so it is cheap; ifree changing is the proven mesh-change signal.
       num_octs = m%ifree - 1
       ! The GPU OWNS the mesh: B.grid is mutated in place by the GPU refine, so we
       ! seed it from the CPU-built initial mesh exactly ONCE and never copy again
       ! (per-step m%grid<->B.grid marshalling is what made refine a regression).
       if (.not. b_grid_seeded) then
          call mtl_drain()                       ! host memcpy into B.grid must not race in-flight GPU
          call mtl_copy_grid_in(c_loc(m%grid(1)), 1, num_octs)
          ! One-shot seed of resident B.flag1 from the CPU-built initial flags at
          ! the init(CPU)->GPU handoff.  init_refine_adaptive builds the adaptive
          ! mesh entirely on the host (GPU flag+refine do not run during init), so
          ! the GPU's B.flag1 is still all-zero when the FIRST GPU refine reads it
          ! -> it sees no flags and derefines the whole adaptive mesh (1D pancake
          ! 25/18/3 -> 22/0/0; box-wide ~1e-4 force shift).  m%flag1 still holds the
          ! flags that built the current mesh; copy them in ONCE, then the GPU
          ! flag->refine resident handoff owns B.flag1.
          block
            integer(c_int), pointer :: mf(:)
            integer :: o, c
            call c_f_pointer(mtl_ptr_flag1(), mf, [g_ncell*twotondim])
            do o = 1, num_octs
               do c = 1, twotondim
                  mf((o-1)*twotondim + c) = m%flag1(c, o)
               end do
            end do
          end block
          ! One-shot seed of resident B.phi / B.phi_old from the CPU mesh.  On a
          ! RESTART the host m%phi holds the loaded converged potential; without this
          ! device B.phi is the memset-0 (alloc) value, so the FIRST step's pre-solve
          ! phi_old snapshot captures 0 -> a finer subcycle's coarse-fine boundary
          ! time-extrapolation reads phi_old~0 -> a large one-time over-energization
          ! kick at restart (coarse phi_old rel-rms 1.0, refined phi blown ~50%).
          ! After this, the GPU owns B.phi (each solve overwrites it; the resident
          ! value carries to the next step's phi_old).  (Fresh-start m%phi=0 here too,
          ! matching the old behaviour; only restarts had a nonzero phi to lose.)
          block
            real(c_float), pointer :: mp(:), mpo(:)
            integer :: o, c
            call c_f_pointer(mtl_ptr_phi(),     mp,  [g_ncell*twotondim])
            call c_f_pointer(mtl_ptr_phi_old(), mpo, [g_ncell*twotondim])
            do o = 1, num_octs
               do c = 1, twotondim
                  mp ((o-1)*twotondim + c) = real(m%phi(c, o),     c_float)
                  mpo((o-1)*twotondim + c) = real(m%phi_old(c, o), c_float)
               end do
            end do
          end block
          b_grid_seeded = .true.
       end if
       call mtl_conn_rebuild_hash(num_octs, r%nlevelmax, g_box_min, g_box_max)
       g_nbor_synced  = -1                       ! hash changed -> nbor/father now stale
       g_synced_ifree = m%ifree
    end if
    end associate
  end subroutine metal_ensure_hash

  ! Mesh sync for the gravity path: ensure the hash is current, then rebuild the
  ! FULL nbor/father once per mesh change (no-op once settled within a step).
  subroutine metal_sync_mesh(pst)
    use ramses_commons, only: pst_t
    type(pst_t), target :: pst
    integer :: num_octs
    call metal_ensure_hash(pst)
    if (pst%s%m%ifree /= g_nbor_synced) then
       num_octs = pst%s%m%ifree - 1
       call mtl_conn_build_range(1, num_octs, 1, num_octs, pst%s%r%nlevelmax)
       g_nbor_synced = pst%s%m%ifree
    end if
  end subroutine metal_sync_mesh

  ! Solve gravity for one AMR level on the GPU: phi (multigrid) + f (gradient),
  ! written into m%phi / m%f for that level's octs.
  subroutine m_metal_poisson(pst, ilevel, icount)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, twotondim, dp
    use poisson_parameters, only: ngs_fine, ngs_coarse
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel, icount
    integer, parameter :: MAXITER = 20      ! matches multigrid_fine_commons.f90
    integer :: head, n, o, c, ddim, goct, ncyc, hascoarse
    integer :: maxiter_use
    real(dp) :: boxlen, dx, fourpi, offset, vol_loc, tfrac, eps_use
    character(len=32) :: envbuf
    real(c_float), pointer :: m_rho(:), m_phi(:), m_f(:)
    logical :: per0,per1,per2
    integer(8) :: tc0, tc1, tc2, trate

    associate(r=>pst%s%r, m=>pst%s%m, g=>pst%s%g)
    call system_clock(tc0, trate)
    ! With cache octs, force a fresh real-oct hash+nbor rebuild before each level's
    ! solve so the cache region (rebuilt below) is clean (no stale ghost octs from a
    ! prior level/step).  Perf cost is the per-solve rebuild; correctness first.
    if (metal_cache_on) then; g_synced_ifree = -1; g_nbor_synced = -1; end if
    call metal_sync_mesh(pst)
    call system_clock(tc1); g_tsync = g_tsync + dble(tc1-tc0)/trate
    head = m%head(ilevel); n = m%noct(ilevel)
    ! Set the free-fall timestep constraint g%rho_max(ilevel).  On the CPU this is
    ! set by force_fine, which the Metal path SKIPS (force runs on the GPU) -> it
    ! was left UNINITIALIZED (a fixed-size type member with no default), giving a
    ! garbage value that drove a ~20x-too-small free-fall dt (run took ~20x too
    ! many steps).  rho is the MONOPOLE (mass/cell) in B.rho; the CPU rho_max is
    ! max DENSITY = monopole/vol_loc = B.rho * 2^(3*ilevel).
    g%rho_max(ilevel) = 0.0_dp
    if (n > 0) then
       call c_f_pointer(mtl_ptr_rho(), m_rho, [g_ncell*twotondim])
       block
         real(dp) :: rmax
         rmax = 0.0_dp
         do o = 1, n
            do c = 1, twotondim
               rmax = max(rmax, abs(real(m_rho((head+o-2)*twotondim + c), dp)))
            end do
         end do
         g%rho_max(ilevel) = rmax * 2.0_dp**(ndim*ilevel)   ! density = monopole/dx^ndim
       end block
    end if
    if (n > 0) then
    boxlen = r%boxlen
    dx     = boxlen / 2.0_dp**ilevel
    fourpi = 4.0_dp*acos(-1.0_dp)
    if (r%cosmo) fourpi = 1.5_dp * g%omega_m * g%aexp
    ! NON-GROUPED (CUDA-faithful): the kernels now build the RHS exactly as the CPU
    ! make_bc_rhs does -- f(:,2) = 4piG*(rho_density - rho_bar), with rho_density =
    ! rho_monopole/vol_loc computed in the kernel, and dx2/oneoverdx2 applied per MG
    ! level.  So we pass the raw 4piG factor, the mean density offset, and vol_loc.
    offset = g%rho_tot                                     ! mean density (charge neutrality)
    if (any(.not. r%periodic(1:ndim))) offset = 0.0_dp
    vol_loc = (boxlen/2.0_dp**ilevel)**ndim                ! fine-level cell volume
    if (any(.not. r%periodic(1:ndim))) offset = 0.0_dp
    hascoarse = merge(1, 0, ilevel > r%levelmin)
    ncyc      = merge(g_ncyc_fine, g_ncyc_base, ilevel > r%levelmin)

    ! Diagnostic overrides for the convergence test (does a tighter solve at the
    ! refined level shrink the +8.5% over-energization? -> SOLVE vs GRADIENT).
    eps_use     = r%epsilon
    maxiter_use = MAXITER
    call get_environment_variable("RAMSES_MG_EPS", envbuf)
    if (len_trim(envbuf) > 0) read(envbuf,*) eps_use
    call get_environment_variable("RAMSES_MG_MAXITER", envbuf)
    if (len_trim(envbuf) > 0) read(envbuf,*) maxiter_use

    per0=r%periodic(1); per1=r%periodic(2); per2=r%periodic(3)
    ! Coarse-fine boundary = interpol_phi: 3rd-order CIC of the coarse phi, linearly
    ! extrapolated in time across a subcycle (icount=2).  tfrac matches CPU
    ! force_fine.f90 gradient_phi: dtnew(ilevel)/dtold(ilevel-1)*(icount-1).  The SAME
    ! tfrac drives the MG-solve boundary condition (make_initial_phi) AND the gradient
    ! ghost, so the held boundary phi is consistent with the force (CPU does both).
    tfrac = 0.0_dp
    if (ilevel > r%levelmin .and. icount > 1) then
       if (g%dtold(ilevel-1) > 0.0_dp) &
            tfrac = g%dtnew(ilevel)/g%dtold(ilevel-1)*dble(icount-1)
    end if
    block
      character(len=8) :: notf
      call get_environment_variable("RAMSES_NO_TFRAC", notf)
      if (len_trim(notf) > 0) tfrac = 0.0_dp     ! diagnostic: disable time-extrap
    end block
    block
      character(len=8) :: tdb
      call get_environment_variable("RAMSES_TFRAC_DBG", tdb)
      if (len_trim(tdb) > 0) write(0,'(A,I3,A,I2,A,ES16.8,A,ES12.4,A,ES12.4)') &
           'TFRACDBG-GPU L',ilevel,' icount',icount,' tfrac=',tfrac, &
           ' dtnew=',g%dtnew(ilevel),' dtold(c)=',g%dtold(ilevel-1)
    end block
    ! PRE-SOLVE warm-start dump (RAMSES_DUMP_SEQ): the phi carried INTO this solve,
    ! BEFORE make_cache/make_initial_phi/MG overwrite it.  Compare GPU-refine vs CPU-
    ! refine to test whether the subcycle refine's phi handoff perturbs the warm start.
    block
      character(len=8) :: dqp
      character(len=40) :: fnp
      integer :: oop, ccp, bp
      integer, save :: pseq(64) = 0
      real(c_float), pointer :: mp_phi(:)
      call get_environment_variable("RAMSES_DUMP_SEQ", dqp)
      if (len_trim(dqp) > 0 .and. ilevel >= 12 .and. ilevel <= 64 .and. pseq(ilevel) < 12) then
         call c_f_pointer(mtl_ptr_phi(), mp_phi, [g_ncell*twotondim])
         write(fnp,'(A,I0,A,I0,A)') 'pre_L', ilevel, '_', pseq(ilevel), '.txt'
         open(unit=89, file=trim(fnp), status='replace', action='write')
         write(89,'(A,I6,A,I2)') '#STEP ', g%nstep_coarse, ' icount ', icount
         do oop = 1, n
            do ccp = 1, twotondim
               bp = (head+oop-2)*twotondim + ccp
               write(89,'(I8,ES22.13)') m%grid(head+oop-1)%ckey(1)*2+(ccp-1), real(mp_phi(bp),dp)
            end do
         end do
         close(89)
         pseq(ilevel) = pseq(ilevel) + 1
      end if
    end block
    ! Materialise the coarse-fine boundary cache (ghost) octs for this level BEFORE the
    ! solve: make_initial_phi / reset_rhs / gauss_seidel / gradient then read the cache
    ! oct phi as the boundary (nbr > ngridmax), the CUDA-faithful boundary that replaced
    ! the inline use_ghost path.  (mtl_make_cache is inert unless a cache region exists.)
    if (metal_cache_on) then
       block
         integer :: ncache
         ncache = mtl_make_cache(ilevel, head, n, r%nlevelmax, &
              merge(1,0,per0), merge(1,0,per1), merge(1,0,per2), real(tfrac,c_float))
       end block
    end if
    ! Snapshot phi -> phi_old AFTER make_cache (the device octs are now in the
    ! order the boundary read uses) and BEFORE make_initial_phi (inside the
    ! multigrid) overwrites B.phi.  Saving before make_cache wrote phi_old to the
    ! wrong device slots -> coarse phi_old read as ~0 by a finer level's icount=2
    ! boundary extrapolation -> deep-level over-energization.  RAMSES_SKIP_SAVE_PHIOLD
    ! lets the solve-isolation test supply phi_old externally instead.
    block
      character(len=8) :: skipsv
      call get_environment_variable("RAMSES_SKIP_SAVE_PHIOLD", skipsv)
      if (len_trim(skipsv) == 0) call mtl_save_phi_old(head, n)
    end block
    ! FAITHFUL path (sole path): m_metal_multigrid mirrors multigrid_fine_commons.f90
    ! multigrid()+recursive_multigrid — the convergence loop / levelmin_mg / safe-mode,
    ! dispatching the per-leaf mtl_mg_* kernels.  The old monolithic mtl_poisson_level
    ! was removed in favour of this faithful per-leaf driver.
    call m_metal_multigrid(pst, ilevel, icount, head, n, fourpi, offset, vol_loc, &
         dx, tfrac, hascoarse, merge(1,0,per0), merge(1,0,per1), merge(1,0,per2))
    call system_clock(tc2); gt_mg = gt_mg + dble(tc2-tc1)/trate       ! tc2 = MG end
    if (ilevel <= 30) then; gt_mg_lvl(ilevel) = gt_mg_lvl(ilevel) + dble(tc2-tc1)/trate
                            gt_mg_cnt(ilevel) = gt_mg_cnt(ilevel) + 1; end if
    call mtl_gradient_phi(head, n, real(dx,c_float), real(tfrac,c_float))
    call system_clock(tk0); gt_grad = gt_grad + dble(tk0-tc2)/trate   ! tk0 = gradient end
    ! STAGE-GATE DIAGNOSTIC (RAMSES_DIAG): per-level max|phi|, max|force|, max(density).
    ! Localizes the deep-level blowup -> is it the SOLVE (phi) or the GRADIENT (f)?
    block
      character(len=8) :: diag
      call get_environment_variable("RAMSES_DIAG", diag)
      if (len_trim(diag) > 0) then
         call mtl_drain()
         call c_f_pointer(mtl_ptr_phi(), m_phi, [g_ncell*twotondim])
         call c_f_pointer(mtl_ptr_f(),   m_f,   [g_ncell*twotondim*3])
         block
           real(dp) :: pmax, fmax, rmx, netfx
           integer :: oo, cc, dd, base2, base3
           pmax=0; fmax=0; rmx=0; netfx=0
           do oo=1,n
              do cc=1,twotondim
                 base2=(head+oo-2)*twotondim+cc
                 pmax=max(pmax, abs(real(m_phi(base2),dp)))
                 rmx =max(rmx,  abs(real(m_rho(base2),dp)))
                 ! net force in dim 1 = sum_cells f(:,1)*mass (mass=monopole=B.rho);
                 ! should be ~0 (momentum conservation).  Nonzero -> spurious net force.
                 netfx = netfx + real(m_f((head+oo-2)*3*twotondim+cc),dp)*real(m_rho(base2),dp)
                 do dd=1,ndim
                    base3=((head+oo-2)*3+(dd-1))*twotondim+cc
                    fmax=max(fmax, abs(real(m_f(base3),dp)))
                 end do
              end do
           end do
           write(*,'(A,I3,A,I8,A,ES11.3,A,ES11.3,A,ES11.3,A,ES11.3)') &
             ' [DIAG] L',ilevel,' noct=',n,' maxphi=',pmax,' maxf=',fmax,' maxrho=',rmx,' NETfx=',netfx
         end block
         ! Per-cell leaf dump (RAMSES_DUMP1D): one line per fine cell at the FIRST
         ! base-level solve -> dump_metal.txt, sorted by 1D cell index = ckey(1)*2+(cell-1).
         ! Columns: idx  rho_monopole  phi  fx.  Diff vs CPU dump_cpu.txt to localise
         ! deposit (rho) vs solve (phi) vs gradient (fx).
         block
           character(len=8) :: d1
           character(len=32) :: fn
           integer :: oo, cc, base2, base3, idx
           logical, save :: done_lv(64) = .false.
           call get_environment_variable("RAMSES_DUMP1D", d1)
           if (len_trim(d1) > 0 .and. ilevel <= 64 .and. .not. done_lv(ilevel)) then
              write(fn,'(A,I0,A)') 'dump_metal_L', ilevel, '.txt'
              open(unit=87, file=trim(fn), status='replace', action='write')
              do oo = 1, n
                 do cc = 1, twotondim
                    base2 = (head+oo-2)*twotondim + cc
                    base3 = ((head+oo-2)*3 + 0)*twotondim + cc
                    idx = m%grid(head+oo-1)%ckey(1)*2 + (cc-1)
                    ! 5th col = mask/distance f(:,3) (idim=3) to check operator symmetry
                    write(87,'(I8,4ES18.9)') idx, real(m_rho(base2),dp), &
                         real(m_phi(base2),dp), real(m_f(base3),dp), &
                         real(m_f(((head+oo-2)*3 + 2)*twotondim + cc),dp)
                 end do
              end do
              close(87)
              if (ilevel == r%levelmin) then
              ! Dump same-level nbor (left,self,right) for a few interior base octs.
              block
                integer(c_int), pointer :: m_nb(:)
                integer :: oq, b3
                call c_f_pointer(mtl_ptr_nbor(), m_nb, [g_ncell*3])
                do oq = 0, 3
                   b3 = (head+oq*128-1)*3
                   write(*,'(A,I6,A,3I8,A,2I8)') ' [NBOR] oct=', head+oq*128, &
                        ' nbor(L,S,R)=', m_nb(b3+1), m_nb(b3+2), m_nb(b3+3), &
                        '  ckey=', m%grid(head+oq*128)%ckey(1), m%grid(head+oq*128)%lev
                end do
              end block
              end if
              done_lv(ilevel) = .true.
           end if
           ! Sequential per-SOLVE dump (RAMSES_DUMP_SEQ): every solve of ilevel>=12,
           ! tagged by a per-level solve counter (first 12).  Catches subcycled solves
           ! that the first-solve-only done_lv guard misses.
           block
             character(len=8) :: dq
             character(len=40) :: fn2
             integer :: oo2, cc2, b2
             integer, save :: seq(64) = 0
             call get_environment_variable("RAMSES_DUMP_SEQ", dq)
             if (len_trim(dq) > 0 .and. ilevel >= 12 .and. ilevel <= 64 .and. seq(ilevel) < 12) then
                write(fn2,'(A,I0,A,I0,A)') 'seq_L', ilevel, '_', seq(ilevel), '.txt'
                open(unit=88, file=trim(fn2), status='replace', action='write')
                write(88,'(A,I6,A,I2,A,I6)') '#STEP ', pst%s%g%nstep_coarse, ' icount ', icount, ' noct ', n
                do oo2 = 1, n
                   do cc2 = 1, twotondim
                      b2 = (head+oo2-2)*twotondim + cc2
                      write(88,'(I8,4ES18.9)') m%grid(head+oo2-1)%ckey(1)*2+(cc2-1), &
                           real(m_rho(b2),dp), real(m_phi(b2),dp), &
                           real(m_f(((head+oo2-2)*3+0)*twotondim+cc2),dp), &
                           real(m_f(((head+oo2-2)*3+2)*twotondim+cc2),dp)
                   end do
                end do
                ! per-oct connectivity: ckey hkey father nbor(L,S,R)
                block
                  integer(c_int), pointer :: m_nb(:), m_fa(:)
                  real(c_float),  pointer :: m_po(:)
                  integer :: b3, fath
                  call c_f_pointer(mtl_ptr_nbor(),    m_nb, [g_ncell*3])
                  call c_f_pointer(mtl_ptr_father(),  m_fa, [g_ncell])
                  call c_f_pointer(mtl_ptr_phi_old(), m_po, [g_ncell*twotondim])
                  do oo2 = 1, n
                     b3 = (head+oo2-2)*3
                     fath = m_fa(head+oo2-1)          ! coarse (L-1) parent oct
                     write(88,'(A,I10,I22,I10,3I10,2L2)') '#C ', m%grid(head+oo2-1)%ckey(1), &
                          m%grid(head+oo2-1)%hkey(1), fath, &
                          m_nb(b3+1), m_nb(b3+2), m_nb(b3+3), &
                          m%grid(head+oo2-1)%refined(1), m%grid(head+oo2-1)%refined(2)
                     ! coarse parent phi_old (what make_initial_phi time-extrap reads)
                     if (fath >= 1) write(88,'(A,I10,2ES22.13)') '#PO ', fath, &
                          real(m_po((fath-1)*twotondim+1),dp), real(m_po((fath-1)*twotondim+2),dp)
                  end do
                end block
                ! MG-hierarchy father map for the fine octs (child -> coarse MG oct).
                block
                  integer(c_int), pointer :: m_fmg(:)
                  integer :: km
                  call c_f_pointer(mtl_ptr_father_mg(), m_fmg, [4*g_ncell])
                  write(88,'(A)',advance='no') '#M '
                  do km = 1, min(n*twotondim, 40)
                     write(88,'(I8)',advance='no') m_fmg(km)
                  end do
                  write(88,'(A)') ''
                end block
                ! cache (coarse-fine ghost) octs: slot > g_ngridmax.  Dump ckey + phi
                ! of every live cache oct -> compare the boundary ghost CONTENT.
                block
                  integer(c_int), pointer :: m_gck(:)
                  integer :: cs, gb
                  type(c_ptr) :: gp
                  ! grid ckey/lev are in m_grid_dev (resident Oct array); read via host m%grid? stale.
                  ! Use the resident phi (m_phi) for cache slots; ckey from resident grid pointer.
                  if (metal_cache_on) then
                    do cs = g_ngridmax+1, g_ngridmax+128
                       gb = (cs-1)*twotondim
                       if (abs(real(m_phi(gb+1),dp)) > 0.0_dp) &
                          write(88,'(A,I8,2ES20.11)') '#G ', cs, &
                               real(m_phi(gb+1),dp), real(m_phi(gb+2),dp)
                    end do
                  end if
                end block
                close(88)
                seq(ilevel) = seq(ilevel) + 1
             end if
           end block
         end block
      end if
    end block
    ! phi/f stay RESIDENT on the GPU (B.phi/B.f): the finer-level coarse boundary
    ! reads B.phi directly, so the host m%phi/m%f are only needed for output and
    ! are synced at I/O (m_metal_fields_to_host) — no per-step marshal/sync here.
    end if   ! n > 0
    call system_clock(tc0); g_tpois = g_tpois + dble(tc0-tc1)/trate
    if (ilevel == r%levelmin .and. mod(g%nstep_coarse, r%ncontrol) == 0) &
       write(*,'(A,F8.2,A,F8.2,A)') ' [METAL] cumulative GPU: sync=', g_tsync, &
            ' s  gravity(solve+marshal)=', g_tpois, ' s'
    end associate
  end subroutine m_metal_poisson

  !====================================================================
  ! GRADIENT-ONLY (no solve): apply just the GPU 4th-order force gradient to the
  ! phi currently resident on the device (e.g. set via ramses_set_field :phi),
  ! materialising the coarse-fine boundary cache exactly as m_metal_poisson does
  ! but SKIPPING the multigrid solve.  tfrac=0 (no subcycle time extrapolation),
  ! so this isolates the gradient OPERATOR: gradient the SAME phi on CPU (fp64,
  ! force_fine icount=1) and GPU (fp32) and diff the resulting f.
  !====================================================================
  subroutine m_metal_gradient_only(pst, ilevel, tfrac)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, dp
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    real(dp), intent(in) :: tfrac              ! subcycle time-extrapolation fraction
    integer :: head, n, ncache
    real(dp) :: boxlen, dx
    logical :: per0, per1, per2
    associate(r=>pst%s%r, m=>pst%s%m)
    if (metal_cache_on) then; g_synced_ifree = -1; g_nbor_synced = -1; end if
    call metal_sync_mesh(pst)        ! hash/nbor only; does NOT touch device phi/phi_old
    head = m%head(ilevel); n = m%noct(ilevel)
    if (n > 0) then
       boxlen = r%boxlen
       dx     = boxlen / 2.0_dp**ilevel
       per0=r%periodic(1); per1=r%periodic(2); per2=r%periodic(3)
       ! NB: do NOT call mtl_save_phi_old here — the caller supplies phi_old (set
       ! via ramses_set_field :phi_old) so the tfrac time-extrapolation is tested
       ! on identical, externally-controlled phi/phi_old.
       if (metal_cache_on) then
          ncache = mtl_make_cache(ilevel, head, n, r%nlevelmax, &
               merge(1,0,per0), merge(1,0,per1), merge(1,0,per2), real(tfrac,c_float))
       end if
       call mtl_gradient_phi(head, n, real(dx,c_float), real(tfrac,c_float))
    end if
    end associate
  end subroutine m_metal_gradient_only

  !====================================================================
  ! Device phi_old snapshot (B.phi -> B.phi_old) for one level — the Metal analog
  ! of CUDA's gpu_save_phi_old, called from r_save_phi_old (interpol_phi.f90) at
  ! the SAME points the CPU saves (amr_step.f90:201 pre-solve, :228 nstep==0
  ! post-solve).  Previously the Metal path had NO device hook in r_save_phi_old
  ! (only #ifdef _CUDA / host #else), so phi_old was saved only by m_metal_poisson's
  ! internal pre-solve snapshot and MISSED the nstep==0 post-solve initialisation —
  ! giving a phi_old that differs from the CPU and a wrong subcycle time-extrapolation
  ! (the over-energization root cause).
  !====================================================================
  subroutine m_metal_save_phi_old(pst, ilevel)
    use ramses_commons, only: pst_t
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    integer :: head, n
    ! NOTE: mtl_save_phi_old(head,n) only lands correctly when the device octs are
    ! in host (head:tail) order, which is established by m_metal_poisson's full
    ! setup (sync+cache), not by a bare sync — so this standalone hook is unused in
    ! the production path (the in-poisson internal save is authoritative).  Kept
    ! for the gradient/solve isolation tests, which set phi_old via ramses_set_field.
    if (.not. (metal_enabled .and. b_grid_seeded)) return
    head = pst%s%m%head(ilevel); n = pst%s%m%noct(ilevel)
    if (n > 0) call mtl_save_phi_old(head, n)
  end subroutine m_metal_save_phi_old

  !====================================================================
  ! CUDA-STYLE multigrid: a faithful Fortran transliteration of the CPU
  ! poisson/multigrid_fine_commons.f90 multigrid() + recursive_multigrid(),
  ! calling the per-leaf mtl_mg_* Metal kernels.  This reuses the EXACT CPU
  ! control flow (prologue, build, mask-restrict -> levelmin_mg, the
  ! main_iteration_loop convergence test, SAFE_FACTOR safe-mode, and the
  ! recursive V-cycle), so convergence is identical to the CPU by construction
  ! -- unlike the monolithic hand-coded V-cycle in mtl_poisson_level.
  ! (The shared Fortran driver carries MPI/cache marshalling the single-process
  ! Metal path does not need, hence this dedicated mirror.)
  !====================================================================
  subroutine m_metal_multigrid(pst, ilevel, icount, head, n, fourpi, offset, vol_loc, &
       dx, tfrac, hascoarse, p0, p1, p2)
    use ramses_commons, only: pst_t
    use amr_parameters, only: dp, twotondim
    use poisson_parameters, only: ngs_fine, ngs_coarse
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel, icount, head, n, hascoarse, p0, p1, p2
    real(dp), intent(in) :: fourpi, offset, vol_loc, dx, tfrac
    integer, parameter :: MAXITER = 20
    real(dp), parameter :: SAFE_FACTOR = 0.5_dp
    integer, parameter :: MINITER = 4              ! min V-cycles before a plateau-exit
    real(dp), save :: plat_factor = -1.0_dp        ! RAMSES_MG_PLATEAU (read once); <=0 disables
    character(len=16) :: pe
    integer :: ifine, iter, i, allmasked, levelmin_mg, bnd, isafe, is_base, ncyc
    real(dp) :: err, last_err, i_res, res, rho_tot, eps
    character(len=8) :: probe
    logical :: mg_monitor
    integer(8) :: ts0, ts1, tsr      ! DIAG: setup-phase timer
    associate(r=>pst%s%r, g=>pst%s%g)
    bnd = r%bound_levelmin
    rho_tot = g%rho_tot
    is_base = merge(1, 0, ilevel == r%levelmin)
    ! Periodic base uses epsilon_base = fp32-achievable floor; refined levels use
    ! epsilon.  IDENTICAL criterion to the CPU multigrid() so both converge in the
    ! same few cycles (the singular base can't reach epsilon=1e-4 in fp32 -> grind).
    eps = merge(r%epsilon_base, r%epsilon, is_base == 1)

    ! Plateau early-exit factor (read once).  DEFAULT 0.0 = OFF so the GPU V-cycle
    ! loop uses the IDENTICAL eps/MAXITER convergence logic as the CPU multigrid()
    ! -- keep both paths the same (residuals + cycle counts) to reason about and to
    ! preserve the unit-tested parity.  Opt in with RAMSES_MG_PLATEAU>0 (e.g. 0.99 =
    ! exit once a V-cycle improves the rel-residual by <1%) for a ~5% perf win; it is
    ! physically neutral but DIVERGES the GPU convergence logic from the CPU, so it is
    ! an experiment knob, not the default.  The real fix is to make the GPU base MG
    ! reach the CPU's residual floor (then it converges in ~4 cycles like the CPU).
    if (plat_factor < 0.0_dp) then
       plat_factor = 0.0_dp
       call get_environment_variable("RAMSES_MG_PLATEAU", pe)
       if (len_trim(pe) > 0) read(pe,*) plat_factor
    end if

    ! --- build the MG hierarchy FIRST (whole thing on the ifine==ilevel call): this
    ! sets MG.fine_head / MG.n_fine, which the prologue kernels below index.  Building
    ! AFTER the prologue left the prologue using the PREVIOUS step's oct range -> on any
    ! refinement (mesh-change) step the initial phi / mask / RHS were written to the
    ! wrong octs -> garbage refined-level solve -> ekin blow-up at refinement onset. ---
    call system_clock(ts0, tsr)            ! DIAG: time the prologue (build + make_* + restrict_mask)
    do ifine = ilevel, bnd+1, -1
       call mtl_mg_build(ilevel, ifine, head, n, n, g_box_min, g_box_max, &
            p0, p1, p2, is_base, real(dx,c_float), hascoarse, real(tfrac,c_float))
    end do

    ! --- prologue: initial guess (coarse-fine BC) + mask + BC-modified RHS ---
    call mtl_mg_make_initial_phi(ilevel, real(dx,c_float), real(tfrac,c_float), hascoarse)
    ! DIAG (RAMSES_DUMP_SEQ): dump the make_initial_phi OUTPUT = the real coarse-fine
    ! boundary/initial condition for this solve, per fine cell.  This is what the warm-
    ! start dump could not see (make_initial_phi overwrites it).
    block
      character(len=8) :: dqi
      character(len=40) :: fni
      integer :: ooi, cci, bi
      integer, save :: iseq(64) = 0
      real(c_float), pointer :: mi_phi(:)
      call get_environment_variable("RAMSES_DUMP_SEQ", dqi)
      if (len_trim(dqi) > 0 .and. ilevel >= 12 .and. ilevel <= 64 .and. iseq(ilevel) < 12) then
         call mtl_drain()
         call c_f_pointer(mtl_ptr_phi(), mi_phi, [g_ncell*twotondim])
         write(fni,'(A,I0,A,I0,A)') 'ini_L', ilevel, '_', iseq(ilevel), '.txt'
         open(unit=90, file=trim(fni), status='replace', action='write')
         write(90,'(A,I6,A,I2,A,4ES22.13)') '#STEP ', pst%s%g%nstep_coarse, ' icount ', icount, &
              ' offset/voloc/dx/fourpi ', offset, vol_loc, dx, fourpi
         do ooi = 1, n
            do cci = 1, twotondim
               bi = (head+ooi-2)*twotondim + cci
               write(90,'(I8,ES22.13)') pst%s%m%grid(head+ooi-1)%ckey(1)*2+(cci-1), real(mi_phi(bi),dp)
            end do
         end do
         close(90)
         iseq(ilevel) = iseq(ilevel) + 1
      end if
    end block
    call mtl_mg_make_mask(ilevel)
    call mtl_mg_make_rhs(ilevel, real(fourpi,c_float), real(offset,c_float), &
         real(vol_loc,c_float), real(dx,c_float), hascoarse, real(tfrac,c_float))

    ! --- restrict the mask up the hierarchy -> levelmin_mg (first fully-masked) ---
    levelmin_mg = bnd
    do ifine = ilevel, bnd+1, -1
       allmasked = mtl_mg_restrict_mask(ilevel, ifine)
       if (allmasked == 1) then
          levelmin_mg = ifine
          exit
       end if
    end do

    ! --- iterate-to-epsilon V-cycle loop: EXACT CPU pattern ---
    ! Mirrors poisson/multigrid_fine_commons.f90 multigrid(): iterate full V-cycles
    ! until the relative residual err < r%epsilon or iter == MAXITER, using the same
    ! initial-residual normalisation i_res, the same err = sqrt(res/(i_res+1e-20*
    ! rho_tot^2)) formula, and the same residual-driven safe-mode escalation.  This
    ! replaces the fixed g_ncyc count so the Metal solve converges to the SAME
    ! tolerance as the CPU (the per-iteration residual readback is intrinsic to the
    ! tolerance test; correctness/parity over the async-pipeline perf optimisation).
    call system_clock(ts1); gt_setup = gt_setup + dble(ts1-ts0)/tsr   ! DIAG: end prologue timing
    iter  = 0
    err   = 1.0_dp
    i_res = 0.0_dp
    main_iteration_loop: do
       iter  = iter + 1
       isafe = merge(1, 0, g%safe_mode(ilevel))

       ! Pre-smoothing (ngs_fine red+black sweeps)
       call mtl_mg_smooth(ilevel, ilevel, isafe, ngs_fine)

       ! Compute new residual (feeds restrict_residual)
       call mtl_mg_cmp_residual(ilevel, ilevel)

       ! Compute initial residual norm (first iteration)
       if (iter == 1) then
          block; integer(8)::tn0,tn1,tnr; call system_clock(tn0,tnr)
          i_res = mtl_mg_residual_norm2(ilevel)
          call system_clock(tn1); gt_norm = gt_norm + dble(tn1-tn0)/tnr; end block
       end if

       ! Coarse-grid correction (one recursive V-cycle)
       if (ilevel > levelmin_mg) then
          call mtl_mg_restrict_residual(ilevel, ilevel)
          ! Periodic base (is_base): project the constant null space out of the
          ! coarse RHS so the singular periodic operator stays consistent.
          if (is_base == 1) call mtl_mg_zeromean_rhs(ilevel, ilevel-1)
          call mtl_mg_reset_corr(ilevel, ilevel-1)
          call metal_recursive_mg(pst, ilevel, ilevel-1, isafe, levelmin_mg, is_base)
          call mtl_mg_interpolate_correct(ilevel, ilevel)
       end if

       ! Post-smoothing
       call mtl_mg_smooth(ilevel, ilevel, isafe, ngs_fine)

       ! Update fine residual + norm for the convergence test
       call mtl_mg_cmp_residual(ilevel, ilevel)
       block; integer(8)::tn0,tn1,tnr; call system_clock(tn0,tnr)
       res = mtl_mg_residual_norm2(ilevel)
       call system_clock(tn1); gt_norm = gt_norm + dble(tn1-tn0)/tnr; end block

       last_err = err
       err = sqrt(res / (i_res + 1.0d-20*rho_tot**2))

       ! Converged?
       if (err < eps .or. iter >= MAXITER) exit

       ! Plateau early-exit: the periodic base can't reach eps in fp32 (it floors
       ! ~4e-3) and the residual plateaus by ~6 cycles, yet the loop otherwise grinds
       ! all MAXITER=20 (400/420 base solves did, 95%).  Once a V-cycle improves the
       ! relative residual by < (1-plat_factor), further cycles only re-grind the fp32
       ! floor -> exiting is physically identical to MAXITER (the floored residual,
       ! hence phi, is unchanged) but saves ~14/20 base V-cycles + their per-cycle
       ! residual readback.  Guard MINITER so well-converging levels still get their
       ! fast initial cycles; safe-mode escalation below still runs for cycles 4..N.
       if (plat_factor > 0.0_dp .and. iter >= MINITER .and. err >= last_err*plat_factor) exit

       ! Not converged: residual-driven safe-mode escalation for the level
       if (err > last_err*SAFE_FACTOR .and. .not. g%safe_mode(ilevel)) then
          g%safe_mode(ilevel) = .true.
       end if
    end do main_iteration_loop

    print '(A,I5,A,I5,A,1pE10.3)', '   ==> Level=', ilevel, ' Step=', iter, ' Error=', err
    if (iter == MAXITER) print *, 'WARN: Metal fine multigrid Poisson failed to converge...'

    ! Zero-mean gauge pin for the periodic base (null-space drift control); the
    ! monolithic mtl_poisson_level does this too.  Refined levels are Dirichlet.
    if (hascoarse == 0) call mtl_mg_gauge_pin(ilevel)
    end associate
  end subroutine m_metal_multigrid

  ! Recursive V-cycle, mirrors recursive_multigrid (multigrid_fine_commons.f90).
  recursive subroutine metal_recursive_mg(pst, ilevel, ifine, isafe, levelmin_mg, is_base)
    use ramses_commons, only: pst_t
    use poisson_parameters, only: ngs_coarse, ncycles_coarse_safe
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel, ifine, isafe, levelmin_mg, is_base
    integer :: i, icycle, ncycle

    if (ifine <= levelmin_mg) then
       ! Coarsest level: solve 'directly' with 2*ngs_coarse sweeps (batched)
       call mtl_mg_smooth(ilevel, ifine, isafe, 2*ngs_coarse)
       return
    end if

    ncycle = 1
    if (isafe == 1) ncycle = ncycles_coarse_safe

    do icycle = 1, ncycle
       ! Pre-smoothing
       call mtl_mg_smooth(ilevel, ifine, isafe, ngs_coarse)
       ! Residual + restrict to coarser level
       call mtl_mg_cmp_residual(ilevel, ifine)
       call mtl_mg_restrict_residual(ilevel, ifine)
       if (is_base == 1) call mtl_mg_zeromean_rhs(ilevel, ifine-1)   ! periodic null-space projection
       ! Reset correction at the coarser level, recurse, interpolate back
       call mtl_mg_reset_corr(ilevel, ifine-1)
       call metal_recursive_mg(pst, ilevel, ifine-1, isafe, levelmin_mg, is_base)
       call mtl_mg_interpolate_correct(ilevel, ifine)
       ! Post-smoothing
       call mtl_mg_smooth(ilevel, ifine, isafe, ngs_coarse)
    end do
  end subroutine metal_recursive_mg

  ! RAMSES_MG_VERBOSE passthrough for the driver's convergence trace.
  function get_mg_verbose() result(v)
    character(len=8) :: v
    call get_environment_variable('RAMSES_MG_VERBOSE', v)
  end function get_mg_verbose

  ! GPU leapfrog kick/drift for the DM particles at level ilevel.  The force
  ! field f is already resident on the GPU (from m_metal_poisson); we marshal
  ! this level's particles xp->ipos(fixed-point)/vp/levelp, run the kernel
  ! (force gather via the resident hash + f, then kick/drift), and marshal the
  ! updated positions/velocities back.  action_part: 1=kick-only, 2=kick+drift.
  subroutine m_metal_kick_drift(pst, ilevel, action_part)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, dp
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel, action_part
    integer :: h, t, np, ip, ddim
    real(dp) :: boxlen, sc, invbl, u
    integer(c_int64_t), pointer :: m_ipos(:)
    real(c_float),      pointer :: m_vp(:)
    integer(c_int),     pointer :: m_levelp(:)
    integer(8) :: c0, c1, rate

    associate(r=>pst%s%r, p=>pst%s%p, g=>pst%s%g, m=>pst%s%m)
    call metal_sync_mesh(pst)                 ! ensure mesh/connectivity current (f is resident)
    h = p%headp(ilevel); t = p%tailp(ilevel); np = t - h + 1
    if (np > 0) then
       boxlen = r%boxlen; sc = 2.0_dp**48
       ! ipos/vp/levelp are RESIDENT (no input marshalling).  The kernel gathers
       ! the force from the resident f-field and updates resident ipos/vp.
       call system_clock(c0, rate)
       block
         real(c_float) :: dtnew_a(r%nlevelmax), dtold_a(r%nlevelmax)
         integer :: lv
         do lv = 1, r%nlevelmax
            dtnew_a(lv) = real(g%dtnew(lv), c_float)
            dtold_a(lv) = real(g%dtold(lv), c_float)
         end do
         ! Pass per-level dt so the action-1 half-kick uses the particle's own
         ! level dt (dtnew(levelp)/dtold(levelp)), matching CPU move_fine — fixes
         ! the level-transition kick (over-energization seed).
         call mtl_kick_drift_part(ilevel, h, np, g_npm, g_hash, action_part, &
              dtnew_a, dtold_a, r%nlevelmax, &
              real(boxlen,c_float), real(boxlen,c_float), real(boxlen,c_float), &
              merge(1,0,r%periodic(1)), merge(1,0,r%periodic(2)), merge(1,0,r%periodic(3)))
       end block
       call system_clock(c1); gt_kick = gt_kick + dble(c1-c0)/rate
       block
         character(len=8) :: kd
         call get_environment_variable("RAMSES_DIAG", kd)
         if (len_trim(kd) > 0) then
            call mtl_drain()
            call c_f_pointer(mtl_ptr_vp(), m_vp, [g_npm*ndim])
            block
              real(dp) :: vmx; integer :: iq
              vmx = 0.0_dp
              do iq = h, t
                 vmx = max(vmx, abs(real(m_vp(iq),dp)))
              end do
              write(*,'(A,I3,A,I2,A,I9,A,ES12.4)') ' [KICKDIAG] L',ilevel,' act=',action_part,' np=',np,' max|vp1|=',vmx
            end block
         end if
       end block
       ! Particle state (ipos/vp/levelp) stays RESIDENT on the GPU.  The CFL
       ! reduction (ekin/vmax) now runs on the GPU too (m_metal_newdt_part), so
       ! the host xp/vp are no longer needed per step — they are synced only at
       ! I/O (m_metal_part_to_host).  Marshal here only if the host owns newdt.
       if (.not. metal_enabled) then
          call c_f_pointer(mtl_ptr_ipos(),   m_ipos,   [g_npm*ndim])
          call c_f_pointer(mtl_ptr_vp(),     m_vp,     [g_npm*ndim])
          call c_f_pointer(mtl_ptr_levelp(), m_levelp, [g_npm])
          call system_clock(c0, rate)
          do ip = h, t
             do ddim = 1, ndim
                p%xp(ip,ddim) = real(m_ipos((ddim-1)*g_npm + ip), dp) / sc * boxlen
                p%vp(ip,ddim) = real(m_vp((ddim-1)*g_npm + ip), dp)
             end do
             p%levelp(ip) = m_levelp(ip)
          end do
          call system_clock(c1); gt_kmar = gt_kmar + dble(c1-c0)/rate
       end if
    end if
    end associate
  end subroutine m_metal_kick_drift

  ! GPU per-level particle CFL reduction (replaces CPU newdt_part): vmax + ekin
  ! over the RESIDENT velocities, so the host vp need not be marshalled each step.
  subroutine m_metal_newdt_part(pst, ilevel, vmax, ekin)
    use ramses_commons, only: pst_t
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    real(8), intent(out) :: vmax, ekin
    integer :: h, t, np
    real(c_double) :: vm, ek
    associate(r=>pst%s%r, p=>pst%s%p)
    h = p%headp(ilevel); t = p%tailp(ilevel); np = t - h + 1
    vm = 0.0d0; ek = 0.0d0
    if (np > 0) call mtl_newdt_part(ilevel, h, np, g_npm, vm, ek)
    vmax = vm; ekin = ek
    end associate
  end subroutine m_metal_newdt_part

  ! Sync RESIDENT particle state (ipos/vp/levelp) -> host xp/vp/levelp.  Called
  ! before I/O (the per-step kick/drift no longer marshals when metal owns them).
  subroutine m_metal_part_to_host(pst)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, twotondim, dp
    type(pst_t), target :: pst
    integer :: ip, ddim, o, c
    real(dp) :: boxlen, sc
    integer(c_int64_t), pointer :: m_ipos(:)
    real(c_float),      pointer :: m_vp(:), m_phi(:), m_f(:)
    integer(c_int),     pointer :: m_levelp(:)
    if (.not. (metal_enabled .and. part_resident)) return
    associate(r=>pst%s%r, p=>pst%s%p, m=>pst%s%m)
    call mtl_drain()                 ! host reads resident GPU buffers below
    boxlen = r%boxlen; sc = 2.0_dp**48
    call c_f_pointer(mtl_ptr_ipos(),   m_ipos,   [g_npm*ndim])
    call c_f_pointer(mtl_ptr_vp(),     m_vp,     [g_npm*ndim])
    call c_f_pointer(mtl_ptr_levelp(), m_levelp, [g_npm])
    do ip = 1, p%npart
       do ddim = 1, ndim
          p%xp(ip,ddim) = real(m_ipos((ddim-1)*g_npm + ip), dp) / sc * boxlen
          p%vp(ip,ddim) = real(m_vp((ddim-1)*g_npm + ip), dp)
       end do
       p%levelp(ip) = m_levelp(ip)
    end do
    ! sync idp back in the GPU particle order (so xp/vp/idp are a consistent triple)
    block
      integer(c_int), pointer :: m_idp(:)
      call c_f_pointer(mtl_ptr_idp(), m_idp, [g_npm])
      do ip = 1, p%npart
         p%idp(ip) = int(m_idp(ip), kind(p%idp(ip)))
      end do
    end block
    ! Also marshal the resident gravity fields phi/f -> host (for output_poisson);
    ! they are kept resident during the run (no per-step marshal).
    call c_f_pointer(mtl_ptr_phi(), m_phi, [g_ncell*twotondim])
    call c_f_pointer(mtl_ptr_f(),   m_f,   [g_ncell*twotondim*3])
    do o = 1, m%noct_used
       do c = 1, twotondim
          m%phi(c, o) = real(m_phi((o-1)*twotondim + c), dp)
          do ddim = 1, ndim
             m%f(c, ddim, o) = real(m_f(((o-1)*3 + (ddim-1))*twotondim + c), dp)
          end do
       end do
    end do
    end associate
  end subroutine m_metal_part_to_host

  ! ---- GPU density deposit (replaces the CPU r_cic_part) -------------------
  ! The multi-level monopole rho is built RESIDENT in B.rho: zero once, deposit
  ! each level (accumulate), finalize once.  Driven from m_rho_fine's level loop.

  ! Upload all particles to the resident GPU arrays ONCE; thereafter the GPU owns
  ! them (sort/split/kick-drift reorder/update in place).  DM particle set is
  ! fixed, so this happens a single time.
  subroutine m_metal_part_upload(pst)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, dp
    type(pst_t), target :: pst
    integer :: ip, ddim
    real(dp) :: boxlen, sc, invbl, u
    integer(c_int64_t), pointer :: m_ipos(:)
    real(c_float),      pointer :: m_vp(:), m_mp(:)
    integer(c_int),     pointer :: m_levelp(:)
    associate(r=>pst%s%r, p=>pst%s%p)
    boxlen = r%boxlen; sc = 2.0_dp**48; invbl = sc/boxlen
    call c_f_pointer(mtl_ptr_ipos(),   m_ipos,   [g_npm*ndim])
    call c_f_pointer(mtl_ptr_vp(),     m_vp,     [g_npm*ndim])
    call c_f_pointer(mtl_ptr_mp(),     m_mp,     [g_npm])
    call c_f_pointer(mtl_ptr_levelp(), m_levelp, [g_npm])
    do ip = 1, p%npart
       do ddim = 1, ndim
          u = p%xp(ip,ddim) * invbl
          m_ipos((ddim-1)*g_npm + ip) = modulo(int(u, c_int64_t), int(sc, c_int64_t))
          m_vp((ddim-1)*g_npm + ip)   = real(p%vp(ip,ddim), c_float)
       end do
       m_mp(ip)     = real(p%mp(ip), c_float)
       m_levelp(ip) = p%levelp(ip)
    end do
    ! upload idp (unique label) so the GPU reorder (sort/split) carries it -> the
    ! phase dump's idp column stays attached to the right particle (diagnostic).
    block
      integer(c_int), pointer :: m_idp(:)
      call c_f_pointer(mtl_ptr_idp(), m_idp, [g_npm])
      do ip = 1, p%npart
         m_idp(ip) = int(p%idp(ip), c_int)
      end do
    end block
    part_resident = .true.
    end associate
  end subroutine m_metal_part_upload

  subroutine m_metal_rho_zero(pst, ilevel)
    use ramses_commons, only: pst_t
    use amr_parameters, only: twotondim
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    integer :: cell_base, ncells
    integer(8) :: c0, c1, rate
    call system_clock(c0, rate)
    call metal_sync_mesh(pst)        ! ensure hash/connectivity current for the deposit
    call system_clock(c1); gt_conn = gt_conn + dble(c1-c0)/rate
    if (.not. part_resident) call m_metal_part_upload(pst)
    ! Zero only the cells of levels [ilevel..nlevelmax] (octs [head(ilevel),
    ! noct_used]); coarser levels' rho/nref must survive a subcycle re-deposit.
    associate(m=>pst%s%m)
    cell_base = (m%head(ilevel)-1)*twotondim
    ncells    = (m%noct_used - m%head(ilevel) + 1)*twotondim
    call mtl_cic_zero(cell_base, ncells)
    end associate
  end subroutine m_metal_rho_zero

  ! GPU Hilbert sort of level-ilevel particles (reorders resident arrays).
  subroutine m_metal_gpu_sort(pst, ilevel)
    use ramses_commons, only: pst_t
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    integer :: h, num
    integer(8) :: c0, c1, rate
    associate(r=>pst%s%r, p=>pst%s%p, g=>pst%s%g)
    ! The Hilbert re-sort only COALESCES the (order-independent) deposit; particles
    ! drift little per step, so re-sorting every step is wasteful.  Skip on
    ! off-cadence steps (RAMSES_SORT_EVERY).
    if (g_sort_every > 1 .and. mod(g%nstep_coarse, g_sort_every) /= 0) return
    h = p%headp(ilevel); num = p%tailp(r%nlevelmax) - h + 1
    call system_clock(c0, rate)
    if (num > 0) call mtl_gpu_sort_part(ilevel, h, num)
    call system_clock(c1); gt_sort = gt_sort + dble(c1-c0)/rate
    end associate
  end subroutine m_metal_gpu_sort

  ! GPU split of [headp(ilevel), tailp(nlevelmax)] -> stay@ilevel / descend@ilevel+1,
  ! reorder resident arrays, update headp/tailp (mirrors the CPU split_part).
  subroutine m_metal_gpu_split(pst, ilevel)
    use ramses_commons, only: pst_t
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    integer :: h, num, n_stay, ilev
    integer(c_int),     pointer :: m_ckmax(:)
    integer(c_int64_t), pointer :: m_koff(:)
    integer(8) :: c0, c1, rate
    associate(r=>pst%s%r, p=>pst%s%p)
    h = p%headp(ilevel); num = p%tailp(r%nlevelmax) - h + 1
    call system_clock(c0, rate)
    if (num > 0) then
       call c_f_pointer(mtl_ptr_ckey_max(), m_ckmax, [r%nlevelmax+2])
       call c_f_pointer(mtl_ptr_key_off(),  m_koff,  [r%nlevelmax+2])
       n_stay = mtl_gpu_split_part(ilevel, h, num, g_hash, m_ckmax(ilevel+1), m_koff(ilevel+1))
    else
       n_stay = 0
    end if
    call system_clock(c1); gt_split = gt_split + dble(c1-c0)/rate
    p%tailp(ilevel) = p%headp(ilevel) + n_stay - 1
    do ilev = ilevel+1, r%nlevelmax
       p%headp(ilev) = p%tailp(ilevel) + 1
       p%tailp(ilev) = p%npart
    end do
    end associate
  end subroutine m_metal_gpu_split

  ! Deposit level-ilevel particles [headp..tailp] into the resident accumulators.
  subroutine m_metal_deposit(pst, ilevel, rtype)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, dp
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel, rtype
    integer :: h, t, np
    integer(c_int),     pointer :: m_ckmax(:)
    integer(c_int64_t), pointer :: m_koff(:)
    integer(8) :: c0, c1, rate

    associate(r=>pst%s%r, p=>pst%s%p)
    ! Deposit ALL particles at this level AND FINER into the level-ilevel grid, exactly
    ! like the CPU cic_part (rho_fine.f90: do i=headp(ilevel),tailp(nlevelmax)).  The
    ! base/coarse grid must see every particle's mass (multilevel rho); using
    ! tailp(ilevel) deposited only the particles RESIDENT at ilevel, dropping all mass
    ! that descended to refined levels -> base rho summed <1 (e.g. 0.766) -> wrong mean
    ! offset -> tilted base phi -> spurious box-wide force.  No-refinement is unaffected
    ! (tailp(nlevelmax)==tailp(ilevel) when ilevel==nlevelmax).
    h = p%headp(ilevel); t = p%tailp(r%nlevelmax); np = t - h + 1
    call system_clock(c0, rate)
    if (np > 0) then
       ! ipos/mp are RESIDENT on the GPU (sort/split keep them current) -> no marshal.
       call c_f_pointer(mtl_ptr_ckey_max(), m_ckmax, [r%nlevelmax+2])
       call c_f_pointer(mtl_ptr_key_off(),  m_koff,  [r%nlevelmax+2])
       ! m_refine=0 (>=0) -> nref accumulates the CIC particle count for the flag.
       call mtl_cic_deposit(ilevel, h, np, g_npm, g_hash, m_ckmax(ilevel+1), m_koff(ilevel+1), &
            0.0_c_float, 0.0_c_float, 1, &
            merge(1,0,r%periodic(1)), merge(1,0,r%periodic(2)), merge(1,0,r%periodic(3)))
    end if
    call system_clock(c1); gt_dep = gt_dep + dble(c1-c0)/rate
    end associate
  end subroutine m_metal_deposit

  ! Finalize the resident rho/nref, set the mean density, and copy nref back to
  ! the host (the CPU refinement flag still reads m%nref).
  subroutine m_metal_rho_finish(pst, ilevel)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, twotondim, dp
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    integer :: ip, o, c, hd, n, lev
    real(dp) :: tot, boxlen
    real(c_float), pointer :: m_nref(:)

    associate(r=>pst%s%r, p=>pst%s%p, g=>pst%s%g, m=>pst%s%m)
    ! Finalize only the cells of levels [ilevel..nlevelmax] (octs [head(ilevel),
    ! noct_used]); coarser levels keep their already-finalized rho/nref.
    call mtl_cic_finalize((m%head(ilevel)-1)*twotondim, &
                          (m%noct_used - m%head(ilevel) + 1)*twotondim)
    boxlen = r%boxlen
    tot = 0.0_dp
    do ip = 1, p%npart
       tot = tot + p%mp(ip)
    end do
    g%rho_tot = tot / boxlen**ndim                      ! mean density (Poisson offset)
    call mtl_drain()                                    ! host reads B.nref (finalize is async)
    call c_f_pointer(mtl_ptr_nref(), m_nref, [g_ncell*twotondim])
    do lev = ilevel, r%nlevelmax
       hd = m%head(lev); n = m%noct(lev)
       do o = 1, n
          do c = 1, twotondim
             m%nref(c, hd+o-1) = real(m_nref((hd+o-2)*twotondim + c), dp)
          end do
       end do
    end do
    end associate
  end subroutine m_metal_rho_finish

  ! GPU refinement flagging: compute flag1 on the GPU (from the resident nref +
  ! grid + nbor + father), then copy it back to the host for the CPU refine.
  subroutine m_metal_flag(pst, ilevel, icount)
    use ramses_commons, only: pst_t
    use amr_parameters, only: twotondim, dp
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel, icount
    integer :: head, num, head1, num1, o, c, do_rules

    associate(r=>pst%s%r, m=>pst%s%m)
    ! Ensure mesh hash + connectivity are current.  With GPU refine (default)
    ! B.grid is GPU-owned and m_metal_refine already invalidates g_synced_ifree
    ! AND g_nbor_synced on any change, so metal_sync_mesh is a NO-OP once the step
    ! has settled: the flag SHARES the gravity's hash+nbor/father cache (which the
    ! same-level rho/poisson just built) instead of forcing a full grid+hash
    ! rebuild + a per-level connectivity rebuild on EVERY call.  That forced
    ! re-sync was ~19% of the full-run wall-clock and entirely redundant here.
    ! The flag's kernels read only B.father (init_flag) and B.nbor (smooth +
    ! enforce_rules); metal_sync_mesh's full build covers both for all levels.
    ! Decide whether to force a full hash + nbor/father rebuild before the flag.  The
    ! flag reads father (init_flag) and nbor (smooth / enforce_rules); feeding it STALE
    ! connectivity gives wrong refinement flags at coarse-fine boundaries -- the dominant
    ! metal-vs-CPU divergence the flag-connectivity fix removed (rms 2.2e-3 -> 1.8e-5).
    ! The hazard is a refine between the gravity solve and this flag pass changing the
    ! mesh with NO net ifree change (kill+make), which the ifree guard misses.
    ! m_metal_refine bumps g_mesh_version on ANY create/derefine (keyed off
    ! ncreate+nkill, so net-zero compaction is caught too).  Rebuild the connectivity
    ! only when the mesh actually mutated since the last flag -- otherwise the flag
    ! SHARES the gravity solve's already-current hash+nbor (forcing a full rebuild on
    ! EVERY call was ~19% of wall-clock, entirely redundant once settled).
    if (g_mesh_version /= g_flag_synced_version) then
       g_synced_ifree = -1
       g_nbor_synced  = -1
    end if
    call metal_sync_mesh(pst)
    g_flag_synced_version = g_mesh_version
    head  = m%head(ilevel);   num  = m%noct(ilevel)
    head1 = m%head(ilevel+1); num1 = m%noct(ilevel+1)
    ! Upload the host's CURRENT-layout flag1 + nref into the resident buffers.
    ! The GPU's B.flag1/B.nref go stale whenever the CPU compacts the mesh
    ! between per-level flag passes (refine_fine runs mid-recursion under
    ! subcycling, shifting every oct index).  m%flag1/m%nref are the source of
    ! truth the CPU flag itself reads, so mirroring them makes the GPU flag read
    ! exactly the CPU inputs: init_flag needs the level-(ilevel+1) children's
    ! flag1, flag_poisson needs the level-ilevel nref.
    head1 = m%head(ilevel+1); num1 = m%noct(ilevel+1)
    ! With GPU refine the mesh + B.flag1 are GPU-owned and stay consistent through the
    ! GPU compaction, and B.nref is freshly deposited each step, so the resident
    ! values are already correct -- uploading stale host copies would corrupt them.
    ! NOTE (2026-05-31): tried ALWAYS uploading children flag1 to fix a GPU-refine
    ! under-refinement (44 vs CPU 50 L11 cells); it CORRUPTED the resident full-GPU
    ! state (per-particle dv up to 0.13, sign flips) -> REVERTED.  The flag<->refine
    ! handoff must keep resident B.flag1 authoritative.  Only the refine_hostmed
    ! EXPERIMENT (default off) host-mediates the flag<->refine handoff.  See PORT_MAP.
    if (refine_hostmed) then
       call mtl_drain()                              ! host writes B.flag1/B.nref below
       block
         integer(c_int), pointer :: mf(:)
         real(c_float),  pointer :: nf(:)
         call c_f_pointer(mtl_ptr_flag1(), mf, [g_ncell*twotondim])
         call c_f_pointer(mtl_ptr_nref(),  nf, [g_ncell*twotondim])
         do o = 1, num1                                   ! children flag1 (for init_flag)
            do c = 1, twotondim
               mf((head1+o-2)*twotondim + c) = m%flag1(c, head1+o-1)
            end do
         end do
         do o = 1, num                                    ! this level's nref (for flag_poisson)
            do c = 1, twotondim
               nf((head+o-2)*twotondim + c) = real(m%nref(c, head+o-1), c_float)
            end do
         end do
       end block
    end if
    if (num > 0) then
       do_rules = 0
       if (ilevel > r%levelmin) then
          if (icount < r%nsubcycle(ilevel-1)) do_rules = 1
       end if
       call mtl_flag(head, num, head1, num1, g_ncell, &
            real(r%m_refine(ilevel), c_float), r%nexpand(ilevel), do_rules)
       ! copy flag1 (32-bit int) back to the host map for the CPU refine
       call mtl_drain()                              ! host reads B.flag1 below
       block
         integer(c_int), pointer :: mf(:)
         call c_f_pointer(mtl_ptr_flag1(), mf, [g_ncell*twotondim])
         do o = 1, num
            do c = 1, twotondim
               m%flag1(c, head+o-1) = mf((head+o-2)*twotondim + c)
            end do
         end do
       end block
       ! DIAG: how many ilevel cells exceed nref threshold vs got flagged?
       block
         character(len=8) :: rdump
         call get_environment_variable('RAMSES_REFINE_DUMP', rdump)
       if (len_trim(rdump)>0) then
         block
           integer(c_int), pointer :: mf(:)
           real(c_float),  pointer :: nf(:)
           integer :: nflag, nover, nref_chld
           call c_f_pointer(mtl_ptr_flag1(), mf, [g_ncell*twotondim])
           call c_f_pointer(mtl_ptr_nref(),  nf, [g_ncell*twotondim])
           nflag=0; nover=0
           do o = 1, num
             do c = 1, twotondim
               if (mf((head+o-2)*twotondim + c)==1) nflag=nflag+1
               if (nf((head+o-2)*twotondim + c) >= real(r%m_refine(ilevel),c_float)) nover=nover+1
             end do
           end do
           ! children (ilevel+1) flag1 count, drives init_flag
           nref_chld=0
           do o = 1, num1
             do c = 1, twotondim
               if (mf((head1+o-2)*twotondim + c)==1) nref_chld=nref_chld+1
             end do
           end do
           write(0,'(A,I3,A,I8,A,I8,A,I8,A,I8)') '[FLAG] ilevel=',ilevel, &
                ' nover(nref>=mref)=',nover,' nflag=',nflag, &
                ' child_flag1=',nref_chld,' num1=',num1
         end block
       end if
       end block
    end if
    end associate
  end subroutine m_metal_flag

  ! GPU AMR refine for levels [ilevel..nlevelmax]: create children for flagged
  ! cells, derefine, compact, rebuild connectivity — all on the GPU — then sync
  ! the mutated mesh back to the host and rebuild the CPU grid_dict so the CPU
  ! consumers (newdt, I/O) stay consistent.  Replaces the CPU r_refine_fine.
  subroutine m_metal_refine(pst, ilevel)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, twotondim
    use hash, only: reset_entire_hash, hash_setp
    type(pst_t), target :: pst
    integer, intent(in) :: ilevel
    integer :: L, o, c, num, nu, ifr, ncreate, nkill
    integer(c_int), pointer :: mf(:)
    integer(kind=8) :: hkey(0:ndim)

    associate(r=>pst%s%r, m=>pst%s%m)
    ! Ensure the GPU mesh + hash are current.  The GPU OWNS the mesh: B.grid is
    ! authoritative and NOT copied from the host — ensure_hash only rebuilds the hash
    ! from B.grid.  Upload the flag map only for the CPU-flag fallback (otherwise
    ! B.flag1 is resident + correct).
    call metal_ensure_hash(pst)
    num = m%ifree - 1
    ! EXPERIMENT (RAMSES_REFINE_HOSTFLAG1=1): force the host-mediated flag1 round-trip
    ! even in the both-GPU path, to test whether the residual fGrG coupling is a
    ! resident-B.flag1 handoff gap (host m%flag1 is the back-copied GPU-flag output).
    if (.not. metal_flag_on .or. refine_hostflag1 .or. refine_hostmed) then
       call mtl_drain()                             ! host writes B.flag1 below
       call c_f_pointer(mtl_ptr_flag1(), mf, [g_ncell*twotondim])
       do o = 1, num
          do c = 1, twotondim
             mf((o-1)*twotondim + c) = m%flag1(c, o)
          end do
       end do
    end if

    ! DIAG (RAMSES_FLAG1_DIAG): per-level flagged-cell count GPU-refine will read
    ! from resident B.flag1 vs the host m%flag1 that CPU-refine reads.  Decides
    ! whether the both-GPU mesh collapse is a flag1 HANDOFF gap (counts differ) or
    ! a refine-LOGIC difference (counts agree, mesh still diverges).
    block
      character(len=8) :: fd
      call get_environment_variable('RAMSES_FLAG1_DIAG', fd)
      if (len_trim(fd) > 0) then
        call mtl_drain()
        block
          integer(c_int), pointer :: mfp(:)
          integer :: LL, oo, cc, dev_cnt, hst_cnt
          call c_f_pointer(mtl_ptr_flag1(), mfp, [g_ncell*twotondim])
          do LL = r%levelmin, r%nlevelmax
            dev_cnt = 0; hst_cnt = 0
            do oo = m%head(LL), m%head(LL)+m%noct(LL)-1
              do cc = 1, twotondim
                if (mfp((oo-1)*twotondim + cc) == 1) dev_cnt = dev_cnt + 1
                if (m%flag1(cc, oo) == 1) hst_cnt = hst_cnt + 1
              end do
            end do
            if (m%noct(LL) > 0) write(0,'(A,I3,A,I2,A,I7,A,I7,A,I7)') &
                 '[FLAG1] refine(ilevel=',ilevel,') L=',LL,' noct=',m%noct(LL), &
                 ' devB.flag1=',dev_cnt,' hostm%flag1=',hst_cnt
          end do
        end block
      end if
    end block

    ! Create / derefine / compact on the GPU.  Returns the # created / killed.
    nu = m%noct_used; ifr = m%ifree
    call system_clock(tk0, tkr)
    call mtl_refine(ilevel, r%levelmin, r%nlevelmax, m%head(r%levelmin), m%noct(r%levelmin), &
         nu, ifr, g_box_min, g_box_max, ncreate, nkill)
    block
      integer(8) :: cc; call system_clock(cc); gt_refine = gt_refine + dble(cc-tk0)/tkr
    end block

    if (ncreate + nkill > 0) then               ! mesh actually changed this call
       m%noct_used = nu; m%ifree = ifr
       do L = r%levelmin, r%nlevelmax
          m%tail(L) = m%head(L) + m%noct(L) - 1
       end do
       ! The GPU now owns B.grid; the host m%grid is left STALE and is synced back
       ! only at I/O (m_metal_grid_to_host).  The grid_dict rebuild needs the host
       ! mesh, so it ALSO syncs first — only done when RAMSES_REFINE_REHASH=1.
       if (refine_rehash .or. refine_hostmed) then
          call mtl_copy_grid_out(c_loc(m%grid(1)), 1, m%noct_used)
          call reset_entire_hash(m%grid_dict, .false.)
          do L = r%levelmin, r%nlevelmax
             do o = m%head(L), m%tail(L)
                hkey(0) = L
                hkey(1:ndim) = m%grid(o)%ckey(1:ndim)
                call hash_setp(m%grid_dict, hkey, o)
             end do
          end do
       end if
       ! Host-mediated both-GPU path: mirror the post-refine resident B.flag1 into
       ! host m%flag1 so the NEXT flag pass's children-flag1 upload reads the current
       ! (reordered) flags -- matching the fGrC path where host is the source of truth.
       if (refine_hostmed) then
          call mtl_drain()
          block
            integer(c_int), pointer :: mf(:)
            integer :: oo, cc
            call c_f_pointer(mtl_ptr_flag1(), mf, [g_ncell*twotondim])
            do oo = 1, m%noct_used
               do cc = 1, twotondim
                  m%flag1(cc, oo) = mf((oo-1)*twotondim + cc)
               end do
            end do
          end block
       end if
       ! Mesh changed -> force the next gravity sync to rebuild hash + nbor from
       ! B.grid (mtl_refine skips the final connectivity rebuild to stay cheap).
       g_synced_ifree = -1
       g_nbor_synced  = -1
       g_mesh_version = g_mesh_version + 1   ! signal the flag that the mesh mutated
    end if
    end associate
  end subroutine m_metal_refine

  ! Sync the GPU-owned mesh (B.grid) back to the host array m%grid + rebuild the
  ! CPU grid_dict.  Called before I/O so the on-disk output reflects the current
  ! mesh (during the run the GPU owns B.grid and m%grid is left stale).  No-op
  ! unless the GPU owns the mesh.
  subroutine m_metal_grid_to_host(pst)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim
    use hash, only: reset_entire_hash, hash_setp
    type(pst_t), target :: pst
    integer :: L, o
    integer(kind=8) :: hkey(0:ndim)
    if (.not. (metal_enabled .and. b_grid_seeded)) return
    associate(r=>pst%s%r, m=>pst%s%m)
    call mtl_drain()                 ! host reads B.grid below
    call mtl_copy_grid_out(c_loc(m%grid(1)), 1, m%noct_used)
    call reset_entire_hash(m%grid_dict, .false.)
    do L = r%levelmin, r%nlevelmax
       do o = m%head(L), m%tail(L)
          hkey(0) = L
          hkey(1:ndim) = m%grid(o)%ckey(1:ndim)
          call hash_setp(m%grid_dict, hkey, o)
       end do
    end do
    end associate
  end subroutine m_metal_grid_to_host

  ! DIAG: copy the resident GPU phi/f -> host m%phi/m%f in the CURRENT GPU slot order
  ! (slot o == m%grid(o) after m_metal_grid_to_host), so a ckey-keyed dump pairs phi
  ! with the right oct.  m_metal_grid_to_host syncs ONLY the grid, leaving m%phi from a
  ! stale layout -> use this right after it for a reliable refined-phi comparison.
  subroutine m_metal_fields_to_host(pst)
    use ramses_commons, only: pst_t
    use amr_parameters, only: ndim, twotondim, dp
    type(pst_t), target :: pst
    integer :: o, c, ddim
    real(c_float), pointer :: m_phi(:), m_f(:), m_nref(:), m_rho(:), m_phiold(:)
    integer(c_int), pointer :: m_flag1(:)
    if (.not. (metal_enabled .and. b_grid_seeded)) return
    associate(m=>pst%s%m)
    call mtl_drain()
    call c_f_pointer(mtl_ptr_phi(),     m_phi,    [g_ncell*twotondim])
    call c_f_pointer(mtl_ptr_phi_old(), m_phiold, [g_ncell*twotondim])
    call c_f_pointer(mtl_ptr_f(),     m_f,     [g_ncell*twotondim*3])
    call c_f_pointer(mtl_ptr_nref(),  m_nref,  [g_ncell*twotondim])
    call c_f_pointer(mtl_ptr_rho(),   m_rho,   [g_ncell*twotondim])
    call c_f_pointer(mtl_ptr_flag1(), m_flag1, [g_ncell*twotondim])
    do o = 1, m%noct_used
       do c = 1, twotondim
          m%phi(c, o)     = real(m_phi((o-1)*twotondim + c), dp)
          m%phi_old(c, o) = real(m_phiold((o-1)*twotondim + c), dp)
          m%rho(c, o)   = real(m_rho((o-1)*twotondim + c), dp)
          m%nref(c, o)  = real(m_nref((o-1)*twotondim + c), dp)
          m%flag1(c, o) = m_flag1((o-1)*twotondim + c)
          do ddim = 1, ndim
             m%f(c, ddim, o) = real(m_f(((o-1)*3 + (ddim-1))*twotondim + c), dp)
          end do
       end do
    end do
    end associate
  end subroutine m_metal_fields_to_host

  ! Print the fine-grained GPU sub-operation profile (cumulative wall seconds).
  subroutine m_metal_prof_report()
    real(8) :: tot
    tot = gt_conn+gt_sort+gt_split+gt_dep+gt_mg+gt_grad+gt_pmar+gt_kick+gt_kmar+gt_refine
    if (tot <= 0.0d0) return
    write(*,'(A)') ' [METAL-PROF] cumulative GPU sub-op wall (s):'
    write(*,'(A,F8.2)') '   rho: connectivity rebuild = ', gt_conn
    write(*,'(A,F8.2)') '   rho: particle sort        = ', gt_sort
    write(*,'(A,F8.2)') '   rho: particle split       = ', gt_split
    write(*,'(A,F8.2)') '   rho: CIC deposit          = ', gt_dep
    block
      real(c_double) :: th, tv
      call mtl_get_mg_times(th, tv)
      write(*,'(A,F8.2,A,F8.2,A,F8.2,A)') '   pois: multigrid total     = ', gt_mg, &
           '  (hierarchy-build=', th, ', V-cycle=', tv, ')'
    end block
    write(*,'(A,F8.2,A)') '   pois:   of which resid-norm READBACK = ', gt_norm, &
         '  (the per-iteration convergence drain)'
    write(*,'(A,F8.2,A)') '   pois:   of which SETUP (build+mask+rhs)= ', gt_setup, &
         '  (prologue; redundantly rebuilt each subcycle solve)'
    block
      integer :: L
      write(*,'(A)') '   pois: multigrid per-level  (level: wall_s  nsolves  s/solve):'
      do L = 0, 30
         if (gt_mg_cnt(L) > 0) write(*,'(A,I3,A,F8.2,I9,A,ES10.2)') &
              '        L', L, ':', gt_mg_lvl(L), gt_mg_cnt(L), '  ', gt_mg_lvl(L)/dble(gt_mg_cnt(L))
      end do
    end block
    write(*,'(A,F8.2)') '   pois: force gradient      = ', gt_grad
    write(*,'(A,F8.2)') '   pois: phi/f marshal->host = ', gt_pmar
    write(*,'(A,F8.2)') '   kick: kernel              = ', gt_kick
    write(*,'(A,F8.2)') '   kick: xp/vp marshal->host = ', gt_kmar
    write(*,'(A,F8.2)') '   refine                    = ', gt_refine
    write(*,'(A,F8.2)') '   --- sum                   = ', tot
  end subroutine m_metal_prof_report

end module metal_gravity_module

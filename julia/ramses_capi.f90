!============================================================================
! ramses_capi.f90 — ISO_C_BINDING C-API that exposes mini-RAMSES routines to
! C / Julia (ccall).  This is the *inverse* of gpu/metal_bridge_iface.f90:
! there we IMPORT C symbols into Fortran; here we EXPORT Fortran routines under
! pinned C names via bind(C, name="ramses_*").  Built into libramses<NDIM>d.dylib
! (see bin/Makefile `libramses` target), loaded from RamsesNG.jl/lib/RamsesLib.
!
! Design (see /Users/tabel/.claude/plans/sharded-bouncing-sifakis.md):
!   * precision contract  — query NPRE (dp bytes) + NDIM, checked on load
!   * pure kernels        — array-in/array-out, no state handle (tightest tests)
!   * state handle + accessors + per-routine wrappers   (added incrementally)
!
! Conventions: scalars passed BY VALUE; arrays as bare pointers (Fortran sees
! them as assumed-size dimension(*)); 1-based Fortran indexing inside.
!============================================================================
module ramses_capi
  use iso_c_binding
  use amr_parameters,  only: ndim, dp, twotondim, threetondim
  use ramses_commons,  only: ramses_t, pst_t, capi_time_cap_active, capi_time_cap_target
  use capi_commons,    only: capi_nml_path, capi_nrestart, capi_setup_only, &
                             capi_last_state, capi_reg, capi_register, CAPI_MAXSTATE, &
                             capi_inject, capi_inject_n, capi_inject_idp, capi_inject_xp, &
                             capi_inject_vp
  implicit none

  ! _CUDA: when .true. (default) every capi getter syncs the whole device-resident
  ! mesh→host before reading (correct but ~0.9s/managed-memory call).  A caller that
  ! reads MANY variables in one host phase (e.g. get_hydro_all for the chem step) can
  ! flip this off, call ramses_sync_to_host ONCE, read all vars, then flip it back —
  ! collapsing N full syncs into 1.  No effect on CPU/Metal builds.
  logical, save :: capi_autosync = .true.

contains

  !--------------------------------------------------------------------------
  ! State lifecycle.  ramses_init drives the REAL launcher (mdl_init) with the
  ! namelist-path / nrestart overrides and capi_setup_only=.true., so it builds
  ! a fully-initialised serial ramses_t (mesh + particles) and returns right
  ! after setup; we register it and hand back an integer handle.
  !--------------------------------------------------------------------------
  function ramses_init(nml_path, nrestart) result(handle) bind(C, name="ramses_init")
    character(kind=c_char), dimension(*), intent(in) :: nml_path
    integer(c_int), value :: nrestart
    integer(c_int) :: handle
    character(len=512) :: fpath
    integer :: i
    type(ramses_t), pointer :: s
    external :: mdl_init
    fpath = ''
    do i = 1, 512
       if (nml_path(i) == c_null_char) exit
       fpath(i:i) = nml_path(i)
    end do
    capi_nml_path   = trim(fpath)
    capi_nrestart   = nrestart
    capi_setup_only = .true.
    nullify(capi_last_state)
    call mdl_init()
    capi_setup_only = .false.
    capi_nml_path   = ''      ! restore the binary default for any later call
    capi_nrestart   = -1
    if (.not. associated(capi_last_state)) then
       handle = 0; return     ! 0 = failure (no state captured)
    end if
    s => capi_last_state
    handle = capi_register(s)
  end function ramses_init

  !--------------------------------------------------------------------------
  ! Like ramses_init, but BEFORE the adaptive refine build the IC particles are
  ! replaced by a caller-supplied deterministic set: n particles with 64-bit ids
  ! `idp` and positions `xp` (column-major n×ndim, box-fraction [0,1)).  Masses
  ! are forced to r%mass_sph inside the hook so nref counts particles (refine at
  ! >8/cell).  RAMSES's own m_init_refine_adaptive then builds the mesh to
  ! nlevelmax from these particles.  The nml supplies levelmin/nlevelmax/units;
  ! set nlevelmax=15 there to allow deep refinement.  Returns a handle (0=fail).
  !--------------------------------------------------------------------------
  function ramses_init_particles(nml_path, n, idp, xp, vp, nrestart) result(handle) &
       bind(C, name="ramses_init_particles")
    character(kind=c_char), dimension(*), intent(in) :: nml_path
    integer(c_int), value :: n, nrestart
    integer(c_int64_t), intent(in) :: idp(*)
    real(c_double),     intent(in) :: xp(*), vp(*)   ! column-major: (d-1)*n + i
    integer(c_int) :: handle
    character(len=512) :: fpath
    integer :: i, d
    type(ramses_t), pointer :: s
    external :: mdl_init
    fpath = ''
    do i = 1, 512
       if (nml_path(i) == c_null_char) exit
       fpath(i:i) = nml_path(i)
    end do
    ! stage the injection arrays for the adaptive_loop hook
    if (allocated(capi_inject_idp)) deallocate(capi_inject_idp)
    if (allocated(capi_inject_xp))  deallocate(capi_inject_xp)
    if (allocated(capi_inject_vp))  deallocate(capi_inject_vp)
    allocate(capi_inject_idp(n), capi_inject_xp(n, ndim), capi_inject_vp(n, ndim))
    do i = 1, n
       capi_inject_idp(i) = idp(i)
       do d = 1, ndim
          capi_inject_xp(i, d) = xp((d-1)*n + i)
          capi_inject_vp(i, d) = vp((d-1)*n + i)
       end do
    end do
    capi_inject_n   = n
    capi_inject     = .true.
    capi_nml_path   = trim(fpath)
    capi_nrestart   = nrestart
    capi_setup_only = .true.
    nullify(capi_last_state)
    call mdl_init()
    capi_inject     = .false.        ! restore defaults for any later call
    capi_setup_only = .false.
    capi_nml_path   = ''
    capi_nrestart   = -1
    if (.not. associated(capi_last_state)) then
       handle = 0; return
    end if
    s => capi_last_state
    handle = capi_register(s)
  end function ramses_init_particles

  ! Release a handle.  (The underlying ramses_t heap is intentionally leaked for
  ! now — full RAMSES teardown is intricate; a test driver inits a handful of
  ! states per process.  TODO: a real r_finalize once needed.)
  subroutine ramses_finalize(handle) bind(C, name="ramses_finalize")
    integer(c_int), value :: handle
    if (handle >= 1 .and. handle <= CAPI_MAXSTATE) nullify(capi_reg(handle)%p)
  end subroutine ramses_finalize

  ! Basic state introspection (also a liveness check after init).
  subroutine ramses_info(handle, levelmin, nlevelmax, npart, nstep_coarse) &
       bind(C, name="ramses_info")
    integer(c_int), value :: handle
    integer(c_int), intent(out) :: levelmin, nlevelmax, npart, nstep_coarse
    type(ramses_t), pointer :: s
    levelmin = -1; nlevelmax = -1; npart = -1; nstep_coarse = -1
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    levelmin     = s%r%levelmin
    nlevelmax    = s%r%nlevelmax
    npart        = s%p%npart
    nstep_coarse = s%g%nstep_coarse
  end subroutine ramses_info

  !--------------------------------------------------------------------------
  ! Particle accessor (idp-keyed): fill caller arrays with idp, xp, vp, levelp.
  ! xp/vp are column-major (n × ndim): component d of particle i at (d-1)*n + i.
  ! For the Metal library the resident GPU particles are synced to host first.
  ! idp is the layout-independent key (reliable CPU↔Metal pairing).
  !--------------------------------------------------------------------------
  subroutine ramses_get_particles(handle, n, idp, xp, vp, levelp) &
       bind(C, name="ramses_get_particles")
    integer(c_int), value :: handle, n
    integer(c_int64_t), intent(out) :: idp(*)
    real(c_double),     intent(out) :: xp(*), vp(*)
    integer(c_int),     intent(out) :: levelp(*)
    type(ramses_t), pointer :: s
    integer :: i, d, np
#ifdef _METAL
    block
      use metal_gravity_module, only: metal_enabled, m_metal_part_to_host
      type(pst_t) :: pst; logical :: ok
      if (metal_enabled) then
         call capi_pst(handle, pst, ok)
         if (ok) call m_metal_part_to_host(pst)
      end if
    end block
#endif
#ifdef _CUDA
    ! CUDA build keeps the mesh+particles device-resident; sync to host so the capi
    ! reads the evolved state (else host arrays are stale → "frozen" particles).
    block
      use gpu_manager, only: r_transfer_grid_host
      type(pst_t) :: pst; logical :: ok
      call capi_pst(handle, pst, ok)
      if (ok .and. capi_autosync) call r_transfer_grid_host(pst)
    end block
#endif
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    np = min(n, s%p%npart)
    do i = 1, np
       idp(i)    = int(s%p%idp(i), c_int64_t)
       levelp(i) = s%p%levelp(i)
       do d = 1, ndim
          xp((d-1)*n + i) = real(s%p%xp(i,d), c_double)
          vp((d-1)*n + i) = real(s%p%vp(i,d), c_double)
       end do
    end do
  end subroutine ramses_get_particles

  ! Add a uniform velocity (CODE units) to ALL particles' vp.  grafic strips the
  ! mean (DC) component of the IC velocity field, so a bulk DM streaming velocity
  ! cannot be imposed through the IC files — this sets it POST-init.  Used to boost
  ! the DM into a different Galilean frame (e.g. the baryon rest frame) for the
  ! streaming-velocity frame-invariance test.
  subroutine ramses_boost_particles(handle, vx, vy, vz) &
       bind(C, name="ramses_boost_particles")
    integer(c_int), value :: handle
    real(c_double), value :: vx, vy, vz
    type(ramses_t), pointer :: s
    integer :: i
    real(kind=8) :: vb(3)
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    vb(1) = vx; vb(2) = vy; vb(3) = vz
    do i = 1, s%p%npart
       s%p%vp(i,1:ndim) = s%p%vp(i,1:ndim) + vb(1:ndim)
    end do
  end subroutine ramses_boost_particles

  ! Set the multigrid/CG convergence tolerance r%epsilon (CPU solve).  Used by
  ! the convergence-tolerance test: tighten eps and watch rel|Δphi| vs Metal.
  subroutine ramses_set_epsilon(handle, eps) bind(C, name="ramses_set_epsilon")
    integer(c_int), value :: handle
    real(c_double), value :: eps
    type(ramses_t), pointer :: s
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    s%r%epsilon = real(eps, dp)
  end subroutine ramses_set_epsilon

  ! Set the per-level time steps g%dtnew(ilevel)/g%dtold(ilevel) directly.  Lets
  ! a Julia driver impose a deterministic subcycle schedule (the kick/drift
  ! kernels read g%dtnew/dtold(ilevel)) for the dynamics-free integrator tests.
  subroutine ramses_set_dt(handle, ilevel, dtnew, dtold) bind(C, name="ramses_set_dt")
    integer(c_int), value :: handle, ilevel
    real(c_double), value :: dtnew, dtold
    type(ramses_t), pointer :: s
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    s%g%dtnew(ilevel) = real(dtnew, dp)
    s%g%dtold(ilevel) = real(dtold, dp)
  end subroutine ramses_set_dt

  subroutine ramses_set_time_cap(handle, active, target_time) bind(C, name="ramses_set_time_cap")
    integer(c_int), value :: handle, active
    real(c_double), value :: target_time
    type(ramses_t), pointer :: s
    capi_time_cap_active = .false.
    capi_time_cap_target = 0.0_dp
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    if (active /= 0) then
       capi_time_cap_active = .true.
       capi_time_cap_target = real(target_time, dp)
    end if
  end subroutine ramses_set_time_cap

  ! Per-particle BINDING level (the level L whose headp(L)..tailp(L) contiguous
  ! range contains the particle — i.e. the subcycle depth kick_drift uses), keyed
  ! by idp so CPU/GPU can be matched despite Hilbert reordering.
  subroutine ramses_get_part_binlevel(handle, n, idp, binlev) &
       bind(C, name="ramses_get_part_binlevel")
    integer(c_int), value :: handle, n
    integer(c_int64_t), intent(out) :: idp(*)
    integer(c_int),     intent(out) :: binlev(*)
    type(ramses_t), pointer :: s
    integer :: L, ip, k
#ifdef _METAL
    block
      use metal_gravity_module, only: metal_enabled, m_metal_part_to_host
      type(pst_t) :: pst; logical :: ok
      if (metal_enabled) then
         call capi_pst(handle, pst, ok)
         if (ok) call m_metal_part_to_host(pst)
      end if
    end block
#endif
    k = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    do L = lbound(s%p%headp,1), ubound(s%p%tailp,1)
       do ip = s%p%headp(L), s%p%tailp(L)
          if (k >= n) return
          k = k + 1
          idp(k)    = int(s%p%idp(ip), c_int64_t)
          binlev(k) = L
       end do
    end do
  end subroutine ramses_get_part_binlevel

  ! Number of particles BOUND to a level (the headp/tailp linked-list range that
  ! kick_drift iterates) — the definitive subcycle-depth diagnostic.
  function ramses_level_npart(handle, ilevel) result(np) bind(C, name="ramses_level_npart")
    integer(c_int), value :: handle, ilevel
    integer(c_int) :: np
    type(ramses_t), pointer :: s
    np = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    if (ilevel < lbound(s%p%headp,1) .or. ilevel > ubound(s%p%tailp,1)) return
    np = s%p%tailp(ilevel) - s%p%headp(ilevel) + 1
  end function ramses_level_npart

  ! Number of octs at a level (so Julia can size get_field buffers).
  function ramses_level_noct(handle, ilevel) result(noct) bind(C, name="ramses_level_noct")
    integer(c_int), value :: handle, ilevel
    integer(c_int) :: noct
    type(ramses_t), pointer :: s
    noct = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    noct = s%m%noct(ilevel)
  end function ramses_level_noct

  !--------------------------------------------------------------------------
  ! Mesh-field accessor (ckey-keyed): for each oct at ilevel emit its ckey
  ! (ndim ints) and the twotondim cell values of field `which`:
  !   0=rho 1=phi 2=phi_old 3=fx 4=fy 5=fz 6=nref.
  ! Returns the number of octs written (≤ nmax).  ckey is the RELIABLE key:
  ! for the Metal library the GPU grid AND fields are synced to host in the same
  ! slot order first (m_metal_grid_to_host + m_metal_fields_to_host), so a
  ! ckey-keyed CPU↔Metal diff is layout-correct (fixes the stale-map artifact).
  !--------------------------------------------------------------------------
  function ramses_get_field(handle, which, ilevel, nmax, ckey, val) result(noct) &
       bind(C, name="ramses_get_field")
    integer(c_int), value :: handle, which, ilevel, nmax
    integer(c_int), intent(out) :: ckey(*)     ! ndim*nmax
    real(c_double), intent(out) :: val(*)       ! twotondim*nmax
    integer(c_int) :: noct
    type(ramses_t), pointer :: s
    integer :: o, c, j, d, hd, no
#ifdef _METAL
    block
      use metal_gravity_module, only: metal_enabled, m_metal_grid_to_host, m_metal_fields_to_host
      type(pst_t) :: pst; logical :: ok
      if (metal_enabled) then
         call capi_pst(handle, pst, ok)
         if (ok) then
            call m_metal_grid_to_host(pst)     ! grid (ckey) in current slot order
            call m_metal_fields_to_host(pst)   ! phi/f in the SAME slot order
         end if
      end if
    end block
#endif
#ifdef _CUDA
    block
      use gpu_manager, only: r_transfer_grid_host
      type(pst_t) :: pst; logical :: ok
      call capi_pst(handle, pst, ok)
      if (ok .and. capi_autosync) call r_transfer_grid_host(pst)   ! grid + phi/f device→host
    end block
#endif
    noct = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    hd = s%m%head(ilevel)
    no = min(s%m%noct(ilevel), nmax)
    do j = 1, no
       o = hd + j - 1
       do d = 1, ndim
          ckey(ndim*(j-1)+d) = s%m%grid(o)%ckey(d)
       end do
       do c = 1, twotondim
          select case (which)
#ifdef GRAV
          case (0); val(twotondim*(j-1)+c) = real(s%m%rho(c,o),     c_double)
          case (1); val(twotondim*(j-1)+c) = real(s%m%phi(c,o),     c_double)
          case (2); val(twotondim*(j-1)+c) = real(s%m%phi_old(c,o), c_double)
          case (3); val(twotondim*(j-1)+c) = real(s%m%f(c,1,o),     c_double)
          case (4); val(twotondim*(j-1)+c) = real(s%m%f(c,2,o),     c_double)
          case (5); val(twotondim*(j-1)+c) = real(s%m%f(c,3,o),     c_double)
          case (6); val(twotondim*(j-1)+c) = real(s%m%nref(c,o),    c_double)
#endif
          case (7); val(twotondim*(j-1)+c) = real(s%m%flag1(c,o),   c_double)
          end select
       end do
    end do
    noct = no
  end function ramses_get_field

  !--------------------------------------------------------------------------
  ! Mesh-field SETTER (inverse of get_field): write field `which` (1=phi 3=fx
  ! 4=fy 5=fz) at ilevel from caller arrays, matched by ckey via the grid hash.
  ! For the Metal library the GPU buffer is written (so the GPU gather/solve sees
  ! it) after syncing the host grid+hash.  Enables the identical-mesh-f gather
  ! diff (copy CPU f → Metal, then gather on both ⇒ pure gather discrepancy) and
  ! the mix-and-match slot swap.  Returns number of octs written.
  !--------------------------------------------------------------------------
  function ramses_set_field(handle, which, ilevel, n, ckey, val) result(nset) &
       bind(C, name="ramses_set_field")
    use hash, only: hash_getp
    integer(c_int), value :: handle, which, ilevel, n
    integer(c_int), intent(in) :: ckey(*)      ! ndim*n
    real(c_double), intent(in) :: val(*)        ! twotondim*n
    integer(c_int) :: nset
    type(ramses_t), pointer :: s
    integer :: j, c, o, d
    integer(8) :: hkey(0:ndim)
    logical :: is_metal
    is_metal = .false.
    nset = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
#ifdef _METAL
    block
      use metal_gravity_module, only: metal_enabled, m_metal_grid_to_host
      type(pst_t) :: pst; logical :: ok
      if (metal_enabled) then
         is_metal = .true.
         call capi_pst(handle, pst, ok)
         if (ok) call m_metal_grid_to_host(pst)   ! grid + grid_dict current
      end if
    end block
#endif
    do j = 1, n
       hkey(0) = ilevel
       do d = 1, ndim
          hkey(d) = ckey(ndim*(j-1)+d)
       end do
       o = hash_getp(s%m%grid_dict, hkey)
       if (o <= 0) cycle
       do c = 1, twotondim
          select case (which)
#ifdef GRAV
          case (1); s%m%phi(c,o)     = real(val(twotondim*(j-1)+c), dp)
          case (2); s%m%phi_old(c,o) = real(val(twotondim*(j-1)+c), dp)
          case (3); s%m%f(c,1,o)     = real(val(twotondim*(j-1)+c), dp)
          case (4); s%m%f(c,2,o)     = real(val(twotondim*(j-1)+c), dp)
          case (5); s%m%f(c,3,o)     = real(val(twotondim*(j-1)+c), dp)
#endif
          ! flag1: lets a driver impose an explicit refinement map and call
          ! ramses_refine_fine directly (geometry-controlled test hierarchies);
          ! note m_flag_fine RESETS flag1, so skip it when driving this way.
          case (7); s%m%flag1(c,o)   = int(val(twotondim*(j-1)+c))
          end select
       end do
       nset = nset + 1
    end do
#ifdef _METAL
    ! Push the just-written host fields into the GPU buffers (slot order matches
    ! the synced grid) so the device gather/solve reads them.
    if (is_metal) then
       block
         use metal_bridge_iface, only: mtl_ptr_phi, mtl_ptr_phi_old, mtl_ptr_f, mtl_drain
         real(c_float), pointer :: gpu_phi(:), gpu_f(:)
         integer :: nn
         call mtl_drain()
         nn = s%m%noct_used
         if (which == 1 .or. which == 2) then
            if (which == 1) then; call c_f_pointer(mtl_ptr_phi(), gpu_phi, [nn*twotondim])
            else;                 call c_f_pointer(mtl_ptr_phi_old(), gpu_phi, [nn*twotondim]); end if
            do j = 1, n
               hkey(0) = ilevel
               do d = 1, ndim; hkey(d) = ckey(ndim*(j-1)+d); end do
               o = hash_getp(s%m%grid_dict, hkey); if (o <= 0) cycle
               do c = 1, twotondim
                  if (which == 1) then; gpu_phi((o-1)*twotondim + c) = real(s%m%phi(c,o),     c_float)
                  else;                 gpu_phi((o-1)*twotondim + c) = real(s%m%phi_old(c,o), c_float); end if
               end do
            end do
         else if (which >= 3 .and. which <= 5) then
            call c_f_pointer(mtl_ptr_f(), gpu_f, [nn*twotondim*3])
            d = which - 2
            do j = 1, n
               hkey(0) = ilevel
               do c = 1, ndim; hkey(c) = ckey(ndim*(j-1)+c); end do
               o = hash_getp(s%m%grid_dict, hkey); if (o <= 0) cycle
               do c = 1, twotondim
                  gpu_f(((o-1)*3 + (d-1))*twotondim + c) = real(s%m%f(c,d,o), c_float)
               end do
            end do
         end if
       end block
    end if
#endif
  end function ramses_set_field

  !--------------------------------------------------------------------------
  ! Build a transient serial pst (nLower=0 ⇒ routines run locally, no MDL
  ! dispatch) bound to a registered state.  ok=.false. for a bad/empty handle.
  !--------------------------------------------------------------------------
  subroutine capi_pst(handle, pst, ok)
    integer, intent(in)        :: handle
    type(pst_t), intent(out)   :: pst        ! intent(out) ⇒ default-initialised
    logical, intent(out)       :: ok
    ok = .false.
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    if (.not. associated(capi_reg(handle)%p)) return
    pst%s => capi_reg(handle)%p
    pst%nLower = 0
    ok = .true.
  end subroutine capi_pst

  !==========================================================================
  ! Per-routine wrappers for the DMO gravity slice.  Each rebuilds a serial pst
  ! and calls the production routine on the live state — so a Julia call is the
  ! real code path.  rho/flag/refine auto-route to the GPU when the Metal
  ! library was initialised with metal_enabled (set in adaptive_loop's setup);
  ! the Poisson solve is explicit: ramses_multigrid / ramses_phi_fine_cg (CPU)
  ! vs ramses_metal_poisson (GPU).
  !==========================================================================
  subroutine ramses_rho_fine(handle, ilevel, rtype) bind(C, name="ramses_rho_fine")
    use rho_fine_module, only: m_rho_fine
    integer(c_int), value :: handle, ilevel, rtype
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_rho_fine(pst, ilevel, rtype)
  end subroutine ramses_rho_fine

  subroutine ramses_save_phi_old(handle, ilevel) bind(C, name="ramses_save_phi_old")
    use interpol_phi_module, only: r_save_phi_old
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_save_phi_old(pst, ilevel, 1)
  end subroutine ramses_save_phi_old

#ifdef GRAV
  subroutine ramses_multigrid(handle, ilevel, icount) bind(C, name="ramses_multigrid")
    use multigrid_fine_commons, only: multigrid
    integer(c_int), value :: handle, ilevel, icount
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call multigrid(pst, ilevel, icount)
  end subroutine ramses_multigrid

  subroutine ramses_phi_fine_cg(handle, ilevel, icount) bind(C, name="ramses_phi_fine_cg")
    use phi_fine_cg_module, only: m_phi_fine_cg
    integer(c_int), value :: handle, ilevel, icount
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_phi_fine_cg(pst, ilevel, icount)
  end subroutine ramses_phi_fine_cg

  subroutine ramses_force_fine(handle, ilevel, icount) bind(C, name="ramses_force_fine")
    use force_fine_module, only: m_force_fine
    integer(c_int), value :: handle, ilevel, icount
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_force_fine(pst, ilevel, icount)
  end subroutine ramses_force_fine
#endif

  ! Level potential energy = sum_{leaf cells, dims} (-dx^ndim/(4pi)/2)*f^2.  CPU:
  ! compute_epot reads m%f (fp64).  Metal: m_metal_epot reads the device B.f.  Both
  ! use the IDENTICAL fact + leaf-cell f^2 sum -> the diff point for "is the GPU
  ! epot reduction faithful given the same force".  Returns the level epot (no
  ! accumulation side effect: g%epot_tot is saved/restored).
#ifdef GRAV
  function ramses_epot(handle, ilevel) result(epot) bind(C, name="ramses_epot")
    use force_fine_module, only: compute_epot
#ifdef _METAL
    use metal_gravity_module, only: metal_enabled, m_metal_epot
#endif
    integer(c_int), value :: handle, ilevel
    real(c_double) :: epot
    type(pst_t) :: pst; logical :: ok
    real(c_double) :: ep            ! compute_epot's epot arg is hardcoded real(8)
    real(dp) :: ep_save             ! keep ep 8-byte so NPRE=4 (dp=real4) still links
    epot = 0.0_c_double
    call capi_pst(handle, pst, ok); if (.not. ok) return
#ifdef _METAL
    if (metal_enabled) then
       ep_save = pst%s%g%epot_tot
       pst%s%g%epot_tot = 0.0_dp
       call m_metal_epot(pst, ilevel)
       epot = pst%s%g%epot_tot
       pst%s%g%epot_tot = ep_save
       return
    end if
#endif
    call compute_epot(pst%s%r, pst%s%g, pst%s%m, ilevel, ep)
    epot = ep
  end function ramses_epot
#endif

  subroutine ramses_kick_drift(handle, ilevel, action) bind(C, name="ramses_kick_drift")
    use move_fine_module, only: m_kick_drift_part
    integer(c_int), value :: handle, ilevel, action
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_kick_drift_part(pst, ilevel, action)   ! 1=kick_only, 2=kick_drift
  end subroutine ramses_kick_drift

  subroutine ramses_flag_fine(handle, ilevel, icount) bind(C, name="ramses_flag_fine")
    use flag_utils, only: m_flag_fine
    integer(c_int), value :: handle, ilevel, icount
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_flag_fine(pst, ilevel, icount)
  end subroutine ramses_flag_fine

  subroutine ramses_refine_fine(handle, ilevel) bind(C, name="ramses_refine_fine")
    use refine_utils, only: m_refine_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_refine_fine(pst, ilevel)
  end subroutine ramses_refine_fine

  !==========================================================================
  ! Per-routine wrappers for the HYDRO slice.  Mirrors the gravity slice above:
  ! each rebuilds a serial pst and calls the production routine on the live state
  ! so a Julia call is the real `r%hydro` code path the time loop runs (see the
  ! `if(r%hydro)` block in amr_step).  The conservative state lives in
  ! m%uold/m%unew (twotondim,nvar,noct); ramses_get_hydro / ramses_set_hydro move
  ! one variable at a time, ckey-keyed like ramses_get_field.  Hydro is CPU-only
  ! today — these are the per-routine handles the Metal hydro port will diff
  ! against (the same harness that cracked the gravity solve).
  !==========================================================================

  ! Number of conservative hydro variables nvar (= 5+nener by default, or NVAR).
  ! No handle needed — it is a compile-time parameter; lets Julia size buffers.
  function ramses_nvar() result(nv) bind(C, name="ramses_nvar")
    use hydro_parameters, only: nvar
    integer(c_int) :: nv
    nv = nvar
  end function ramses_nvar

  ! GETTER for one hydro variable: read m%uold(:,ivar,:) (field=0) or
  ! m%unew(:,ivar,:) (field=1) at ilevel, ivar in 1..nvar.  Layout matches
  ! ramses_get_field: ckey is ndim*noct, val is twotondim*noct.  Returns noct.
  function ramses_get_hydro(handle, field, ivar, ilevel, nmax, ckey, val) result(noct) &
       bind(C, name="ramses_get_hydro")
    use hydro_parameters, only: nvar
    integer(c_int), value :: handle, field, ivar, ilevel, nmax
    integer(c_int), intent(out) :: ckey(*)     ! ndim*nmax
    real(c_double), intent(out) :: val(*)       ! twotondim*nmax
    integer(c_int) :: noct
    type(ramses_t), pointer :: s
    integer :: o, c, j, d, hd, no
#ifdef _CUDA
    block
      use gpu_manager, only: r_transfer_grid_host
      type(pst_t) :: pst; logical :: ok
      call capi_pst(handle, pst, ok)
      if (ok .and. capi_autosync) call r_transfer_grid_host(pst)   ! uold device→host for the capi read
    end block
#endif
    noct = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    if (ivar < 1 .or. ivar > nvar) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    hd = s%m%head(ilevel)
    no = min(s%m%noct(ilevel), nmax)
    do j = 1, no
       o = hd + j - 1
       do d = 1, ndim
          ckey(ndim*(j-1)+d) = s%m%grid(o)%ckey(d)
       end do
       do c = 1, twotondim
          if (field == 0) then
             val(twotondim*(j-1)+c) = real(s%m%uold(c, ivar, o), c_double)
          else
             val(twotondim*(j-1)+c) = real(s%m%unew(c, ivar, o), c_double)
          end if
       end do
    end do
    noct = no
  end function ramses_get_hydro

  ! SETTER (inverse of ramses_get_hydro): write m%uold/m%unew(:,ivar,:) at ilevel
  ! from caller arrays, matched by ckey via the grid hash.  Returns octs written.
  function ramses_set_hydro(handle, field, ivar, ilevel, n, ckey, val) result(nset) &
       bind(C, name="ramses_set_hydro")
    use hash, only: hash_getp
    use hydro_parameters, only: nvar
    integer(c_int), value :: handle, field, ivar, ilevel, n
    integer(c_int), intent(in) :: ckey(*)      ! ndim*n
    real(c_double), intent(in) :: val(*)        ! twotondim*n
    integer(c_int) :: nset
    type(ramses_t), pointer :: s
    integer :: j, c, o, d
    integer(8) :: hkey(0:ndim)
    nset = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    if (ivar < 1 .or. ivar > nvar) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    do j = 1, n
       hkey(0) = ilevel
       do d = 1, ndim
          hkey(d) = ckey(ndim*(j-1)+d)
       end do
       o = hash_getp(s%m%grid_dict, hkey)
       if (o <= 0) cycle
       do c = 1, twotondim
          if (field == 0) then
             s%m%uold(c, ivar, o) = real(val(twotondim*(j-1)+c), dp)
          else
             s%m%unew(c, ivar, o) = real(val(twotondim*(j-1)+c), dp)
          end if
       end do
       nset = nset + 1
    end do
  end function ramses_set_hydro

  ! Upload the host uold to the device after host-side edits (e.g. operator-split
  ! chemistry via set_hydro).  No-op on the CPU/Metal builds (host IS authoritative);
  ! on the _CUDA build the mesh is device-resident, so without this the chem write-back
  ! never reaches the GPU.  Grid/hash and device particles are left untouched.
  subroutine ramses_set_uold_device(handle) bind(C, name="ramses_set_uold_device")
    integer(c_int), value :: handle
#ifdef _CUDA
    block
      use gpu_manager, only: set_uold_device
      type(pst_t) :: pst; logical :: ok
      call capi_pst(handle, pst, ok)
      if (ok) call set_uold_device(pst)
    end block
#endif
  end subroutine ramses_set_uold_device

  ! Toggle the per-getter device→host autosync (see capi_autosync).  flag/=0 → on.
  ! A host phase that reads many variables flips it OFF, calls ramses_sync_to_host
  ! ONCE, reads, then flips it back ON — turning N full managed-memory syncs into 1.
  subroutine ramses_set_autosync(flag) bind(C, name="ramses_set_autosync")
    integer(c_int), value :: flag
    capi_autosync = (flag /= 0)
  end subroutine ramses_set_autosync

  ! Explicitly sync the device-resident mesh→host once (no-op on CPU/Metal).  Used
  ! with autosync OFF to amortize one transfer across many getter calls.
  subroutine ramses_sync_to_host(handle) bind(C, name="ramses_sync_to_host")
    integer(c_int), value :: handle
#ifdef _CUDA
    block
      use gpu_manager, only: r_transfer_grid_host
      type(pst_t) :: pst; logical :: ok
      call capi_pst(handle, pst, ok)
      if (ok) call r_transfer_grid_host(pst)
    end block
#endif
  end subroutine ramses_sync_to_host

  ! ZERO-COPY chem: expose the DEVICE address of the module uold array at the first
  ! oct of `ilevel`, plus the oct count, so Julia (CUDA.jl) can unsafe_wrap it as a
  ! CuArray(twotondim,nvar,noct) and run ChemistryKernels in place — no host round-trip.
  ! Returns 0 on non-_CUDA / failure.  Device uold(c,iv,o) mirrors host indices (whole-
  ! array H→D at set_grid_device), so the level slice starts at o=head(ilevel).
  function ramses_uold_devptr(handle, ilevel, noct_out, head_out) result(addr) &
       bind(C, name="ramses_uold_devptr")
    integer(c_int), value :: handle, ilevel
    integer(c_int), intent(out) :: noct_out, head_out
    integer(c_intptr_t) :: addr
    addr = 0_c_intptr_t; noct_out = 0; head_out = 0
#ifdef _CUDA
    block
      use gpu_runner, only: gpu_uold_devptr
      type(ramses_t), pointer :: s
      integer :: hd
      if (handle < 1 .or. handle > CAPI_MAXSTATE) return
      s => capi_reg(handle)%p
      if (.not. associated(s)) return
      hd = s%m%head(ilevel)
      noct_out = s%m%noct(ilevel)
      head_out = hd
      addr = transfer(gpu_uold_devptr(hd), addr)   ! current buffer base addr (via .cuf helper)
    end block
#endif
  end function ramses_uold_devptr

  ! Hyperbolic solver (unsplit Godunov): the heavy hydro kernel, THE target of
  ! the Metal port.  Reads m%uold, writes fluxes into m%unew (amr_step calls it
  ! with input_size=1 after set_unew copies uold->unew).
  subroutine ramses_godunov_fine(handle, ilevel) bind(C, name="ramses_godunov_fine")
    use godunov_fine_module, only: r_godunov_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_godunov_fine(pst, ilevel, 1)
  end subroutine ramses_godunov_fine

  ! unew <- uold (start of the hydro update).
  subroutine ramses_set_unew(handle, ilevel) bind(C, name="ramses_set_unew")
    use godunov_fine_module, only: r_set_unew
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_set_unew(pst, ilevel, 1)
  end subroutine ramses_set_unew

  ! uold <- unew (commit the hydro update).
  subroutine ramses_set_uold(handle, ilevel) bind(C, name="ramses_set_uold")
    use godunov_fine_module, only: r_set_uold
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_set_uold(pst, ilevel, 1)
  end subroutine ramses_set_uold

  ! Add gravity source terms to unew with the new force (half time step).
  subroutine ramses_gravity_hydro_fine(handle, ilevel) bind(C, name="ramses_gravity_hydro_fine")
    use synchro_hydro_fine_module, only: r_gravity_hydro_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_gravity_hydro_fine(pst, ilevel, 1)
  end subroutine ramses_gravity_hydro_fine

  ! Add the non-gravity hydro source terms to unew.
  subroutine ramses_source_hydro_fine(handle, ilevel) bind(C, name="ramses_source_hydro_fine")
    use source_hydro_fine_module, only: r_source_hydro_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_source_hydro_fine(pst, ilevel, 1)
  end subroutine ramses_source_hydro_fine

  ! Add gravity source terms to uold over time step dteff (the +-0.5*dt
  ! synchro half-steps in amr_step; pass the signed dteff from Julia).
  subroutine ramses_synchro_hydro_fine(handle, ilevel, dteff) bind(C, name="ramses_synchro_hydro_fine")
    use synchro_hydro_fine_module, only: m_synchro_hydro_fine
    integer(c_int), value :: handle, ilevel
    real(c_double), value :: dteff
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_synchro_hydro_fine(pst, ilevel, real(dteff, 8))
  end subroutine ramses_synchro_hydro_fine

  ! Restriction operator: average fine uold up to the coarser level (ilevel<nlevelmax).
  subroutine ramses_upload_fine(handle, ilevel) bind(C, name="ramses_upload_fine")
    use upload_module, only: m_upload_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_upload_fine(pst, ilevel)
  end subroutine ramses_upload_fine

  ! Cooling / heating update (no-op unless cooling/neq_chem/isothermal is on).
  subroutine ramses_cooling_fine(handle, ilevel) bind(C, name="ramses_cooling_fine")
    use cooling_fine_module, only: r_cooling_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_cooling_fine(pst, ilevel, 1)
  end subroutine ramses_cooling_fine

  ! Recompute the per-level time step (Courant + particle dt) into g%dtnew(ilevel);
  ! read it back with ramses_get_dt.
  subroutine ramses_newdt_fine(handle, ilevel) bind(C, name="ramses_newdt_fine")
    use newdt_fine_module, only: m_newdt_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_newdt_fine(pst, ilevel)
  end subroutine ramses_newdt_fine

  ! Run ONE production AMR step at ilevel — the full recursive subcycle the time
  ! loop calls: refine + rho + Poisson + force + kick + drift + newdt +
  ! update_time.  CPU runs the host path; the Metal library routes deposit/solve/
  ! force/refine/kick to the GPU.  Loop this from levelmin for a controlled
  ! multi-step CPU-vs-GPU evolution.  `done` is ignored (caller bounds the loop).
  subroutine ramses_amr_step(handle, ilevel, icount) bind(C, name="ramses_amr_step")
    use amr_step, only: m_amr_step
    integer(c_int), value :: handle, ilevel, icount
    type(pst_t) :: pst; logical :: ok, done
    call capi_pst(handle, pst, ok); if (.not. ok) return
    done = .false.
    call m_amr_step(pst, ilevel, icount, done)
  end subroutine ramses_amr_step

  ! Read per-level time steps + current expansion factor — to record the dt each
  ! path picks during the multi-step run (the newdt CPU-vs-GPU comparison).
  subroutine ramses_get_dt(handle, ilevel, dtnew, dtold, aexp) bind(C, name="ramses_get_dt")
    integer(c_int), value :: handle, ilevel
    real(c_double), intent(out) :: dtnew, dtold, aexp
    type(ramses_t), pointer :: s
    dtnew = 0.0_c_double; dtold = 0.0_c_double; aexp = 0.0_c_double
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    dtnew = real(s%g%dtnew(ilevel), c_double)
    dtold = real(s%g%dtold(ilevel), c_double)
    aexp  = real(s%g%aexp, c_double)
  end subroutine ramses_get_dt

  ! Code-unit conversion factors to cgs (calls amr/units.f90::units for the current
  ! aexp/cosmology) — so an external chemistry/cooling service (Grackle) gets the
  ! exact density/length/time units.  scale_v=scale_l/scale_t; scale_nH, scale_T2
  ! are the number-density and T/mu conversions.
  subroutine ramses_get_units(handle, sd, sl, st, sv, snh, st2) bind(C, name="ramses_get_units")
    integer(c_int), value :: handle
    real(c_double), intent(out) :: sd, sl, st, sv, snh, st2
    type(ramses_t), pointer :: s
    real(kind=8) :: scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2
    external :: units
    sd = 0.0_c_double; sl = 0.0_c_double; st = 0.0_c_double
    sv = 0.0_c_double; snh = 0.0_c_double; st2 = 0.0_c_double
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    call units(s%r, s%g, scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2)
    sd = real(scale_d, c_double); sl = real(scale_l, c_double); st = real(scale_t, c_double)
    sv = real(scale_v, c_double); snh = real(scale_nH, c_double); st2 = real(scale_T2, c_double)
  end subroutine ramses_get_units

  subroutine ramses_get_time(handle, t, texp, aexp, nstep) bind(C, name="ramses_get_time")
    integer(c_int), value :: handle
    real(c_double), intent(out) :: t, texp, aexp
    integer(c_int), intent(out) :: nstep
    type(ramses_t), pointer :: s
    t = 0.0_c_double; texp = 0.0_c_double; aexp = 0.0_c_double; nstep = 0_c_int
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    t     = real(s%g%t, c_double)
    texp  = real(s%g%texp, c_double)
    aexp  = real(s%g%aexp, c_double)
    nstep = int(s%g%nstep, c_int)
  end subroutine ramses_get_time

  ! Code (conformal) time corresponding to a target expansion factor, by interpolating
  ! the Friedman lookup table (aexp_frw, tau_frw) — exactly as init_time.f90 inverts it.
  ! Lets a Julia driver cap the step (via ramses_set_time_cap) to land EXACTLY on an
  ! output scale factor, mirroring how amr_step honours capi_time_cap_target.
  subroutine ramses_t_from_aexp(handle, aexp_t, t_out) bind(C, name="ramses_t_from_aexp")
    integer(c_int), value :: handle
    real(c_double), value :: aexp_t
    real(c_double), intent(out) :: t_out
    type(ramses_t), pointer :: s
    integer :: i, nfrw
    real(dp) :: at
    t_out = 0.0_c_double
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    nfrw = size(s%g%aexp_frw); at = real(aexp_t, dp)
    i = 1
    do while (s%g%aexp_frw(i) > at .and. i < nfrw)
       i = i + 1
    end do
    if (i < 2) i = 2
    t_out = real( s%g%tau_frw(i)*(at - s%g%aexp_frw(i-1))/(s%g%aexp_frw(i)-s%g%aexp_frw(i-1)) + &
                  s%g%tau_frw(i-1)*(at - s%g%aexp_frw(i))/(s%g%aexp_frw(i-1)-s%g%aexp_frw(i)), c_double)
  end subroutine ramses_t_from_aexp

#ifdef _METAL
  ! GPU gravity for one level (deposit's GPU side runs in m_rho_fine; this does
  ! the grouped-multigrid Poisson + 4th-order gradient on the device).
  subroutine ramses_metal_poisson(handle, ilevel, icount) bind(C, name="ramses_metal_poisson")
    use metal_gravity_module, only: m_metal_poisson
    integer(c_int), value :: handle, ilevel, icount
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_metal_poisson(pst, ilevel, icount)
  end subroutine ramses_metal_poisson

  ! GPU gradient-only: apply the 4th-order force stencil to the phi already on the
  ! device (set via ramses_set_field :phi), WITHOUT solving.  Isolates the gradient
  ! operator for a CPU-vs-GPU diff on identical phi.  tfrac=0 (no time-extrap).
  subroutine ramses_metal_gradient(handle, ilevel, tfrac) bind(C, name="ramses_metal_gradient")
    use metal_gravity_module, only: m_metal_gradient_only
    integer(c_int), value :: handle, ilevel
    real(c_double), value :: tfrac
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_metal_gradient_only(pst, ilevel, real(tfrac, dp))
  end subroutine ramses_metal_gradient

  ! GPU hydro Godunov for one level: the Metal port's set_unew -> AMR Godunov ->
  ! grav_hydro -> set_uold sequence (gpu_hydro.cuf), with host<->device uold
  ! transfer.  The direct CPU-vs-Metal diff point for godunov_fine: run
  ! ramses_set_unew + ramses_godunov_fine + ramses_set_uold on one state and this
  ! on another, then compare uold (ramses_get_hydro field=0) by ckey.
  subroutine ramses_metal_godunov_fine(handle, ilevel) bind(C, name="ramses_metal_godunov_fine")
    use metal_gravity_module, only: m_metal_godunov_fine
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_metal_godunov_fine(pst, ilevel, .true.)   ! pure-hydro diff: zero the force
  end subroutine ramses_metal_godunov_fine

  ! GPU coarse-fine reflux isolation: with unew=uold at both levels, run the FINE
  ! level Godunov + reflux finalize, leaving the correction in unew(ilevel-1).
  ! Diff against CPU set_unew(ilevel-1)+set_unew(ilevel)+godunov_fine(ilevel),
  ! reading unew (field=1) at the coarse level ilevel-1.
  subroutine ramses_metal_godunov_reflux(handle, ilevel) bind(C, name="ramses_metal_godunov_reflux")
    use metal_gravity_module, only: m_metal_godunov_reflux
    integer(c_int), value :: handle, ilevel
    type(pst_t) :: pst; logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_metal_godunov_reflux(pst, ilevel)
  end subroutine ramses_metal_godunov_reflux
#endif

  !--------------------------------------------------------------------------
  ! Build the CIC parent-cell index table `ccc` and weight table `bbb` exactly
  ! as poisson/force_fine.f90 does (NDIM=3).  Shared by the interpol_phi kernel
  ! so the boundary CIC is bit-identical to the production path.
  !--------------------------------------------------------------------------
  subroutine build_ccc_bbb(ccc, bbb)
    integer, intent(out) :: ccc(1:8,1:8)
    real(c_double), intent(out) :: bbb(1:8)
    real(dp) :: aa, bb, cc, dd
    aa = 1.0_dp/4.0_dp**ndim
    bb = 3.0_dp*aa; cc = 9.0_dp*aa; dd = 27.0_dp*aa
    bbb(:) = (/aa, bb, bb, cc, bb, cc, cc, dd/)
    ccc(:,1)=(/1 ,2 ,4 ,5 ,10,11,13,14/)
    ccc(:,2)=(/3 ,2 ,6 ,5 ,12,11,15,14/)
    ccc(:,3)=(/7 ,8 ,4 ,5 ,16,17,13,14/)
    ccc(:,4)=(/9 ,8 ,6 ,5 ,18,17,15,14/)
    ccc(:,5)=(/19,20,22,23,10,11,13,14/)
    ccc(:,6)=(/21,20,24,23,12,11,15,14/)
    ccc(:,7)=(/25,26,22,23,16,17,13,14/)
    ccc(:,8)=(/27,26,24,23,18,17,15,14/)
  end subroutine build_ccc_bbb

  !--------------------------------------------------------------------------
  ! PURE KERNEL: coarse-fine boundary interpolation (interpol_phi).
  ! Given the 3^NDIM coarse-parent-cube phi and phi_old (1-based cube index)
  ! and the subcycle time fraction tfrac, return the 2^NDIM boundary-ghost phi
  ! exactly as poisson/interpol_phi.f90 computes it.  By-construction parity:
  ! we build a minimal mesh whose oct k / cell 1 holds cube-cell k, then call
  ! the REAL interpol_phi (igrid_nbor(k)=k, ind_nbor(k)=1).  Array-in/array-out,
  ! no state handle — the tightest possible unit, and the direct CPU-vs-Metal
  ! diff point for the over-energization boundary suspect.
  !--------------------------------------------------------------------------
#ifdef GRAV
  subroutine ramses_interpol_phi_kernel(phi_cube, phiold_cube, tfrac, phi_int) &
       bind(C, name="ramses_interpol_phi_kernel")
    use amr_commons,         only: mesh_t
    use interpol_phi_module, only: interpol_phi
    real(c_double), intent(in)  :: phi_cube(threetondim), phiold_cube(threetondim)
    real(c_double), value       :: tfrac
    real(c_double), intent(out) :: phi_int(twotondim)
    type(mesh_t) :: m
    integer  :: ccc(1:8,1:8)
    real(c_double) :: bbb(1:8)            ! interpol_phi's bbb/tfrac/phi are real(8)
    real(c_double) :: phint(twotondim)
    integer  :: igrid_nbor(threetondim), ind_nbor(threetondim), k
    call build_ccc_bbb(ccc, bbb)
    allocate(m%phi(twotondim, threetondim), m%phi_old(twotondim, threetondim))
    m%phi = 0.0_dp; m%phi_old = 0.0_dp
    do k = 1, threetondim
       m%phi(1, k)     = real(phi_cube(k), dp)
       m%phi_old(1, k) = real(phiold_cube(k), dp)
       igrid_nbor(k)   = k          ! cube cell k lives in oct k ...
       ind_nbor(k)     = 1          ! ... at cell 1
    end do
    call interpol_phi(m, igrid_nbor, ind_nbor, ccc, bbb, real(tfrac, c_double), phint)
    phi_int(1:twotondim) = real(phint(1:twotondim), c_double)
    deallocate(m%phi, m%phi_old)
  end subroutine ramses_interpol_phi_kernel
#endif

  !--------------------------------------------------------------------------
  ! Radiative transfer (RAMSES-RT) slice — ADR-0006 Phase 4 (RamsesNG.jl).
  ! Exposes the M1 photon state (m%rtuold/m%rtunew, nrtvar = nrtgrp*(1+ndim)
  ! per-group density+flux) and the per-level RT step (rt_setup mirrors the
  ! amr_step preamble: rtunew <- rtuold + emissivity; rt_step subcycles the
  ! hyperbolic solve + non-eq photo-chemistry to g%dtnew(ilevel)).  The state
  ! arrays exist only in -DRT builds, so their accessors are guarded; in a
  ! non-RT build they return 0 / no-op (the Julia side gates on ramses_nrtvar).
  ! Ionization fractions ride as hydro passive scalars (uold ivar 5+1..5+NION),
  ! already reachable through ramses_get_hydro.
  !--------------------------------------------------------------------------
  function ramses_nrtvar() result(nv) bind(C, name="ramses_nrtvar")
    use rt_parameters, only: nrtvar
    integer(c_int) :: nv
#ifdef RT
    nv = nrtvar
#else
    nv = 0
#endif
  end function ramses_nrtvar

  ! GETTER for one RT variable: m%rtuold(:,ivar,:) (field=0) / m%rtunew (field=1)
  ! at ilevel, ivar in 1..nrtvar.  Layout matches ramses_get_hydro.
  function ramses_get_rt(handle, field, ivar, ilevel, nmax, ckey, val) result(noct) &
       bind(C, name="ramses_get_rt")
    use rt_parameters, only: nrtvar
    integer(c_int), value :: handle, field, ivar, ilevel, nmax
    integer(c_int), intent(out) :: ckey(*)     ! ndim*nmax
    real(c_double), intent(out) :: val(*)       ! twotondim*nmax
    integer(c_int) :: noct
#ifdef RT
    type(ramses_t), pointer :: s
    integer :: o, c, j, d, hd, no
    noct = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    if (ivar < 1 .or. ivar > nrtvar) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    hd = s%m%head(ilevel)
    no = min(s%m%noct(ilevel), nmax)
    do j = 1, no
       o = hd + j - 1
       do d = 1, ndim
          ckey(ndim*(j-1)+d) = s%m%grid(o)%ckey(d)
       end do
       do c = 1, twotondim
          if (field == 0) then
             val(twotondim*(j-1)+c) = real(s%m%rtuold(c, ivar, o), c_double)
          else
             val(twotondim*(j-1)+c) = real(s%m%rtunew(c, ivar, o), c_double)
          end if
       end do
    end do
    noct = no
#else
    noct = 0
#endif
  end function ramses_get_rt

  ! SETTER (inverse of ramses_get_rt): write m%rtuold/m%rtunew(:,ivar,:) at
  ! ilevel from caller arrays, matched by ckey via the grid hash.
  function ramses_set_rt(handle, field, ivar, ilevel, n, ckey, val) result(nset) &
       bind(C, name="ramses_set_rt")
    use hash, only: hash_getp
    use rt_parameters, only: nrtvar
    integer(c_int), value :: handle, field, ivar, ilevel, n
    integer(c_int), intent(in) :: ckey(*)      ! ndim*n
    real(c_double), intent(in) :: val(*)        ! twotondim*n
    integer(c_int) :: nset
#ifdef RT
    type(ramses_t), pointer :: s
    integer :: j, c, o, d
    integer(8) :: hkey(0:ndim)
    nset = 0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    if (ivar < 1 .or. ivar > nrtvar) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    do j = 1, n
       hkey(0) = ilevel
       do d = 1, ndim
          hkey(d) = ckey(ndim*(j-1)+d)
       end do
       o = hash_getp(s%m%grid_dict, hkey)
       if (o <= 0) cycle
       do c = 1, twotondim
          if (field == 0) then
             s%m%rtuold(c, ivar, o) = real(val(twotondim*(j-1)+c), dp)
          else
             s%m%rtunew(c, ivar, o) = real(val(twotondim*(j-1)+c), dp)
          end if
       end do
       nset = nset + 1
    end do
#else
    nset = 0
#endif
  end function ramses_set_rt

  ! The amr_step RT preamble for one level: rtunew <- rtuold, then the stellar/
  ! source emissivity (amr_step.f90 "Set rtunew equal to rtuold" block).
  subroutine ramses_rt_setup(handle, ilevel) bind(C, name="ramses_rt_setup")
    use rt_godunov_fine_module, only: r_set_rtunew, r_set_emissivity
    integer(c_int), value :: handle, ilevel
#ifdef RT
    type(pst_t) :: pst
    logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call r_set_rtunew(pst, ilevel, 1)
    call r_set_emissivity(pst, ilevel, 1)
#endif
  end subroutine ramses_rt_setup

  ! Per-coarse-step RT/chemistry constant updates (update_time.f90:94): the
  ! reduced speed of light, the cross-section×c tables (signc — WITHOUT this
  ! the photo-chemistry sees zero cross sections), Compton/UV constants.  Call
  ! once after init and once per outer step when driving rt_step directly.
  subroutine ramses_rt_neq_updates(handle, nstep) bind(C, name="ramses_rt_neq_updates")
    use update_rt_c_module, only: r_rt_neq_updates
    integer(c_int), value :: handle, nstep
#ifdef RT
    type(pst_t) :: pst
    logical :: ok
    integer :: ns
    call capi_pst(handle, pst, ok); if (.not. ok) return
    ns = nstep
    call r_rt_neq_updates(pst, ns, 1)
#endif
  end subroutine ramses_rt_neq_updates

  ! One RT step at ilevel: subcycles the M1 hyperbolic solve (+ photo-chemistry
  ! when neq_chem) to the current g%dtnew(ilevel) — set it via ramses_set_dt.
  subroutine ramses_rt_step(handle, ilevel) bind(C, name="ramses_rt_step")
    use rt_step_module, only: m_rt_step
    integer(c_int), value :: handle, ilevel
#ifdef RT
    type(pst_t) :: pst
    logical :: ok
    call capi_pst(handle, pst, ok); if (.not. ok) return
    call m_rt_step(pst, ilevel)
#endif
  end subroutine ramses_rt_step

  !--------------------------------------------------------------------------
  ! Precision / shape contract.  The Julia bindings call this on load and abort
  ! on mismatch — exactly the silent-bug class (fp32 vs fp64, NDIM) this
  ! framework exists to surface.  npre_bytes = sizeof(real(dp)); 8=fp64, 4=fp32.
  !--------------------------------------------------------------------------
  subroutine ramses_precision_bytes(npre_bytes, ndim_out, twotondim_out) &
       bind(C, name="ramses_precision_bytes")
    integer(c_int), intent(out) :: npre_bytes, ndim_out, twotondim_out
    real(dp) :: x
    npre_bytes    = storage_size(x) / 8
    ndim_out      = ndim
    twotondim_out = twotondim
  end subroutine ramses_precision_bytes

  ! Cumulative wallclock (GPU-synced) of the 'hydro - godunov' timer slot, for
  ! apples-to-apples solver benchmarking vs upstream's identical m_timer slot.
  function ramses_get_timer_godunov() result(secs) bind(C, name="ramses_get_timer_godunov")
    use timer_module, only: time, labels, ntimer
    real(c_double) :: secs
    integer :: i
    secs = 0.0_c_double
    do i = 1, ntimer
       if (trim(labels(i)) == 'hydro - godunov') secs = real(time(i), c_double)
    end do
  end function ramses_get_timer_godunov

end module ramses_capi

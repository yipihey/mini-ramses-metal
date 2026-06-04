!============================================================================
! capi_commons.f90 — shared state for the Julia/C-callable API (ramses_capi.f90).
!
! The library reuses the REAL launcher (mdl_init -> worker_init -> master ->
! adaptive_loop) to build a fully-initialised serial `ramses_t`, but stops right
! after setup instead of running the time loop.  Three hooks make that possible
! without disturbing the normal binary (all default to the no-op binary path):
!   * capi_nml_path / capi_nrestart  — m_read_params uses these instead of argv
!   * capi_setup_only                — adaptive_loop returns after setup
!   * capi_last_state                — adaptive_loop stashes the live state here
! A small integer-handle registry then hands opaque state handles to Julia
! (avoids c_loc on the non-interoperable ramses_t).
!============================================================================
module capi_commons
  use ramses_commons, only: ramses_t
  implicit none

  ! --- launcher overrides (set by ramses_init before calling mdl_init) ---
  character(len=512), save :: capi_nml_path = ''   ! namelist path; '' => use argv
  integer,            save :: capi_nrestart = -1   ! restart number; <0 => use argv
  logical,            save :: capi_setup_only = .false.   ! stop after setup
  type(ramses_t), pointer, save :: capi_last_state => null()  ! set by adaptive_loop

  ! --- deterministic particle injection (set by ramses_init_particles) ---
  ! When capi_inject=.true., adaptive_loop overwrites the IC particles with these
  ! (all bound to levelmin) right after m_input_part and BEFORE the adaptive
  ! refine build, so RAMSES's own refine machinery builds the mesh from them.
  ! Masses are set to r%mass_sph inside the hook (so nref counts particles).
  logical,                       save :: capi_inject   = .false.
  integer,                       save :: capi_inject_n = 0
  integer(kind=8), allocatable,  save :: capi_inject_idp(:)
  real(kind=8),    allocatable,  save :: capi_inject_xp(:,:)   ! (n, ndim)
  real(kind=8),    allocatable,  save :: capi_inject_vp(:,:)   ! (n, ndim) initial velocities

  ! --- integer-handle registry of live states ---
  integer, parameter :: CAPI_MAXSTATE = 8
  type capi_slot
     type(ramses_t), pointer :: p => null()
  end type capi_slot
  type(capi_slot), save :: capi_reg(CAPI_MAXSTATE)

contains

  ! Register a state, return its 1-based handle (0 on overflow).
  integer function capi_register(s) result(h)
    type(ramses_t), pointer, intent(in) :: s
    integer :: i
    h = 0
    do i = 1, CAPI_MAXSTATE
       if (.not. associated(capi_reg(i)%p)) then
          capi_reg(i)%p => s
          h = i
          return
       end if
    end do
  end function capi_register

end module capi_commons

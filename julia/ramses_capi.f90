! Minimal C-API for upstream CT-MHD benchmarking: init (setup-only), step, time.
module ramses_capi
  use iso_c_binding
  use amr_parameters, only: ndim, dp, twotondim
  use ramses_commons, only: ramses_t, pst_t
  use capi_commons,   only: capi_nml_path, capi_nrestart, capi_setup_only, &
                            capi_last_state, capi_reg, CAPI_MAXSTATE, capi_register
  implicit none
contains
  subroutine capi_pst(handle, pst, ok)
    integer, intent(in) :: handle
    type(pst_t), intent(out) :: pst
    logical, intent(out) :: ok
    ok = .false.
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    if (.not. associated(capi_reg(handle)%p)) return
    pst%s => capi_reg(handle)%p
    pst%nLower = 0
    ok = .true.
  end subroutine capi_pst

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
    capi_nml_path   = ''
    capi_nrestart   = -1
    if (.not. associated(capi_last_state)) then
       handle = 0; return
    end if
    s => capi_last_state
    handle = capi_register(s)
  end function ramses_init

  subroutine ramses_amr_step(handle, ilevel, icount) bind(C, name="ramses_amr_step")
    use amr_step, only: m_amr_step
    integer(c_int), value :: handle, ilevel, icount
    type(pst_t) :: pst; logical :: ok, done
    call capi_pst(handle, pst, ok); if (.not. ok) return
    done = .false.
    call m_amr_step(pst, ilevel, icount, done)
  end subroutine ramses_amr_step

  subroutine ramses_get_time(handle, t, texp, aexp, nstep) bind(C, name="ramses_get_time")
    integer(c_int), value :: handle
    real(c_double), intent(out) :: t, texp, aexp
    integer(c_int), intent(out) :: nstep
    type(ramses_t), pointer :: s
    t=0; texp=0; aexp=0; nstep=0
    if (handle < 1 .or. handle > CAPI_MAXSTATE) return
    s => capi_reg(handle)%p
    if (.not. associated(s)) return
    t     = real(s%g%t, c_double)
    aexp  = real(s%g%aexp, c_double)
    nstep = int(s%g%nstep_coarse, c_int)
  end subroutine ramses_get_time

  ! Cumulative wallclock (GPU-synced) of the 'hydro - godunov' timer slot.
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

! Minimal C-API shared state for benchmarking upstream CT-MHD via libramses.
! (subset of the fork's capi_commons: nml override + setup_only + handle registry)
module capi_commons
  use ramses_commons, only: ramses_t
  implicit none
  character(len=512), save :: capi_nml_path = ''   ! namelist path; '' => use argv
  integer,            save :: capi_nrestart = -1
  logical,            save :: capi_setup_only = .false.  ! adaptive_loop returns after setup
  type(ramses_t), pointer, save :: capi_last_state => null()
  integer, parameter :: CAPI_MAXSTATE = 8
  type capi_slot
     type(ramses_t), pointer :: p => null()
  end type capi_slot
  type(capi_slot), save :: capi_reg(CAPI_MAXSTATE)
contains
  integer function capi_register(s) result(h)
    type(ramses_t), pointer, intent(in) :: s
    integer :: i
    h = 0
    do i = 1, CAPI_MAXSTATE
       if (.not. associated(capi_reg(i)%p)) then
          capi_reg(i)%p => s; h = i; return
       end if
    end do
  end function capi_register
end module capi_commons

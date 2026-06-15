module turb_update_module
  use amr_commons, only: run_t, global_t
  use turb_commons, only: turb_t, turb_next_field
contains
  !################################################################
  !################################################################
  !################################################################
  !################################################################
  recursive subroutine r_update_turb(pst)
    use mdl_module
    use ramses_commons, only: pst_t
    use mdl_parameters
#if defined(_CUDA) && defined(TURB)
    use gpu_manager, only: gpu_update_turb
#endif
    implicit none
    type(pst_t)::pst

    integer::rID

    if(pst%nLower>0)then
       rID = mdl_send_request(pst%s%mdl,MDL_UPDATE_TURB,pst%iUpper+1)
       call r_update_turb(pst%pLower)
       call mdl_get_reply(pst%s%mdl,rID,0)
    else
#if defined(_CUDA) && defined(TURB)
       call gpu_update_turb(pst)
#else
       call turb_check_time(pst%s%r, pst%s%g, pst%s%turb)
#endif
    endif

  end subroutine r_update_turb
  !################################################################
  !################################################################
  !################################################################
  !################################################################
  subroutine turb_check_time(run, global, turb)
    implicit none
    type(run_t)    :: run
    type(global_t) :: global
    type(turb_t)   :: turb

    real(kind=8)   :: turb_last_tfrac        ! Time fraction since last
    real(kind=8)   :: turb_next_tfrac        ! Time fraction until next

    ! evolving forced turbulence
    do
       if (global%t >= turb%turb_next_time) then
          call turb_next_field(run, turb)
       else
          exit
       end if
    end do

    turb_last_tfrac = real((global%t - turb%turb_last_time) / turb%turb_dt, kind=8)
    turb_next_tfrac = 1.0 - turb_last_tfrac

    turb%afield_now = turb_last_tfrac*turb%afield_last + turb_next_tfrac*turb%afield_next

  end subroutine turb_check_time
  !################################################################
  !################################################################
  !################################################################
  !################################################################
end module turb_update_module

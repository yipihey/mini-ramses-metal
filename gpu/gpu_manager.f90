module gpu_manager
  use cudafor
  use nvtx
  use gpu_utils
  use gpu_runner
contains
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_set_grid_device(pst)
  use mdl_module
  use ramses_commons, only: pst_t
  use refine_device, only: insert_hash_kernel, update_nbor_array
  use mdl_parameters
  implicit none
  type(pst_t)::pst

  integer::rID
  integer::head_idx, num_octs, num_subgrids, num_threads, num_blocks, ind
  
  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SET_GRID_DEVICE,pst%iUpper+1)
     call r_set_grid_device(pst%pLower)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else

     ! Copy grid from host to device
     call nvtxStartRange("Copy entire mesh from host to device", color=5)!red
     grid = pst%s%m%grid
     ! flag1 is only allocated/used for adaptive mesh refinement
     if(pst%s%r%nlevelmax > pst%s%r%levelmin) flag1 = pst%s%m%flag1
#ifdef HYDRO
     uold = pst%s%m%uold
#ifdef MHD
     bold = pst%s%m%bold
#endif
#endif
     call GPU_Error_Check(__FILE__, __LINE__)
     call nvtxEndRange()

     ! Particle H→D (allocated in init_part).
     if (pst%s%r%part .and. allocated(pst%s%p%xp) .and. allocated(xp)) then
        call nvtxStartRange("Copy particles from host to device", color=5)!red
        xp     = pst%s%p%xp
        vp     = pst%s%p%vp
        mp     = pst%s%p%mp
        levelp = pst%s%p%levelp
        sortp  = pst%s%p%sortp
        if (allocated(idp)) idp = pst%s%p%idp
        call GPU_Error_Check(__FILE__, __LINE__)
        call nvtxEndRange()
     endif

     ! Star particle H→D
     if (pst%s%r%star .and. allocated(pst%s%star%xp) .and. allocated(star_xp)) then
        call nvtxStartRange("Copy stars from host to device", color=5)!red
        star_xp     = pst%s%star%xp
        star_vp     = pst%s%star%vp
        star_mp     = pst%s%star%mp
        star_tp     = pst%s%star%tp
        star_zp     = pst%s%star%zp
        star_levelp = pst%s%star%levelp
        star_sortp  = pst%s%star%sortp
        if (allocated(star_idp)) star_idp = pst%s%star%idp
        call GPU_Error_Check(__FILE__, __LINE__)
        call nvtxEndRange()
     endif

     ! Insert entire grid in the device hash table
     head_idx = 1
     num_octs = pst%s%m%ifree - 1
     num_threads = 128
     num_blocks = (num_octs + num_threads - 1) / num_threads

     call nvtxStartRange("Insert grid in hash table", color=5)!red
     call insert_hash_kernel<<<num_blocks, num_threads>>>(grid, hash_key, hash_val, pst%s%m%hash_size, &
          & ckey_max, key_off, head_idx, num_octs)
     call GPU_Error_Check(__FILE__, __LINE__)
     call nvtxEndRange()

     ! Compute nbor grids array for coarse level only
     head_idx = 1
     num_subgrids = pst%s%m%noct(pst%s%r%levelmin) / nsubgridtondim
     num_threads = 128
     num_blocks = (num_subgrids + num_threads - 1) / num_threads
     call nvtxStartRange("Compute nbor array", color=5)!red
     do ind = 1, subgridsize
        call update_nbor_array<<<num_blocks, num_threads>>>(nbor, grid, hash_key, hash_val, pst%s%m%hash_size, &
             & ckey_max, key_off, box_ckey_min, box_ckey_max, periodic, head_idx, num_subgrids, ind)
     end do
     call GPU_Error_Check(__FILE__, __LINE__)
     call nvtxEndRange()

     pst%s%m%data_on_device=.true.

  endif

end subroutine r_set_grid_device
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_transfer_grid_host(pst)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_TRANSFER_GRID_HOST,pst%iUpper+1)
     call r_transfer_grid_host(pst%pLower)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else

     ! Copy grid from device to host
     call nvtxStartRange("Copy entire mesh from device to host", color=5)!red
     pst%s%m%grid = grid
     ! flag1 is only allocated/used for adaptive mesh refinement
     if(pst%s%r%nlevelmax > pst%s%r%levelmin) pst%s%m%flag1 = flag1
#ifdef HYDRO
     pst%s%m%uold = uold
#ifdef MHD
     pst%s%m%bold = bold
#endif
#endif
#ifdef GRAV
     pst%s%m%f = f
     pst%s%m%phi = phi
#endif
     call GPU_Error_Check(__FILE__, __LINE__)
     call nvtxEndRange()

#ifdef _CUDA
     call gpu_to_host_part(pst)
#endif

  endif

end subroutine r_transfer_grid_host
!###########################################################
!###########################################################
!###########################################################
!###########################################################
#ifdef _CUDA
!> D→H particle copy at host I/O boundaries.
subroutine gpu_to_host_part(pst)
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t)::pst

  if (.not. allocated(xp)) return

  call nvtxStartRange("Copy particles from device to host", color=5)!red
  pst%s%p%xp     = xp
  pst%s%p%vp     = vp
  pst%s%p%mp     = mp
  pst%s%p%levelp = levelp
  pst%s%p%sortp  = sortp
  if (allocated(idp)) pst%s%p%idp = idp
  call GPU_Error_Check(__FILE__, __LINE__)
  call nvtxEndRange()

  if (.not. allocated(star_xp)) return

  call nvtxStartRange("Copy stars from device to host", color=5)!red
  pst%s%star%xp     = star_xp
  pst%s%star%vp     = star_vp
  pst%s%star%mp     = star_mp
  pst%s%star%tp     = star_tp
  pst%s%star%zp     = star_zp
  pst%s%star%levelp = star_levelp
  pst%s%star%sortp  = star_sortp
  if (allocated(star_idp)) pst%s%star%idp = star_idp
  call GPU_Error_Check(__FILE__, __LINE__)
  call nvtxEndRange()

end subroutine gpu_to_host_part
#endif
!###########################################################
!###########################################################
!###########################################################
#if defined(_CUDA) && defined(TURB)
!> One-time seed of both device turbulence fields from the host (after init_turb).
subroutine gpu_turb_init_fields(pst)
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t)::pst

  if (.not. allocated(afield_next_d)) return

  call nvtxStartRange("Copy initial turb fields host to device", color=5)!red
  afield_last_d = pst%s%turb%afield_last
  afield_next_d = pst%s%turb%afield_next
  call GPU_Error_Check(__FILE__, __LINE__)
  call nvtxEndRange()

end subroutine gpu_turb_init_fields
!###########################################################
!###########################################################
!###########################################################
!> Device-side companion to turb_next_field: rotate next->last on the device (no H2D
!> traffic) then upload only the freshly generated host afield_next.
subroutine gpu_turb_next_field(pst)
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t)::pst

  if (.not. allocated(afield_next_d)) return

  call nvtxStartRange("Rotate + upload turb afield_next", color=5)!red
  afield_last_d = afield_next_d                  ! device-to-device rotate
  afield_next_d = pst%s%turb%afield_next         ! H2D: only the new 'next'
  call GPU_Error_Check(__FILE__, __LINE__)
  call nvtxEndRange()

end subroutine gpu_turb_next_field
!###########################################################
!###########################################################
!###########################################################
!> GPU turbulence update (replaces host turb_check_time on the leaf):
!> generate new field(s) on the host as time advances, mirror to the device, and
!> time-interpolate afield_now on the device (Kernel A). The host afield_now is also
!> updated so host-side diagnostics (e.g. current_turb_rms) stay correct.
subroutine gpu_update_turb(pst)
  use ramses_commons, only: pst_t
  use turb_commons, only: turb_next_field
  implicit none
  type(pst_t)::pst
  real(kind=8) :: last_tfrac, next_tfrac

  ! Advance the host field(s) and mirror each new one to the device.
  do
     if (pst%s%g%t >= pst%s%turb%turb_next_time) then
        call turb_next_field(pst%s%r, pst%s%turb)
        call gpu_turb_next_field(pst)
     else
        exit
     end if
  end do

  ! Device time-interpolation (Kernel A).
  call gpu_turb_interp(pst%s)

  ! Keep the host afield_now in sync for host-side diagnostics.
  last_tfrac = (pst%s%g%t - pst%s%turb%turb_last_time)/pst%s%turb%turb_dt
  next_tfrac = 1d0 - last_tfrac
  pst%s%turb%afield_now = last_tfrac*pst%s%turb%afield_last + next_tfrac*pst%s%turb%afield_next

end subroutine gpu_update_turb
#endif
!###########################################################
!###########################################################
!###########################################################
!###########################################################
end module gpu_manager

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
     ! Face-centred B field H→D, parallel with uold (mirrors the host #ifdef MHD).
     bold = pst%s%m%bold
#endif
#endif
     call GPU_Error_Check(__FILE__, __LINE__)
     call nvtxEndRange()

#ifdef _CUDA
     ! Particle H→D (allocated in init_part).
     if (pst%s%r%part .and. allocated(pst%s%p%xp) .and. allocated(xp)) then
        call nvtxStartRange("Copy particles from host to device", color=5)!red
        xp     = pst%s%p%xp
        vp     = pst%s%p%vp
        mp     = pst%s%p%mp
        levelp = pst%s%p%levelp
        sortp  = pst%s%p%sortp
        if (allocated(idp)) idp = pst%s%p%idp
!!$        if (allocated(pst%s%p%jp)     .and. allocated(jp))     jp     = pst%s%p%jp
!!$        if (allocated(pst%s%p%zp)     .and. allocated(zp))     zp     = pst%s%p%zp
!!$        if (allocated(pst%s%p%tp)     .and. allocated(tp))     tp     = pst%s%p%tp
!!$        if (allocated(pst%s%p%tm)     .and. allocated(tm))     tm     = pst%s%p%tm
!!$        if (allocated(pst%s%p%size)   .and. allocated(size_p)) size_p = pst%s%p%size
!!$        if (allocated(pst%s%p%charge) .and. allocated(charge)) charge = pst%s%p%charge
!!$        if (allocated(pst%s%p%idm)    .and. allocated(idm))    idm    = pst%s%p%idm
!!$        if (allocated(pst%s%p%idt)    .and. allocated(idt))    idt    = pst%s%p%idt
        call GPU_Error_Check(__FILE__, __LINE__)
        call nvtxEndRange()
     endif
#endif

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
     ! Face-centred B field D→H, parallel with uold (mirrors the host #ifdef MHD).
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
!!$  if (allocated(jp)     .and. allocated(pst%s%p%jp))     pst%s%p%jp     = jp
!!$  if (allocated(zp)     .and. allocated(pst%s%p%zp))     pst%s%p%zp     = zp
!!$  if (allocated(tp)     .and. allocated(pst%s%p%tp))     pst%s%p%tp     = tp
!!$  if (allocated(tm)     .and. allocated(pst%s%p%tm))     pst%s%p%tm     = tm
!!$  if (allocated(size_p) .and. allocated(pst%s%p%size))   pst%s%p%size   = size_p
!!$  if (allocated(charge) .and. allocated(pst%s%p%charge)) pst%s%p%charge = charge
!!$  if (allocated(idm)    .and. allocated(pst%s%p%idm))    pst%s%p%idm    = idm
!!$  if (allocated(idt)    .and. allocated(pst%s%p%idt))    pst%s%p%idt    = idt
  call GPU_Error_Check(__FILE__, __LINE__)
  call nvtxEndRange()

end subroutine gpu_to_host_part
#endif
!###########################################################
!###########################################################
!###########################################################
!###########################################################
end module gpu_manager

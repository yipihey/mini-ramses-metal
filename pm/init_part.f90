module init_part_module

contains
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
recursive subroutine r_init_part(pst)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  !--------------------------------------------------------------------
  ! This routine is the recursive slave procedure to allocate
  ! particle-based arrays.
  !--------------------------------------------------------------------
  integer::rID
  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_INIT_PART,pst%iUpper+1)
     call r_init_part(pst%pLower)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     if(pst%s%r%part)then
        call init_part(pst%s%r,pst%s%g,pst%s%m,pst%s%p)
     endif
     if(pst%s%r%star)then
        call init_star(pst%s%r,pst%s%g,pst%s%star)
     end if
     if(pst%s%r%sink)then
        call init_sink(pst%s%r,pst%s%g,pst%s%sink)
     end if
     if(pst%s%r%tree)then
        call init_tree(pst%s%r,pst%s%g,pst%s%tree)
     end if
     if(pst%s%r%trac)then
        call init_trac(pst%s%r,pst%s%g,pst%s%trac)
     end if
     if(pst%s%r%dust)then
        call init_dust(pst%s%r,pst%s%g,pst%s%dust)
     end if
  endif

end subroutine r_init_part
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine init_part(r,g,m,p)
  use amr_parameters, only: ndim, twotondim, threetondim
  use amr_commons, only: run_t,global_t,mesh_t
  use pm_parameters, only: PART_TYPE
  use pm_commons, only: part_t
#ifdef _CUDA
  use gpu_runner
  use part_device, only: ensure_scan_capacity_part
  use cudafor
#endif
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  type(part_t)::p
#ifdef _CUDA
  integer::scan_size
#endif
  !---------------------------------
  ! Allocate PART particle variables
  !---------------------------------
  p%type=PART_TYPE
  allocate(p%xp    (r%npartmax,ndim))
  allocate(p%vp    (r%npartmax,ndim))
  allocate(p%mp    (r%npartmax))
  allocate(p%levelp(r%npartmax))
  allocate(p%idp   (r%npartmax))
  p%nvaralloc=2*ndim+3
#ifdef OUTPUT_PARTICLE_POTENTIAL
  allocate(p%phip  (r%npartmax))
  p%nvaralloc=p%nvaralloc+1
#endif
  ! ALlocate workspace variables
  allocate(p%sortp (r%npartmax))
  allocate(p%workp (r%npartmax))
  ! Allocate head and tail of particle levels
  if(ANY(.not.r%periodic(1:ndim)))then
     allocate(p%headp(r%levelmin-1:r%nlevelmax))
     allocate(p%tailp(r%levelmin-1:r%nlevelmax))
  else
     allocate(p%headp(r%levelmin:r%nlevelmax))
     allocate(p%tailp(r%levelmin:r%nlevelmax))
  endif
  ! No particle just yet
  p%headp=1
  p%tailp=0

  ! Device mirrors/scratch; H→D in r_set_grid_device.
#ifdef _CUDA
  allocate(xp(1:r%npartmax, 1:ndim))
  allocate(vp(1:r%npartmax, 1:ndim))
  allocate(mp(1:r%npartmax))
  allocate(levelp(1:r%npartmax))
  allocate(sortp(1:r%npartmax))
  if (r%nlevelmax > r%levelmin) allocate(idp(1:r%npartmax))
#ifdef OUTPUT_PARTICLE_POTENTIAL
  allocate(phip(1:r%npartmax))
#endif
  ! gpu_cic_part source map.
  allocate(xp_swap(1:r%npartmax))
  allocate(isp_swap(1:r%npartmax))
  allocate(idp_swap(1:r%npartmax))
! CUB workspace
#if defined(CUB_SORT_PART)
  allocate(hkeyp(1:r%npartmax))
#endif
  ! Prefix sum arrays
  scan_size = max(r%npartmax, m%ngridmax + m%ncachemax)
  call ensure_scan_capacity_part(scan_size, r%part_dep_algo)
#endif
end subroutine init_part
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine init_star(r,g,p)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t,global_t
  use pm_parameters, only: STAR_TYPE
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  !-----------------------------------
  ! Allocate star particle variables
  !------------------------------------
  p%type=STAR_TYPE
  allocate(p%xp    (r%nstarmax,ndim))
  allocate(p%vp    (r%nstarmax,ndim))
  allocate(p%mp    (r%nstarmax))
  allocate(p%zp    (r%nstarmax))
  allocate(p%tp    (r%nstarmax))
  allocate(p%levelp(r%nstarmax))
  allocate(p%idp   (r%nstarmax))
  p%nvaralloc=2*ndim+5
#ifdef OUTPUT_PARTICLE_POTENTIAL
  allocate(p%phip  (r%nstarmax))
  p%nvaralloc=p%nvaralloc+1
#endif
  ! Allocate workspace variables
  allocate(p%sortp (r%nstarmax))
  allocate(p%workp (r%nstarmax))
  ! Allocate head and tail of particle levels
  if(ANY(.not.r%periodic(1:ndim)))then
     allocate(p%headp(r%levelmin-1:r%nlevelmax))
     allocate(p%tailp(r%levelmin-1:r%nlevelmax))
  else
     allocate(p%headp(r%levelmin:r%nlevelmax))
     allocate(p%tailp(r%levelmin:r%nlevelmax))
  endif
  ! No particle just yet
  p%headp=1
  p%tailp=0
end subroutine init_star
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine init_sink(r,g,p)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t,global_t
  use pm_parameters, only: SINK_TYPE
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  !-----------------------------------
  ! Allocate sink particle variables
  !------------------------------------
  p%type=SINK_TYPE
  p%static=r%static_sink
  allocate(p%xp    (r%nsinkmax,ndim))
  allocate(p%vp    (r%nsinkmax,ndim))
  allocate(p%fp    (r%nsinkmax,ndim))
  allocate(p%jp    (r%nsinkmax,ndim))
  allocate(p%mp    (r%nsinkmax))
  allocate(p%tp    (r%nsinkmax))
  allocate(p%levelp(r%nsinkmax))
  allocate(p%idp   (r%nsinkmax))
  p%nvaralloc=4*ndim+4
#ifdef OUTPUT_PARTICLE_POTENTIAL
  allocate(p%phip  (r%nsinkmax))
  p%nvaralloc=p%nvaralloc+1
#endif
  ! Allocate workspace variables
  allocate(p%sortp (r%nsinkmax))
  allocate(p%workp (r%nsinkmax))
  ! Allocate head and tail of particle levels
  if(ANY(.not.r%periodic(1:ndim)))then
     allocate(p%headp(r%levelmin-1:r%nlevelmax))
     allocate(p%tailp(r%levelmin-1:r%nlevelmax))
  else
     allocate(p%headp(r%levelmin:r%nlevelmax))
     allocate(p%tailp(r%levelmin:r%nlevelmax))
  endif
  ! No particle just yet
  p%headp=1
  p%tailp=0
  ! Set high frequency dump counter
  p%step_counter=-1
end subroutine init_sink
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine init_tree(r,g,p)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t,global_t
  use pm_parameters, only: TREE_TYPE
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  !-----------------------------------
  ! Allocate tree particle variables
  !------------------------------------
  p%type=TREE_TYPE
  allocate(p%xp    (r%ntreemax,ndim))
  allocate(p%vp    (r%ntreemax,ndim))
  allocate(p%mp    (r%ntreemax))
  allocate(p%tp    (r%ntreemax))
  allocate(p%tm    (r%ntreemax))
  allocate(p%levelp(r%ntreemax))
  allocate(p%idp   (r%ntreemax))
  allocate(p%idm   (r%ntreemax))
  allocate(p%idt   (r%ntreemax))
  p%nvaralloc=2*ndim+7
#ifdef OUTPUT_PARTICLE_POTENTIAL
  allocate(p%phip  (r%ntreemax))
  p%nvaralloc=p%nvaralloc+1
#endif
  ! Allocate workspace variables
  allocate(p%sortp (r%ntreemax))
  allocate(p%workp (r%ntreemax))
  ! Allocate head and tail of particle levels
  if(ANY(.not.r%periodic(1:ndim)))then
     allocate(p%headp(r%levelmin-1:r%nlevelmax))
     allocate(p%tailp(r%levelmin-1:r%nlevelmax))
  else
     allocate(p%headp(r%levelmin:r%nlevelmax))
     allocate(p%tailp(r%levelmin:r%nlevelmax))
  endif
  ! No particle just yet
  p%headp=1
  p%tailp=0
end subroutine init_tree
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine init_trac(r,g,p)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t,global_t
  use pm_parameters, only: TRAC_TYPE
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  !-----------------------------------
  ! Allocate tracer particle variables
  !------------------------------------
  p%type=TRAC_TYPE
  allocate(p%xp    (r%ntracmax,ndim))
  allocate(p%vp    (r%ntracmax,ndim))
  allocate(p%mp    (r%ntracmax))
  allocate(p%levelp(r%ntracmax))
  allocate(p%idp   (r%ntracmax))
  p%nvaralloc=2*ndim+3

  allocate(p%sortp (r%ntracmax))
  allocate(p%workp (r%ntracmax))
  if(ANY(.not.r%periodic(1:ndim)))then
     allocate(p%headp(r%levelmin-1:r%nlevelmax))
     allocate(p%tailp(r%levelmin-1:r%nlevelmax))
  else
     allocate(p%headp(r%levelmin:r%nlevelmax))
     allocate(p%tailp(r%levelmin:r%nlevelmax))
  endif
  p%headp=1
  p%tailp=0
end subroutine init_trac
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine init_dust(r,g,p)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t,global_t
  use pm_parameters, only: DUST_TYPE
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  !-----------------------------------
  ! Allocate dust particle variables
  !------------------------------------
  p%type=DUST_TYPE
  allocate(p%xp    (r%ndustmax,ndim))
  allocate(p%vp    (r%ndustmax,ndim))
  allocate(p%mp    (r%ndustmax))
  allocate(p%levelp(r%ndustmax))
  allocate(p%idp   (r%ndustmax))
  allocate(p%size  (r%ndustmax))
  allocate(p%charge(r%ndustmax)) 
  p%nvaralloc=2*ndim+5

  allocate(p%sortp (r%ndustmax))
  allocate(p%workp (r%ndustmax))
  allocate(p%headp(r%levelmin:r%nlevelmax))
  allocate(p%tailp(r%levelmin:r%nlevelmax))
  p%headp=1
  p%tailp=0
end subroutine init_dust
!#########################################################################
!#########################################################################
subroutine allocate_gas(r,g,p)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t,global_t
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  !-----------------------------------
  ! Allocate gas sph particle variables
  !------------------------------------
  allocate(p%xp    (p%npart,ndim))
  allocate(p%vp    (p%npart,ndim))
  allocate(p%mp    (p%npart))
  allocate(p%zp    (p%npart))
  allocate(p%up    (p%npart))
  allocate(p%levelp(p%npart))
  p%nvaralloc=2*ndim+5
  ! Allocate workspace variables
  allocate(p%sortp (p%npart))
  allocate(p%workp (p%npart))
  ! Allocate head and tail of particle levels
  if(ANY(.not.r%periodic(1:ndim)))then
     allocate(p%headp(r%levelmin-1:r%nlevelmax))
     allocate(p%tailp(r%levelmin-1:r%nlevelmax))
  else
     allocate(p%headp(r%levelmin:r%nlevelmax))
     allocate(p%tailp(r%levelmin:r%nlevelmax))
  endif
  ! No particle just yet
  p%headp=1
  p%tailp=0
end subroutine allocate_gas
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
recursive subroutine r_deallocate_gas(pst)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  !--------------------------------------------------------------------
  ! This routine is the recursive slave procedure to deallocate
  ! gas sph particle used for Gadget initial conditions.
  !--------------------------------------------------------------------
  integer::rID
  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_DEALLOCATE_GAS,pst%iUpper+1)
     call r_deallocate_gas(pst%pLower)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call deallocate_gas(pst%s%r,pst%s%g,pst%s%gas)
  endif

end subroutine r_deallocate_gas
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine deallocate_gas(r,g,p)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t,global_t
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  !-----------------------------------------
  ! Deallocate gas sph particle variables
  !-----------------------------------------
  deallocate(p%xp)
  deallocate(p%vp)
  deallocate(p%mp)
  deallocate(p%zp)
  deallocate(p%up)
  deallocate(p%levelp)
  deallocate(p%sortp)
  deallocate(p%workp)
  deallocate(p%headp)
  deallocate(p%tailp)
  p%nvaralloc=0
  p%npart=0
end subroutine deallocate_gas
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
end module init_part_module

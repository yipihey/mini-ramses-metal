module newdt_fine_module
#ifdef _CUDA
  use part_device, only: gpu_newdt_part
#endif

type :: out_newdt_part_t
  real(kind=8)::ekin,vmax
end type out_newdt_part_t

type :: out_max_bq_t
  real(kind=8)::max_b, max_q
end type out_max_bq_t

type :: in_broadcast_dt_t
  integer::ilevel
  real(kind=8)::dtnew,dtold
end type in_broadcast_dt_t

contains
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine m_newdt_fine(pst,ilevel)
  use amr_parameters, only: nvector
  use ramses_commons, only: pst_t
  use courant_fine_module, only: r_courant_fine, out_courant_fine_t
  use constants, only: twopi
  implicit none
  type(pst_t)::pst
  integer::ilevel
  !-----------------------------------------------------------
  ! This routine compute the time step using 3 constraints:
  ! 1- a Courant-type condition using particle velocity
  ! 2- the gravity free-fall time
  ! 3- 10% maximum variation for aexp 
  ! This routine also compute the particle kinetic energy.
  !-----------------------------------------------------------
  real(kind=8)::dx,tff,fourpi,threepi2
  real(kind=8)::ekin,vmax
  real(kind=8)::dt_gyro, max_b, max_q
  type(out_courant_fine_t)::out_courant_fine
  type(out_newdt_part_t)::out_newdt_part
  type(in_broadcast_dt_t)::in_broadcast_dt

  associate(r=>pst%s%r,g=>pst%s%g,m=>pst%s%m,p=>pst%s%p,mdl=>pst%s%mdl)

  if(m%noct_tot(ilevel)==0)return
  if(r%verbose)write(*,'("   Entering newdt_fine for level ",I2)')ilevel

  ! Save old time step
  g%dtold(ilevel)=g%dtnew(ilevel)

  ! Compute local cell spacing
  dx=r%boxlen/2**ilevel

  ! Maximum time step
  g%dtnew(ilevel)=dx/r%smallc

  ! Gravity-based Courant condition
  if(r%poisson.and.r%gravity_type<=0)then
     fourpi=4.0d0*ACOS(-1.0d0)
     if(r%cosmo)fourpi=1.5d0*g%omega_m*g%aexp
     threepi2=3.0d0*ACOS(-1.0d0)**2
     if(g%rho_max(ilevel)>0)then
        tff=sqrt(threepi2/8./fourpi/g%rho_max(ilevel))
        g%dtnew(ilevel)=MIN(g%dtnew(ilevel),r%courant_factor*tff)
     endif
  end if

  ! Cosmic-expansion-based Courant condition
  if(r%cosmo)then
     g%dtnew(ilevel)=MIN(g%dtnew(ilevel),0.1/g%hexp)
  end if

  ! Turbulence driving condition
  if(r%turb)then
     g%dtnew(ilevel)=MIN(g%dtnew(ilevel),pst%s%turb%turb_dt)
  end if

  ! Particle-based Courant condition
  if(r%pic)then
     if(r%verbose)write(*,'("   Entering newdt_part for level ",I2)')ilevel
     call r_newdt_part(pst,ilevel,1,out_newdt_part,storage_size(out_newdt_part)/32)
     ekin=out_newdt_part%ekin
     g%ekin_tot=g%ekin_tot+ekin
     vmax=out_newdt_part%vmax
     if(vmax>0.0d0)then
        g%dtnew(ilevel)=MIN(real(g%dtnew(ilevel),kind=8),r%courant_factor*dx/vmax)
     endif
  endif

  ! Hydro-based Courant condition
  if(r%hydro)then
     if(r%verbose)write(*,'("   Entering newdt_hydro for level ",I2)')ilevel
     call r_courant_fine(pst,ilevel,1,out_courant_fine,storage_size(out_courant_fine)/32)
     g%mass_tot=g%mass_tot+out_courant_fine%mass
     g%ekin_tot=g%ekin_tot+out_courant_fine%ekin
     g%eint_tot=g%eint_tot+out_courant_fine%eint
     g%emag_tot=g%emag_tot+out_courant_fine%emag
     g%dtnew(ilevel)=MIN(real(g%dtnew(ilevel),kind=8),out_courant_fine%dt)
  endif

  ! Dust gyro-frequency timestep constraint (MHD + dust only)
#ifdef MHD
  if(r%dust)then
     ! Compute maximum |B| at this level and maximum particle charge |q|
     block
       type(out_max_bq_t)::bq
       call r_max_B_and_Q(pst, ilevel, 1, bq, storage_size(bq)/32)
       max_b = bq%max_b
       max_q = bq%max_q
     end block
     if(max_b>0.0d0 .and. max_q>0.0d0)then
        dt_gyro = r%dust_gyro_factor * twopi / (max_q * max_b)
        g%dtnew(ilevel)=MIN(real(g%dtnew(ilevel),kind=8), dt_gyro)
     endif
  endif
#endif

  if(r%rt.and.r%rt_advect)then
     if(r%verbose)write(*,'("   Entering newdt_rt for level ",I2)')ilevel
     g%dtnew(ilevel)=MIN(real(g%dtnew(ilevel),kind=8),r%rt_nsubcycle*r%rt_courant_factor*dx/3d0/g%rt_c(ilevel))
  endif

  ! Adaptive time step condition
  if(ilevel>r%levelmin)then
     g%dtnew(ilevel)=MIN(g%dtnew(ilevel-1)/real(r%nsubcycle(ilevel-1)),g%dtnew(ilevel))
  end if
  if(r%verbose)write(*,'("   New time step for level ",I2," is ",1PE12.5)')ilevel,g%dtnew(ilevel)

  ! Broadcast new and old time steps to all CPUs
  in_broadcast_dt%ilevel=ilevel
  in_broadcast_dt%dtnew=g%dtnew(ilevel)
  in_broadcast_dt%dtold=g%dtold(ilevel)
  call r_broadcast_dt(pst,in_broadcast_dt,storage_size(in_broadcast_dt)/32)

  end associate

end subroutine m_newdt_fine
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
recursive subroutine r_max_B_and_Q(pst, ilevel, input_size, output, output_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  use amr_parameters, only: ndim, twotondim
  implicit none
  type(pst_t)::pst
  integer::ilevel
  integer,VALUE::input_size
  integer::output_size
  type(out_max_bq_t)::output, next_output

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_MAX_B_AND_Q,pst%iUpper+1,input_size,output_size,ilevel)
     call r_max_B_and_Q(pst%pLower,ilevel,input_size,output,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_output)
     output%max_b = MAX(output%max_b, next_output%max_b)
     output%max_q = MAX(output%max_q, next_output%max_q)
  else
     call max_B_and_Q(pst%s%r,pst%s%g,pst%s%m,pst%s%dust,ilevel,output%max_b,output%max_q)
  endif

end subroutine r_max_B_and_Q
!#####################################################################
!#####################################################################
subroutine max_B_and_Q(r,g,m,p,ilevel,max_b,max_q)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: run_t, global_t, mesh_t
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  type(part_t)::p
  integer::ilevel
  real(kind=8)::max_b, max_q

  integer::igrid, ind, idim, ipart
  real(kind=8)::bx, by, bz, bmag

  max_b = 0.0d0
  max_q = 0.0d0

#ifdef MHD
  do igrid=m%head(ilevel), m%tail(ilevel)
     do ind=1,twotondim
        bx=0.5d0*(m%bold(ind,1,igrid)+m%bold(ind,4,igrid))
        by=0.5d0*(m%bold(ind,2,igrid)+m%bold(ind,5,igrid))
        bz=0.5d0*(m%bold(ind,3,igrid)+m%bold(ind,6,igrid))
        bmag = sqrt(bx*bx + by*by + bz*bz)
        if(bmag>max_b) max_b=bmag
     end do
  end do
#endif

  if(r%dust)then
     do ipart=p%headp(ilevel), p%tailp(ilevel)
        if (abs(p%charge(ipart))>max_q) max_q=abs(p%charge(ipart))
     end do
  end if

end subroutine max_B_and_Q
recursive subroutine r_newdt_part(pst,ilevel,input_size,output,output_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
#ifdef _CUDA
  use pm_parameters, only: PART_TYPE
#endif
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  type(out_newdt_part_t)::output,next_output
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_NEWDT_PART,pst%iUpper+1,input_size,output_size,ilevel)
     call r_newdt_part(pst%pLower,ilevel,input_size,output,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_output)
     output%ekin=output%ekin+next_output%ekin
     output%vmax=MAX(output%vmax,next_output%vmax)
  else
     output%vmax=0.0d0
     output%ekin=0.0d0
#ifdef _CUDA
     if(pst%s%r%part)then
        call gpu_newdt_part(pst%s, ilevel, output%vmax, output%ekin)
     endif
     return
#endif
     if(pst%s%r%part)then
        call newdt_part(pst%s%r,pst%s%g,pst%s%p   ,ilevel,output%ekin,output%vmax)
     endif
     if(pst%s%r%star)then
        call newdt_part(pst%s%r,pst%s%g,pst%s%star,ilevel,output%ekin,output%vmax)
     endif
     if(pst%s%r%sink)then
        call newdt_part(pst%s%r,pst%s%g,pst%s%sink,ilevel,output%ekin,output%vmax)
     endif
     if(pst%s%r%tree)then
        call newdt_part(pst%s%r,pst%s%g,pst%s%tree,ilevel,output%ekin,output%vmax)
     endif
     if(pst%s%r%trac)then
        call newdt_part(pst%s%r,pst%s%g,pst%s%trac,ilevel,output%ekin,output%vmax)
     endif
     if(pst%s%r%dust)then
        call newdt_part(pst%s%r,pst%s%g,pst%s%dust,ilevel,output%ekin,output%vmax)
     endif
  endif

end subroutine r_newdt_part
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine newdt_part(r,g,p,ilevel,ekin,vmax)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t, global_t
  use pm_commons, only: part_t
  use pm_parameters, only: TREE_TYPE, TRAC_TYPE
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  integer::ilevel
  real(kind=8)::ekin,vmax

  integer::ipart,idim

  ! Compute maximum particle velocity
  do idim = 1, ndim
     do ipart = p%headp(ilevel), p%tailp(ilevel)
        vmax = MAX(vmax, ABS(dble(p%vp(ipart, idim))))
     end do
  end do

  if(p%type==TREE_TYPE .or. p%type==TRAC_TYPE)return

  ! Compute kinetic energy
  do idim = 1, ndim
     do ipart = p%headp(ilevel), p%tailp(ilevel)
        ekin = ekin + 0.5D0 * p%mp(ipart) * p%vp(ipart, idim)**2
     end do
  end do

end subroutine newdt_part
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
recursive subroutine r_broadcast_dt(pst,input,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  type(in_broadcast_dt_t)::input

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_BROADCAST_DT,pst%iUpper+1,input_size,0,input)
     call r_broadcast_dt(pst%pLower,input,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     pst%s%g%dtnew(input%ilevel)=input%dtnew
     pst%s%g%dtold(input%ilevel)=input%dtold
  endif

end subroutine r_broadcast_dt
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
end module newdt_fine_module

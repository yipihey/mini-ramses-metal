module synchro_hydro_fine_module
#ifdef _CUDA
  use gpu_runner, only: gpu_sync_hydro, gpu_grav_hydro
#endif
contains
!################################################################
!################################################################
!################################################################
!################################################################
subroutine m_synchro_hydro_fine(pst,ilevel,dteff)
  use amr_parameters, only: ndim,twotondim
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t)::pst
  integer::ilevel
  real(kind=8)::dteff
  !--------------------------------------------------------------
  ! Add gravity source terms to uold with time step dteff.
  !--------------------------------------------------------------
  integer,dimension(1:3)::input_array,dummy
  
  if(pst%s%m%noct_tot(ilevel)==0)return
  if(pst%s%r%verbose)write(*,'("   Entering synchro_hydro_fine for level",i2," and time step dt=",1PE12.5)')ilevel,dteff

  input_array(1)=ilevel
  input_array(2:3)=transfer(dteff,input_array)
  call r_synchro_hydro_fine(pst,input_array,3,dummy,0)
  
end subroutine m_synchro_hydro_fine
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_synchro_hydro_fine(pst,input_array,input_size,output_array,output_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  integer,dimension(1:input_size)::input_array
  integer,dimension(1:output_size)::output_array

  integer::ilevel
  real(kind=8)::dteff
  integer::rID
  
  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SYNCHRO_HYDRO_FINE,pst%iUpper+1,input_size,output_size,input_array)
     call r_synchro_hydro_fine(pst%pLower,input_array,input_size,input_array,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size)
  else
     ilevel=input_array(1)
     dteff=transfer(input_array(2:3),dteff)
#ifdef _CUDA
     call gpu_sync_hydro(pst%s, ilevel, dteff)
#else
     call synchro_hydro_fine(pst%s%r,pst%s%m,ilevel,dteff)
#endif
  endif

end subroutine r_synchro_hydro_fine
!################################################################
!################################################################
!################################################################
!################################################################
subroutine synchro_hydro_fine(r,m,ilevel,dteff)
  use amr_parameters, only: ndim, twotondim, dp
  use amr_commons, only: run_t, mesh_t
  implicit none
  type(run_t)::r
  type(mesh_t)::m
  integer::ilevel
  real(kind=8)::dteff
  !--------------------------------------------------------------
  ! Add gravity source terms to uold with time step dteff.
  !--------------------------------------------------------------
  integer::igrid,ind
  integer::idim
  real(kind=8)::ener

#ifdef HYDRO
  ! Loop over octs
  do igrid=m%head(ilevel),m%tail(ilevel)
     ! Loop over cells
     do ind=1,twotondim

        ! Remove kinetic energy from total energy
        ener=m%uold(ind,5,igrid)
        do idim=1,3
           ener=ener-0.5d0*m%uold(ind,idim+1,igrid)**2/max(dble(m%uold(ind,1,igrid)),r%smallr)
        end do

        ! Update momentum
#ifdef GRAV
        do idim=1,ndim
           m%uold(ind,idim+1,igrid)=m%uold(ind,idim+1,igrid)+&
                & max(m%uold(ind,1,igrid),real(r%smallr,kind=dp))*m%f(ind,idim,igrid)*dteff
        end do
#else
        do idim=1,ndim
           m%uold(ind,idim+1,igrid)=m%uold(ind,idim+1,igrid)+&
                & max(dble(m%uold(ind,1,igrid)),r%smallr)*r%constant_gravity(idim)*dteff
        end do        
#endif
        ! Update total energy
        do idim=1,3
           ener=ener+0.5d0*m%uold(ind,idim+1,igrid)**2/max(dble(m%uold(ind,1,igrid)),r%smallr)
        end do
        m%uold(ind,5,igrid)=ener

     end do
     ! End loop over cells
  end do
  ! End loop over grids

#endif

end subroutine synchro_hydro_fine
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_gravity_hydro_fine(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_GRAVITY_HYDRO_FINE,pst%iUpper+1,input_size,0,ilevel)
     call r_gravity_hydro_fine(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     call gpu_grav_hydro(pst%s, ilevel)
#else
     call gravity_hydro_fine(pst%s%r,pst%s%g,pst%s%m,ilevel)
#endif
  endif

end subroutine r_gravity_hydro_fine
!################################################################
!################################################################
!################################################################
!################################################################
subroutine gravity_hydro_fine(r,g,m,ilevel)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: run_t, global_t, mesh_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  integer::ilevel
  !--------------------------------------------------------------
  ! This routine adds to unew the gravity source terms to unew
  ! with only half a time step. Only the momentum and the
  ! total energy are modified in array unew.
  !--------------------------------------------------------------
  integer::igrid,ind
  real(kind=8)::d,u,v,w,e_kin,e_prim,d_old,fact

#ifdef HYDRO

  if(r%verbose.and.g%myid==1)write(*,'("   Entering gravity_hydro_fine for level ",I2)')ilevel

  ! Add gravity source term at time t with half time step
  do igrid=m%head(ilevel),m%tail(ilevel)
     do ind=1,twotondim

        d=max(dble(m%unew(ind,1,igrid)),r%smallr)
        u=0.0d0; v=0.0d0; w=0.0d0
        u=m%unew(ind,2,igrid)/d
        v=m%unew(ind,3,igrid)/d
        w=m%unew(ind,4,igrid)/d
        e_kin=0.5d0*d*(u**2+v**2+w**2)
        e_prim=m%unew(ind,5,igrid)-e_kin
        d_old=max(dble(m%uold(ind,1,igrid)),r%smallr)
        fact=d_old/d*0.5d0*g%dtnew(ilevel)
#ifdef GRAV
        u=u+m%f(ind,1,igrid)*fact
#if NDIM>1
        v=v+m%f(ind,2,igrid)*fact
#endif
#if NDIM>2
        w=w+m%f(ind,3,igrid)*fact
#endif
#else
        u=u+r%constant_gravity(1)*fact
#if NDIM>1
        v=v+r%constant_gravity(2)*fact
#endif
#if NDIM>2
        w=w+r%constant_gravity(3)*fact
#endif
#endif
        m%unew(ind,2,igrid)=d*u
        m%unew(ind,3,igrid)=d*v
        m%unew(ind,4,igrid)=d*w
        e_kin=0.5d0*d*(u**2+v**2+w**2)
        m%unew(ind,5,igrid)=e_prim+e_kin
     end do
  end do

#endif

end subroutine gravity_hydro_fine
!###########################################################
!###########################################################
!###########################################################
!###########################################################
end module synchro_hydro_fine_module

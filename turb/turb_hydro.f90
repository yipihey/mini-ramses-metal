module turb_hydro_module
contains
!################################################################
!################################################################
!################################################################
!################################################################
subroutine m_turb_hydro(pst,ilevel,dteff)
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
  if(pst%s%r%verbose)write(*,'("   Entering turb_hydro for level",i2," and time step dt=",1PE12.5)')ilevel,dteff

  input_array(1)=ilevel
  input_array(2:3)=transfer(dteff,input_array)
  call r_turb_hydro(pst,input_array,3,dummy,0)
  
end subroutine m_turb_hydro
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_turb_hydro(pst,input_array,input_size,output_array,output_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
#if defined(_CUDA) && defined(TURB)
  use gpu_runner, only: gpu_turb_hydro
#endif
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
     rID = mdl_send_request(pst%s%mdl,MDL_TURB_HYDRO,pst%iUpper+1,input_size,output_size,input_array)
     call r_turb_hydro(pst%pLower,input_array,input_size,input_array,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size)
  else
     ilevel=input_array(1)
     dteff=transfer(input_array(2:3),dteff)
#if defined(_CUDA) && defined(TURB)
     call gpu_turb_hydro(pst%s,ilevel,dteff)
#else
     call turb_hydro(pst%s%r,pst%s%m,ilevel,dteff)
#endif
  endif

end subroutine r_turb_hydro
!################################################################
!################################################################
!################################################################
!################################################################
subroutine turb_hydro(r,m,ilevel,dteff)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: run_t, mesh_t
  implicit none
  type(run_t)::r
  type(mesh_t)::m
  integer::ilevel
  real(kind=8)::dteff
  !------------------------------------------------------------------
  ! Add turbulent driving source terms to uold with time step dteff.
  !------------------------------------------------------------------
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
           ener=max(ener-0.5d0*m%uold(ind,idim+1,igrid)**2/max(dble(m%uold(ind,1,igrid)),r%smallr),&
           & m%uold(ind,1,igrid)*r%smallc**2)
        end do
#ifdef TURB
        ! Update momentum
        do idim=1,3
           m%uold(ind,idim+1,igrid)=m%uold(ind,idim+1,igrid)+&
                & max(m%uold(ind,1,igrid),r%smallr)*m%fturb(ind,idim,igrid)*dteff
        end do
#endif
        ! Update total energy
        do idim=1,3
           ener=max(ener+0.5d0*m%uold(ind,idim+1,igrid)**2/max(dble(m%uold(ind,1,igrid)),r%smallr),&
           & m%uold(ind,1,igrid)*r%smallc**2)
        end do
        m%uold(ind,5,igrid)=ener

     end do
     ! End loop over cells
  end do
  ! End loop over grids

#endif

end subroutine turb_hydro
!################################################################
!################################################################
!################################################################
!################################################################
end module turb_hydro_module

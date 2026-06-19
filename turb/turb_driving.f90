module turb_driving
contains
!#########################################################
!#########################################################
!#########################################################
!#########################################################
recursive subroutine r_drive_turb(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
#if defined(_CUDA) && defined(TURB)
  use gpu_runner, only: gpu_drive_turb
#endif
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_DRIVE_TURB,pst%iUpper+1,input_size,0,ilevel)
     call r_drive_turb(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#if defined(_CUDA) && defined(TURB)
     call gpu_drive_turb(pst%s,ilevel)
#else
     call drive_turb(pst%s%r,pst%s%g,pst%s%m,pst%s%turb,ilevel)
#endif
  endif

end subroutine r_drive_turb
!#########################################################
!#########################################################
!#########################################################
!#########################################################
subroutine drive_turb(r,g,m,t,ilevel)
  use amr_parameters, only: ndim, twotondim, nvector
  use amr_commons, only: run_t, global_t, mesh_t
  use turb_commons, only: turb_t, turb_force_calc
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  type(turb_t)::t
  integer::ilevel
  !-------------------------------------
  ! Compute analytical gravity force
  !-------------------------------------
  integer::igrid,ind,i,ngrid,idim,nstride
  real(kind=8)::dx
  real(kind=8),dimension(1:nvector)::rho
  real(kind=8),dimension(1:nvector,1:ndim)::xx,ff
 
  ! Mesh size at level ilevel in code units
  dx=r%boxlen/2**ilevel

  ! Loop over grids by vector sweeps
  do igrid=m%head(ilevel),m%tail(ilevel),nvector
     ngrid=MIN(nvector,m%tail(ilevel)-igrid+1)

     ! Loop over cells
     do ind=1,twotondim

        ! Compute cell centre position in code units
        do idim=1,ndim
           nstride=2**(idim-1)
           do i=1,ngrid
              xx(i,idim)=(2*m%grid(igrid+i-1)%ckey(idim)+MOD((ind-1)/nstride,2)+0.5)*dx-m%skip(idim)
           end do
        end do

        ! Collect gas density
#ifdef HYDRO
        do i=1,ngrid
           rho(i)=m%uold(ind,1,igrid+i-1)
        end do
#endif
        ! Interpolate turbulent driving force field
        call turb_force_calc(r,t,ngrid,xx,rho,ff)

        ! Scatter variables to main memory
#ifdef TURB
        do idim=1,ndim
           do i=1,ngrid
              m%fturb(ind,idim,igrid+i-1)=ff(i,idim)
           end do
        end do
#endif

     end do
     ! End loop over cells

  end do
  ! End loop over grid

end subroutine drive_turb
!#########################################################
!#########################################################
!#########################################################
!#########################################################
end module turb_driving

module interpol_phi_module
#ifdef _CUDA
  use gpu_runner, only: gpu_save_phi_old
#endif
contains
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine interpol_phi(m,igrid_nbor,ind_nbor,ccc,bbb,tfrac,phi_int)
  use amr_parameters, only: ndim, twotondim, threetondim
  use amr_commons, only: mesh_t
  implicit none
  type(mesh_t)::m
  integer,dimension(1:threetondim)::igrid_nbor,ind_nbor
  integer,dimension(1:8,1:8)::ccc
  real(kind=8),dimension(1:8)::bbb
  real(kind=8)::tfrac
  real(kind=8),dimension(1:twotondim)::phi_int
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Routine for interpolation at level-boundaries. Interpolation is used for
  ! - boundary conditions for solving poisson equation at fine level
  ! - computing force (gradient_phi) at fine level for cells close to boundary
  ! Interpolation is performed in space (using CIC) and - if adaptive 
  ! timestepping is on - also in time (using linear extrapolation 
  ! of the change in phi during the last coarse step onto the first fine step)
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  integer::ind,ind_average,ind_father
  integer::igrid_nbr,ind_nbr,igrid_cen,ind_cen
  real(kind=8)::coeff,add

#ifdef GRAV

  ! Store central cell
  igrid_cen=igrid_nbor(threetondim/2+1)
  ind_cen=ind_nbor(threetondim/2+1)

  ! Third order phi interpolation
  do ind=1,twotondim
     phi_int(ind)=0d0
     do ind_average=1,twotondim
        ind_father=ccc(ind_average,ind)
        coeff=bbb(ind_average)
        igrid_nbr=igrid_nbor(ind_father)
        ind_nbr=ind_nbor(ind_father)
        if (igrid_nbr==0) then 
           write(*,*)'no all neighbors present in interpol_phi...'
           write(*,*)igrid_nbor
           stop
           add=coeff*(m%phi(ind_cen,igrid_cen)+(m%phi(ind_cen,igrid_cen)-m%phi_old(ind_cen,igrid_cen))*tfrac)
        else
           add=coeff*(m%phi(ind_nbr,igrid_nbr)+(m%phi(ind_nbr,igrid_nbr)-m%phi_old(ind_nbr,igrid_nbr))*tfrac)
        endif
        phi_int(ind)=phi_int(ind)+add
     end do
  end do

#endif

end subroutine interpol_phi
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_save_phi_old(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size

  integer::ilevel
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SAVE_PHI_OLD,pst%iUpper+1,input_size,0,ilevel)
     call r_save_phi_old(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     call gpu_save_phi_old(pst%s, ilevel)
#else
     ! For _METAL the authoritative phi_old snapshot is done inside m_metal_poisson
     ! (after its full device sync); a bare-sync save here does not land correctly
     ! (device-oct ordering).  The host save is harmless (host phi_old is unused on
     ! the GPU path) and keeps non-Metal builds correct.
     call save_phi_old(pst%s%m,ilevel)
#endif
  endif

end subroutine r_save_phi_old
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine save_phi_old(m,ilevel)
  use amr_parameters, only: ndim, twotondim, threetondim
  use amr_commons, only: mesh_t
  implicit none
  type(mesh_t)::m
  integer ilevel
  ! Save the old potential for time extrapolation in case of subcycling
  integer::ind,igrid

#ifdef GRAV

  ! Loop over level grids
  do igrid=m%head(ilevel),m%tail(ilevel)
     ! Loop over cells
     do ind=1,twotondim
        ! Save phi      
        m%phi_old(ind,igrid)=m%phi(ind,igrid)
     end do
  end do

#endif

end subroutine save_phi_old
!###########################################################
!###########################################################
!###########################################################
!###########################################################
end module interpol_phi_module

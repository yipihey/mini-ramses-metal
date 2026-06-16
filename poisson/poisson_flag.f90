!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine poisson_flag(s,ilevel)
  use amr_parameters, only: ndim,twotondim
  use ramses_commons, only: ramses_t
  use hydro_parameters, only: nvar
  use hash
  implicit none
  type(ramses_t)::s
  integer::ilevel
  ! -------------------------------------------------------------------
  ! This routine flag for refinement cells that satisfies
  ! some user-defined physical criteria at the level ilevel. 
  ! -------------------------------------------------------------------
  real(kind=8)::dx_loc,vol_loc,d_scale,factG,twopi
  real(kind=8),dimension(1:nvar)::uu
  real(kind=8),dimension(1:6)::bb
  integer::igrid,ind,ivar
  logical::ok

  associate(r=>s%r,g=>s%g,m=>s%m)

  if(        r%m_refine(ilevel).LE.-1.0 .and.&
       & r%jeans_refine(ilevel).LE.-1.0 )return

  ! Constants
  twopi=2.0d0*ACOS(-1.0d0)
  factG=1
  if(r%cosmo)factG=3d0/4d0/twopi*g%omega_m*g%aexp
  dx_loc=r%boxlen/2**ilevel
  vol_loc=dx_loc**3
  d_scale=r%mass_sph/vol_loc

  ! Loop over active grids
  do igrid=m%head(ilevel),m%tail(ilevel)

     ! Loop over cells
     do ind=1,twotondim

        ! Initialize refinement to false
        ok=.false.

        ! Flag cells with density beyond the threshold
#ifndef GRAV
#ifdef HYDRO
        if(r%mass_sph>0.and.r%m_refine(ilevel)>=0)then
           ok=(ok .or. m%uold(ind,1,igrid)>=r%m_refine(ilevel)*d_scale)
        endif
#endif
#endif
        ! Flag cells with density beyond the threshold
#ifdef GRAV
        if(r%m_refine(ilevel)>=0)then
           ok=(ok .or. m%nref(ind,igrid)>=r%m_refine(ilevel))
        endif
#endif

        if(r%jeans_refine(ilevel)>=0)then
           ! Gather hydro variables
#ifdef HYDRO
           do ivar=1,nvar
              uu(ivar)=m%uold(ind,ivar,igrid)
           end do
           bb=0.0
#ifdef MHD
           do ivar=1,6
              bb(ivar)=m%bold(ind,ivar,igrid)
           end do
#endif
           call jeans_length_refine(r,uu,bb,factG,dx_loc,r%jeans_refine(ilevel),ok)
#endif
        endif
        
        ! Count only newly flagged cells
        if(m%flag1(ind,igrid)==0.and.ok)g%nflag=g%nflag+1
        if(ok)m%flag1(ind,igrid)=1

     end do
     ! End loop over cells
  end do
  ! End loop over grids

  end associate

end subroutine poisson_flag
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################
subroutine jeans_length_refine(r,uu,bb,factG,size_cell,n_jeans,ok)
  use amr_parameters, only: ndim
  use amr_commons, only: run_t
  use hydro_parameters, only: nvar, nener
  implicit none
  type(run_t)::r
  real(kind=8)::uu(1:nvar)
  real(kind=8)::bb(1:6)
  real(kind=8)::n_jeans,factG,size_cell
  logical ::ok
  !
  integer::irad
  real(kind=8)::lamb_jeans,twopi
  real(kind=8)::dens,tempe,etherm

  twopi=2.0d0*ACOS(-1.0d0)
  ! compute the thermal energy
  dens = max( uu(1) , r%smallr )
  etherm = uu(5)
  etherm = etherm - 0.5d0*uu(2)**2/dens
  etherm = etherm - 0.5d0*uu(3)**2/dens
  etherm = etherm - 0.5d0*uu(4)**2/dens
#if NENER>0
  do irad=1,nener
     etherm=etherm-uu(5+irad)
  end do
#endif
#ifdef MHD
  etherm=etherm-0.125d0*(bb(1)+bb(4))**2
  etherm=etherm-0.125d0*(bb(2)+bb(5))**2
  etherm=etherm-0.125d0*(bb(3)+bb(6))**2
#endif

  ! compute the temperature
  tempe =  etherm / dens * (r%gamma - 1d0)
  ! prevent numerical crash due to negative temperature
  tempe = max( tempe , r%smallc**2 )

  ! compute the Jeans length
  lamb_jeans = sqrt( tempe * twopi / 2d0 / dens / factG )

  ! the Jeans length must be smaller
  ! than n_jeans times the size of the pixel
  ok = ok .or. ( n_jeans*size_cell >= lamb_jeans )

end subroutine jeans_length_refine
!#####################################################################
!#####################################################################
!#####################################################################
!#####################################################################


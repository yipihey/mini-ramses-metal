module phi_fine_cg_module
#ifdef _CUDA
  use gpu_runner, only: gpu_init_phi
#endif
  use multigrid_fine_coarse, only: level_count_t
  type recurrence_t
     integer::ilevel
     real(kind=8)::cg
  end type recurrence_t
contains
#ifdef GRAV
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine m_phi_fine_cg(pst,ilevel,icount)
  use amr_parameters, only: ndim, twondim, twotondim, threetondim, nvector
  use ramses_commons, only: pst_t
  use cache_commons
  implicit none
  type(pst_t)::pst
  integer::ilevel,icount
  !=========================================================
  ! Iterative Poisson solver with Conjugate Gradient method 
  ! to solve A x = b
  ! r  : stored in f(1)
  ! p  : stored in f(2)
  ! A p: stored in f(3)
  ! x  : stored in phi
  ! b  : stored in rho
  !=========================================================
  integer::igrid,ind,iter,itermax
  integer,dimension(1)::dummy
  real(kind=8)::error,error_ini
  real(kind=8)::r2_old=0,alpha_cg,beta_cg
  real(kind=8)::r2,pAp,rhs_norm
  type(level_count_t)::level_count
  type(recurrence_t)::recurrence

  associate(r=>pst%s%r,g=>pst%s%g,m=>pst%s%m,mdl=>pst%s%mdl)
    
  if(r%gravity_type>0)return
  if(m%noct_tot(ilevel)==0)return
  if(r%verbose)write(*,111)ilevel
111 format('   Entering phi_fine_cg for level ',I2)

  !===============================
  ! Compute initial phi
  !===============================
  level_count%ilevel=ilevel
  level_count%icount=icount
  call r_make_initial_phi(pst,level_count,storage_size(level_count)/32)

  !===============================
  ! Compute right-hand side norm
  !===============================
  call r_cmp_rhs_norm(pst,ilevel,1,rhs_norm,storage_size(rhs_norm)/32)
  rhs_norm=DSQRT(rhs_norm/dble(twotondim*m%noct_tot(ilevel)))

  !==============================================
  ! Compute r = b - Ax and store it into f(i,1)
  ! Also set p = r and store it into f(i,2)
  !==============================================
  call r_cmp_residual_cg(pst,level_count,storage_size(level_count)/32)

  !====================================
  ! Main iteration loop
  !====================================
  iter=0; itermax=10000
  error=1.0D0; error_ini=1.0D0
  do while(error>r%epsilon*error_ini.and.iter<itermax)

     iter=iter+1

     !====================================
     ! Compute residual norm
     !====================================
     call r_cmp_r2_cg(pst,ilevel,1,r2,storage_size(r2)/32)

     !====================================
     ! Compute beta factor
     !====================================
     if(iter==1)then
        beta_cg=0.
     else
        beta_cg=r2/r2_old
     end if
     r2_old=r2

     !====================================
     ! Recurrence on p
     !====================================
     recurrence%ilevel=ilevel
     recurrence%cg=beta_cg
     call r_recurrence_on_p(pst,recurrence,storage_size(recurrence)/32)

     !==============================================
     ! Compute z = Ap and store it into f(i,3)
     !==============================================
     call r_cmp_Ap_cg(pst,ilevel,1)

     !====================================
     ! Compute p.Ap scalar product
     !====================================
     call r_cmp_pAp_cg(pst,ilevel,1,pAp,storage_size(pAp)/32)

     !====================================
     ! Compute alpha factor
     !====================================
     alpha_cg = r2/pAp

     !====================================
     ! Recurrence on x and r
     !====================================
     recurrence%ilevel=ilevel
     recurrence%cg=alpha_cg
     call r_recurrence_x_and_r(pst,recurrence,storage_size(recurrence)/32)

     !====================================
     ! Compute error
     !====================================
     error=DSQRT(r2/dble(twotondim*m%noct_tot(ilevel)))
     if(iter==1)error_ini=error
     if(r%verbose)write(*,112)iter,error/rhs_norm,error/error_ini
112  format('   ==> Step=',i5,' Error=',2(1pe10.3,1x))

  end do
  ! End main iteration loop

  write(*,115)ilevel,iter,error/rhs_norm,error/error_ini
115 format('   ==> Level=',i5,' Step=',i5,' Error=',2(1pe10.3,1x))
  if(iter>=itermax)then
     write(*,*)'Poisson failed to converge...'
  end if

  end associate
  
end subroutine m_phi_fine_cg
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_recurrence_on_p(pst,input,input_size)
  use mdl_module
  use amr_parameters, only: twotondim
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  type(recurrence_t)::input

  integer::ind,igrid,ilevel
  real(kind=8)::beta_cg
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_RECURRENCE_ON_P,pst%iUpper+1,input_size,0,input)
     call r_recurrence_on_p(pst%pLower,input,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     ilevel=input%ilevel
     beta_cg=input%cg
     do igrid=pst%s%m%head(ilevel),pst%s%m%tail(ilevel)
        do ind=1,twotondim
           pst%s%m%f(ind,2,igrid)=pst%s%m%f(ind,1,igrid)+beta_cg*pst%s%m%f(ind,2,igrid)
        end do
     end do
  endif

end subroutine r_recurrence_on_p
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_recurrence_x_and_r(pst,input,input_size)
  use mdl_module
  use amr_parameters, only: twotondim
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  type(recurrence_t)::input

  integer::ind,igrid,ilevel
  real(kind=8)::alpha_cg
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_RECURRENCE_X_AND_R,pst%iUpper+1,input_size,0,input)
     call r_recurrence_x_and_r(pst%pLower,input,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     ilevel=input%ilevel
     alpha_cg=input%cg
     ! Recurrence on x
     do igrid=pst%s%m%head(ilevel),pst%s%m%tail(ilevel)
        do ind=1,twotondim
           pst%s%m%phi(ind,igrid)=pst%s%m%phi(ind,igrid)+alpha_cg*pst%s%m%f(ind,2,igrid)
        end do
     end do
     ! Recurrence on r
     do igrid=pst%s%m%head(ilevel),pst%s%m%tail(ilevel)
        do ind=1,twotondim
           pst%s%m%f(ind,1,igrid)=pst%s%m%f(ind,1,igrid)-alpha_cg*pst%s%m%f(ind,3,igrid)
        end do
     end do
  endif

end subroutine r_recurrence_x_and_r
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_cmp_residual_cg(pst,input,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  use call_back
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  type(level_count_t)::input

  integer::ilevel,icount
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CMP_RESIDUAL_CG,pst%iUpper+1,input_size,0,input)
     call r_cmp_residual_cg(pst%pLower,input,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call cmp_residual_cg(pst%s,input%ilevel,input%icount)
  endif

end subroutine r_cmp_residual_cg

subroutine cmp_residual_cg(s,ilevel,icount)
  use mdl_module
  use amr_parameters, only: ndim, twondim, twotondim, threetondim, nvector
  use ramses_commons, only: ramses_t
  use interpol_phi_module, only: interpol_phi
  use nbors_utils
  use cache_commons
  use cache
  implicit none
  type(ramses_t)::s
  integer::ilevel,icount
  !------------------------------------------------------------------
  ! This routine computes the residual for the Conjugate Gradient
  ! Poisson solver. The residual is stored in f(i,1) and f(i,2).
  !------------------------------------------------------------------
  integer::inbor,igrid,idim,ind,igridn
  integer::id1,id2,ig1,ig2
  integer,dimension(1:8,1:8)::ccc
  real(kind=8)::dx2,fourpi,oneoversix,fact,residu
  real(kind=8)::aa,bb,cc,dd,tfrac
  real(kind=8),dimension(1:8)::bbb
  integer,dimension(1:3,1:2,1:8)::iii,jjj
  real(kind=8),dimension(1:twotondim,0:twondim)::phi_nbor
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer,dimension(1:threetondim)::ind_nbor
  integer,dimension(1:threetondim)::igrid_nbor
  integer,dimension(1:3,1:6),save::shift=reshape(&
       & (/-1,0,0,1,0,0,0,-1,0,0,1,0,0,0,-1,0,0,1/),(/3,6/))
  type(msg_three_realdp)::dummy_three_realdp

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)
    
  ! Set constants
  dx2=(r%boxlen/2**ilevel)**2
  fourpi=4.D0*ACOS(-1.0D0)
  if(r%cosmo)fourpi=1.5D0*g%omega_m*g%aexp
  oneoversix=1.0D0/dble(twondim)
  fact=oneoversix*fourpi*dx2

  iii(1,1,1:8)=(/1,0,1,0,1,0,1,0/); jjj(1,1,1:8)=(/2,1,4,3,6,5,8,7/)
  iii(1,2,1:8)=(/0,2,0,2,0,2,0,2/); jjj(1,2,1:8)=(/2,1,4,3,6,5,8,7/)
  iii(2,1,1:8)=(/3,3,0,0,3,3,0,0/); jjj(2,1,1:8)=(/3,4,1,2,7,8,5,6/)
  iii(2,2,1:8)=(/0,0,4,4,0,0,4,4/); jjj(2,2,1:8)=(/3,4,1,2,7,8,5,6/)
  iii(3,1,1:8)=(/5,5,5,5,0,0,0,0/); jjj(3,1,1:8)=(/5,6,7,8,1,2,3,4/)
  iii(3,2,1:8)=(/0,0,0,0,6,6,6,6/); jjj(3,2,1:8)=(/5,6,7,8,1,2,3,4/)

  ! CIC method constants
  aa = 1.0D0/4.0D0**ndim
  bb = 3.0D0*aa
  cc = 9.0D0*aa
  dd = 27.D0*aa
  bbb(:)  =(/aa ,bb ,bb ,cc ,bb ,cc ,cc ,dd/)

  ! Sampling positions in the 3x3x3 father cell cube
  ccc(:,1)=(/1 ,2 ,4 ,5 ,10,11,13,14/)
  ccc(:,2)=(/3 ,2 ,6 ,5 ,12,11,15,14/)
  ccc(:,3)=(/7 ,8 ,4 ,5 ,16,17,13,14/)
  ccc(:,4)=(/9 ,8 ,6 ,5 ,18,17,15,14/)
  ccc(:,5)=(/19,20,22,23,10,11,13,14/)
  ccc(:,6)=(/21,20,24,23,12,11,15,14/)
  ccc(:,7)=(/25,26,22,23,16,17,13,14/)
  ccc(:,8)=(/27,26,24,23,18,17,15,14/)

  if (icount .ne. 1 .and. icount .ne. 2)then
     write(*,*)'icount has bad value'
     call mdl_abort(mdl)
  endif

  ! Compute fraction of time steps for interpolation
  if (g%dtold(ilevel-1)>0.0)then
     tfrac=g%dtnew(ilevel)/g%dtold(ilevel-1)*(icount-1)
  else
     tfrac=0.0
  end if
  block; character(len=8)::ntf; call get_environment_variable("RAMSES_NO_TFRAC",ntf)
     if(len_trim(ntf)>0) tfrac=0.0d0; end block   ! DIAG: disable time-extrap to match Metal test

  call open_cache(mdl, m, pack_size=storage_size(dummy_three_realdp)/32, &
       pack=pack_fetch_interpol, unpack=unpack_fetch_interpol)

  hash_nbor(0)=ilevel

  ! Loop over grids
  do igrid=m%head(ilevel),m%tail(ilevel)

     ! Get central oct potential
     do ind=1,twotondim
        phi_nbor(ind,0)=m%phi(ind,igrid)
     end do

     ! Get neighboring octs potential
     do inbor=1,twondim

        ! Get neighboring grid
        hash_nbor(1:ndim)=m%grid(igrid)%ckey(1:ndim)+shift(1:ndim,inbor)

        ! Periodic boundary conditions
        do idim=1,ndim
           if(r%periodic(idim))then
              if(hash_nbor(idim)< m%box_ckey_min(idim,ilevel))hash_nbor(idim)=m%box_ckey_max(idim,ilevel)-1
              if(hash_nbor(idim)>=m%box_ckey_max(idim,ilevel))hash_nbor(idim)=m%box_ckey_min(idim,ilevel)
           endif
        enddo

        ! Get neighbouring grid using a read-only cache
        call get_grid(s,hash_nbor,igridn,flush_cache=.false.,fetch_cache=.true.)

        ! If grid exists, then copy into array
        if(igridn>0)then
           do ind=1,twotondim
              phi_nbor(ind,inbor)=m%phi(ind,igridn)
           end do

        ! Otherwise interpolate from coarser level
        else
           ! Get 3**ndim neighbouring parent cell using a read-only cache
           call get_threetondim_nbor_parent_cell(s,hash_nbor,igrid_nbor,ind_nbor,flush_cache=.false.,fetch_cache=.true.)
           call interpol_phi(m,igrid_nbor,ind_nbor,ccc,bbb,tfrac,phi_nbor(1,inbor))
           do ind=1,threetondim
              call unlock_cache(m,igrid_nbor(ind))
           end do
        endif

     end do
     ! End loop over neighboring octs

     ! Loop over cells
     do ind=1,twotondim

        ! Compute residual using 6 neighbors potential
        residu=m%phi(ind,igrid)
        do idim=1,ndim
           id1=jjj(idim,1,ind); ig1=iii(idim,1,ind)
           id2=jjj(idim,2,ind); ig2=iii(idim,2,ind)
           residu=residu-oneoversix*(phi_nbor(id1,ig1)+phi_nbor(id2,ig2))
        end do
        residu=residu+fact*(m%rho(ind,igrid)-g%rho_tot)

        ! Store results in f(ind,1)
        m%f(ind,1,igrid)=residu

        ! Store results in f(ind,2)
        m%f(ind,2,igrid)=residu

     end do
     ! End loop over cells

  end do
  ! End loop over grids

  call close_cache(mdl)

  end associate

end subroutine cmp_residual_cg
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_cmp_Ap_cg(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CMP_AP_CG,pst%iUpper+1,input_size,0,ilevel)
     call r_cmp_Ap_cg(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     call cmp_Ap_cg(pst%s,ilevel)
  endif

end subroutine r_cmp_Ap_cg

subroutine cmp_Ap_cg(s,ilevel)
  use amr_parameters, only: ndim, twondim, twotondim, threetondim, nvector
  use ramses_commons, only: ramses_t
  use nbors_utils
  use cache_commons
  use cache
  implicit none
  type(ramses_t)::s
  integer::ilevel
  !------------------------------------------------------------------
  ! This routine computes Ap for the Conjugate Gradient
  ! Poisson Solver and store the result into f(i,3).
  !------------------------------------------------------------------
  integer::inbor,igrid,idim,ind,igridn
  integer::id1,id2,ig1,ig2
  real(kind=8)::oneoversix,residu
  integer,dimension(1:3,1:2,1:8)::iii,jjj
  real(kind=8),dimension(1:twotondim,0:twondim)::phi_nbor
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer,dimension(1:3,1:6),save::shift=reshape(&
       & (/-1,0,0,1,0,0,0,-1,0,0,1,0,0,0,-1,0,0,1/),(/3,6/))
  type(msg_small_realdp)::dummy_small_realdp

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  ! Set constants
  oneoversix=1.0D0/dble(twondim)

  iii(1,1,1:8)=(/1,0,1,0,1,0,1,0/); jjj(1,1,1:8)=(/2,1,4,3,6,5,8,7/)
  iii(1,2,1:8)=(/0,2,0,2,0,2,0,2/); jjj(1,2,1:8)=(/2,1,4,3,6,5,8,7/)
  iii(2,1,1:8)=(/3,3,0,0,3,3,0,0/); jjj(2,1,1:8)=(/3,4,1,2,7,8,5,6/)
  iii(2,2,1:8)=(/0,0,4,4,0,0,4,4/); jjj(2,2,1:8)=(/3,4,1,2,7,8,5,6/)
  iii(3,1,1:8)=(/5,5,5,5,0,0,0,0/); jjj(3,1,1:8)=(/5,6,7,8,1,2,3,4/)
  iii(3,2,1:8)=(/0,0,0,0,6,6,6,6/); jjj(3,2,1:8)=(/5,6,7,8,1,2,3,4/)

  call open_cache(mdl, m, pack_size=storage_size(dummy_small_realdp)/32, &
       pack=pack_fetch_cg, unpack=unpack_fetch_cg)

  hash_nbor(0)=ilevel

  ! Loop over grids
  do igrid=m%head(ilevel),m%tail(ilevel)

     ! Get central oct potential
     do ind=1,twotondim
        phi_nbor(ind,0)=m%f(ind,2,igrid)
     end do

     ! Get neighboring octs potential
     do inbor=1,twondim

        ! Get neighboring grid
        hash_nbor(1:ndim)=m%grid(igrid)%ckey(1:ndim)+shift(1:ndim,inbor)

        ! Periodic boundary conditions
        do idim=1,ndim
           if(r%periodic(idim))then
              if(hash_nbor(idim)< m%box_ckey_min(idim,ilevel))hash_nbor(idim)=m%box_ckey_max(idim,ilevel)-1
              if(hash_nbor(idim)>=m%box_ckey_max(idim,ilevel))hash_nbor(idim)=m%box_ckey_min(idim,ilevel)
           endif
        enddo

        ! Get neighbouring grid using read-only cache
        call get_grid(s,hash_nbor,igridn,flush_cache=.false.,fetch_cache=.true.)

        ! If grid exists, then copy into array
        if(igridn>0)then
           do ind=1,twotondim
              phi_nbor(ind,inbor)=m%f(ind,2,igridn)
           end do
        else
           do ind=1,twotondim
              phi_nbor(ind,inbor)=0.0
           end do
        endif

     end do
     ! End loop over neighboring octs

     ! Loop over cells
     do ind=1,twotondim

        ! Compute Ap using neighbors potential
        residu=-m%f(ind,2,igrid)
        do idim=1,ndim
           id1=jjj(idim,1,ind); ig1=iii(idim,1,ind)
           id2=jjj(idim,2,ind); ig2=iii(idim,2,ind)
           residu=residu+oneoversix*(phi_nbor(id1,ig1)+phi_nbor(id2,ig2))
        end do

        ! Store results in f(ind,3)
        m%f(ind,3,igrid)=residu

     end do
     ! End loop over cells

  end do
  ! End loop over grids

  call close_cache(mdl)

  end associate

end subroutine cmp_Ap_cg
!###########################################################
!###########################################################
!###########################################################
!###########################################################
recursive subroutine r_make_initial_phi(pst,input,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  use call_back
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  type(level_count_t)::input

  integer::ilevel,icount
  integer::rID
  
  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_MAKE_INITIAL_PHI,pst%iUpper+1,input_size,0,input)
     call r_make_initial_phi(pst%pLower,input,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     call gpu_init_phi(pst%s,input%ilevel,input%icount)
#else
     call make_initial_phi(pst%s,input%ilevel,input%icount)
#endif
  endif

end subroutine r_make_initial_phi
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine make_initial_phi(s,ilevel,icount)
  use mdl_module
  use amr_parameters, only: ndim, twondim, twotondim, threetondim, nvector
  use ramses_commons, only: ramses_t
  use interpol_phi_module, only: interpol_phi
  use nbors_utils
  use cache_commons
  use cache
  use boundaries, only: init_bound_phi
  implicit none
  type(ramses_t)::s
  integer::ilevel,icount
  !
  !
  !
  integer::igrid,idim,ind, nstride
  integer,dimension(1:8,1:8)::ccc
  real(kind=8)::aa,bb,cc,dd,tfrac,dx
  real(kind=8),dimension(1:8)::bbb
  real(kind=8),dimension(1:nvector,1:ndim)::xx
  real(kind=8),dimension(1:nvector)::pp
  integer(kind=8),dimension(0:ndim)::hash_key
  integer,dimension(1:threetondim)::ind_nbor
  integer,dimension(1:threetondim)::igrid_nbor
  real(kind=8),dimension(1:twotondim)::phi_int
  type(msg_three_realdp)::dummy_three_realdp

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  ! CIC method constants
  aa = 1.0D0/4.0D0**ndim
  bb = 3.0D0*aa
  cc = 9.0D0*aa
  dd = 27.D0*aa
  bbb(:)  =(/aa ,bb ,bb ,cc ,bb ,cc ,cc ,dd/)

  ! Sampling positions in the 3x3x3 father cell cube
  ccc(:,1)=(/1 ,2 ,4 ,5 ,10,11,13,14/)
  ccc(:,2)=(/3 ,2 ,6 ,5 ,12,11,15,14/)
  ccc(:,3)=(/7 ,8 ,4 ,5 ,16,17,13,14/)
  ccc(:,4)=(/9 ,8 ,6 ,5 ,18,17,15,14/)
  ccc(:,5)=(/19,20,22,23,10,11,13,14/)
  ccc(:,6)=(/21,20,24,23,12,11,15,14/)
  ccc(:,7)=(/25,26,22,23,16,17,13,14/)
  ccc(:,8)=(/27,26,24,23,18,17,15,14/)

  if (icount .ne. 1 .and. icount .ne. 2)then
     write(*,*)'icount has bad value'
     call mdl_abort(mdl)
  endif

  ! Compute fraction of time steps for interpolation
  if (g%dtold(ilevel-1)>0.0)then
     tfrac=g%dtnew(ilevel)/g%dtold(ilevel-1)*(icount-1)
  else
     tfrac=0.0
  end if
  block; character(len=8)::ntf; call get_environment_variable("RAMSES_NO_TFRAC",ntf)
     if(len_trim(ntf)>0) tfrac=0.0d0; end block   ! DIAG: disable time-extrap to match Metal test

  call open_cache(mdl, m, pack_size=storage_size(dummy_three_realdp)/32, &
       pack=pack_fetch_interpol, unpack=unpack_fetch_interpol, &
       bound=init_bound_phi)

  hash_key(0)=ilevel

  ! Mesh size at level ilevel in code units
  dx=r%boxlen/2**ilevel

  ! Loop over grids
  do igrid=m%head(ilevel),m%tail(ilevel)

     ! By default, initial phi is equal to zero

     ! Loop over cells
     do ind=1,twotondim
        m%phi(ind,igrid)=0.0d0
        do idim=1,ndim
           m%f(ind,idim,igrid)=0.0
        end do
     end do
     ! End loop over cells

     ! For fine levels, initial phi is interpolated from coarser level
     if(ilevel.GT.r%levelmin)then

        hash_key(1:ndim)=m%grid(igrid)%ckey(1:ndim)
        ! Get 3**ndim neghbouring parent cell using read-only cache
        call get_threetondim_nbor_parent_cell(s,hash_key,igrid_nbor,ind_nbor,flush_cache=.false.,fetch_cache=.true.)
        call interpol_phi(m,igrid_nbor,ind_nbor,ccc,bbb,tfrac,phi_int)
        do ind=1,threetondim
           call unlock_cache(m,igrid_nbor(ind))
        end do

        ! Loop over cells
        do ind=1,twotondim
           m%phi(ind,igrid)=phi_int(ind)
        end do
        ! End loop over cells

     ! For coarse level and for non-periodic boundary conditions, set multipole potential
     else if (any(.not. r%periodic(1:ndim)))  then

        ! Loop over cells
        do ind=1,twotondim
           do idim=1,ndim
              nstride=2**(idim-1)
              xx(1,idim)=(2*m%grid(igrid)%ckey(idim)+MOD((ind-1)/nstride,2)+0.5)*dx-m%skip(idim)
           end do
           ! Call analytical potential routine
           call phiana(r,g,xx,pp,dx,1)
           m%phi(ind,igrid)=pp(1)
        end do
        ! End loop over cells

     end if

  end do
  ! End loop over grids

  call close_cache(mdl)

  end associate

end subroutine make_initial_phi

subroutine pack_fetch_interpol(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_three_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind
  type(msg_three_realdp)::msg

#ifdef GRAV
  do ind=1,twotondim
     msg%realdp_phi(ind)=mesh%phi(ind,igrid)
     msg%realdp_phi_old(ind)=mesh%phi_old(ind,igrid)
     msg%realdp_dis(ind)=mesh%f(ind,3,igrid)
  end do
#endif

  msg_array=transfer(msg,msg_array)

end subroutine pack_fetch_interpol

subroutine unpack_fetch_interpol(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim,twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_three_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind
  type(msg_three_realdp)::msg

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)
  msg=transfer(msg_array,msg)

#ifdef GRAV
  do ind=1,twotondim
     mesh%phi(ind,igrid)=msg%realdp_phi(ind)
     mesh%phi_old(ind,igrid)=msg%realdp_phi_old(ind)
     mesh%f(ind,3,igrid)=msg%realdp_dis(ind)
  end do
#endif

end subroutine unpack_fetch_interpol
! ########################################################################
! ########################################################################
! ########################################################################
! ########################################################################
subroutine pack_fetch_cg(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_small_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind
  type(msg_small_realdp)::msg

#ifdef GRAV
  do ind=1,twotondim
     msg%realdp(ind)=mesh%f(ind,2,igrid)
  end do
#endif

  msg_array=transfer(msg,msg_array)

end subroutine pack_fetch_cg
! ########################################################################
! ########################################################################
! ########################################################################
! ########################################################################
subroutine unpack_fetch_cg(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_small_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind
  type(msg_small_realdp)::msg

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)
  msg=transfer(msg_array,msg)

#ifdef GRAV
  do ind=1,twotondim
     mesh%f(ind,2,igrid)=msg%realdp(ind)
  end do
#endif

end subroutine unpack_fetch_cg
! ########################################################################
! ########################################################################
! ########################################################################
! ########################################################################
recursive subroutine r_cmp_rhs_norm(pst,ilevel,input_size,rhs_norm,output_size)
  use mdl_module
  use amr_parameters, only: twotondim,twondim
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  ! ------------------------------------------------------------------------
  ! Compute norm of residual 
  ! ------------------------------------------------------------------------
  integer::ind,igrid,ilevel
  real(kind=8)::rhs_norm,next_rhs_norm
  real(kind=8)::dx2,fourpi,oneoversix,fact,fact2

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CMP_RHS_NORM,pst%iUpper+1,input_size,output_size,ilevel)
     call r_cmp_rhs_norm(pst%pLower,ilevel,input_size,rhs_norm,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_rhs_norm)
     rhs_norm=rhs_norm+next_rhs_norm
  else
     ! Set constants
     dx2=(pst%s%r%boxlen/2.0d0**ilevel)**2
     fourpi=4.D0*ACOS(-1.0D0)
     if(pst%s%r%cosmo)fourpi=1.5D0*pst%s%g%omega_m*pst%s%g%aexp
     oneoversix=1.0D0/dble(twondim)
     fact=oneoversix*fourpi*dx2
     fact2=fact*fact
     rhs_norm=0.d0
     do igrid=pst%s%m%head(ilevel),pst%s%m%tail(ilevel)
        do ind=1,twotondim
           rhs_norm=rhs_norm+fact2*(pst%s%m%rho(ind,igrid)-pst%s%g%rho_tot)**2
        end do
     end do
  endif

end subroutine r_cmp_rhs_norm
! ########################################################################
! ########################################################################
! ########################################################################
! ########################################################################
recursive subroutine r_cmp_r2_cg(pst,ilevel,input_size,r2,output_size)
  use mdl_module
  use amr_parameters, only: twotondim
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  ! ------------------------------------------------------------------------
  ! Compute norm of residual 
  ! ------------------------------------------------------------------------
  integer::ind,igrid,ilevel
  real(kind=8)::r2,next_r2
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CMP_R2_CG,pst%iUpper+1,input_size,output_size,ilevel)
     call r_cmp_r2_cg(pst%pLower,ilevel,input_size,r2,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_r2)
     r2=r2+next_r2
  else
     r2=0.0d0
     do igrid=pst%s%m%head(ilevel),pst%s%m%tail(ilevel)
        do ind=1,twotondim
           r2=r2+pst%s%m%f(ind,1,igrid)**2
        end do
     end do
  endif

end subroutine r_cmp_r2_cg
! ########################################################################
! ########################################################################
! ########################################################################
! ########################################################################
recursive subroutine r_cmp_pAp_cg(pst,ilevel,input_size,pAp,output_size)
  use mdl_module
  use amr_parameters, only: twotondim
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  ! ------------------------------------------------------------------------
  ! Compute norm of residual 
  ! ------------------------------------------------------------------------
  integer::igrid,ind,ilevel
  real(kind=8)::pAp,next_pAp
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CMP_PAP_CG,pst%iUpper+1,input_size,output_size,ilevel)
     call r_cmp_pAp_cg(pst%pLower,ilevel,input_size,pAp,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_pAp)
     pAp=pAp+next_pAp
  else
     pAp=0.0d0
     do igrid=pst%s%m%head(ilevel),pst%s%m%tail(ilevel)
        do ind=1,twotondim
           pAp=pAp+pst%s%m%f(ind,2,igrid)*pst%s%m%f(ind,3,igrid)
        end do
     end do
  endif

end subroutine r_cmp_pAp_cg
! ########################################################################
! ########################################################################
! ########################################################################
! ########################################################################
#endif
end module phi_fine_cg_module

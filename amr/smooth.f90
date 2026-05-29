module smooth_module
#ifdef _CUDA
  use gpu_runner, only: gpu_smooth_flag
#endif
contains
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_smooth_fine(pst,ilevel,input_size,noct,output_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  integer::ilevel,noct

  integer::next_noct
  integer::nflag
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SMOOTH_FINE,pst%iUpper+1,input_size,output_size,ilevel)
     call r_smooth_fine(pst%pLower,ilevel,input_size,noct,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_noct)
     noct=noct+next_noct
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        call gpu_smooth_flag(pst%s, ilevel, nflag)
     else
        call smooth_fine(pst%s,ilevel,nflag)
     endif
#else
     call smooth_fine(pst%s,ilevel,nflag)
#endif
     noct=nflag
  endif

end subroutine r_smooth_fine
!############################################################
!############################################################
!############################################################
!############################################################
subroutine smooth_fine(s,ilevel,nflag)
  use amr_parameters, only: ndim, twotondim, twondim
  use ramses_commons, only: ramses_t
  use cache_commons
  use cache
  use marshal, only: pack_fetch_flag, unpack_fetch_flag
  use boundaries, only: init_bound_flag
  use nbors_utils
  implicit none
  type(ramses_t)::s
  integer::ilevel,nflag
  ! -------------------------------------------------------------------
  ! Dilatation operator.
  ! This routine makes one cell width cubic buffer around flag1 cells 
  ! at level ilevel by following these 3 steps:
  ! step 1: flag1 cells with at least 1 flag1 neighbors (if ndim > 0) 
  ! step 2: flag1 cells with at least 2 flag1 neighbors (if ndim > 1) 
  ! step 3: flag1 cells with at least 2 flag1 neighbors (if ndim > 2) 
  ! Array flag2 is used as temporary workspace.
  ! -------------------------------------------------------------------
  integer::ismooth,count_nbor,ig,in
  integer::igrid,idim,ind,i_nbor,igrid_nbor,icell_nbor
  integer,dimension(1:3)::n_nbor=(/1,2,2/)
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer,dimension(0:twondim)::igridn
  type(msg_int4)::dummy_int4

  integer,dimension(1:3,1:6),save::shift=reshape(&
       & (/-1,0,0,1,0,0,&
       &   0,-1,0,0,1,0,&
       &   0,0,-1,0,0,1/),(/3,6/))
  integer,dimension(1:8,1:6),save::ggg=reshape(&
       & (/1,0,1,0,1,0,1,0,&
       &   0,2,0,2,0,2,0,2,&
       &   3,3,0,0,3,3,0,0,&
       &   0,0,4,4,0,0,4,4,&
       &   5,5,5,5,0,0,0,0,&
       &   0,0,0,0,6,6,6,6/),(/8,6/))
  integer,dimension(1:8,1:6),save::hhh=reshape(&
       & (/2,1,4,3,6,5,8,7,&
       &   2,1,4,3,6,5,8,7,&
       &   3,4,1,2,7,8,5,6,&
       &   3,4,1,2,7,8,5,6,&
       &   5,6,7,8,1,2,3,4,&
       &   5,6,7,8,1,2,3,4/),(/8,6/))

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  hash_nbor(0)=ilevel

  ! Loop over steps
  do ismooth=1,ndim

     ! Initialize flag2 to 0
     do igrid=m%head(ilevel),m%tail(ilevel)
        do ind=1,twotondim
           m%flag2(ind,igrid)=0
        end do
     end do

     call open_cache(mdl, m, pack_size=storage_size(dummy_int4)/32, &
          pack=pack_fetch_flag, unpack=unpack_fetch_flag, &
          bound=init_bound_flag)

     ! Count neighbors and set flag2 accordingly
     do igrid=m%head(ilevel),m%tail(ilevel)

        ! Get neighboring octs
        igridn(0)=igrid
        do i_nbor=1,twondim
           hash_nbor(1:ndim)=m%grid(igrid)%ckey(1:ndim)+shift(1:ndim,i_nbor)
           ! Periodic boundary conditions
           do idim=1,ndim
              if(r%periodic(idim))then
                 if(hash_nbor(idim)< m%box_ckey_min(idim,ilevel))hash_nbor(idim)=m%box_ckey_max(idim,ilevel)-1
                 if(hash_nbor(idim)>=m%box_ckey_max(idim,ilevel))hash_nbor(idim)=m%box_ckey_min(idim,ilevel)
              endif
           enddo
           call get_grid(s,hash_nbor,igridn(i_nbor),flush_cache=.false.,fetch_cache=.true.,lock=.true.)
        end do

        ! Count neighbors and set flag2 accordingly
        do ind=1,twotondim
           count_nbor=0
           do in=1,twondim
              ig=ggg(ind,in)
              icell_nbor=hhh(ind,in)
              if(igridn(ig)>0)then
                 count_nbor=count_nbor+m%flag1(icell_nbor,igridn(ig))
              endif
           end do
           ! flag2 cell if necessary
           if(count_nbor>=n_nbor(ismooth))then
              m%flag2(ind,igrid)=1
           endif
        end do

        do i_nbor=1,twondim
           call unlock_cache(m,igridn(i_nbor))
        end do

     end do
     ! End loop over grids

    call close_cache(mdl)

     ! Set flag1=1 for cells with flag2=1
     do igrid=m%head(ilevel),m%tail(ilevel)
        do ind=1,twotondim
           if(m%flag1(ind,igrid)==1)m%flag2(ind,igrid)=0
        end do
        do ind=1,twotondim
           if(m%flag2(ind,igrid)==1)then
              m%flag1(ind,igrid)=1
              g%nflag=g%nflag+1
           endif
        end do
     end do

  end do
  ! End loop over steps

  nflag=g%nflag

  end associate

end subroutine smooth_fine
!############################################################
!############################################################
!############################################################
!############################################################
end module smooth_module

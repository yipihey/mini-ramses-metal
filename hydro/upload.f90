module upload_module
#ifdef _CUDA
  use gpu_runner, only: gpu_upload
#endif
contains
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine m_upload_fine(pst,ilevel)
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t)::pst
  integer::ilevel
  !--------------------------------------------------------------------
  ! This routine is the master procedure to upload HYDRO variables
  ! from level ilevel+1 to ilevel (averaging down or restriction).
  !--------------------------------------------------------------------
  integer::dummy

  if(ilevel==pst%s%r%nlevelmax)return
  if(pst%s%m%noct_tot(ilevel)==0)return
  if(pst%s%m%noct_tot(ilevel+1)==0)return
  if(pst%s%r%verbose)write(*,111)ilevel
111 format('   Entering upload_fine for level',i2)

  call r_upload_fine(pst,ilevel,1)

end subroutine m_upload_fine
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_upload_fine(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size

  integer::ilevel
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_UPLOAD_FINE,pst%iUpper+1,input_size,0,ilevel)
     call r_upload_fine(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        call gpu_upload(pst%s,ilevel)
     else
        call upload_fine(pst%s,ilevel)
     endif
#else
     call upload_fine(pst%s,ilevel)
#endif
  endif

end subroutine r_upload_fine
!###########################################################
!########################################################### 
!###########################################################
!###########################################################
subroutine upload_fine(s,ilevel)
  use mdl_module
  use hydro_parameters, only: nvar,nener
  use amr_parameters, only: ndim, twotondim
  use ramses_commons, only: ramses_t
  use nbors_utils
  use cache_commons
  use cache
  implicit none
  type(ramses_t)::s
  integer::ilevel
  !----------------------------------------------------------------------
  ! This routine performs a restriction operation (averaging down)
  ! for the hydro variables.
  !----------------------------------------------------------------------
#if NENER>0
  integer::irad
#endif
  integer::ioct,ind,ivar,icell,idim,igrid
  integer(kind=8),dimension(0:ndim)::hash_key
  integer,dimension(1:6,1:4)::hh
  real(kind=8)::average,ekin,erad,emag
  type(msg_realdp)::dummy_realdp

#ifdef HYDRO

  hh(1,1:4)=(/1,3,5,7/)
  hh(2,1:4)=(/2,4,6,8/)
  hh(3,1:4)=(/1,2,5,6/)
  hh(4,1:4)=(/3,4,7,8/)
  hh(5,1:4)=(/1,2,3,4/)
  hh(6,1:4)=(/5,6,7,8/)

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  ! Set conservative variable to zero in refined cells
  do ioct=m%head(ilevel),m%tail(ilevel)
     do ivar=1,nvar
        do ind=1,twotondim
           if(m%grid(ioct)%refined(ind))then
              m%uold(ind,ivar,ioct)=0.0
           endif
        end do
     end do
     ! Zero mflux for refined cells
#ifdef TRCFLX
     do ind=1,twotondim
        if(m%grid(ioct)%refined(ind))then
           m%mflux(ind,1:2*ndim+1,ioct)=0.0d0
        endif
     end do
#endif
#ifdef MHD
     do ivar=1,6
        do ind=1,twotondim
           if(m%grid(ioct)%refined(ind))then
              m%bold(ind,ivar,ioct)=0.0
           endif
        end do
     end do
#endif
  end do

  call open_cache(mdl, m, pack_size=storage_size(dummy_realdp)/32, &
       init=init_flush_upload, flush=pack_flush_upload, combine=unpack_flush_upload)

  ! Loop over finer level grids
  hash_key(0)=ilevel+1
  do ioct=m%head(ilevel+1),m%tail(ilevel+1)

     ! Get parent cell and grid index
     hash_key(1:ndim)=m%grid(ioct)%ckey(1:ndim)
     call get_parent_cell(s,hash_key,igrid,icell,flush_cache=.true.,fetch_cache=.false.)

     ! Average conservative variables
     do ivar=1,nvar
        average=0.0d0
        do ind=1,twotondim
           average=average+m%uold(ind,ivar,ioct)
        end do
        ! Scatter result to parent cell
        m%uold(icell,ivar,igrid)=average/dble(twotondim)
     end do

     ! Average cell-centered magnetic field
#ifdef MHD
#if NDIM<3
     ! Average cell-centered Bz
     average=0.0d0
     do ind=1,twotondim
        average=average+m%bold(ind,3,ioct)
     end do
     ! Scatter result to parent cell
     m%bold(icell,3,igrid)=average/dble(twotondim)
     m%bold(icell,6,igrid)=average/dble(twotondim)
#if NDIM==1
     ! Average cell-centered Bx and By
     do idim=1,2
        average=0.0d0
        do ind=1,twotondim
           average=average+m%bold(ind,idim,ioct)
        end do
        ! Scatter result to parent cell
        m%bold(icell,idim,igrid)=average/dble(twotondim)
        m%bold(icell,idim+3,igrid)=average/dble(twotondim)
     end do
#endif
#endif
#endif

     ! Average face-centered magnetic field
#ifdef MHD
#if NDIM>1
     ! Loop over dimensions
     do idim=1,ndim
        ! Left magnetic field in parent cell
        average=0.0d0
        do ind=1,twotondim/2
           average=average+m%bold(hh(2*idim-1,ind),idim,ioct)
        end do
        m%bold(icell,idim,igrid)=average/dble(twotondim/2)
        ! Right magnetic field in parent cell
        average=0.0d0
        do ind=1,twotondim/2
           average=average+m%bold(hh(2*idim,ind),idim+3,ioct)
        end do
        m%bold(icell,idim+3,igrid)=average/dble(twotondim/2)
     end do
#endif
#endif

     ! Average mflux for tracer particles
#ifdef TRCFLX
     ! Average old density (stored in mflux(:,1))
     average=0.0d0
     do ind=1,twotondim
        average=average+m%mflux(ind,1,ioct)
     end do
     m%mflux(icell,1,igrid)=average/dble(twotondim)

     ! Average face-centered time-integrated mass flux
     do idim=1,ndim
        ! Left face in parent cell
        average=0.0d0
        do ind=1,twotondim/2
           average=average+m%mflux(hh(2*idim-1,ind),1+idim,ioct)
        end do
        m%mflux(icell,1+idim,igrid)=average/dble(twotondim/2)
        ! Right face in parent cell
        average=0.0d0
        do ind=1,twotondim/2
           average=average+m%mflux(hh(2*idim,ind),1+idim+ndim,ioct)
        end do
        m%mflux(icell,1+idim+ndim,igrid)=average/dble(twotondim/2)
     end do
#endif
     ! Average internal energy instead of total energy
     if(r%interpol_var==1)then
        average=0.0d0
        do ind=1,twotondim
           ekin=0.0d0
           do idim=1,3
              ekin=ekin+0.5d0*m%uold(ind,idim+1,ioct)**2/max(dble(m%uold(ind,1,ioct)),r%smallr)
           end do
           emag=0.0d0
#ifdef MHD
           do idim=1,3
              emag=emag+0.125d0*(m%bold(ind,idim,ioct)+m%bold(ind,idim+3,ioct))**2
           end do
#endif
           erad=0.0d0
#if NENER>0
           do irad=1,nener
              erad=erad+m%uold(ind,5+irad,ioct)
           end do
#endif
           average=average+m%uold(ind,5,ioct)-ekin-erad-emag
        end do
        ! Scatter result to parent cell
        ekin=0.0d0
        do idim=1,3
           ekin=ekin+0.5d0*m%uold(icell,idim+1,igrid)**2/max(dble(m%uold(icell,1,igrid)),r%smallr)
        end do
        emag=0.0d0
#ifdef MHD
        do idim=1,3
           emag=emag+0.125d0*(m%bold(icell,idim,igrid)+m%bold(icell,idim+3,igrid))**2
        end do
#endif
        erad=0.0d0
#if NENER>0
        do irad=1,nener
           erad=erad+m%uold(icell,5+irad,igrid)
        end do
#endif
        m%uold(icell,5,igrid)=average/dble(twotondim)+ekin+erad+emag
     endif
  end do

  call close_cache(mdl)

  end associate

#endif

end subroutine upload_fine
!##########################################################################
!##########################################################################
!##########################################################################
!##########################################################################
subroutine init_flush_upload(mesh,igrid,hash_key)
  use amr_parameters, only: ndim, twotondim
  use hydro_parameters, only: nvar
  use amr_commons, only: mesh_t
  type(mesh_t)::mesh
  integer::igrid
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind,ivar

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)

#ifdef HYDRO
  do ivar=1,nvar
     do ind=1,twotondim
        mesh%uold(ind,ivar,igrid)=0.0d0
     end do
  end do
#ifdef TRCFLX
  mesh%mflux(:,:,igrid)=0.0d0
#endif
#endif

#ifdef MHD
  do ivar=1,6
     do ind=1,twotondim
        mesh%bold(ind,ivar,igrid)=0.0d0
     end do
  end do
#endif

end subroutine init_flush_upload
!##########################################################################
!##########################################################################
!##########################################################################
!##########################################################################
subroutine pack_flush_upload(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: twotondim
  use hydro_parameters, only: nvar
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind,ivar
  type(msg_realdp)::msg

#ifdef HYDRO
  do ivar=1,nvar
     do ind=1,twotondim
        msg%realdp(ind,ivar)=mesh%uold(ind,ivar,igrid)
     end do
  end do
#ifdef TRCFLX
  msg%realdp_mflux=mesh%mflux(:,:,igrid)
#endif
#endif

#ifdef MHD
  do ivar=1,6
     do ind=1,twotondim
        msg%realdp_mhd(ind,ivar)=mesh%bold(ind,ivar,igrid)
     end do
  end do
#endif

  msg_array=transfer(msg,msg_array)

end subroutine pack_flush_upload
!##########################################################################
!##########################################################################
!##########################################################################
!##########################################################################
subroutine unpack_flush_upload(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim, twotondim
  use hydro_parameters, only: nvar
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind,ivar
  type(msg_realdp)::msg

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)
  msg=transfer(msg_array,msg)

#ifdef HYDRO
  do ivar=1,nvar
     do ind=1,twotondim
        if(mesh%grid(igrid)%refined(ind))then
           mesh%uold(ind,ivar,igrid)=mesh%uold(ind,ivar,igrid)+msg%realdp(ind,ivar)
        endif
     end do
  end do
#ifdef TRCFLX
  do ind=1,twotondim
     if(mesh%grid(igrid)%refined(ind))then
        mesh%mflux(ind,:,igrid)=mesh%mflux(ind,:,igrid)+msg%realdp_mflux(ind,:)
     endif
  end do
#endif
#endif
#ifdef MHD
  do ivar=1,6
     do ind=1,twotondim
        if(mesh%grid(igrid)%refined(ind))then
           mesh%bold(ind,ivar,igrid)=mesh%bold(ind,ivar,igrid)+msg%realdp_mhd(ind,ivar)
        endif
     end do
  end do
#endif

end subroutine unpack_flush_upload
!##########################################################################
!##########################################################################
!##########################################################################
!##########################################################################
end module upload_module

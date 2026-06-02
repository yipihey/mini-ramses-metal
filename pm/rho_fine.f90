module rho_fine_module
#ifdef _CUDA
  use gpu_runner, only: gpu_multipole_leaf, gpu_multipole_split, gpu_reset_rho, gpu_cic_multipole, gpu_cic_multipole2
  use part_device, only: gpu_split_part, gpu_sort_part, gpu_cic_part
#endif
contains
!###############################################
!###############################################
!###############################################
!###############################################
subroutine m_rho_fine(pst,ilevel,rtype)
  use amr_parameters, only: ndim
  use ramses_commons, only: pst_t
  use amr_commons, only: multipole_t
  implicit none
  type(pst_t)::pst
  integer::ilevel
  integer::rtype ! rtype 0=all, 1=dm, 2=star, 3=sink, 4=gas
  !------------------------------------------------------------------
  ! This master routine computes the mass density field to be used
  ! as source term in the Poisson solver.
  ! The density field is computed for all levels greater than ilevel.
  ! On output, particles are sorted according to their grid level of
  ! refinement, and inside their level, they are sorted according to
  ! their grid Hilbert order.
  !------------------------------------------------------------------
  type(multipole_t)::multipole_tot
  integer::i,input_size
  integer,dimension(1:2)::input_array
  associate(r=>pst%s%r,g=>pst%s%g,m=>pst%s%m,p=>pst%s%p,mdl=>pst%s%mdl)

  if(m%noct_tot(ilevel)==0)return
  if(.not. r%poisson .and. .not. r%pic)return

  if(r%verbose)write(*,'("   Entering rho_fine for level ",I2)')ilevel

  !---------------------------
  ! Reset multipole to zero
  !---------------------------
#ifdef GRAV
  if(r%poisson)then
     if(ilevel==r%levelmin)then
        multipole_tot%q=0d0
        input_size=storage_size(multipole_tot)/32
        call r_broadcast_multipole(pst,multipole_tot,input_size)
     endif
     !-------------------------------------------------------
     ! Initialize rho to analytical and baryon density field
     !-------------------------------------------------------
     ! Loop over all finer levels from fine to coarse
     do i=r%nlevelmax,ilevel,-1

        ! Compute gas multipole expansion
        if(r%hydro)then

           ! Set multipoles in all leaf cells
           if(m%noct_tot(i)>0)then
              if(r%verbose)write(*,'(" Compute leaf multipoles for level ",I2)')i
              call r_multipole_leaf_cells(pst,i,1)
           endif

           ! Average down multipoles in all split cells
           if(i<r%nlevelmax)then
              if(m%noct_tot(i+1)>0)then
                 if(r%verbose)write(*,'(" Compute split multipoles for level ",I2)')i
                 call r_multipole_split_cells(pst,i,1)
              endif
           endif

        endif

        ! Reset array rho to zero
        if(m%noct_tot(i)>0)then
           call r_reset_rho(pst,i,1)
        endif

        ! Mass deposition into array rho using gas pseudo-particles
        if(r%hydro)then

           if(m%noct_tot(i)>0.AND.(rtype==0 .or. rtype==4))then
              if(r%verbose)write(*,'(" Compute rho from multipoles for level ",I2)')i
              call r_cic_multipole(pst,i,1)
           endif

        endif

     end do
  endif
  ! End loop over finer levels
#endif
  !-------------------------------------------------------
  ! Compute particle contribution to density field
  !-------------------------------------------------------
  if(r%pic)then

     ! For non periodic BC, split particles that left the box
     if(ilevel==r%levelmin.AND.ANY(.not.r%periodic(1:ndim)))then
        call r_split_part(pst,ilevel-1,1)
     endif

     ! Loop over all finer levels from coarse to fine
     do i=ilevel,r%nlevelmax

        ! Sort particles according to their Hilbert index
        if(m%noct_tot(i)>0)then
           if(r%verbose)write(*,'(" Sort particles for level ",I2)')i
           call r_sort_part(pst,i,1)
        endif

        ! Mass deposition into array rho using all massive particle types
#ifdef GRAV
        if(m%noct_tot(i)>0 .and. r%poisson)then
           if(r%verbose)write(*,'(" Compute rho from particles for level ",I2)')i
           input_array(1)=i
           input_array(2)=rtype
           call r_cic_part(pst,input_array,2)
        endif
#endif

        ! Sort particles between coarse and fine levels
        if(m%noct_tot(i)>0.AND.i<r%nlevelmax)then
           if(r%verbose)write(*,'(" Split particles for level ",I2)')i
           call r_split_part(pst,i,1)
        endif

     end do
  endif

  !---------------------------------------------------------------------
  ! Collect multipole contribution from all CPU and broadcast rho_tot
  !---------------------------------------------------------------------
#ifdef GRAV
  if(ilevel==r%levelmin .and. r%poisson)then

     ! Collect local multipole from all CPU
     call r_collect_multipole(pst,ilevel,1,multipole_tot,storage_size(multipole_tot)/32)

     ! Broadcast total multipole to all CPU
     call r_broadcast_multipole(pst,multipole_tot,storage_size(multipole_tot)/32)

     if(r%verbose)write(*,*)'rho_average=',g%rho_tot
  endif  
#endif

  end associate

end subroutine m_rho_fine
!################################################################
!################################################################
!################################################################
!################################################################
#ifdef GRAV
recursive subroutine r_multipole_leaf_cells(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_MULTIPOLE_LEAF_CELLS,pst%iUpper+1,input_size,0,ilevel)
     call r_multipole_leaf_cells(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        call gpu_multipole_leaf(pst%s, ilevel)
        return
     endif
#endif
     call multipole_leaf_cells(pst%s%r,pst%s%g,pst%s%m,ilevel)
  endif

end subroutine r_multipole_leaf_cells
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine multipole_leaf_cells(r,g,m,ilevel)
  use amr_parameters, only: ndim, twotondim, dp
  use amr_commons, only: run_t, global_t, mesh_t
  use cache_commons
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  integer::ilevel
  !-------------------------------------------------------------------
  ! This routine compute the monopole and dipole of the gas mass and
  ! the analytical profile (if any) within each cell.
  ! For pure particle runs, this is not necessary and the
  ! routine is not even called.
  !-------------------------------------------------------------------
  integer::igrid,ind,idim,ivar,nstride,icell
  real(kind=8),dimension(1:ndim)::xx
  real(kind=8)::dx_loc,vol_loc,mmm,dd
  logical::leaf_cell

  ! Mesh spacing in that level
  dx_loc=r%boxlen/2**ilevel
  vol_loc=dx_loc**ndim

#ifdef HYDRO
  ! Initialize multipole fields to zero
  do igrid=m%head(ilevel),m%tail(ilevel)
     do ind=1,twotondim
        do idim=1,ndim+1
           m%unew(ind,idim,igrid)=0.0D0
        end do
     end do
  end do
#endif

  !-------------------------------------------------------
  ! Compute contribution of leaf cells to mass multipoles
  !-------------------------------------------------------
  do igrid=m%head(ilevel),m%tail(ilevel)
     ! Loop over cells
     do ind=1,twotondim

        leaf_cell=m%grid(igrid)%refined(ind).EQV..FALSE.

        ! For leaf cells only
        if(leaf_cell)then

           ! Cell coordinates
           do idim=1,ndim
              nstride=2**(idim-1)
              xx(idim)=(2*m%grid(igrid)%ckey(idim)+MOD((ind-1)/nstride,2)+0.5)*dx_loc-m%skip(idim)
           end do
#ifdef HYDRO
           ! Add gas mass
           mmm=max(m%uold(ind,1,igrid),real(r%smallr,kind=dp))*vol_loc
           m%unew(ind,1,igrid)=m%unew(ind,1,igrid)+mmm
           do idim=1,ndim
              m%unew(ind,idim+1,igrid)=m%unew(ind,idim+1,igrid)+mmm*xx(idim)
           end do
#endif
           ! Add analytical density profile
           if(r%gravity_test)then
              call rho_ana(xx,dd,dx_loc,r%gravity_params)
              mmm=max(dd,r%smallr)*vol_loc
#ifdef HYDRO
              m%unew(ind,1,igrid)=m%unew(ind,1,igrid)+mmm
              do idim=1,ndim
                 m%unew(ind,idim+1,igrid)=m%unew(ind,idim+1,igrid)+mmm*xx(idim)
              end do
#endif
           end if
        endif
     end do
     ! End loop over cells
  end do
  ! End loop over grids

end subroutine multipole_leaf_cells
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_multipole_split_cells(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_MULTIPOLE_SPLIT_CELLS,pst%iUpper+1,input_size,0,ilevel)
     call r_multipole_split_cells(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        call gpu_multipole_split(pst%s, ilevel)
        return
     endif
#endif
     call multipole_split_cells(pst%s,ilevel)
  endif

end subroutine r_multipole_split_cells
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine multipole_split_cells(s,ilevel)
  use amr_parameters, only: ndim, twotondim
  use ramses_commons, only: ramses_t
  use nbors_utils
  use cache_commons
  use cache
  implicit none
  type(ramses_t)::s
  integer::ilevel
  !-------------------------------------------------------------------
  ! This routine compute the monopole and dipole of the gas mass and
  ! the analytical profile (if any) within each cell.
  ! For pure particle runs, this is not necessary and the
  ! routine is not even called.
  !-------------------------------------------------------------------
  integer::ind,idim,ivar,ioct,icell,igrid
  real(kind=8)::average
  integer(kind=8),dimension(0:ndim)::hash_key
  logical::leaf_cell
  type(msg_realdp)::dummy_realdp

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  !-------------------------------------------------------
  ! Perform Multigrid restriction from level ilevel+1
  !-------------------------------------------------------
  call open_cache(mdl, m, pack_size=storage_size(dummy_realdp)/32, &
       init=init_flush_multipole, flush=pack_flush_multipole, combine=unpack_flush_multipole)

  ! Loop over finer level grids
  hash_key(0)=ilevel+1
  do ioct=m%head(ilevel+1),m%tail(ilevel+1)
     hash_key(1:ndim)=m%grid(ioct)%ckey(1:ndim)
     ! Get parent cell using a write-only cache
     call get_parent_cell(s,hash_key,igrid,icell,flush_cache=.true.,fetch_cache=.false.)
#ifdef HYDRO
     ! Average conservative variables
     do ivar=1,ndim+1
        average=0.0d0
        do ind=1,twotondim
           average=average+m%unew(ind,ivar,ioct)
        end do
        ! Scatter result to cell
        m%unew(icell,ivar,igrid)=average
     end do
#endif
  end do

  call close_cache(mdl)

  end associate

end subroutine multipole_split_cells
!################################################################
!################################################################
!################################################################
!################################################################
subroutine init_flush_multipole(mesh,igrid,hash_key)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: mesh_t
  type(mesh_t)::mesh
  integer::igrid
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind,ivar

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)

#ifdef HYDRO
  do ivar=1,ndim+1
     do ind=1,twotondim
        mesh%unew(ind,ivar,igrid)=0.0
     end do
  end do
#endif

end subroutine init_flush_multipole
!################################################################
!################################################################
!################################################################
!################################################################
subroutine pack_flush_multipole(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind,ivar
  type(msg_realdp)::msg

#ifdef HYDRO
  do ivar=1,ndim+1
     do ind=1,twotondim
        msg%realdp(ind,ivar)=mesh%unew(ind,ivar,igrid)
     end do
  end do
#endif

  msg_array=transfer(msg,msg_array)

end subroutine pack_flush_multipole
!################################################################
!################################################################
!################################################################
!################################################################
subroutine unpack_flush_multipole(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim,twotondim
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
  do ivar=1,ndim+1
     do ind=1,twotondim
        if(mesh%grid(igrid)%refined(ind))then
           mesh%unew(ind,ivar,igrid)=mesh%unew(ind,ivar,igrid)+msg%realdp(ind,ivar)
        endif
     end do
  end do
#endif

end subroutine unpack_flush_multipole
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_reset_rho(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_RESET_RHO,pst%iUpper+1,input_size,0,ilevel)
     call r_reset_rho(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        call gpu_reset_rho(pst%s, ilevel)
        return
     endif
#endif
     call reset_rho(pst%s%r,pst%s%g,pst%s%m,ilevel)
  endif

end subroutine r_reset_rho
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine reset_rho(r,g,m,ilevel)
  use amr_parameters, only: twotondim
  use amr_commons, only: run_t,global_t,mesh_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  integer::ilevel
  !-------------------------------------------------------------------
  ! This routine compute array rho (source term for Poisson equation)
  ! by first reseting array rho to zero, then 
  ! by depositing the gas multipole mass in each cells using CIC.
  ! For pure particle runs, the gas mass deposition is not done
  ! and the routine only set rho to zero.
  !-------------------------------------------------------------------
  integer::igrid,ind

#ifdef GRAV
  ! Initialize density field to zero
  do igrid=m%head(ilevel),m%tail(ilevel)
     do ind=1,twotondim
        m%rho(ind,igrid)=0.0D0
        m%nref(ind,igrid)=0.0D0
     end do
  end do
#endif

end subroutine reset_rho
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_cic_multipole(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CIC_MULTIPOLE,pst%iUpper+1,input_size,0,ilevel)
     call r_cic_multipole(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        call gpu_cic_multipole2(pst%s, ilevel)
        return
     endif
#endif
     call cic_multipole(pst%s,ilevel)
  endif

end subroutine r_cic_multipole
!###########################################################
!###########################################################
!###########################################################
!###########################################################
subroutine cic_multipole(s,ilevel)
  use mdl_module
  use amr_parameters, only: ndim, twotondim
  use ramses_commons, only: ramses_t
  use nbors_utils
  use cache_commons
  use cache
  implicit none
  type(ramses_t)::s
  integer::ilevel
  !
  ! Local variables
  real(kind=8),dimension(1:ndim)::x,dd,dg
  integer,dimension(1:ndim)::ig,id
  real(kind=8),dimension(1:twotondim)::vol
  integer,dimension(1:ndim,1:twotondim)::ckey
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer::inbor,ioct,ind,idim,icell,igrid
  real(kind=8)::dx_loc,vol_loc,mmm,mask
  type(msg_twin_realdp)::dummy_twin_realdp

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)
    
  ! Mesh spacing in that level
  dx_loc=r%boxlen/2**ilevel
  vol_loc=dx_loc**ndim

  ! Use hash table directly for cells (not for grids)
  hash_nbor(0)=ilevel+1

  call open_cache(mdl, m, pack_size=storage_size(dummy_twin_realdp)/32, &
       init=init_flush_rho, flush=pack_flush_rho, combine=unpack_flush_rho)

  ! Loop over grids
  do ioct=m%head(ilevel),m%tail(ilevel)

     ! Loop over cells
     do ind=1,twotondim

#ifdef HYDRO        
        ! Compute pseudo particle mass
        mmm=m%unew(ind,1,ioct)

        ! Compute pseudo particle (centre of mass) position
        if(mmm==0)then
           write(*,*)'Sorry empty cell in cic_multipole'
           call mdl_abort(mdl)
        endif
        x(1:ndim)=m%unew(ind,2:ndim+1,ioct)/mmm

        ! Compute total multipole
        if(ilevel==r%levelmin)then
           do idim=1,ndim+1
              g%multipole%q(idim)=g%multipole%q(idim)+m%unew(ind,idim,ioct)
           end do
        endif
#endif
        ! Rescale particle position at level ilevel
        do idim=1,ndim
           x(idim)=(x(idim)+m%skip(idim))/dx_loc
        end do

        ! CIC at level ilevel (dd: right cloud boundary; dg: left cloud boundary)
        do idim=1,ndim
           dd(idim)=x(idim)+0.5D0
           id(idim)=int(dd(idim))
           dd(idim)=dd(idim)-id(idim)
           dg(idim)=1.0D0-dd(idim)
           ig(idim)=id(idim)-1
        end do

        ! Periodic boundary conditions
        do idim=1,ndim
           if(r%periodic(idim))then
              if(ig(idim)< m%box_ckey_min(idim,ilevel+1))ig(idim)=m%box_ckey_max(idim,ilevel+1)-1
              if(id(idim)>=m%box_ckey_max(idim,ilevel+1))id(idim)=m%box_ckey_min(idim,ilevel+1)
           endif
        enddo

        ! Compute cloud volumes
        vol = cic_weight(dg,dd)

        ! Compute cells Cartesian key
        ckey = cic_index(ig,id)

#ifdef GRAV
        ! Update mass density
        do inbor=1,twotondim
           hash_nbor(1:ndim)=ckey(1:ndim,inbor)
           ! Get parent cell using write-only cache
           call get_parent_cell(s,hash_nbor,igrid,icell,flush_cache=.true.,fetch_cache=.false.)
           if(igrid>0)then
              m%rho(icell,igrid)=m%rho(icell,igrid)+mmm*vol(inbor)/vol_loc
#ifdef HYDRO
              if(r%m_refine(ilevel)>=0)then
                 if(r%ivar_refine>0)then
                    mask=m%uold(ind,r%ivar_refine,ioct)/m%uold(ind,1,ioct)
                    if(mask.gt.r%var_cut_refine)then
                       m%nref(icell,igrid)=m%nref(icell,igrid)+mmm*vol(inbor)/r%mass_sph
                    endif
                 else
                    m%nref(icell,igrid)=m%nref(icell,igrid)+mmm*vol(inbor)/r%mass_sph
                 endif
              endif
#endif
           end if
        end do
#endif     
     end do
     ! End loop over cells

  end do
  ! End loop over grids

  call close_cache(mdl)

  end associate

end subroutine cic_multipole
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_cic_part(pst,input_array,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
#ifdef _CUDA
  use pm_parameters, only: PART_TYPE
#endif
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer,dimension(1:input_size)::input_array

  integer::rID
  integer::ilevel,rtype

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_CIC_PART,pst%iUpper+1,input_size,0,input_array)
     call r_cic_part(pst%pLower,input_array,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     ilevel=input_array(1)
     rtype=input_array(2)
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        if(pst%s%r%part)then
           if(pst%s%p%type/=PART_TYPE)then
              write(*,*)'ERROR: r_cic_part: DM PART_TYPE only.'
              call abort
           endif
           if(pst%s%r%part_mass_deposition_scheme/=1)then
              write(*,'(A,I0,A)')'ERROR: r_cic_part: CIC only (scheme ', &
                   & pst%s%r%part_mass_deposition_scheme,').'
              call abort
           endif
           call gpu_cic_part(pst%s, ilevel, rtype)
        endif
        if(pst%s%r%star.or.pst%s%r%sink)then
           write(*,*)'ERROR: r_cic_part: star/sink not supported on GPU.'
           call abort
        endif
        return
     endif
#endif
     ! Mass deposition for various components (DM particles, star, sink)
     ! based on their respective deposition schemes (CIC 1, TSC 2 or PCS 3)
     if(pst%s%r%part)then
        if(pst%s%r%part_mass_deposition_scheme==1)then
           call cic_part(pst%s,pst%s%p   ,ilevel,rtype)
        else if(pst%s%r%part_mass_deposition_scheme==2)then
           call tsc_part(pst%s,pst%s%p   ,ilevel,rtype)
        else if(pst%s%r%part_mass_deposition_scheme==3)then
           call pcs_part(pst%s,pst%s%p   ,ilevel,rtype)
        endif
     endif
     if(pst%s%r%star)then
        if(pst%s%r%star_mass_deposition_scheme==1)then
           call cic_part(pst%s,pst%s%star,ilevel,rtype)
        elseif(pst%s%r%star_mass_deposition_scheme==2)then
           call tsc_part(pst%s,pst%s%star,ilevel,rtype)
        elseif(pst%s%r%star_mass_deposition_scheme==3)then
           call pcs_part(pst%s,pst%s%star,ilevel,rtype)
        endif
     endif
     if(pst%s%r%sink)then
        if(pst%s%r%sink_mass_deposition_scheme==1)then
           call cic_part(pst%s,pst%s%sink,ilevel,rtype)
        elseif(pst%s%r%sink_mass_deposition_scheme==2)then
           call tsc_part(pst%s,pst%s%sink,ilevel,rtype)
        elseif(pst%s%r%sink_mass_deposition_scheme==3)then
           call pcs_part(pst%s,pst%s%sink,ilevel,rtype)
        endif
     endif
  endif

end subroutine r_cic_part
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
subroutine cic_part(s,p,ilevel,rtype)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: oct
  use ramses_commons, only: ramses_t
  use pm_parameters
  use pm_commons, only: part_t
  use nbors_utils
  use cache_commons
  use cache
  use multigrid_fine_coarse, only:pack_fetch_phi,unpack_fetch_phi
  use hilbert
  implicit none
  type(ramses_t)::s
  type(part_t)::p
  integer::ilevel,rtype
  !
  ! Local variables
  integer,dimension(1:ndim)::ir,il,ix
  real(kind=8)::dx_loc,vol_loc
  real(kind=8),dimension(1:ndim)::x,dr,dl
  real(kind=8),dimension(1:twotondim)::vol
  integer,dimension(1:ndim,1:twotondim)::ckey
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer::i,ipart,icell,igrid,ind,idim
  type(msg_twin_realdp)::dummy_twin_realdp
  logical::part,star,sink

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  ! Mesh spacing in that level
  dx_loc=r%boxlen/2**ilevel
  vol_loc=dx_loc**ndim

  ! Are particles dark matter, stars or sinks?
  part = p%type.eq.PART_TYPE
  star = p%type.eq.STAR_TYPE
  sink = p%type.eq.SINK_TYPE

  ! Don't deposit mass depending on rho action type and paticle type
  if(part.and.rtype.NE.0.and.rtype.NE.1)return
  if(star.and.rtype.NE.0.and.rtype.NE.2)return
  if(sink.and.rtype.NE.0.and.rtype.NE.3)return

  ! Compute contribution to multipole
  if(ilevel==r%levelmin)then
     do i=1,p%npart
        g%multipole%q(1)=g%multipole%q(1)+p%mp(i)
     end do
     do idim=1,ndim
        do i=1,p%npart
           g%multipole%q(idim+1)=g%multipole%q(idim+1)+p%mp(i)*p%xp(i,idim)
        end do
     end do
  endif

  ! Open write-only cache for array rho
  hash_nbor(0)=ilevel+1
  call open_cache(mdl, m, pack_size=storage_size(dummy_twin_realdp)/32, &
       init=init_flush_rho, flush=pack_flush_rho, combine=unpack_flush_rho)

  ! Loop over particles in Hilbert order
  do i=p%headp(ilevel),p%tailp(r%nlevelmax)
     ipart=p%sortp(i)

     ! Rescale particle position at level ilevel
     do idim=1,ndim
        x(idim)=(p%xp(ipart,idim)+m%skip(idim))/dx_loc
     end do

     ! CIC at level ilevel (dr: right cloud boundary; dl: left cloud boundary)
     do idim=1,ndim
        dr(idim)=x(idim)+0.5D0
        ir(idim)=int(dr(idim))
        dr(idim)=dr(idim)-ir(idim)
        dl(idim)=1.0D0-dr(idim)
        il(idim)=ir(idim)-1
     end do
     
     ! Periodic boundary conditions
     do idim=1,ndim
        if(r%periodic(idim))then
           if(il(idim)< m%box_ckey_min(idim,ilevel+1))il(idim)=m%box_ckey_max(idim,ilevel+1)-1
           if(ir(idim)>=m%box_ckey_max(idim,ilevel+1))ir(idim)=m%box_ckey_min(idim,ilevel+1)
        endif
     enddo

     ! Compute cloud volumes
     vol = cic_weight(dl,dr)

     ! Compute cells Cartesian key
     ckey = cic_index(il,ir)

#ifdef GRAV
     ! Update mass density
     do ind=1,twotondim
        hash_nbor(1:ndim)=ckey(1:ndim,ind)
        ! Get parent cell using write-only cache
        call get_parent_cell(s,hash_nbor,igrid,icell,flush_cache=.true.,fetch_cache=.false.)
        if(igrid>0)then
           m%rho(icell,igrid)=m%rho(icell,igrid)+p%mp(ipart)*vol(ind)/vol_loc
           if(r%m_refine(ilevel)>=0)then
              if(star.or.sink)then
                 m%nref(icell,igrid)=m%nref(icell,igrid)+p%mp(ipart)*vol(ind)/r%mass_sph
              endif
              if(part)then
                 if(r%mass_cut_refine>0)then
                    if(p%mp(ipart)<r%mass_cut_refine)then
                       m%nref(icell,igrid)=m%nref(icell,igrid)+vol(ind)
                    endif
                 else
                    m%nref(icell,igrid)=m%nref(icell,igrid)+vol(ind)
                 endif
              endif
           endif
        endif
     end do
#endif

  end do
  ! End loop over particles

  call close_cache(mdl)

  end associate

end subroutine cic_part
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
subroutine tsc_part(s,p,ilevel,rtype)
  use amr_parameters, only: ndim, twotondim, threetondim
  use ramses_commons, only: ramses_t
  use pm_parameters
  use pm_commons, only: part_t
  use nbors_utils
  use cache_commons
  use cache
  use hilbert
  implicit none
  type(ramses_t)::s
  type(part_t)::p
  integer::ilevel,rtype
  !
  ! Local variables
  integer,dimension(1:ndim)::ix,cl,cc,cr
  real(kind=8)::dx_loc,vol_loc,xl,xc,xr
  real(kind=8),dimension(1:ndim)::x,wl,wr,wc
  real(kind=8),dimension(1:threetondim)::vol
  integer,dimension(1:ndim,1:threetondim)::ckey
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer::i,ipart,icell,igrid,ind,idim
  type(msg_twin_realdp)::dummy_twin_realdp
  logical::part,star,sink

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  ! Mesh spacing in that level
  dx_loc=r%boxlen/2**ilevel
  vol_loc=dx_loc**ndim

  ! Are particles dark matter, stars or sinks?
  part = p%type.eq.PART_TYPE
  star = p%type.eq.STAR_TYPE
  sink = p%type.eq.SINK_TYPE

  ! Don't deposit mass depending on rho action type and paticle type
  if(part.and.rtype.NE.0.and.rtype.NE.1)return
  if(star.and.rtype.NE.0.and.rtype.NE.2)return
  if(sink.and.rtype.NE.0.and.rtype.NE.3)return

  ! Compute contribution to multipole
  if(ilevel==r%levelmin)then
     do i=1,p%npart
        g%multipole%q(1)=g%multipole%q(1)+p%mp(i)
     end do
     do idim=1,ndim
        do i=1,p%npart
           g%multipole%q(idim+1)=g%multipole%q(idim+1)+p%mp(i)*p%xp(i,idim)
        end do
     end do
  endif

  ! Open write-only cache for array rho
  hash_nbor(0)=ilevel+1
  call open_cache(mdl, m, pack_size=storage_size(dummy_twin_realdp)/32, &
       init=init_flush_rho, flush=pack_flush_rho, combine=unpack_flush_rho)

  ! Loop over particles in Hilbert order
  do i=p%headp(ilevel),p%tailp(r%nlevelmax)
     ipart=p%sortp(i)

     ! Rescale particle position at level ilevel
     do idim=1,ndim
        x(idim)=(p%xp(ipart,idim)+m%skip(idim))/dx_loc
     end do

     ! TSC at level ilevel; a particle contributes to 3 cells in each direction
     do idim=1,ndim
        cl(idim)=int(x(idim))-1 ! cell index
        cc(idim)=int(x(idim))
        cr(idim)=int(x(idim))+1
        xl=dble(cl(idim))+0.5D0 ! cell coordinate
        xc=dble(cc(idim))+0.5D0
        xr=dble(cr(idim))+0.5D0
        wl(idim)=0.5D0*(1.5D0-abs(x(idim)-xl))**2 ! weight
        wc(idim)=0.75D0-         (x(idim)-xc) **2
        wr(idim)=0.5D0*(1.5D0-abs(x(idim)-xr))**2
     end do

     ! Periodic boundary conditions
     do idim=1,ndim
        if(r%periodic(idim))then
           if(cl(idim)< m%box_ckey_min(idim,ilevel+1))cl(idim)=m%box_ckey_max(idim,ilevel+1)-1
           if(cr(idim)>=m%box_ckey_max(idim,ilevel+1))cr(idim)=m%box_ckey_min(idim,ilevel+1)
        endif
     enddo

     ! Compute cloud volumes
     vol = tsc_weight(wl,wc,wr)

     ! Compute cells Cartesian key
     ckey = tsc_index(cl,cc,cr)

#ifdef GRAV
     ! Update mass density
     do ind=1,threetondim
        hash_nbor(1:ndim)=ckey(1:ndim,ind)
        ! Get parent cell using write-only cache
        call get_parent_cell(s,hash_nbor,igrid,icell,flush_cache=.true.,fetch_cache=.false.)
        if(igrid>0)then
           m%rho(icell,igrid)=m%rho(icell,igrid)+p%mp(ipart)*vol(ind)/vol_loc
           if(r%m_refine(ilevel)>=0)then
              if(star.or.sink)then
                 m%nref(icell,igrid)=m%nref(icell,igrid)+p%mp(ipart)*vol(ind)/r%mass_sph
              endif
              if(part)then
                 if(r%mass_cut_refine>0)then
                    if(p%mp(ipart)<r%mass_cut_refine)then
                       m%nref(icell,igrid)=m%nref(icell,igrid)+vol(ind)
                    endif
                 else
                    m%nref(icell,igrid)=m%nref(icell,igrid)+vol(ind)
                 endif
              endif
           endif
        endif
     end do
#endif

  end do
  ! End loop over particles

  call close_cache(mdl)

  end associate

end subroutine tsc_part
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
subroutine pcs_part(s,p,ilevel,rtype)
  use amr_parameters, only: ndim, twotondim, fourtondim
  use ramses_commons, only: ramses_t
  use pm_parameters
  use pm_commons, only: part_t
  use nbors_utils
  use cache_commons
  use cache
  use hilbert
  implicit none
  type(ramses_t)::s
  type(part_t)::p
  integer::ilevel,rtype
  !
  ! Local variables
  integer,dimension(1:ndim)::ix,cll,cl,cr,crr
  real(kind=8)::dx_loc,vol_loc,xll,xl,xr,xrr
  real(kind=8),dimension(1:ndim)::x,wll,wl,wr,wrr
  real(kind=8),dimension(1:fourtondim)::vol
  integer,dimension(1:ndim,1:fourtondim)::ckey
  integer(kind=8),dimension(0:ndim)::hash_nbor
  integer::i,ipart,icell,igrid,ind,idim
  type(msg_twin_realdp)::dummy_twin_realdp
  logical::part,star,sink

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  ! Mesh spacing in that level
  dx_loc=r%boxlen/2**ilevel
  vol_loc=dx_loc**ndim

  ! Are particles dark matter, stars or sinks?
  part = p%type.eq.PART_TYPE
  star = p%type.eq.STAR_TYPE
  sink = p%type.eq.SINK_TYPE

  ! Don't deposit mass depending on rho action type and paticle type
  if(part.and.rtype.NE.0.and.rtype.NE.1)return
  if(star.and.rtype.NE.0.and.rtype.NE.2)return
  if(sink.and.rtype.NE.0.and.rtype.NE.3)return

  ! Compute contribution to multipole
  if(ilevel==r%levelmin)then
     do i=1,p%npart
        g%multipole%q(1)=g%multipole%q(1)+p%mp(i)
     end do
     do idim=1,ndim
        do i=1,p%npart
           g%multipole%q(idim+1)=g%multipole%q(idim+1)+p%mp(i)*p%xp(i,idim)
        end do
     end do
  endif

  ! Open write-only cache for array rho
  hash_nbor(0)=ilevel+1
  call open_cache(mdl, m, pack_size=storage_size(dummy_twin_realdp)/32, &
       init=init_flush_rho, flush=pack_flush_rho, combine=unpack_flush_rho)

  ! Loop over particles in Hilbert order
  do i=p%headp(ilevel),p%tailp(r%nlevelmax)
     ipart=p%sortp(i)

     ! Rescale particle position at level ilevel
     do idim=1,ndim
        x(idim)=(p%xp(ipart,idim)+m%skip(idim))/dx_loc
     end do

     ! PCS at level ilevel; a particle contributes to 4 cells in each direction
     do idim=1,ndim
        crr(idim)=int(x(idim)+1.5D0) ! rightermost cell index
        cr (idim)=crr(idim)-1
        cl (idim)=crr(idim)-2
        cll(idim)=crr(idim)-3
        xll=dble(cll(idim))+0.5D0 ! cell coordinate
        xl =dble(cl (idim))+0.5D0
        xr =dble(cr (idim))+0.5D0
        xrr=dble(crr(idim))+0.5D0
        wll(idim)=(2D0                        -abs(x(idim)-xll))**3/6D0 ! weight
        wl (idim)=(4D0-6D0*(x(idim)-xl)**2+3d0*abs(x(idim)-xl )**3)/6D0
        wr (idim)=(4D0-6D0*(x(idim)-xr)**2+3d0*abs(x(idim)-xr )**3)/6D0
        wrr(idim)=(2D0                        -abs(x(idim)-xrr))**3/6D0
     end do

     ! Periodic boundary conditions
     do idim=1,ndim
        if(r%periodic(idim))then
           if(cll(idim)< m%box_ckey_min(idim,ilevel+1))cll(idim)=cll(idim)-m%box_ckey_min(idim,ilevel+1)+m%box_ckey_max(idim,ilevel+1)
           if(cl (idim)< m%box_ckey_min(idim,ilevel+1))cl (idim)=cl (idim)-m%box_ckey_min(idim,ilevel+1)+m%box_ckey_max(idim,ilevel+1)
           if(cr (idim)>=m%box_ckey_max(idim,ilevel+1))cr (idim)=cr (idim)+m%box_ckey_min(idim,ilevel+1)-m%box_ckey_max(idim,ilevel+1)
           if(crr(idim)>=m%box_ckey_max(idim,ilevel+1))crr(idim)=crr(idim)+m%box_ckey_min(idim,ilevel+1)-m%box_ckey_max(idim,ilevel+1)
        endif
     enddo

     ! Compute cloud volumes
     vol = pcs_weight(wll,wl,wr,wrr)

     ! Compute cells Cartesian key
     ckey = pcs_index(cll,cl,cr,crr)

#ifdef GRAV
     ! Update mass density
     do ind=1,fourtondim
        hash_nbor(1:ndim)=ckey(1:ndim,ind)
        ! Get parent cell using write-only cache
        call get_parent_cell(s,hash_nbor,igrid,icell,flush_cache=.true.,fetch_cache=.false.)
        if(igrid>0)then
           m%rho(icell,igrid)=m%rho(icell,igrid)+p%mp(ipart)*vol(ind)/vol_loc
           if(r%m_refine(ilevel)>=0)then
              if(star.or.sink)then
                 m%nref(icell,igrid)=m%nref(icell,igrid)+p%mp(ipart)*vol(ind)/r%mass_sph
              endif
              if(part)then
                 if(r%mass_cut_refine>0)then
                    if(p%mp(ipart)<r%mass_cut_refine)then
                       m%nref(icell,igrid)=m%nref(icell,igrid)+vol(ind)
                    endif
                 else
                    m%nref(icell,igrid)=m%nref(icell,igrid)+vol(ind)
                 endif
              endif
           endif
        endif
     end do
#endif

  end do
  ! End loop over particles

  call close_cache(mdl)

  end associate

end subroutine pcs_part
!################################################################
!################################################################
!################################################################
!################################################################
subroutine init_flush_rho(mesh,igrid,hash_key)
  use amr_parameters, only: ndim,twotondim
  use amr_commons, only: mesh_t
  type(mesh_t)::mesh
  integer::igrid
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)

#ifdef GRAV
  do ind=1,twotondim
     mesh%rho(ind,igrid)=0.0
     mesh%nref(ind,igrid)=0.0
  end do
#endif

end subroutine init_flush_rho
!################################################################
!################################################################
!################################################################
!################################################################
subroutine pack_flush_rho(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_twin_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind
  type(msg_twin_realdp)::msg

#ifdef GRAV
  do ind=1,twotondim
     msg%realdp_phi(ind)=mesh%rho(ind,igrid)
     msg%realdp_dis(ind)=mesh%nref(ind,igrid)
  end do
#endif

  msg_array=transfer(msg,msg_array)

end subroutine pack_flush_rho
!################################################################
!################################################################
!################################################################
!################################################################
subroutine unpack_flush_rho(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_twin_realdp
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind
  type(msg_twin_realdp)::msg

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)
  msg=transfer(msg_array,msg)
  
#ifdef GRAV
  do ind=1,twotondim
     mesh%rho(ind,igrid)=mesh%rho(ind,igrid)+msg%realdp_phi(ind)
     mesh%nref(ind,igrid)=mesh%nref(ind,igrid)+msg%realdp_dis(ind)
  end do
#endif

end subroutine unpack_flush_rho
#endif
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_split_part(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SPLIT_PART,pst%iUpper+1,input_size,0,ilevel)
     call r_split_part(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        if(pst%s%r%part)call gpu_split_part(pst%s, ilevel)
        if(pst%s%r%star.or.pst%s%r%sink.or.pst%s%r%tree.or.pst%s%r%trac)then
           write(*,*)'ERROR: r_split_part: non-DM not supported on GPU.'
           call abort
        endif
        return
     endif
#endif
     if(pst%s%r%part)call split_part(pst%s,pst%s%p   ,ilevel)
     if(pst%s%r%star)call split_part(pst%s,pst%s%star,ilevel)
     if(pst%s%r%sink)call split_part(pst%s,pst%s%sink,ilevel)
     if(pst%s%r%tree)call split_part(pst%s,pst%s%tree,ilevel)
     if(pst%s%r%trac)call split_part(pst%s,pst%s%trac,ilevel)
  endif

end subroutine r_split_part
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
subroutine split_part(s,p,ilevel)
  use amr_parameters, only: ndim, twotondim, i8b
  use amr_commons, only: oct
  use ramses_commons, only: ramses_t
  use pm_commons, only: part_t
  use nbors_utils
  use cache_commons
  use hilbert
  use cache
  implicit none
  type(ramses_t)::s
  type(part_t)::p
  integer::ilevel
  !
  ! Local variables
  real(kind=8),dimension(1:ndim)::x,xp_tmp,vp_tmp,fp_tmp,jp_tmp
  integer,dimension(1:ndim)::ii,ix,ix_ref
  integer(kind=8),dimension(0:ndim)::hash_key
  integer::i,ipart,jpart,idim,icell,igrid,ilev
  integer::npart_coarse,npart_fine
  real(kind=8)::dx_loc,vol_loc
  real(kind=8)::mp_tmp
  integer::levelp_tmp
  integer(i8b)::idp_tmp
  type(msg_int4)::dummy_int4
  logical::in_domain

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

  if(ilevel==r%levelmin-1)then

     ! Loop over particles
     npart_coarse=0
     do ipart=p%headp(ilevel),p%tailp(r%nlevelmax)
        p%sortp(ipart)=ipart
        in_domain=.true.
        do idim=1,ndim
           if(.not. r%periodic(idim))then
              if(p%xp(ipart,idim)<0.0d0.or.p%xp(ipart,idim)>=r%box_size(idim))in_domain=.false.
           endif
        end do
        if(.not. in_domain)then
           npart_coarse=npart_coarse+1
           p%levelp(ipart)=-p%levelp(ipart)
        else
           p%sortp(ipart)=-p%sortp(ipart)
        endif
     end do
     ! End loop over particles

  else

     ! Mesh spacing in that level
     dx_loc=r%boxlen/2**ilevel
     vol_loc=dx_loc**ndim

     ! Open read-only cache for array refined
     hash_key(0)=ilevel
     call open_cache(mdl, m, pack_size=storage_size(dummy_int4)/32, &
          pack=pack_fetch_split,unpack=unpack_fetch_split)

     ! Loop over particles
     ix_ref=-1
     npart_coarse=0
     do i=p%headp(ilevel),p%tailp(r%nlevelmax)
        ipart=p%sortp(i)

        ! Acquire grid using read-only cache
        ix = int((p%xp(ipart,1:ndim)+m%skip(1:ndim))/(2*dx_loc))
        if(.NOT. ALL(ix.EQ.ix_ref))then
           hash_key(1:ndim)=ix(1:ndim)
           call get_grid(s,hash_key,igrid,flush_cache=.false.,fetch_cache=.true.)
           ix_ref=ix
        endif

        ! If particle sits outside current level,
        ! then it is clearly not in a refined cell.
        ! This can happen during second adaptive step
        if(igrid==0)then
           npart_coarse=npart_coarse+1
           p%levelp(ipart)=-p%levelp(ipart)
        else
           ! Rescale particle position at level ilevel
           do idim=1,ndim
              x(idim)=(p%xp(ipart,idim)+m%skip(idim))/dx_loc
           end do

           ! Shift particle position to to 2x2x2 grid corner
           do idim=1,ndim
              ii(idim)=int(x(idim)-2*ix_ref(idim))
           end do

           ! Compute parent cell index
#if NDIM==1
           icell=1+ii(1)
#endif
#if NDIM==2
           icell=1+ii(1)+2*ii(2)
#endif
#if NDIM==3
           icell=1+ii(1)+2*ii(2)+4*ii(3)
#endif
           ! Increase counter if cell is not refined
           if(.NOT.m%grid(igrid)%refined(icell))then
              npart_coarse=npart_coarse+1
              p%levelp(ipart)=-p%levelp(ipart)
           else
              p%sortp(i)=-p%sortp(i)
           endif
        endif

     end do
     ! End loop over particles

     call close_cache(mdl)

  endif

  p%tailp(ilevel)=p%headp(ilevel)+npart_coarse-1
  do ilev=ilevel+1,r%nlevelmax
     p%headp(ilev)=p%tailp(ilevel)+1
     p%tailp(ilev)=p%npart
  end do

  ! Loop over fine level particles
  ! This preserves the initial ordering after partioning
  npart_fine=0
  do ipart=p%headp(ilevel),p%tailp(r%nlevelmax)
     if(p%levelp(ipart)>0)then
        npart_fine=npart_fine+1
        p%workp(ipart)=p%headp(ilevel+1)+npart_fine-1
     endif
  end do

  ! Loop over coarse level particles
  ! This enforces Hilbert ordering after partioning
  npart_coarse=0
  do i=p%headp(ilevel),p%tailp(r%nlevelmax)
     ipart=p%sortp(i)
     if(ipart>0)then
        npart_coarse=npart_coarse+1
        p%workp(ipart)=p%headp(ilevel)+npart_coarse-1
        p%levelp(ipart)=-p%levelp(ipart)
     endif
  end do

  ! Swap particles using new index table
  do ipart=p%headp(ilevel),p%tailp(r%nlevelmax)
     do while(p%workp(ipart).NE.ipart)
        ! Swap new index
        jpart=p%workp(ipart)
        p%workp(ipart)=p%workp(jpart)
        p%workp(jpart)=jpart
        ! Swap positions
        xp_tmp(1:ndim)=p%xp(ipart,1:ndim)
        p%xp(ipart,1:ndim)=p%xp(jpart,1:ndim)
        p%xp(jpart,1:ndim)=xp_tmp(1:ndim)
        ! Swap velocities
        vp_tmp(1:ndim)=p%vp(ipart,1:ndim)
        p%vp(ipart,1:ndim)=p%vp(jpart,1:ndim)
        p%vp(jpart,1:ndim)=vp_tmp(1:ndim)
        ! Swap masses
        mp_tmp=p%mp(ipart)
        p%mp(ipart)=p%mp(jpart)
        p%mp(jpart)=mp_tmp
        ! Swap metallicity
        if(allocated(p%zp))then
           mp_tmp=p%zp(ipart)
           p%zp(ipart)=p%zp(jpart)
           p%zp(jpart)=mp_tmp
        endif
        ! Swap acceleration
        if(allocated(p%fp))then
           fp_tmp(1:ndim)=p%fp(ipart,1:ndim)
           p%fp(ipart,1:ndim)=p%fp(jpart,1:ndim)
           p%fp(jpart,1:ndim)=fp_tmp(1:ndim)
        endif
        ! Swap angular momentum
        if(allocated(p%jp))then
           jp_tmp(1:ndim)=p%jp(ipart,1:ndim)
           p%jp(ipart,1:ndim)=p%jp(jpart,1:ndim)
           p%jp(jpart,1:ndim)=jp_tmp(1:ndim)
        endif
        ! Swap age
        if(allocated(p%tp))then
           mp_tmp=p%tp(ipart)
           p%tp(ipart)=p%tp(jpart)
           p%tp(jpart)=mp_tmp
        endif
        ! Swap merging age
        if(allocated(p%tm))then
           mp_tmp=p%tm(ipart)
           p%tm(ipart)=p%tm(jpart)
           p%tm(jpart)=mp_tmp
        endif
        ! Swap potential (per-particle; must move with idp so the
        ! gathered potential stays attached to its particle).
        if(allocated(p%phip))then
           mp_tmp=p%phip(ipart)
           p%phip(ipart)=p%phip(jpart)
           p%phip(jpart)=mp_tmp
        endif
        ! Swap levels
        levelp_tmp=p%levelp(ipart)
        p%levelp(ipart)=p%levelp(jpart)
        p%levelp(jpart)=levelp_tmp
        ! Swap ids
        idp_tmp=p%idp(ipart)
        p%idp(ipart)=p%idp(jpart)
        p%idp(jpart)=idp_tmp
        ! Swap merging ids
        if(allocated(p%idm))then
           idp_tmp=p%idm(ipart)
           p%idm(ipart)=p%idm(jpart)
           p%idm(jpart)=idp_tmp
        endif
        ! Swap tracking ids
        if(allocated(p%idt))then
           idp_tmp=p%idt(ipart)
           p%idt(ipart)=p%idt(jpart)
           p%idt(jpart)=idp_tmp
        endif
        if(allocated(p%size))then
           mp_tmp=p%size(ipart)
           p%size(ipart)=p%size(jpart)
           p%size(jpart)=mp_tmp
        endif
        if(allocated(p%charge))then
           mp_tmp=p%charge(ipart)
           p%charge(ipart)=p%charge(jpart)
           p%charge(jpart)=mp_tmp
        endif
     end do
  end do

  end associate

end subroutine split_part
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
subroutine pack_fetch_split(mesh,igrid,msg_size,msg_array)
  use amr_parameters, only: twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_int4
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array

  integer::ind
  type(msg_int4)::msg

  do ind=1,twotondim
     if(mesh%grid(igrid)%refined(ind))then
        msg%int4(ind)=1
     else
        msg%int4(ind)=0
     endif
  enddo
  msg_array=transfer(msg,msg_array)
  
end subroutine pack_fetch_split
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
subroutine unpack_fetch_split(mesh,igrid,msg_size,msg_array,hash_key)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: mesh_t
  use cache_commons, only: msg_int4
  type(mesh_t)::mesh
  integer::igrid
  integer::msg_size
  integer,dimension(1:msg_size),optional::msg_array
  integer(kind=8),dimension(0:ndim)::hash_key

  integer::ind
  type(msg_int4)::msg

  mesh%grid(igrid)%lev=hash_key(0)
  mesh%grid(igrid)%ckey(1:ndim)=hash_key(1:ndim)
  msg=transfer(msg_array,msg)

  do ind=1,twotondim
     if(msg%int4(ind)==1)then
        mesh%grid(igrid)%refined(ind)=.true.
     else
        mesh%grid(igrid)%refined(ind)=.false.
     endif
  enddo

end subroutine unpack_fetch_split
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
#ifdef GRAV
recursive subroutine r_collect_multipole(pst,ilevel,input_size,multipole,output_size)
  use mdl_module
  use amr_parameters, only: ndim
  use ramses_commons, only: pst_t
  use amr_commons, only: multipole_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  integer::ilevel
  type(multipole_t)::multipole,next_multipole

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_COLLECT_MULTIPOLE,pst%iUpper+1,input_size,output_size,ilevel)
     call r_collect_multipole(pst%pLower,ilevel,input_size,multipole,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_multipole)
     multipole%q = multipole%q+next_multipole%q
  else
     multipole%q = pst%s%g%multipole%q
  endif

end subroutine r_collect_multipole
!###############################################
!###############################################
!###############################################
!###############################################
recursive subroutine r_broadcast_multipole(pst,multipole,input_size)
  use mdl_module
  use amr_parameters, only: ndim
  use ramses_commons, only: pst_t
  use amr_commons, only: multipole_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  type(multipole_t)::multipole

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_BROADCAST_MULTIPOLE,pst%iUpper+1,input_size,0,multipole)
     call r_broadcast_multipole(pst%pLower,multipole,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
     pst%s%g%multipole=multipole
     pst%s%g%rho_tot=pst%s%g%multipole%q(1)/PRODUCT(pst%s%r%box_size(1:ndim))
  endif

end subroutine r_broadcast_multipole
#endif
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
function cic_weight(dl,dr)
  use amr_parameters, only: ndim, twotondim
  real(kind=8),dimension(1:twotondim)::cic_weight
  real(kind=8),dimension(1:ndim)::dl,dr
#if NDIM==1
  cic_weight(1)=dl(1)
  cic_weight(2)=dr(1)
#endif
#if NDIM==2
  cic_weight(1)=dl(1)*dl(2)
  cic_weight(2)=dr(1)*dl(2)
  cic_weight(3)=dl(1)*dr(2)
  cic_weight(4)=dr(1)*dr(2)
#endif
#if NDIM==3
  cic_weight(1)=dl(1)*dl(2)*dl(3)
  cic_weight(2)=dr(1)*dl(2)*dl(3)
  cic_weight(3)=dl(1)*dr(2)*dl(3)
  cic_weight(4)=dr(1)*dr(2)*dl(3)
  cic_weight(5)=dl(1)*dl(2)*dr(3)
  cic_weight(6)=dr(1)*dl(2)*dr(3)
  cic_weight(7)=dl(1)*dr(2)*dr(3)
  cic_weight(8)=dr(1)*dr(2)*dr(3)
#endif
end function cic_weight
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
function cic_index(il,ir)
  use amr_parameters, only: ndim, twotondim
  integer,dimension(1:ndim,1:twotondim)::cic_index
  integer,dimension(1:ndim)::il,ir
#if NDIM==1
  cic_index(1,1)=il(1)
  cic_index(1,2)=ir(1)
#endif
#if NDIM==2
  cic_index(1:2,1)=(/il(1),il(2)/)
  cic_index(1:2,2)=(/ir(1),il(2)/)
  cic_index(1:2,3)=(/il(1),ir(2)/)
  cic_index(1:2,4)=(/ir(1),ir(2)/)
#endif
#if NDIM==3
  cic_index(1:3,1)=(/il(1),il(2),il(3)/)
  cic_index(1:3,2)=(/ir(1),il(2),il(3)/)
  cic_index(1:3,3)=(/il(1),ir(2),il(3)/)
  cic_index(1:3,4)=(/ir(1),ir(2),il(3)/)
  cic_index(1:3,5)=(/il(1),il(2),ir(3)/)
  cic_index(1:3,6)=(/ir(1),il(2),ir(3)/)
  cic_index(1:3,7)=(/il(1),ir(2),ir(3)/)
  cic_index(1:3,8)=(/ir(1),ir(2),ir(3)/)
#endif
end function cic_index
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
function tsc_weight(wl,wc,wr)
  use amr_parameters, only: ndim, threetondim
  real(kind=8),dimension(1:threetondim)::tsc_weight
  real(kind=8),dimension(1:ndim)::wl,wc,wr
#if NDIM==1
  tsc_weight(1)=wl(1)
  tsc_weight(2)=wc(1)
  tsc_weight(3)=wr(1)
#endif
#if NDIM==2
  tsc_weight(1)=wl(1)*wl(2)
  tsc_weight(2)=wc(1)*wl(2)
  tsc_weight(3)=wr(1)*wl(2)
  tsc_weight(4)=wl(1)*wc(2)
  tsc_weight(5)=wc(1)*wc(2)
  tsc_weight(6)=wr(1)*wc(2)
  tsc_weight(7)=wl(1)*wr(2)
  tsc_weight(8)=wc(1)*wr(2)
  tsc_weight(9)=wr(1)*wr(2)
#endif
#if NDIM==3
  tsc_weight(1) =wl(1)*wl(2)*wl(3)
  tsc_weight(2) =wc(1)*wl(2)*wl(3)
  tsc_weight(3) =wr(1)*wl(2)*wl(3)
  tsc_weight(4) =wl(1)*wc(2)*wl(3)
  tsc_weight(5) =wc(1)*wc(2)*wl(3)
  tsc_weight(6) =wr(1)*wc(2)*wl(3)
  tsc_weight(7) =wl(1)*wr(2)*wl(3)
  tsc_weight(8) =wc(1)*wr(2)*wl(3)
  tsc_weight(9) =wr(1)*wr(2)*wl(3)
  tsc_weight(10)=wl(1)*wl(2)*wc(3)
  tsc_weight(11)=wc(1)*wl(2)*wc(3)
  tsc_weight(12)=wr(1)*wl(2)*wc(3)
  tsc_weight(13)=wl(1)*wc(2)*wc(3)
  tsc_weight(14)=wc(1)*wc(2)*wc(3)
  tsc_weight(15)=wr(1)*wc(2)*wc(3)
  tsc_weight(16)=wl(1)*wr(2)*wc(3)
  tsc_weight(17)=wc(1)*wr(2)*wc(3)
  tsc_weight(18)=wr(1)*wr(2)*wc(3)
  tsc_weight(19)=wl(1)*wl(2)*wr(3)
  tsc_weight(20)=wc(1)*wl(2)*wr(3)
  tsc_weight(21)=wr(1)*wl(2)*wr(3)
  tsc_weight(22)=wl(1)*wc(2)*wr(3)
  tsc_weight(23)=wc(1)*wc(2)*wr(3)
  tsc_weight(24)=wr(1)*wc(2)*wr(3)
  tsc_weight(25)=wl(1)*wr(2)*wr(3)
  tsc_weight(26)=wc(1)*wr(2)*wr(3)
  tsc_weight(27)=wr(1)*wr(2)*wr(3)
#endif
end function tsc_weight
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
function tsc_index(cl,cc,cr)
  use amr_parameters, only: ndim, threetondim
  integer,dimension(1:ndim,1:threetondim)::tsc_index
  integer,dimension(1:ndim)::cl,cc,cr
#if NDIM==1
  tsc_index(1,1)=cl(1)
  tsc_index(1,2)=cc(1)
  tsc_index(1,3)=cr(1)
#endif
#if NDIM==2
  tsc_index(1:2,1)=(/cl(1),cl(2)/)
  tsc_index(1:2,2)=(/cc(1),cl(2)/)
  tsc_index(1:2,3)=(/cr(1),cl(2)/)
  tsc_index(1:2,4)=(/cl(1),cc(2)/)
  tsc_index(1:2,5)=(/cc(1),cc(2)/)
  tsc_index(1:2,6)=(/cr(1),cc(2)/)
  tsc_index(1:2,7)=(/cl(1),cr(2)/)
  tsc_index(1:2,8)=(/cc(1),cr(2)/)
  tsc_index(1:2,9)=(/cr(1),cr(2)/)
#endif
#if NDIM==3
  tsc_index(1:3,1) =(/cl(1),cl(2),cl(3)/)
  tsc_index(1:3,2) =(/cc(1),cl(2),cl(3)/)
  tsc_index(1:3,3) =(/cr(1),cl(2),cl(3)/)
  tsc_index(1:3,4) =(/cl(1),cc(2),cl(3)/)
  tsc_index(1:3,5) =(/cc(1),cc(2),cl(3)/)
  tsc_index(1:3,6) =(/cr(1),cc(2),cl(3)/)
  tsc_index(1:3,7) =(/cl(1),cr(2),cl(3)/)
  tsc_index(1:3,8) =(/cc(1),cr(2),cl(3)/)
  tsc_index(1:3,9) =(/cr(1),cr(2),cl(3)/)
  tsc_index(1:3,10)=(/cl(1),cl(2),cc(3)/)
  tsc_index(1:3,11)=(/cc(1),cl(2),cc(3)/)
  tsc_index(1:3,12)=(/cr(1),cl(2),cc(3)/)
  tsc_index(1:3,13)=(/cl(1),cc(2),cc(3)/)
  tsc_index(1:3,14)=(/cc(1),cc(2),cc(3)/)
  tsc_index(1:3,15)=(/cr(1),cc(2),cc(3)/)
  tsc_index(1:3,16)=(/cl(1),cr(2),cc(3)/)
  tsc_index(1:3,17)=(/cc(1),cr(2),cc(3)/)
  tsc_index(1:3,18)=(/cr(1),cr(2),cc(3)/)
  tsc_index(1:3,19)=(/cl(1),cl(2),cr(3)/)
  tsc_index(1:3,20)=(/cc(1),cl(2),cr(3)/)
  tsc_index(1:3,21)=(/cr(1),cl(2),cr(3)/)
  tsc_index(1:3,22)=(/cl(1),cc(2),cr(3)/)
  tsc_index(1:3,23)=(/cc(1),cc(2),cr(3)/)
  tsc_index(1:3,24)=(/cr(1),cc(2),cr(3)/)
  tsc_index(1:3,25)=(/cl(1),cr(2),cr(3)/)
  tsc_index(1:3,26)=(/cc(1),cr(2),cr(3)/)
  tsc_index(1:3,27)=(/cr(1),cr(2),cr(3)/)
#endif
end function tsc_index
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
function pcs_weight(wll,wl,wr,wrr)
  use amr_parameters, only: ndim, fourtondim
  real(kind=8),dimension(1:fourtondim)::pcs_weight
  real(kind=8),dimension(1:ndim)::wll,wl,wr,wrr
#if NDIM==1
  pcs_weight(1)=wll(1)
  pcs_weight(2)=wl (1)
  pcs_weight(3)=wr (1)
  pcs_weight(4)=wrr(1)
#endif
#if NDIM==2
  pcs_weight(1) =wll(1)*wll(2)
  pcs_weight(2) =wl (1)*wll(2)
  pcs_weight(3) =wr (1)*wll(2)
  pcs_weight(4) =wrr(1)*wll(2)
  pcs_weight(5) =wll(1)*wl (2)
  pcs_weight(6) =wl (1)*wl (2)
  pcs_weight(7) =wr (1)*wl (2)
  pcs_weight(8) =wrr(1)*wl (2)
  pcs_weight(9) =wll(1)*wr (2)
  pcs_weight(10)=wl (1)*wr (2)
  pcs_weight(11)=wr (1)*wr (2)
  pcs_weight(12)=wrr(1)*wr (2)
  pcs_weight(13)=wll(1)*wrr(2)
  pcs_weight(14)=wl (1)*wrr(2)
  pcs_weight(15)=wr (1)*wrr(2)
  pcs_weight(16)=wrr(1)*wrr(2)
#endif
#if NDIM==3
  pcs_weight(1) =wll(1)*wll(2)*wll(3)
  pcs_weight(2) =wl (1)*wll(2)*wll(3)
  pcs_weight(3) =wr (1)*wll(2)*wll(3)
  pcs_weight(4) =wrr(1)*wll(2)*wll(3)
  pcs_weight(5) =wll(1)*wl (2)*wll(3)
  pcs_weight(6) =wl (1)*wl (2)*wll(3)
  pcs_weight(7) =wr (1)*wl (2)*wll(3)
  pcs_weight(8) =wrr(1)*wl (2)*wll(3)
  pcs_weight(9) =wll(1)*wr (2)*wll(3)
  pcs_weight(10)=wl (1)*wr (2)*wll(3)
  pcs_weight(11)=wr (1)*wr (2)*wll(3)
  pcs_weight(12)=wrr(1)*wr (2)*wll(3)
  pcs_weight(13)=wll(1)*wrr(2)*wll(3)
  pcs_weight(14)=wl (1)*wrr(2)*wll(3)
  pcs_weight(15)=wr (1)*wrr(2)*wll(3)
  pcs_weight(16)=wrr(1)*wrr(2)*wll(3)
  pcs_weight(17)=wll(1)*wll(2)*wl (3)
  pcs_weight(18)=wl (1)*wll(2)*wl (3)
  pcs_weight(19)=wr (1)*wll(2)*wl (3)
  pcs_weight(20)=wrr(1)*wll(2)*wl (3)
  pcs_weight(21)=wll(1)*wl (2)*wl (3)
  pcs_weight(22)=wl (1)*wl (2)*wl (3)
  pcs_weight(23)=wr (1)*wl (2)*wl (3)
  pcs_weight(24)=wrr(1)*wl (2)*wl (3)
  pcs_weight(25)=wll(1)*wr (2)*wl (3)
  pcs_weight(26)=wl (1)*wr (2)*wl (3)
  pcs_weight(27)=wr (1)*wr (2)*wl (3)
  pcs_weight(28)=wrr(1)*wr (2)*wl (3)
  pcs_weight(29)=wll(1)*wrr(2)*wl (3)
  pcs_weight(30)=wl (1)*wrr(2)*wl (3)
  pcs_weight(31)=wr (1)*wrr(2)*wl (3)
  pcs_weight(32)=wrr(1)*wrr(2)*wl (3)
  pcs_weight(33)=wll(1)*wll(2)*wr (3)
  pcs_weight(34)=wl (1)*wll(2)*wr (3)
  pcs_weight(35)=wr (1)*wll(2)*wr (3)
  pcs_weight(36)=wrr(1)*wll(2)*wr (3)
  pcs_weight(37)=wll(1)*wl (2)*wr (3)
  pcs_weight(38)=wl (1)*wl (2)*wr (3)
  pcs_weight(39)=wr (1)*wl (2)*wr (3)
  pcs_weight(40)=wrr(1)*wl (2)*wr (3)
  pcs_weight(41)=wll(1)*wr (2)*wr (3)
  pcs_weight(42)=wl (1)*wr (2)*wr (3)
  pcs_weight(43)=wr (1)*wr (2)*wr (3)
  pcs_weight(44)=wrr(1)*wr (2)*wr (3)
  pcs_weight(45)=wll(1)*wrr(2)*wr (3)
  pcs_weight(46)=wl (1)*wrr(2)*wr (3)
  pcs_weight(47)=wr (1)*wrr(2)*wr (3)
  pcs_weight(48)=wrr(1)*wrr(2)*wr (3)
  pcs_weight(49)=wll(1)*wll(2)*wrr(3)
  pcs_weight(50)=wl (1)*wll(2)*wrr(3)
  pcs_weight(51)=wr (1)*wll(2)*wrr(3)
  pcs_weight(52)=wrr(1)*wll(2)*wrr(3)
  pcs_weight(53)=wll(1)*wl (2)*wrr(3)
  pcs_weight(54)=wl (1)*wl (2)*wrr(3)
  pcs_weight(55)=wr (1)*wl (2)*wrr(3)
  pcs_weight(56)=wrr(1)*wl (2)*wrr(3)
  pcs_weight(57)=wll(1)*wr (2)*wrr(3)
  pcs_weight(58)=wl (1)*wr (2)*wrr(3)
  pcs_weight(59)=wr (1)*wr (2)*wrr(3)
  pcs_weight(60)=wrr(1)*wr (2)*wrr(3)
  pcs_weight(61)=wll(1)*wrr(2)*wrr(3)
  pcs_weight(62)=wl (1)*wrr(2)*wrr(3)
  pcs_weight(63)=wr (1)*wrr(2)*wrr(3)
  pcs_weight(64)=wrr(1)*wrr(2)*wrr(3)
#endif
end function pcs_weight
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
function pcs_index(cll,cl,cr,crr)
  use amr_parameters, only: ndim, fourtondim
  integer,dimension(1:ndim,1:fourtondim)::pcs_index
  integer,dimension(1:ndim)::cll,cl,cr,crr
#if NDIM==1
  pcs_index(1,1)=cll(1)
  pcs_index(1,2)=cl (1)
  pcs_index(1,3)=cr (1)
  pcs_index(1,4)=crr(1)
#endif
#if NDIM==2
  pcs_index(1:2,1) =(/cll(1),cll(2)/)
  pcs_index(1:2,2) =(/cl (1),cll(2)/)
  pcs_index(1:2,3) =(/cr (1),cll(2)/)
  pcs_index(1:2,4) =(/crr(1),cll(2)/)
  pcs_index(1:2,5) =(/cll(1),cl (2)/)
  pcs_index(1:2,6) =(/cl (1),cl (2)/)
  pcs_index(1:2,7) =(/cr (1),cl (2)/)
  pcs_index(1:2,8) =(/crr(1),cl (2)/)
  pcs_index(1:2,9) =(/cll(1),cr (2)/)
  pcs_index(1:2,10)=(/cl (1),cr (2)/)
  pcs_index(1:2,11)=(/cr (1),cr (2)/)
  pcs_index(1:2,12)=(/crr(1),cr (2)/)
  pcs_index(1:2,13)=(/cll(1),crr(2)/)
  pcs_index(1:2,14)=(/cl (1),crr(2)/)
  pcs_index(1:2,15)=(/cr (1),crr(2)/)
  pcs_index(1:2,16)=(/crr(1),crr(2)/)
#endif
#if NDIM==3
  pcs_index(1:3,1) =(/cll(1),cll(2),cll(3)/)
  pcs_index(1:3,2) =(/cl (1),cll(2),cll(3)/)
  pcs_index(1:3,3) =(/cr (1),cll(2),cll(3)/)
  pcs_index(1:3,4) =(/crr(1),cll(2),cll(3)/)
  pcs_index(1:3,5) =(/cll(1),cl (2),cll(3)/)
  pcs_index(1:3,6) =(/cl (1),cl (2),cll(3)/)
  pcs_index(1:3,7) =(/cr (1),cl (2),cll(3)/)
  pcs_index(1:3,8) =(/crr(1),cl (2),cll(3)/)
  pcs_index(1:3,9) =(/cll(1),cr (2),cll(3)/)
  pcs_index(1:3,10)=(/cl (1),cr (2),cll(3)/)
  pcs_index(1:3,11)=(/cr (1),cr (2),cll(3)/)
  pcs_index(1:3,12)=(/crr(1),cr (2),cll(3)/)
  pcs_index(1:3,13)=(/cll(1),crr(2),cll(3)/)
  pcs_index(1:3,14)=(/cl (1),crr(2),cll(3)/)
  pcs_index(1:3,15)=(/cr (1),crr(2),cll(3)/)
  pcs_index(1:3,16)=(/crr(1),crr(2),cll(3)/)
  pcs_index(1:3,17)=(/cll(1),cll(2),cl (3)/)
  pcs_index(1:3,18)=(/cl (1),cll(2),cl (3)/)
  pcs_index(1:3,19)=(/cr (1),cll(2),cl (3)/)
  pcs_index(1:3,20)=(/crr(1),cll(2),cl (3)/)
  pcs_index(1:3,21)=(/cll(1),cl (2),cl (3)/)
  pcs_index(1:3,22)=(/cl (1),cl (2),cl (3)/)
  pcs_index(1:3,23)=(/cr (1),cl (2),cl (3)/)
  pcs_index(1:3,24)=(/crr(1),cl (2),cl (3)/)
  pcs_index(1:3,25)=(/cll(1),cr (2),cl (3)/)
  pcs_index(1:3,26)=(/cl (1),cr (2),cl (3)/)
  pcs_index(1:3,27)=(/cr (1),cr (2),cl (3)/)
  pcs_index(1:3,28)=(/crr(1),cr (2),cl (3)/)
  pcs_index(1:3,29)=(/cll(1),crr(2),cl (3)/)
  pcs_index(1:3,30)=(/cl (1),crr(2),cl (3)/)
  pcs_index(1:3,31)=(/cr (1),crr(2),cl (3)/)
  pcs_index(1:3,32)=(/crr(1),crr(2),cl (3)/)
  pcs_index(1:3,33)=(/cll(1),cll(2),cr (3)/)
  pcs_index(1:3,34)=(/cl (1),cll(2),cr (3)/)
  pcs_index(1:3,35)=(/cr (1),cll(2),cr (3)/)
  pcs_index(1:3,36)=(/crr(1),cll(2),cr (3)/)
  pcs_index(1:3,37)=(/cll(1),cl (2),cr (3)/)
  pcs_index(1:3,38)=(/cl (1),cl (2),cr (3)/)
  pcs_index(1:3,39)=(/cr (1),cl (2),cr (3)/)
  pcs_index(1:3,40)=(/crr(1),cl (2),cr (3)/)
  pcs_index(1:3,41)=(/cll(1),cr (2),cr (3)/)
  pcs_index(1:3,42)=(/cl (1),cr (2),cr (3)/)
  pcs_index(1:3,43)=(/cr (1),cr (2),cr (3)/)
  pcs_index(1:3,44)=(/crr(1),cr (2),cr (3)/)
  pcs_index(1:3,45)=(/cll(1),crr(2),cr (3)/)
  pcs_index(1:3,46)=(/cl (1),crr(2),cr (3)/)
  pcs_index(1:3,47)=(/cr (1),crr(2),cr (3)/)
  pcs_index(1:3,48)=(/crr(1),crr(2),cr (3)/)
  pcs_index(1:3,49)=(/cll(1),cll(2),crr(3)/)
  pcs_index(1:3,50)=(/cl (1),cll(2),crr(3)/)
  pcs_index(1:3,51)=(/cr (1),cll(2),crr(3)/)
  pcs_index(1:3,52)=(/crr(1),cll(2),crr(3)/)
  pcs_index(1:3,53)=(/cll(1),cl (2),crr(3)/)
  pcs_index(1:3,54)=(/cl (1),cl (2),crr(3)/)
  pcs_index(1:3,55)=(/cr (1),cl (2),crr(3)/)
  pcs_index(1:3,56)=(/crr(1),cl (2),crr(3)/)
  pcs_index(1:3,57)=(/cll(1),cr (2),crr(3)/)
  pcs_index(1:3,58)=(/cl (1),cr (2),crr(3)/)
  pcs_index(1:3,59)=(/cr (1),cr (2),crr(3)/)
  pcs_index(1:3,60)=(/crr(1),cr (2),crr(3)/)
  pcs_index(1:3,61)=(/cll(1),crr(2),crr(3)/)
  pcs_index(1:3,62)=(/cl (1),crr(2),crr(3)/)
  pcs_index(1:3,63)=(/cr (1),crr(2),crr(3)/)
  pcs_index(1:3,64)=(/crr(1),crr(2),crr(3)/)
#endif
end function pcs_index
!################################################################
!################################################################
!################################################################
!################################################################
recursive subroutine r_sort_part(pst,ilevel,input_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::ilevel

  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_SORT_PART,pst%iUpper+1,input_size,0,ilevel)
     call r_sort_part(pst%pLower,ilevel,input_size)
     call mdl_get_reply(pst%s%mdl,rID,0)
  else
#ifdef _CUDA
     if(pst%s%m%data_on_device)then
        if(pst%s%r%part)call gpu_sort_part(pst%s, ilevel)
        if(pst%s%r%star.or.pst%s%r%sink.or.pst%s%r%tree.or.pst%s%r%trac)then
           write(*,*)'ERROR: r_sort_part: non-DM not supported on GPU.'
           call abort
        endif
        return
     endif
#endif
     if(pst%s%r%part)call sort_part(pst%s,pst%s%p   ,ilevel)
     if(pst%s%r%star)call sort_part(pst%s,pst%s%star,ilevel)
     if(pst%s%r%sink)call sort_part(pst%s,pst%s%sink,ilevel)
     if(pst%s%r%tree)call sort_part(pst%s,pst%s%tree,ilevel)
     if(pst%s%r%trac)call sort_part(pst%s,pst%s%trac,ilevel)
  endif

end subroutine r_sort_part
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
subroutine sort_part(s,p,ilevel)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: oct
  use ramses_commons, only: ramses_t
  use pm_parameters
  use pm_commons, only: part_t
  use nbors_utils
  use cache_commons
  use cache
  use hilbert
  implicit none
  type(ramses_t)::s
  type(part_t)::p
  integer::ilevel
  !
  ! Local variables
  integer,dimension(1:ndim)::ix
  integer::i

  associate(r=>s%r,g=>s%g,m=>s%m)

  ! Sort particle according to current level Hilbert key
  do i=p%headp(ilevel),p%tailp(r%nlevelmax)
     p%sortp(i)=i
  end do
  ix=0
  call sort_hilbert(r,g,m,p,p%headp(ilevel),p%tailp(r%nlevelmax),ix,0,1,ilevel-1)

  end associate

end subroutine sort_part
!##############################################################################
!##############################################################################
!##############################################################################
!##############################################################################
recursive subroutine sort_hilbert(r,g,m,p,head_part, tail_part, ix_coarse, cstate_coarse, ilevel, final_level)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: run_t, global_t, mesh_t
  use pm_commons, only: part_t
  use hilbert, only: next_state_diagram_reverse, one_digit_diagram
  implicit none

  type(run_t),intent(in)::r
  type(global_t),intent(in)::g
  type(mesh_t)::m
  type(part_t)::p
  integer, intent(in) :: ilevel, final_level
  integer, intent(in) :: head_part, tail_part
  integer, dimension(1:ndim), intent(in) :: ix_coarse
  integer, intent(in) :: cstate_coarse

  ! Description:
  ! This subroutine sort particles along the Hilbert key at the resolution
  ! set by final_level. It should be called first with ilevel=levelmin.

  ! Iputs: 
  ! - Head_part and tail_part are head and tail of particle distribution to work on.
  ! - Array sortp must be initialized with sortp(i)=i between head_part and tail_part.
  ! - Cartesian key of coarse cell in which these particles are contained.
  ! - State of the coarse cell for Hilbert ordering
  ! - Current and final level

  ! Example: 
  ! ix=(/0,0,0/)
  ! call sort_hilbert(1, npart, ix, 0, 1, nlevelmax) 
  ! will sort all particles according to their Hilbert key at levelmax.
  ! On output, array sortp is modified.

  ! Local variables
  integer :: ip, ind_part, idim, ipart, new_ipart
  integer :: ckey_max, cstate_fine, ind_cart_part, head_fine, tail_fine
  integer, dimension(1:ndim) :: ix_fine, ix_ref, ix_part
  integer, dimension(0:twotondim-1,1:ndim) :: ix, ix_child
  integer, dimension(0:twotondim-1) :: nstate, sdigit, ind, ind_cart, ind_hilbert
  integer, dimension(0:twotondim-1) :: numb_part, offset
  real(kind=8) :: ckey_factor

  ! Compute particle position to cartesian key factor
  ckey_max = 2**ilevel
  ckey_factor = 2.0**ilevel / r%boxlen

  ! Initial Cartesian offset for fine cells
  do idim = 1, ndim
     ix_ref(idim) = ISHFT(ix_coarse(idim),1)
  end do

  ! Compute the Hilbert index for fine cells
  do ip = 0, twotondim-1
     sdigit(ip) = ip
  end do

  ! Compute lookup index in state diagrams
  do ip = 0, twotondim-1
     ind(ip) = cstate_coarse * twotondim + sdigit(ip)
  end do

  ! Save next state
  do ip = 0, twotondim-1
     nstate(ip) = next_state_diagram_reverse(ind(ip))
  end do

  ! Add one integer key digit each
  do idim = 1, ndim
     do ip = 0, twotondim-1
        ix(ip, idim) = one_digit_diagram(ind(ip), idim)
     end do
  end do

  ! Compute Cartesian index for children cells
  ind_cart = 0
  do idim = 1, ndim
     do ip = 0, twotondim-1
        ix_child(ip, idim) = ix_ref(idim) + ix(ip, idim)
        ind_cart(ip) = ind_cart(ip) + ix(ip, idim) * 2**(idim-1)
     end do
  end do

  ! Compute mapping from Cartesian to Hilbert order
  ind_hilbert = 0
  do ip = 0, twotondim-1
     ind_hilbert(ind_cart(ip))=ip
  end do

  ! Count particles per children cell
  numb_part = 0
  do ipart = head_part, tail_part
     ind_part = p%sortp(ipart)
     ind_cart_part = 0
     do idim = 1,ndim
        ix_part(idim) = int((p%xp(ind_part,idim)+m%skip(idim))*ckey_factor) - ix_ref(idim)
        ind_cart_part = ind_cart_part + ix_part(idim) * 2**(idim-1)
     end do
     ip = ind_hilbert(ind_cart_part)
     numb_part(ip) = numb_part(ip) + 1
  end do

  offset = head_part-1
  do ip = 1, twotondim-1
     offset(ip) = offset(ip-1) + numb_part(ip-1)
  end do

  ! Compute new sortp array
  numb_part = 0
  do ipart = head_part, tail_part
     ind_part = p%sortp(ipart)
     ind_cart_part = 0
     do idim = 1,ndim
        ix_part(idim) = int((p%xp(ind_part,idim)+m%skip(idim))*ckey_factor) - ix_ref(idim)
        ind_cart_part = ind_cart_part + ix_part(idim) * 2**(idim-1)
     end do
     ip = ind_hilbert(ind_cart_part)
     numb_part(ip) = numb_part(ip) + 1
     new_ipart = offset(ip) + numb_part(ip)
     p%workp(new_ipart) = ind_part
  end do
  do ipart = head_part,tail_part
     p%sortp(ipart) = p%workp(ipart)
  end do

  ! Recursive call
  if(ilevel < final_level)then
     do ip = 0, twotondim-1
        if(numb_part(ip) > 0)then
           head_fine = offset(ip) + 1
           tail_fine = offset(ip) + numb_part(ip)
           ix_fine(1:ndim) = ix_child(ip,1:ndim)
           cstate_fine = nstate(ip)
           call sort_hilbert(r,g,m,p,head_fine,tail_fine,ix_fine,cstate_fine,ilevel+1,final_level)
        endif
     end do
  endif

end subroutine sort_hilbert
!###############################################
!###############################################
!###############################################
!###############################################
end module rho_fine_module

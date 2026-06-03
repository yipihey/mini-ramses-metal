module init_refine_adaptive_module
contains
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine m_init_refine_adaptive(pst)
  use ramses_commons, only: pst_t
  use flag_utils, only: m_flag_fine
  use refine_utils, only: m_refine_fine
  use upload_module, only: m_upload_fine
  use rt_upload_module, only: m_rt_upload_fine
#ifdef GRAV
  use rho_fine_module, only: m_rho_fine
#endif
  use init_part_module, only: r_deallocate_gas
  use input_part_zoom_module, only: m_input_part_zoom
  use input_part_module, only: r_mass_min_part, r_broadcast_mp_min
  use input_hydro_grafic_module, only: r_input_refmap_grafic
  implicit none
  type(pst_t)::pst
  !--------------------------------------------------------------------
  ! This routine is the master procedure to set the base grid
  ! and initialize all cell-based variables within it.
  !--------------------------------------------------------------------
  real(kind=8)::mp_min
  integer::istep,ilevel

  if(pst%s%r%filetype=='grafic')return

  if(pst%s%r%verbose)write(*,*)'Entering init_refine_adaptive'

  do istep=pst%s%r%levelmin,pst%s%r%nlevelmax+1

     if(pst%s%r%filetype=='grafic_zoom'.and.pst%s%r%initfile(istep+1).eq.' ')exit

     if(istep<pst%s%r%nlevelmax)then
        write(*,'(" Building initial fine grid at level ",I0)')istep+1
     else if(istep==pst%s%r%nlevelmax)then
        write(*,'(" Finalizing initial grid at level ",I0)')istep
     else
        write(*,*)'Finalizing initial grid at all levels'
     endif

     ! Refine all level cells from levelmin
     call m_refine_fine(pst,pst%s%r%levelmin)

     ! Initialize hydro variables on the fine grids
     if(pst%s%r%hydro)then
        do ilevel=pst%s%r%nlevelmax,pst%s%r%levelmin+1,-1
           call m_init_flow_fine(pst,ilevel)
        end do
        do ilevel=pst%s%r%nlevelmax-1,pst%s%r%levelmin,-1
           call m_upload_fine(pst,ilevel)
        end do
     endif

     ! Initialize rt variables on the fine grids
     if(pst%s%r%rt)then
        do ilevel=pst%s%r%nlevelmax,pst%s%r%levelmin+1,-1
           call m_rt_init_flow_fine(pst,ilevel)
        end do
        do ilevel=pst%s%r%nlevelmax-1,pst%s%r%levelmin,-1
           call m_rt_upload_fine(pst,ilevel)
        end do
     endif

     ! Initialize refinement map on the fine grids
     if(pst%s%r%filetype=='grafic_zoom'.and.pst%s%r%ivar_refine==0)then
        do ilevel=pst%s%r%nlevelmax,pst%s%r%levelmin+1,-1
           call r_input_refmap_grafic(pst,ilevel,1)
        end do
     endif

     ! Compute total mass density from gas and particles on the fine grids
#ifdef GRAV
     if(pst%s%r%filetype.NE.'grafic_zoom')then
        call m_rho_fine(pst,pst%s%r%levelmin,0)
     endif
#endif

     ! Flag all level cells for refinement
     do ilevel=pst%s%r%nlevelmax-1,pst%s%r%levelmin,-1
        call m_flag_fine(pst,ilevel,2)
     end do

  end do

  if(pst%s%r%filetype=='gadget'.and.pst%s%r%hydro)then
     ! Deallocate gas particles after gadget IC completed
     call r_deallocate_gas(pst)
  endif

  if(pst%s%r%filetype=='grafic_zoom'.and.pst%s%r%pic)then
     call m_input_part_zoom(pst)
     ! Compute minimum particle mass and initial coarse particle level
     call r_mass_min_part(pst,pst%s%r%levelmin,1,mp_min,2)
     call r_broadcast_mp_min(pst,mp_min,2)
  endif

end subroutine m_init_refine_adaptive
!###############################################
!###############################################
!###############################################
!###############################################
end module init_refine_adaptive_module

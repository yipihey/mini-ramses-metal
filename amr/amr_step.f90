module amr_step
contains
!#####################################################
!#####################################################
!#####################################################
!#####################################################
recursive subroutine m_amr_step(pst,ilevel,icount,done)
  use ramses_commons, only: pst_t
  use pm_parameters
  use flag_utils, only: m_flag_fine
  use update_time_module, only: m_update_time
  use refine_utils, only: m_refine_fine
  use upload_module, only: m_upload_fine
  use rho_fine_module, only: m_rho_fine
#ifdef GRAV
  use phi_fine_cg_module, only: m_phi_fine_cg
  use multigrid_fine_commons, only: multigrid
  use force_fine_module, only: m_force_fine
#endif
  use move_fine_module, only: m_kick_drift_part
  use output_amr_module, only: m_dump_all
  use synchro_hydro_fine_module, only: m_synchro_hydro_fine, r_gravity_hydro_fine
  use source_hydro_fine_module, only: r_source_hydro_fine
  use interpol_phi_module, only: r_save_phi_old
  use godunov_fine_module, only: r_godunov_fine,r_set_unew,r_set_uold
  use cooling_fine_module, only: r_cooling_fine
  use newdt_fine_module, only: m_newdt_fine,r_broadcast_dt,in_broadcast_dt_t
  use movie_module, only: m_output_frame
  use star_formation_module, only: out_star_formation_t, r_star_formation
  use sink_formation_module, only: m_sink_formation
  use tree_formation_module, only: m_tree_formation
  use feedback_module, only: out_feedback_t, r_thermal_feedback, m_mechanical_feedback
  use clump_finder_module, only: m_clump_finder
  use lightcone_module, only: m_output_lightcone
  use rt_godunov_fine_module, only: r_set_rtunew,r_set_emissivity
  use rt_step_module, only: m_rt_step
  use sink_evolution_module, only: r_sink_evolution, out_accretion_t
  use sink_merger_module, only: r_sink_merger
  use turb_driving, only: r_drive_turb
  use turb_hydro_module, only: m_turb_hydro
#ifdef _CUDA
  use gpu_manager, only: r_transfer_grid_host
#endif

  implicit none

  type(pst_t) :: pst
  integer :: ilevel,icount
  logical :: done,ok_fbk
  !-------------------------------------------------------------------!
  ! This routine is the adaptive-mesh/adaptive-time-step main driver. !
  ! Each routine is called using a specific order, don't change it,   !
  ! unless you check all consequences first                           !
  !-------------------------------------------------------------------!
  type(in_broadcast_dt_t) :: in_broadcast_dt
  type(out_star_formation_t) :: output_star
  type(out_feedback_t) :: output_fbk
  type(out_accretion_t) :: output_acc
  real(kind=8) :: mass_fbk
  real(kind=8) :: tcurr=0
  real(kind=8), save :: tprev=0.
  real(kind=8), external :: wallclock
  logical, save :: bkp_last_done=.false.

  associate(r=>pst%s%r, g=>pst%s%g, m=>pst%s%m, mdl=>pst%s%mdl)

  if(m%noct_tot(ilevel)==0)return
  if(r%verbose)write(*,'(" Entering amr_step",i1," for level",i2)')icount,ilevel
  g%isubcycle(ilevel)=icount ! only in master 

  !------------------------------
  ! Make new refinements and load
  ! balance grids and particles.
  !------------------------------
  if(ilevel==r%levelmin.or.icount>1)then
     if(.not.r%static_mesh.and.r%nlevelmax>r%levelmin)then
        call m_timer('refine','start')
        call m_refine_fine(pst,ilevel)
     endif
  endif

  !-------------------------
  ! Sink formation in clumps
  !-------------------------
  if(r%sink.and.ilevel==r%levelmin.and.r%sink_form)then
     call m_timer('sink - formation','start')
     call m_sink_formation(pst)
  endif

  !--------------------------------
  ! Merging tree particle formation
  !--------------------------------
  if(r%tree.and.ilevel==r%levelmin.and.mod(g%nstep_coarse, r%nsteps_per_tree)==0)then
     call m_timer('tree - formation','start')
     call m_tree_formation(pst)
  endif

  if(ilevel==r%levelmin)then
     if(r%foutput>0)then
        if(mod(g%nstep_coarse,r%foutput)==0.or.g%aexp>=r%aout(g%iout).or.g%t>=r%tout(g%iout))then
           !----------------------------
           ! Call the clump finder
           !----------------------------
           if(r%clump_finder)then ! Create output and no need to keep alive
              call m_timer('clump','start')
              call m_clump_finder(pst,.true.,.false.)
           endif
           !---------------------------
           ! Write output files to disk
           !---------------------------
           call m_timer('output','start')
#ifdef _CUDA
           call r_transfer_grid_host(pst)
#endif
           call m_dump_all(pst,.false.)
        endif
     endif
     !----------------------------
     ! Write restart files to disk
     !----------------------------
     tcurr=wallclock()
     if(tcurr>tprev+r%bkp_time_hrs*3600)then
        call m_timer('backup','start')
#ifdef _CUDA
           call r_transfer_grid_host(pst)
#endif
        call m_dump_all(pst,.true.)
        tprev=tcurr
     endif
     if(r%run_time_hrs>0.and..not.bkp_last_done)then
        if(tcurr>r%run_time_hrs*3600-r%bkp_last_min*60)then
           call m_timer('backup','start')
#ifdef _CUDA
           call r_transfer_grid_host(pst)
#endif
           call m_dump_all(pst,.true.)
           bkp_last_done=.true.
        endif
     endif
     ! Lightcone
     if (r%lightcone) then
        call m_timer('lightcone','start')
        call m_output_lightcone(pst)
     endif
  endif

  !--------------------------
  ! Write movie frame to disk
  !--------------------------
  if(r%movie) then
     if(r%imov.le.r%imovout)then 
        if((r%aendmov>0.and.g%aexp>=(r%aendmov-r%astartmov)*dble(r%imov)/dble(r%imovout)+r%astartmov) &
             & .or.(r%tendmov>0.and.g%t>=(r%tendmov-r%tstartmov)*dble(r%imov)/dble(r%imovout)+r%tstartmov))then
           call m_output_frame(pst)
        endif
     endif
  end if

  !------------------------------------
  ! Poisson source term for gravity or
  ! just for particle list for pic only
  !------------------------------------
  if(ilevel==r%levelmin.or.icount>1)then
     call m_timer('rho','start')
     call m_rho_fine(pst,ilevel,0)
  endif

  ! Remove gravity source term with half time step and old force
  if(r%hydro.and..not.r%static_gas)then
     if(r%poisson.or.maxval(abs(r%constant_gravity))>0)then
        call m_timer('hydro - gravity','start')
        call m_synchro_hydro_fine(pst,ilevel,-0.5d0*dble(g%dtnew(ilevel)))
     end if
  endif

  !---------------
  ! Gravity solver
  !---------------
#ifdef GRAV
  if(r%poisson.and.r%gravity_type<=0)then
     call m_timer('poisson','start')

     ! Save old potential for time-extrapolation at level boundaries
     call r_save_phi_old(pst,ilevel,1)

     ! Compute new gravitational potential
     if(ilevel > r%levelmin)then
        if(ilevel >= r%cg_levelmin) then
           call m_phi_fine_cg(pst,ilevel,icount)
        else
           call multigrid(pst,ilevel,icount)
        end if
     else
        call multigrid(pst,r%levelmin,icount)
     end if

     ! Initial old potential
     if (g%nstep==0)call r_save_phi_old(pst,ilevel,1)
  endif

  ! Compute gravitational acceleration
  if(r%poisson)then
     call m_timer('grav force','start')
     call m_force_fine(pst,ilevel,icount)
  end if
#endif

  ! Perform second kick for particles
  if(r%pic)then
     call m_timer('particle - kickdrift','start')
     call m_kick_drift_part(pst,ilevel,action_kick_only)
  endif

  ! Add gravity source term with half time step and new force
  if(r%hydro.and..not.r%static_gas)then
     if(r%poisson.or.maxval(abs(r%constant_gravity))>0)then
        call m_timer('hydro - gravity','start')
        call m_synchro_hydro_fine(pst,ilevel,+0.5d0*dble(g%dtnew(ilevel)))
     endif
  end if

  !--------------------------
  ! Compute turbulent driving
  !--------------------------
  if(r%turb)then
     call m_timer('hydro - turbulence','start')
     call r_drive_turb(pst,ilevel,1)
  endif

  !----------------------
  ! Compute new time step
  !----------------------
  call m_timer('compute dt','start')
  call m_newdt_fine(pst,ilevel)

  !-----------------------
  ! Set unew equal to uold
  !-----------------------
  if(r%hydro.and..not.r%static_gas)then
     call m_timer('hydro - set unew','start')
     call r_set_unew(pst,ilevel,1)
  endif

  !---------------------------
  ! Set rtunew equal to rtuold
  !---------------------------
  if(r%rt)then
     call m_timer('radiative transfer','start')
     call r_set_rtunew(pst,ilevel,1)
     call r_set_emissivity(pst,ilevel,1)
  endif

  !---------------------------
  ! Recursive call to amr_step
  !---------------------------
  if(ilevel<r%nlevelmax)then
     call m_timer('recursive call','start')
     if(m%noct_tot(ilevel+1)>0)then
        if(r%nsubcycle(ilevel)==2)then
           call m_amr_step(pst,ilevel+1,1,done)
           if (done)return
           call m_amr_step(pst,ilevel+1,2,done)
        else
           call m_amr_step(pst,ilevel+1,1,done)
        endif
     else 
        ! Otherwise, modify finer level time-step
        g%dtold(ilevel+1)=g%dtnew(ilevel)/dble(r%nsubcycle(ilevel))
        g%dtnew(ilevel+1)=g%dtnew(ilevel)/dble(r%nsubcycle(ilevel))

        ! Broadcast modified time step to all CPUs
        in_broadcast_dt%ilevel=ilevel+1
        in_broadcast_dt%dtnew=g%dtnew(ilevel+1)
        in_broadcast_dt%dtold=g%dtold(ilevel+1)
        call r_broadcast_dt(pst,in_broadcast_dt,storage_size(in_broadcast_dt)/32)

        ! Update time variable
        call m_timer('update time','start')
        call m_update_time(pst,ilevel,done)
     end if
  else
     call m_timer('update time','start')
     call m_update_time(pst,ilevel,done)
  end if
  if (done)return

  !------------------
  ! Thermal feedback
  !------------------
  if(r%star.and.r%thermal_feedback)then
     call m_timer('star - feedback','start')
     call r_thermal_feedback(pst,ilevel,1,output_fbk,2)
     if(output_fbk%mass>0)then
        g%mass_star_tot=g%mass_star_tot-output_fbk%mass
     endif
  endif

  !---------------------
  ! Mechanical feedback
  !---------------------
  if(r%star.and.r%mechanical_feedback)then
     ok_fbk=.false.
     if(ilevel==r%nlevelmax)then
        ok_fbk=.true.
     else
        if(m%noct_tot(ilevel+1)==0)then
           ok_fbk=.true.
        endif
     end if
     if(ok_fbk)then
        call m_timer('star - feedback','start')
        call m_mechanical_feedback(pst,ilevel,mass_fbk)           
        if(mass_fbk>0)then
           g%mass_star_tot=g%mass_star_tot-mass_fbk
        endif
     endif
  endif

  !----------------------------
  ! Sink accretion and feedback
  !----------------------------
  if(r%sink)then
     call m_timer('sink - evolution','start')
     call r_sink_evolution(pst,ilevel,1,output_acc,2)
     if(output_acc%mass>0)then
        g%mass_sink_tot=g%mass_sink_tot+output_acc%mass
     end if
  end if

  !-----------
  ! Hydro step
  !-----------
  if(r%hydro)then

     if(.not.r%static_gas)then
        ! Hyperbolic solver
        call m_timer('hydro - godunov','start')
        call r_godunov_fine(pst,ilevel,1)

        ! Add gravity source terms to unew with half time step
        if(r%poisson.or.maxval(abs(r%constant_gravity))>0)then
           call m_timer('hydro - gravity','start')
           call r_gravity_hydro_fine(pst,ilevel,1)
        endif

        ! Add other hydro source terms to unew
        call m_timer('hydro - source','start')
        call r_source_hydro_fine(pst,ilevel,1)

        ! Set uold equal to unew
        call m_timer('hydro - set uold','start')
        call r_set_uold(pst,ilevel,1)

        ! Add gravity source terms to uold with half time step
        ! to complete the time step with old force (will be removed later)
        if(r%poisson.or.maxval(abs(r%constant_gravity))>0)then
           call m_timer('hydro - gravity','start')
           call m_synchro_hydro_fine(pst,ilevel,+0.5d0*dble(g%dtnew(ilevel)))
        endif

        ! Add turbulent driving source terms to uold with full time step
        if(r%turb)then
           call m_timer('hydro - turbulence','start')
           call m_turb_hydro(pst,ilevel,dble(g%dtnew(ilevel)))
        endif
     endif

     ! Restriction operator
     if(ilevel<r%nlevelmax)then
        call m_timer('hydro - upload','start')
        call m_upload_fine(pst,ilevel)
     endif
  endif

  !------------------------
  ! Radiative transfer step
  !------------------------
  if(r%rt)then
     if(r%rt_advect)then
        call m_timer('radiative transfer','start')
        call m_rt_step(pst,ilevel)
     else
        if(r%hydro .and. (r%neq_chem.or.r%cooling_ism.or.r%cooling.or.r%isothermal))call r_cooling_fine(pst,ilevel,1)
     endif
  endif

  !------------------------
  ! Compute cooling/heating
  !------------------------
  if(r%hydro .and. (.not.r%rt) .and. (r%cooling.or.r%cooling_ism.or.r%isothermal.or.r%neq_chem))then
     call m_timer('cooling','start')
     call r_cooling_fine(pst,ilevel,1)
  endif

  !----------------------------
  ! Sink merger
  !----------------------------
  if(r%sink.and.r%sink_merge)then
     call m_timer('sink - merger','start')
     call r_sink_merger(pst,ilevel,1)
  end if

  !-------------------------------------------
  ! Perform first kick and drift for particles
  !-------------------------------------------
  if(r%pic)then
     call m_timer('particle - kickdrift','start')
     call m_kick_drift_part(pst,ilevel,action_kick_drift)
  endif

  !----------------------------------
  ! Star formation in leaf cells only
  !----------------------------------
  if(r%star.and.r%hydro)then
     call m_timer('star - formation','start')
     call r_star_formation(pst,ilevel,1,output_star,2)
     if(output_star%mass>0)then
        g%mass_star_tot=g%mass_star_tot+output_star%mass
     endif
  endif

  !-----------------------
  ! Compute refinement map
  !-----------------------
  if(ilevel<r%nlevelmax)then
     call m_timer('flag','start')
     if(.not.r%static_mesh)call m_flag_fine(pst,ilevel,icount)
  endif

  !-------------------------------
  ! Update coarser level time-step
  !-------------------------------
  if(ilevel>r%levelmin)then
     ! Impose adaptive time step constraints
     if(r%nsubcycle(ilevel-1)==1)g%dtnew(ilevel-1)=g%dtnew(ilevel)
     if(icount==2)g%dtnew(ilevel-1)=g%dtold(ilevel)+g%dtnew(ilevel)

     ! Broadcast updated time step to all CPUs
     call m_timer('recursive call','start')
     in_broadcast_dt%ilevel=ilevel-1
     in_broadcast_dt%dtnew=g%dtnew(ilevel-1)
     in_broadcast_dt%dtold=g%dtold(ilevel-1)
     call r_broadcast_dt(pst,in_broadcast_dt,storage_size(in_broadcast_dt)/32)
  end if

  end associate

end subroutine m_amr_step
end module amr_step

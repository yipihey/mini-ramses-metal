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
#ifdef _METAL
  use metal_gravity_module, only: m_metal_poisson, metal_enabled, m_metal_grid_to_host, m_metal_part_to_host, metal_hydro_on, metal_inited, metal_hydro_resident
#ifdef HYDRO
  use metal_gravity_module, only: m_metal_hydro_level, m_metal_hydro_setunew, &
       m_metal_hydro_resident, m_metal_hydro_setuold, m_metal_hydro_upload_dev, m_metal_uold_to_host
#endif
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
  logical :: gpu_hydro, resident

  associate(r=>pst%s%r, g=>pst%s%g, m=>pst%s%m, mdl=>pst%s%mdl)

  gpu_hydro = .false.; resident = .false.
#ifdef _METAL
  gpu_hydro = metal_inited .and. metal_hydro_on   ! route the godunov step to the GPU (no pic needed)
  ! uold/unew GPU-RESIDENCY (m_metal_hydro_resident path) is wired but DISABLED:
  ! it leaves host uold stale, which breaks the CPU consumers that read uold every
  ! step (newdt CFL, write_screen conservation, output).  Making it correct needs
  ! device newdt (mtl_hydro_cmpdt) + a per-step uold->host diagnostic sync; the net
  ! win is modest since the kernel already dominates and is ~115x faster than CPU.
  ! Set RAMSES_GPU_HYDRO_RESIDENT=1 to exercise it (incl. the stale-host caveat).
  if (metal_hydro_resident) resident = gpu_hydro .and. .not. metal_enabled
#endif

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
#ifdef _METAL
           call m_metal_grid_to_host(pst)
           call m_metal_part_to_host(pst)
           call m_metal_uold_to_host(pst)   ! sync resident hydro state for output (no-op if not resident)
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
#ifdef _METAL
           call m_metal_grid_to_host(pst)
           call m_metal_part_to_host(pst)
           call m_metal_uold_to_host(pst)   ! sync resident hydro state for output (no-op if not resident)
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
#ifdef _METAL
           call m_metal_grid_to_host(pst)
           call m_metal_part_to_host(pst)
           call m_metal_uold_to_host(pst)   ! sync resident hydro state for output (no-op if not resident)
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
     call m_part_trace(pst,ilevel,icount,'pre')   ! DIAG (RAMSES_PART_TRACE): per-particle vp BEFORE kick
     call m_trace_dump(pst,ilevel,icount,'dep')   ! DIAG: nref right after deposit, pre-solve/kick
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
#ifdef _METAL
     ! Hybrid: GPU computes this level's potential AND force.  The whole mesh is
     ! mirrored on the GPU (AMR father + same-level nbor); the level consumes the
     ! CPU density rho, takes the coarser level's phi (resident from its earlier
     ! solve) as the coarse-fine boundary, runs the grouped multigrid V-cycle and
     ! the 4th-order gradient.  CPU keeps managing the AMR mesh.
     if(metal_enabled)then
        call m_metal_poisson(pst, ilevel, icount)
     else
#endif
     if(ilevel > r%levelmin)then
        if(ilevel >= r%cg_levelmin) then
           call m_phi_fine_cg(pst,ilevel,icount)
        else
           call multigrid(pst,ilevel,icount)
        end if
     else
        call multigrid(pst,r%levelmin,icount)
     end if
#ifdef _METAL
     endif
#endif

     ! Initial old potential
     if (g%nstep==0)call r_save_phi_old(pst,ilevel,1)
     call m_trace_dump(pst,ilevel,icount,'sol')   ! DIAG: phi/nref right after solve, PRE-kick
  endif

  ! Compute gravitational acceleration
  if(r%poisson)then
     call m_timer('grav force','start')
#ifdef _METAL
     ! force already filled by m_metal_poisson above (GPU gradient)
     if(.not.metal_enabled) call m_force_fine(pst,ilevel,icount)
#else
     call m_force_fine(pst,ilevel,icount)
#endif
  end if
#endif

  ! Perform second kick for particles
  if(r%pic)then
     call m_timer('particle - kickdrift','start')
     call m_kick_drift_part(pst,ilevel,action_kick_only)
     call m_part_trace(pst,ilevel,icount,'pst')   ! DIAG: per-particle vp AFTER the kick, by idp (Δvp=ff*0.5*dt)
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
  call m_timer('time step','start')
  call m_newdt_fine(pst,ilevel)

  !-----------------------
  ! Set unew equal to uold
  !-----------------------
  if(r%hydro.and..not.r%static_gas)then
     call m_timer('hydro - set unew','start')
#ifdef _METAL
     if(resident)then
        call m_metal_hydro_setunew(pst,ilevel)   ! device set_unew + (1st call) upload uold resident
     else
#endif
        call r_set_unew(pst,ilevel,1)   ! host unew=uold (GPU per-call path uploads this, +finer reflux)
#ifdef _METAL
     endif
#endif
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
#ifdef _METAL
      if(gpu_hydro)then
        ! GPU hydro: godunov_only (unew += du, gravity predictor from B.f, coarse-fine
        ! reflux scatter via cache octs) + grav_hydro, on the device.  source is a
        ! no-op here (nvar=5).  RESIDENT path (pure hydro): uold/unew stay GPU-resident
        ! -- set_unew/set_uold are device, no host copy.  Per-call path (cosmo): host
        ! set_unew (pre-recursion) + finer reflux are in host unew; upload/run/download.
        call m_timer('hydro - godunov','start')
        if(resident)then
           call m_metal_hydro_resident(pst,ilevel)
        else
           call m_metal_hydro_level(pst,ilevel)
        endif
        call m_timer('hydro - source','start')
        call r_source_hydro_fine(pst,ilevel,1)
        call m_timer('hydro - set uold','start')
        if(resident)then
           call m_metal_hydro_setuold(pst,ilevel)
        else
           call r_set_uold(pst,ilevel,1)
        endif
      else
#endif
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
#ifdef _METAL
      endif
#endif

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
#ifdef _METAL
        if(resident)then
           call m_metal_hydro_upload_dev(pst,ilevel)   ! device fine->coarse restriction
        else
#endif
           call m_upload_fine(pst,ilevel)
#ifdef _METAL
        endif
#endif
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
     call m_trace_dump(pst,ilevel,icount,'flg')   ! DIAG (RAMSES_TRACE_DUMP): nref/phi/flag1/refined by ckey
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

! DIAG: matched step-by-step trace.  Dumps per-oct (keyed by Cartesian key) the
! refinement density nref, the potential phi, the refine flag1, and refined-status
! for level ilevel, at the FIRST coarse steps.  Shared by CPU + Metal (both sync
! m%nref/m%phi/m%flag1 to host; the first refine creates nothing from the restart
! flag1 so m%grid is the identical restart mesh at step 0).  Gated RAMSES_TRACE_DUMP.
subroutine m_trace_dump(pst,ilevel,icount,tag)
  use ramses_commons, only: pst_t
  use amr_parameters, only: twotondim
#ifdef _METAL
  use metal_gravity_module, only: metal_enabled, m_metal_grid_to_host, m_metal_fields_to_host
#endif
  implicit none
  type(pst_t), target :: pst
  integer, intent(in) :: ilevel, icount
  character(len=*), intent(in) :: tag
  integer :: o, c, u
  integer, save :: ndump = 0
  character(len=16) :: te
  character(len=72) :: fn
  call get_environment_variable('RAMSES_TRACE_DUMP', te)
  if (len_trim(te) == 0) return
  if (ilevel < 8) return        ! trace L8 (near-floor) through L9+ (the 125x jump) + divergent levels
  ndump = ndump + 1
  if (ndump > 200) return       ! cap total dump files (works on restart, where nstep_coarse is large)
#ifdef _METAL
  ! GPU owns B.grid (host m%grid goes stale after GPU refine reorders); sync it to
  ! the CURRENT layout so the dumped ckey matches the slot-indexed m%nref/m%flag1/m%phi
  ! (which rho_finish/flag/poisson synced in that same current layout).
  if (metal_enabled) call m_metal_grid_to_host(pst)
  if (metal_enabled) call m_metal_fields_to_host(pst)   ! sync phi/f in the SAME slot order as grid
#endif
  associate(m=>pst%s%m, g=>pst%s%g)
  write(fn,'(A,A,A,I2.2,A,I4.4,A,I1,A)') 'trace_',trim(tag),'_L',ilevel,'_s',g%nstep_coarse,'_c',icount,'.dat'
  open(newunit=u, file=trim(fn), status='replace', action='write')
  do o = m%head(ilevel), m%tail(ilevel)
     do c = 1, twotondim
        write(u,'(3I9,I3,5ES23.15,I3,I2)') &
             m%grid(o)%ckey(1), m%grid(o)%ckey(2), m%grid(o)%ckey(3), c, &
             m%nref(c,o), m%phi(c,o), m%f(c,1,o), m%f(c,2,o), m%f(c,3,o), &
             m%flag1(c,o), merge(1,0,m%grid(o)%refined(c))
     end do
  end do
  close(u)
  end associate
end subroutine m_trace_dump

! DIAG: per-particle (idp, xp, levelp) dump after the first deposit+split on the
! restart state (pre-kick), to compare CPU vs GPU oct/level assignment by id.
subroutine m_part_trace(pst, ilevel, icount, tag)
  use ramses_commons, only: pst_t
  use amr_parameters, only: ndim
#ifdef _METAL
  use metal_gravity_module, only: metal_enabled, m_metal_part_to_host
#endif
  implicit none
  type(pst_t), target :: pst
  integer, intent(in) :: ilevel, icount
  character(len=*), intent(in) :: tag
  integer :: ip, u
  integer, save :: done = 0
  character(len=16) :: te
  character(len=48) :: fn
  call get_environment_variable('RAMSES_PART_TRACE', te)
  if (len_trim(te) == 0) return
  done = done + 1
  if (done > 200) return                 ! pre/post kick pairs tagged by level+icount
#ifdef _METAL
  if (metal_enabled) call m_metal_part_to_host(pst)   ! sync resident ipos->xp + vp + levelp to host
#endif
  associate(p=>pst%s%p)
  ! tag (pre/pst) + level + icount so a clean single-kick pre/post pair can be differenced: Δvp=ff*0.5*dt
  write(fn,'(A,A,A,I2.2,A,I1,A,I3.3,A)') 'pt_',trim(tag),'_L',ilevel,'_c',icount,'_',done,'.dat'
  open(newunit=u, file=trim(fn), status='replace', action='write')
  do ip = 1, p%npart                   ! idp, xp(3), vp(3), levelp
     write(u,'(I12,6ES24.16,I4)') p%idp(ip), p%xp(ip,1),p%xp(ip,2),p%xp(ip,3), &
          p%vp(ip,1),p%vp(ip,2),p%vp(ip,3), p%levelp(ip)
  end do
  close(u)
  end associate
end subroutine m_part_trace
end module amr_step

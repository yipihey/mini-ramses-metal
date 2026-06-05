subroutine adaptive_loop(pst)
  use mdl_module
  use ramses_commons, only: pst_t
  use capi_commons, only: capi_setup_only, capi_last_state, &
                          capi_inject, capi_inject_n, capi_inject_idp, capi_inject_xp, capi_inject_vp
  use init_amr_module, only: r_init_amr
  use params_module, only: m_read_params
  use init_time_module, only: r_init_time
  use init_hydro_module, only: r_init_hydro
  use init_rt_module, only: r_init_rt
  use init_part_module, only: r_init_part
  use init_xion_module, only: m_init_xion
  use input_part_module, only: m_input_part
  use init_refine_basegrid_module, only: m_init_refine_basegrid
  use init_refine_adaptive_module, only: m_init_refine_adaptive
  use init_refine_restart_module, only: m_init_refine_restart
  use init_refine_ramses_module, only: m_init_refine_ramses
  use turb_init_module, only: r_init_turb
  use amr_step, only: m_amr_step
  use update_time_module, only: getmem, writemem, r_hash_stats
  use load_balance_module, only: r_balance_part
  use clump_finder_module, only: m_clump_finder
#ifdef _CUDA
  use gpu_manager, only: r_set_grid_device
  use nvtx
#endif
#ifdef _METAL
  use metal_bridge_iface
  use metal_gravity_module, only: metal_enabled, metal_flag_on, metal_hydro_on, refine_rehash, refine_hostflag1, refine_hostmed, m_metal_prof_report, g_ncyc_fine, g_ncyc_base, g_mg_check_every, g_sort_every, metal_mg_driver_on, metal_mg_all_levels, m_metal_part_to_host, m_metal_grid_to_host
  use iso_c_binding
  use amr_parameters, only: ndim
#endif

  implicit none
  type(pst_t)::pst
  logical::done
#ifdef _METAL
  integer(c_int),     pointer :: m_sortp(:)
  integer(c_int64_t), pointer :: m_ipos(:), m_hkey(:)
  integer :: m_npart, m_npm, m_ip, m_dd, m_nbad, m_ierr
  integer(c_int64_t) :: m_prevk
  real(c_double) :: m_u, m_sc, m_bl
  character(len=512) :: m_mlib
#endif

  ! Local variables
  integer::ilevel, dummy
  double precision::tt1,tt2
  real(kind=4)::core_mem
#ifdef _CUDA
  character(len=32) :: str_step
  integer step
  step = 0
#endif

  associate(r=>pst%s%r,g=>pst%s%g,mdl=>pst%s%mdl)

  tt1 = mdl_wtime(mdl)

  ! Read run parameters
  call m_read_params(pst)

  ! Initialize grid variables
  call r_init_amr(pst)

  ! Initialize time variables and cooling tables
  call r_init_time(pst)

  ! Initialize hydro kernel workspace
  if(r%hydro)call r_init_hydro(pst)

  ! Initialize rt kernel workspace
  if(r%rt)call r_init_rt(pst)

  ! Initialize particle variables
  if(r%pic)call r_init_part(pst)

  ! Initialize turbulent driveing
  if(r%turb)call r_init_turb(pst)

  ! Read initial particle properties from files
  if(r%pic)call m_input_part(pst)

  ! [C-API] Overwrite the IC particles with a deterministic injected set (all
  ! bound to levelmin, mp=mass_sph so nref counts particles) BEFORE the adaptive
  ! refine build, so RAMSES's own refine machinery builds the mesh from them.
  ! Set by ramses_init_particles (julia/ramses_capi.f90).
  if(capi_inject)then
     block
       integer :: ip, dd, nd, nn
       real(kind=8) :: mp0
       associate(p=>pst%s%p)
         nd = size(p%xp,2)
         ! Use the grafic DM particle mass (loaded by m_input_part) so density
         ! is nonzero and in the same fp32-safe regime as production.  DMO has
         ! mass_sph=0 (omega_b=0); refinement here is count-based anyway
         ! (nref += vol, rho_fine.f90:870), so only gravity needs mp>0.
         mp0 = 0.0d0
         if(p%npart>=1) mp0 = p%mp(1)
         if(mp0<=0.0d0) mp0 = 1.0d0/dble(max(1,capi_inject_n))
         nn = min(capi_inject_n, r%npartmax)
         if(nn < capi_inject_n) &
              write(*,*)' [C-API] WARNING: injected count clamped to npartmax',r%npartmax
         p%npart = nn
         do ip=1,nn
            p%idp(ip)    = capi_inject_idp(ip)
            p%mp(ip)     = mp0
            p%levelp(ip) = r%levelmin
            do dd=1,nd
               p%xp(ip,dd) = capi_inject_xp(ip,dd)
               p%vp(ip,dd) = capi_inject_vp(ip,dd)
            end do
         end do
         ! Bind all particles to levelmin (mirror input_part_grafic:316-324)
         p%headp = p%npart+1
         p%tailp = p%npart
         p%headp(r%levelmin) = 1
         p%tailp(r%levelmin) = p%npart
         if(ANY(.not.r%periodic(1:nd)))then
            p%headp(r%levelmin-1) = 1
            p%tailp(r%levelmin-1) = 0
         end if
       end associate
     end block
     write(*,'(A,I9,A)')' [C-API] injected ',pst%s%p%npart,' deterministic particles'
  end if

  ! Build initial AMR grid
  if(r%nrestart==0)then
     if(r%filetype=='ramses'.and.r%hydro)then
        call m_init_refine_ramses(pst) ! Build AMR grid from output file
     else
        call m_init_refine_basegrid(pst) ! Build coarse grid
        call m_init_refine_adaptive(pst) ! Build adaptive grid
     endif
  else
     call m_init_refine_restart(pst) ! Build AMR grid from restart file
     g%first_coarse_restart = .true. ! Indicate the initial course step post restart
  endif

  ! Initialization of ionization fractions
  if(r%neq_chem .and. r%is_init_xion) call m_init_xion(pst)

  ! Timing since startup
  tt2 = mdl_wtime(mdl)
  print '(A,F14.7)',' Time elapsed since startup:',tt2-tt1

  ! Output mesh structure
  do ilevel=r%levelmin,r%nlevelmax
     if(pst%s%m%noct_tot(ilevel)>0)write(*,999)&
          & ilevel,pst%s%m%noct_tot(ilevel),pst%s%m%noct_min(ilevel),pst%s%m%noct_max(ilevel),pst%s%m%noct_tot(ilevel)/mdl_threads(mdl)
  end do
999 format(' Level ',I2,' has ',I11,' grids (',3(I8,','),')')

  g%nstep_coarse_old=g%nstep_coarse

#ifdef _CUDA
  ! Copy entire grid from host to device
  call r_set_grid_device(pst)
#endif

#ifdef _METAL
  ! Initialise the Metal GPU bridge and exercise the validated DM pipeline on the
  ! real initial-condition particles: convert positions to 64-bit fixed-point and
  ! Hilbert-sort them on the device.  (The full adaptive loop still runs on the
  ! CPU pending the AMR-management kernels; this proves the integrated binary
  ! drives Metal correctly on the actual problem data.)
  call get_environment_variable('RAMSES_METALLIB', m_mlib)
  if (len_trim(m_mlib) == 0) m_mlib = '../../bin/ramses_kernels.metallib'
  m_ierr = mtl_init(trim(m_mlib)//c_null_char)
  if (m_ierr == 0 .and. r%pic) then
     ! Hybrid: enable the in-loop Metal base-level gravity (deposit -> grouped
     ! multigrid Poisson -> force).  amr_step drives it each base step; the CPU
     ! still manages the AMR mesh and solves the finer levels.
     metal_enabled = .true.
     ! GPU refinement flagging defaults ON (correct + ~4x faster than the CPU flag;
     ! see metal_gravity_module).  RAMSES_GPU_FLAG=0 routes flagging back to the CPU
     ! (bit-reproducible vs the historical CPU runs).
     call get_environment_variable('RAMSES_GPU_FLAG', m_mlib)
     if (trim(m_mlib) == '0') metal_flag_on = .false.
     ! Route the hydro godunov step to the GPU (OPT-IN; default CPU hydro).
     call get_environment_variable('RAMSES_GPU_HYDRO', m_mlib)
     if (trim(m_mlib) == '1') metal_hydro_on = .true.
     ! AMR refine ALWAYS runs on the GPU under metal (the RAMSES_GPU_REFINE=0
     ! "CPU refine in a GPU run" hybrid was removed -- known-wrong: hot high-v tail).
     ! grid_dict rebuild after GPU refine (only needed if a CPU consumer reads it).
     call get_environment_variable('RAMSES_REFINE_REHASH', m_mlib)
     if (trim(m_mlib) == '1') refine_rehash = .true.
     call get_environment_variable('RAMSES_REFINE_HOSTFLAG1', m_mlib)
     if (trim(m_mlib) == '1') refine_hostflag1 = .true.
     call get_environment_variable('RAMSES_REFINE_HOSTMED', m_mlib)
     if (trim(m_mlib) == '1') refine_hostmed = .true.
     call get_environment_variable('RAMSES_NCYC_FINE', m_mlib)
     if (len_trim(m_mlib) > 0) read(m_mlib,*) g_ncyc_fine
     call get_environment_variable('RAMSES_NCYC_BASE', m_mlib)
     if (len_trim(m_mlib) > 0) read(m_mlib,*) g_ncyc_base
     call get_environment_variable('RAMSES_SORT_EVERY', m_mlib)
     if (len_trim(m_mlib) > 0) read(m_mlib,*) g_sort_every
     ! Stage 2: convergence-monitor cadence for the fixed-cycle MG (0 = off).
     call get_environment_variable('RAMSES_MG_CHECK_EVERY', m_mlib)
     if (len_trim(m_mlib) > 0) read(m_mlib,*) g_mg_check_every
     ! CUDA-style multigrid driver (Fortran V-cycle + per-leaf Metal kernels).
     call get_environment_variable('RAMSES_METAL_MG', m_mlib)
     if (trim(m_mlib) == '1') metal_mg_driver_on = .true.
     if (trim(m_mlib) == '2') then            ! 2 = driver for ALL levels
        metal_mg_driver_on = .true.; metal_mg_all_levels = .true.
     end if
     write(*,'(A,L1,A,L1)') ' [METAL] in-loop GPU gravity ENABLED; gpu_flag=', metal_flag_on, &
          ' mg_driver=', metal_mg_driver_on
  end if
#endif

  ! C-API (RamsesNG.jl): stop right after setup and hand the fully-initialised
  ! live state to ramses_init, which drives the routines itself.  Default path
  ! (capi_setup_only=.false.) is unchanged — the binary runs the time loop below.
  if (capi_setup_only) then
     capi_last_state => pst%s
     return
  end if

  ! Just in case we only do clump finding
  if(r%clump_only)then
     write(*,*)'Load balancing particle distribution'
     tt1 = mdl_wtime(mdl)
     call r_balance_part(pst,r%levelmin,1,dummy,0)
     tt2 = mdl_wtime(mdl)
     print '(A,F14.7)',' Time elapsed load balancing:',tt2-tt1
     call m_clump_finder(pst,.true.,.false.)
     return
  endif

  write(*,*)'Starting time integration' 

  done = .false.
  do while(.not.done) ! Main time loop

#ifdef _CUDA
     write(str_step,'(A,I0)'),"step_",step
     call nvtxStartRange(trim(str_step), color=5)
     step = step + 1
#endif

     tt1 = mdl_wtime(mdl)

     if(r%verbose)write(*,*)'Entering amr_step_coarse'

     g%epot_tot=0.0D0  ! Reset total potential energy
     g%ekin_tot=0.0D0  ! Reset total kinetic energy
     g%mass_tot=0.0D0  ! Reset total mass
     g%eint_tot=0.0D0  ! Reset total internal energy
     g%emag_tot=0.0D0  ! Reset total magnetic energy

     ! Call base level
     call m_amr_step(pst,r%levelmin,1,done)

     ! New coarse time-step
     g%nstep_coarse=g%nstep_coarse+1

     ! Periodic phase dump (RAMSES_DUMP_PHASE_EVERY=<N>): every N coarse steps write
     ! a phase file phase_s<nstep>_a<aexp>.txt into RAMSES_DUMP_PHASE_DIR (default cwd)
     ! to trace WHEN specific particles diverge (find the scale factor of a bad step).
     block
       character(len=256) :: pdir, pevery, pfn
       integer :: ip, plun, nevery
       call get_environment_variable('RAMSES_DUMP_PHASE_EVERY', pevery)
       if (len_trim(pevery) > 0) then
          read(pevery,*) nevery
          if (nevery > 0 .and. mod(g%nstep_coarse, nevery) == 0) then
             call get_environment_variable('RAMSES_DUMP_PHASE_DIR', pdir)
             if (len_trim(pdir) == 0) pdir = '.'
#ifdef _METAL
             if (metal_enabled) call m_metal_part_to_host(pst)
#endif
             write(pfn,'(A,A,I6.6,A,F8.6,A)') trim(pdir),'/phase_s',g%nstep_coarse,'_a',g%aexp,'.txt'
             open(newunit=plun, file=trim(pfn), status='replace', action='write')
             do ip = 1, pst%s%p%npart
                write(plun,'(2ES20.10,1X,I10)') pst%s%p%xp(ip,1), pst%s%p%vp(ip,1), pst%s%p%idp(ip)
             end do
             close(plun)
             ! per-step oct layout: level + ckey + hkey per oct (to diff GPU vs CPU
             ! refine layout/set/order).  Sync the GPU-owned grid to host first.
             block
               integer :: LL, oo
#ifdef _METAL
               if (metal_enabled) call m_metal_grid_to_host(pst)
#endif
               write(pfn,'(A,A,I6.6,A)') trim(pdir),'/mesh_s',g%nstep_coarse,'.txt'
               open(newunit=plun, file=trim(pfn), status='replace', action='write')
               do LL = pst%s%r%levelmin, pst%s%r%nlevelmax
                  write(plun,'(A,I3,1X,I8)') '#L ', LL, pst%s%m%noct(LL)
               end do
               do LL = pst%s%r%levelmin+1, pst%s%r%nlevelmax
                  do oo = pst%s%m%head(LL), pst%s%m%head(LL)+pst%s%m%noct(LL)-1
                     write(plun,'(I3,1X,I12,1X,I20,4(1X,ES22.13))') LL, pst%s%m%grid(oo)%ckey(1), &
                          pst%s%m%grid(oo)%hkey(1), pst%s%m%phi(1,oo), pst%s%m%phi(2,oo), &
                          pst%s%m%phi_old(1,oo), pst%s%m%phi_old(2,oo)
                  end do
               end do
               close(plun)
               ! 2:1 grid topology dump (RAMSES_DUMP_GRID2TO1): level, ckey, refined(1),
               ! refined(2) for ALL octs all levels -> reconstruct leaf cells + check that
               ! adjacent leaves differ by <=1 level (valid AMR topology).
               block
                 character(len=256) :: gfn
                 integer :: g2lun
                 call get_environment_variable('RAMSES_DUMP_GRID2TO1', gfn)
                 if (len_trim(gfn) > 0) then
                    write(pfn,'(A,A,I6.6,A)') trim(pdir),'/grid2to1_s',g%nstep_coarse,'.txt'
                    open(newunit=g2lun, file=trim(pfn), status='replace', action='write')
                    do LL = pst%s%r%levelmin, pst%s%r%nlevelmax
                       do oo = pst%s%m%head(LL), pst%s%m%head(LL)+pst%s%m%noct(LL)-1
                          write(g2lun,'(I3,1X,I12,2(1X,L1))') LL, pst%s%m%grid(oo)%ckey(1), &
                               pst%s%m%grid(oo)%refined(1), pst%s%m%grid(oo)%refined(2)
                       end do
                    end do
                    close(g2lun)
                 end if
               end block
             end block
          end if
       end if
     end block

     tt2 = mdl_wtime(mdl)
     if(mod(g%nstep_coarse,r%ncontrol)==0)then
        if(.not. done)print '(A,F14.7)',' Time elapsed since last coarse step:',tt2-tt1
#ifdef _METAL
        if(.not. done)call m_output_timer(.false.,'dummy')   ! periodic breakdown (profiling)
        if(.not. done)call m_metal_prof_report()             ! GPU sub-op breakdown
#endif
     endif

     call getmem(core_mem)
     call writemem(core_mem)

#ifdef _CUDA
     call nvtxEndRange()
#endif
  end do

  ! Phase-space dump (RAMSES_DUMP_PHASE=<file>): write x and vx per particle for the
  ! vx-vs-x phase-space plot.  Works for both CPU and GPU (GPU syncs particles to host
  ! first).  1D runs write no output_ dirs, so this is how we extract phase space.
  block
    character(len=256) :: pfn
    integer :: ip, plun
    call get_environment_variable('RAMSES_DUMP_PHASE', pfn)
    if (len_trim(pfn) > 0) then
#ifdef _METAL
       if (metal_enabled) call m_metal_part_to_host(pst)
#endif
       open(newunit=plun, file=trim(pfn), status='replace', action='write')
       do ip = 1, pst%s%p%npart
          ! cols: x  vx  idp  (idp = unique particle label -> exact CPU<->GPU match,
          ! immune to caustic particle-crossing where nearest-x pairing fails)
          write(plun,'(2ES20.10,1X,I10)') pst%s%p%xp(ip,1), pst%s%p%vp(ip,1), pst%s%p%idp(ip)
       end do
       close(plun)
       write(*,'(A,I9,A,A)') ' [PHASE] wrote ', pst%s%p%npart, ' particles to ', trim(pfn)
    end if
  end block

  call m_output_timer(.false.,'dummy')

!  call r_hash_stats(pst)

  return

  end associate

end subroutine adaptive_loop

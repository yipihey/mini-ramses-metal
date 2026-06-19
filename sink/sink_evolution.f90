module sink_evolution_module
  use rho_fine_module, only: cic_weight, cic_index, tsc_weight, tsc_index, pcs_weight, pcs_index
  type :: out_accretion_t
     real(kind=8)::mass
  end type out_accretion_t
contains
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################`
  recursive subroutine r_sink_evolution(pst,ilevel,input_size,output,output_size)
    use mdl_module
    use ramses_commons, only: pst_t
    use mdl_parameters
    implicit none
    type(pst_t)::pst
    integer,VALUE::input_size
    integer::output_size
    type(out_accretion_t)::output,next_output

    integer::ilevel
    integer::rID

    if(pst%nLower>0)then
       rID = mdl_send_request(pst%s%mdl,MDL_SINK_EVOLUTION,pst%iUpper+1,input_size,output_size,ilevel)
       call r_sink_evolution(pst%pLower,ilevel,input_size,output,output_size)
       call mdl_get_reply(pst%s%mdl,rID,output_size,next_output)
       output%mass=output%mass+next_output%mass
    else
       call sink_evolution(pst%s,pst%s%sink,ilevel,output%mass)
    endif

  end subroutine r_sink_evolution
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine sink_evolution(s,p,ilevel,macc_loc)
    use constants
    use amr_parameters, only: ndim, twotondim
    use hydro_parameters, only: nvar, nener
    use ramses_commons, only: ramses_t
    use pm_commons, only: part_t
    use params_module
    use nbors_utils
    use cache_commons
    use cache
    use marshal, only: pack_fetch_refine,unpack_fetch_refine
    use boundaries, only: init_bound_refine
    use godunov_fine_module, only: init_flush_godunov,pack_flush_godunov,unpack_flush_godunov
    use hilbert
    implicit none
    type(ramses_t)::s
    type(part_t)::p
    integer::ilevel
    real(kind=8)::macc_loc
    !==================================================================
    ! This is the RAMSES routine for sink evolution, calling further accretion,
    ! evolution and feedback routines.
    ! Written by Nicholas Choustikov (Apr 2025)
    !==================================================================
    ! Local variables
    real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v,factG ! Units
    real(kind=8)::dx_loc,vol_loc
    integer::nBHnei,nBH_fb_nei
    real(kind=8)::jet_angle,tan_theta,lambda_sonic
    integer::kk,jj,ii,ipart,istep,idim
    real(kind=8),dimension(1:ndim)::x_rel
    real(kind=8)::r_rel,dmacc_loc
    type(msg_large_realdp)::dummy_large_realdp
    real(kind=8)::dMBH_overdt,dMED_overdt,rho_gas,cs_gas,rho_inf,rho_av_all
    real(kind=8)::fbk_ener_agn,fbk_mom_agn
    real(kind=8),dimension(1:ndim)::vel_gas
    logical::output_file

#ifdef HYDRO
#if NDIM==3
    associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

    if(r%verbose)write(*,*)'Entering sink_evolution...'

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Check if high frequency dump
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    output_file = .false.
    if(r%sink_delta_tout>0)then
       istep = int(g%t / r%sink_delta_tout)
       if(istep > p%step_counter)then
          p%step_counter = p%step_counter + 1
          output_file = .true.
       endif
    endif

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Get all units, constants and cell sizes
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    ! Conversion factor from user units to cgs units
    call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)

    ! Gravitational constant
    factG=1
    if(r%cosmo)factG=3d0/4d0/twopi*g%omega_m*g%aexp

    ! Bondi sonic constant
    if(r%use_bondi_lambda)then
       if(abs(r%gamma - 1).le.0.01)then
          lambda_sonic = 0.25d0*exp(1.5d0)
       else if(abs(r%gamma - 5.0d0/3.0d0).le.0.01)then
          lambda_sonic = 0.25d0
       else
          lambda_sonic = 0.5d0**((r%gamma + 1)/(2d0*(r%gamma - 1))) &
               & * (0.25d0*(5d0-3d0*r%gamma))**(-(5d0-3d0*r%gamma)/(2d0*(r%gamma - 1)))
       end if
    else
       lambda_sonic = 1.0d0
    end if

    ! Mesh spacing in that level
    dx_loc=r%boxlen/2**ilevel
    vol_loc=dx_loc**ndim

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Prepare for the B-spline interpolation
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    ! Compute number of cells within B-spline region
    nBHnei = int(r%sink_b_spline_order**ndim)

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Prepare for feedback
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Compute the number of neighbouring cells which may be affected by feedback
    if(r%agn)then
       nBH_fb_nei = 0
       do kk=-r%agn_feedback_radius,r%agn_feedback_radius
          x_rel(3)=dble(kk)
          do jj=-r%agn_feedback_radius,r%agn_feedback_radius
             x_rel(2)=dble(jj)
             do ii=-r%agn_feedback_radius,r%agn_feedback_radius
                x_rel(1)=dble(ii)
                r_rel = norm2(x_rel(:))
                if(r_rel.lt.dble(r%agn_feedback_radius))then
                   nBH_fb_nei = nBH_fb_nei + 1
                end if
             end do
          end do
       end do

       ! Jet geometry safety net
       jet_angle = max(tiny(0.0d0),dble(r%agn_jet_opening_angle))
       jet_angle = min(jet_angle, 180d0)
       tan_theta = tan(pi/180d0*jet_angle/2) ! tangent of half of the opening angle
    end if

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Open Cache
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Open cache for array uold (fetch) and unew (flush)
    call open_cache(mdl, m, pack_size=storage_size(dummy_large_realdp)/32, &
         pack=pack_fetch_refine, unpack=unpack_fetch_refine, &
         init=init_flush_godunov, flush=pack_flush_godunov, &
         combine=unpack_flush_godunov, bound=init_bound_refine)

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Begin loop over sink particles
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    macc_loc=0
    do ipart = p%headp(ilevel), p%tailp(ilevel)

       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
       ! Sink Accretion
       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
       if(r%accretion_type>0)then
          call sink_accretion(s,p,ilevel,ipart,dx_loc,vol_loc,nBHnei,scale_l,scale_t,scale_d, &
               &              factG,lambda_sonic,dmacc_loc,dMBH_overdt,dMED_overdt, &
               &              rho_inf,cs_gas,vel_gas,rho_av_all)
          macc_loc = macc_loc + dmacc_loc
       endif

       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
       ! Dynamics
       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
       if(r%drag_sink)then
          call dynamical_friction(s,p,ilevel,ipart,rho_av_all,cs_gas,vel_gas,factG)
       endif

       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
       ! Sink/AGN Feedback
       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
       if(r%agn)then
          call AGN_feedback(s,p,ilevel,ipart,dx_loc,vol_loc,nBH_fb_nei,scale_v, &
               &            dMBH_overdt,dMED_overdt,tan_theta,fbk_mom_agn,fbk_ener_agn)
       endif

       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
       ! Save sink data at a high cadence if needed
       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
       if(output_file)then
          if(r%agn)then
             call dump_sink_data_fine_AGN(s,p,ipart,ilevel,scale_l,scale_t,scale_d, &
                  &                       dMBH_overdt,dMED_overdt,rho_inf,cs_gas, &
                  &                       fbk_mom_agn,fbk_ener_agn)
          else
             call dump_sink_data_fine(s,p,ipart,ilevel,scale_l,scale_t,scale_d, &
                  &                   dMBH_overdt,dMED_overdt,rho_inf,cs_gas)
          end if
       end if

       ! Periodic box
       do idim=1,ndim
          if(r%periodic(idim))then
             if(p%xp(ipart,idim)< 0.0d0           )p%xp(ipart,idim)=p%xp(ipart,idim)+r%box_size(idim)
             if(p%xp(ipart,idim)>=r%box_size(idim))p%xp(ipart,idim)=p%xp(ipart,idim)-r%box_size(idim)
          endif
       end do

    end do ! End loop over ipart

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Close cache
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    call close_cache(mdl)

    end associate
#endif
#endif
  end subroutine sink_evolution
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine sink_accretion(s,p,ilevel,ipart,dx_loc,vol_loc,nBHnei,scale_l,scale_t,scale_d, &
       &                    factG,lambda_sonic,macc_loc,dMBH_overdt,dMED_overdt, &
       &                    rho_inf,cs_gas,vel_gas,rho_av_all)
    use constants
    use amr_parameters, only: ndim, twotondim
    use hydro_parameters, only: nvar, nener
    use ramses_commons, only: ramses_t
    use pm_commons, only: part_t, cross
    use params_module
    use nbors_utils
    implicit none
    type(ramses_t)::s
    type(part_t)::p
    integer::ilevel,ipart,nBHnei
    real(kind=8)::scale_l,scale_t,scale_d,factG
    real(kind=8)::dx_loc,vol_loc
    real(kind=8)::macc_loc,lambda_sonic
    real(kind=8)::dMBH_overdt,dMEd_overdt,rho_inf,cs_gas,rho_av_all
    !==================================================================
    ! This is the RAMSES routine for sink (black hole) particle accretion.
    ! For now, it is focused on a simple Bondi-Hoyle-Lyttleton accretion scheme.
    ! The routine modifies hydro variables unew, as well as sink particle properties.
    ! Written by Nicholas Choustikov (Apr 2025)
    !==================================================================
    ! Local variables
    real(kind=8),dimension(1:ndim,1:nBHnei)::xBHnei
    integer,dimension(1:ndim,1:nBHnei)::ckeynei
    real(kind=8),dimension(1:nBHnei)::vol
    real(kind=8),dimension(1:ndim)::xcen,xnei,x_rel
    real(kind=8),dimension(1:ndim)::vv,v_rel,x_acc
    integer,dimension(1:ndim)::ckey,ckey_nbor,ckey_div
    integer(kind=8),dimension(0:ndim)::hash_nbor
    real(kind=8)::m_acc,e_acc
    real(kind=8),dimension(1:ndim)::p_acc,l_acc,vel_gas
    real(kind=8),dimension(1:nvar)::passive_acc
    integer::i,j,k,ii,jj,kk,icelln,igridn,ind,idim,ivar,irad
    real(kind=8)::d,e,ethermal,r2_sink,v_bondi,cs,rho_gas,velocity,erad
    real(kind=8)::weight,r_rel
    real(kind=8)::d_acc,m_gas,bondi_mass
    real(kind=8)::weighted_bondi,dMdt_freefall,t_ff
    real(kind=8)::div_cell,total_divergence,div_right,div_left
#ifdef MHD
    real(kind=8)::bx,by,bz,emag
#endif   
   
#ifdef HYDRO
#if NDIM==3
    if(s%r%accretion_type==0)return
    associate(r=>s%r,g=>s%g,m=>s%m)

    if(r%verbose)write(*,*)'Entering sink_accretion...'

    hash_nbor(0) = ilevel+1
    xBHnei=0d0; ckeynei=0d0; vol=0d0
    macc_loc=0d0

    ! Black hole position
    xcen(1:ndim) = (p%xp(ipart,1:ndim)+m%skip(1:ndim)) / dx_loc

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Initialise B-spline interpolation
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    if      (r%sink_b_spline_order==2)then
       call sink_B_spline_weights_CIC(s,xcen(1:ndim),xBHnei,ckeynei,vol,ilevel)
    else if (r%sink_b_spline_order==3)then
       call sink_B_spline_weights_TSC(s,xcen(1:ndim),xBHnei,ckeynei,vol,ilevel)
    else if (r%sink_b_spline_order==4)then
       call sink_B_spline_weights_PCS(s,xcen(1:ndim),xBHnei,ckeynei,vol,ilevel)
    else
       write(*,*)'This is an unknown B-spline order'
       ckeynei = -1 ! To cause a seg-fault
    end if

    ! Initialise sink information at zero
    rho_gas=0d0; vel_gas=0d0; cs_gas=0d0; m_gas=0d0; weighted_bondi=0d0; total_divergence=0d0
    rho_av_all=0d0

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Collect local gas information
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    ! Loop over all cells in the accretion region, collecting physical properties and hash IDs
    do j = 1,nBHnei

       ! Get neighbouring cell coordinates
       ! Note, periodic BCs for xnei are already enforced in sink_B_spline_weights_PCS etc.
       xnei(1:ndim) = xBHnei(1:ndim,j)
       x_rel(1:ndim) = xnei(1:ndim) - xcen(1:ndim)

       ! Get neighboring cell at current level
       hash_nbor(1:ndim)  = ckeynei(1:ndim,j)
       call get_parent_cell(s,hash_nbor,igridn,icelln,flush_cache=.true.,fetch_cache=.true.)

       ! If missing then cycle
       if(igridn==0)cycle

       ! Get the B-spline weights for this cell (they should already be normalised)
       weight = vol(j)

       ! Get physical information
       d                = max(dble(m%uold(icelln,1,igridn)),r%smallr)
       vv(1)            = m%uold(icelln,2,igridn)/d
       vv(2)            = m%uold(icelln,3,igridn)/d
       vv(3)            = m%uold(icelln,4,igridn)/d
       e                = m%uold(icelln,5,igridn)
      
       ! We need to remove all non-thermal energies as they are not accreted
#ifdef MHD
       ! Deal with MHD
       bx=0.5d0*(m%bold(icelln,1,igridn) + m%bold(icelln,4,igridn))
       by=0.5d0*(m%bold(icelln,2,igridn) + m%bold(icelln,5,igridn))
       bz=0.5d0*(m%bold(icelln,3,igridn) + m%bold(icelln,6,igridn))
       emag=0.5d0*(bx**2+by**2+bz**2)
       e = e - emag
#endif

#if NENER>0
       ! Deal with RT
       erad = 0.0d0
       do irad=1,nener
          erad = erad + m%uold(icelln,5+irad,igridn)
       end do
       e = e - erad
#endif

       ethermal         = (e - 0.5d0*d*sum(vv(:)**2)) / d
       cs               = sqrt(max(dble(r%gamma)*(dble(r%gamma)-1.0d0)*ethermal,dble(r%smallc)**2) &
            &           * dble(r%acc_sink_boost)**(-2d0/3d0))

       ! Add to average (weighted) information
       rho_gas          = rho_gas         + d          * weight
       vel_gas(1:ndim)  = vel_gas(1:ndim) + vv(1:ndim) * weight
       cs_gas           = cs_gas          + cs         * weight
       m_gas            = m_gas           + d          * weight * vol_loc
#ifdef GRAV
       rho_av_all       = rho_av_all      + m%rho(icelln,igridn) * weight
#endif
       if(r%accretion_type==2)then
          ! Compute local mass divergence for Bleuler+14 flux accretion
          div_cell = 0
          do idim =1,ndim
             ! 'Right' value
             ckey_div(1:ndim) = ckeynei(1:ndim,j)
             ckey_div(idim) = ckey_div(idim) + 1

             ! Get the cell information
             hash_nbor(1:ndim)  = ckey_div(1:ndim)
             call get_parent_cell(s,hash_nbor,igridn,icelln,flush_cache=.true.,fetch_cache=.true.)
             ! If missing then cycle
             if(igridn==0)cycle

             ! Compute the 'Right' contribution
             div_right = (m%uold(icelln,1+idim,igridn) - max(dble(m%uold(icelln,1,igridn)),r%smallr)*p%vp(ipart,idim))/dx_loc

             ! 'Left' value
             ckey_div(1:ndim) = ckeynei(1:ndim,j)
             ckey_div(idim) = ckey_div(idim) - 1

             ! Get the cell information
             hash_nbor(1:ndim)  = ckey_div(1:ndim)
             call get_parent_cell(s,hash_nbor,igridn,icelln,flush_cache=.true.,fetch_cache=.true.)
             ! If missing then cycle
             if(igridn==0)cycle

             ! Compute the 'Left' contribution
             div_left = (m%uold(icelln,1+idim,igridn) - max(dble(m%uold(icelln,1,igridn)),r%smallr)*p%vp(ipart,idim))/dx_loc

             ! Compute the 'Total' contribution
             div_cell = (div_right - div_left)
          end do
          total_divergence = total_divergence + div_cell
       end if

       ! Compute local bondi rate
       if(r%use_local_bondi_rate)then
          v_rel(1:ndim) = vv(1:ndim) - p%vp(ipart,1:ndim)
          if(r%bondi_use_vrel)then
             v_bondi    = sqrt(sum(v_rel(:)**2) + cs**2)
          else
             v_bondi    = cs
          end if
          if(r%bondi_use_gas_mass)then
             bondi_mass = p%mp(ipart) + d*vol_loc
          else
             bondi_mass = p%mp(ipart)
          end if
          r2_sink     = (factG * bondi_mass / v_bondi**2)**2
          if(r%use_rho_inf)then
             r_rel = norm2(x_rel(:))*dx_loc
             rho_inf = d / (bondi_alpha(r_rel/(r2_sink+tiny(0.0d0))**0.5d0))
          else
             rho_inf = d
          end if
          dMBH_overdt = 4.0d0 * pi * rho_inf * r2_sink * v_bondi * lambda_sonic
          weighted_bondi = weighted_bondi + dMBH_overdt * weight
       end if
    end do

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Compute overall accretion rate
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    ! Compute BHL accretion rate
    if(r%accretion_type==1)then
       if(r%use_local_bondi_rate)then
          dMBH_overdt = weighted_bondi
       else
          ! Bondi velocity
          v_rel(1:ndim) = vel_gas(1:ndim) - p%vp(ipart,1:ndim)
          if(r%bondi_use_vrel)then
             velocity   = norm2(v_rel(:))
             v_bondi    = sqrt(velocity + cs_gas**2)
          else
             v_bondi    = cs_gas
          end if

          ! Bondi mass
          if(r%bondi_use_gas_mass)then
             bondi_mass = p%mp(ipart) + m_gas
          else
             bondi_mass = p%mp(ipart)
          end if

          ! Bondi radius
          r2_sink     = (factG * bondi_mass / v_bondi**2)**2

          ! Density at infinity (using extrapolation)
          if(r%use_rho_inf)then
             rho_inf = rho_gas / (bondi_alpha(dble(r%sink_b_spline_order)*0.5d0*dx_loc/(r2_sink+tiny(0.0d0))**0.5d0))
          else
             rho_inf = rho_gas
          end if

          ! Bondi-Hoyle-Lyttleton accretion rate
          dMBH_overdt = 4.0d0 * pi * rho_inf * r2_sink * v_bondi * lambda_sonic
       end if
       if(r%verbose_sink)write(*,*)'Bondi: ',dMBH_overdt

    ! Compute flux accretion rate
    else if(r%accretion_type==2)then
       ! Use Divergence of the flow as your accretion rate
       dMBH_overdt = -1.0*total_divergence*vol_loc

       ! Applying the correction from Bleuler+14
       if(r%sink_density_threshold.gt.0.0)then
          dMBH_overdt = dMBH_overdt &
               &      * (1 + 0.1d0*log(rho_gas / r%sink_density_threshold))
       endif
       if(r%verbose_sink)write(*,*)'Flux: ',dMBH_overdt

    ! Compute threshold accretion rate
    else if(r%accretion_type==3)then
       dMBH_overdt = 0.5d0*(rho_gas - r%sink_density_threshold)*vol_loc*dble(nBHnei)
    end if

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Limit overall accretion rate
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Add accretion limiters across the entire accretion region
    ! (this preserves the scheme we are using, as compared to cell-specific limiters)

    ! Eddington accretion rate, which introduces an optional cap
    dMED_overdt = 4.0d0 * pi * factG_in_cgs * p%mp(ipart) * mH / (0.1d0 * sigma_T * c_cgs) * scale_t
    if(r%eddington_cap>0)dMBH_overdt = min(dMBH_overdt, dMED_overdt*r%eddington_cap)

    ! If the accretion rate is too low, do nothing
    if((r%eddington_floor>0).and.(dMBH_overdt/dMED_overdt<r%eddington_floor))then
       return
    end if

    ! limiting total accreted mass to 75% of the weighted mass of the accretion region (c.f. Beckmann+2018)
    dMBH_overdt = min(dMBH_overdt, 0.75d0*rho_gas*vol_loc*dble(nBHnei) / g%dtnew(ilevel))

    if(((g%t - p%tp(ipart)).lt.r%t_start_black_hole).and.r%t_start_black_hole.gt.0.0)then
       dMBH_overdt = exp(g%t - p%tp(ipart) - r%t_start_black_hole) * dMBH_overdt
    end if

    if(r%manual_accretion_rate.gt.0.0d0)then
       dMBH_overdt = r%manual_accretion_rate * dMED_overdt
    end if

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Accrete from local cells
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    ! Loop over all cells in the accretion region, proceeding with accretion
    m_acc=0.0d0; e_acc = 0.0d0; x_acc=0.0d0; p_acc=0.0d0; l_acc=0.0d0; passive_acc=0.0d0
    do j = 1, nBHnei

       ! Compute neighbouring cell coordinates
       ! Note, periodic BCs for xnei are already enforced in sink_B_spline_weights_PCS etc.
       xnei(1:ndim) = xBHnei(1:ndim,j)
       x_rel(1:ndim) = xnei(1:ndim) - xcen(1:ndim)

       ! Get neighboring cell at current level
       hash_nbor(1:ndim)  = ckeynei(1:ndim,j)
       call get_parent_cell(s,hash_nbor,igridn,icelln,flush_cache=.true.,fetch_cache=.true.)

       ! If missing cycle
       if(igridn==0)cycle

       ! Get physical information
       d          = max(dble(m%uold(icelln,1,igridn)),r%smallr)
       vv(1:ndim) = m%uold(icelln,2:ndim+1,igridn)/d
       e          = m%uold(icelln,5,igridn)/d

       ! We need to remove all non-thermal energies as they are not accreted
#ifdef MHD
       ! Deal with MHD
       bx = 0.5d0*(m%bold(icelln,1,igridn) + m%bold(icelln,4,igridn))
       by = 0.5d0*(m%bold(icelln,2,igridn) + m%bold(icelln,5,igridn))
       bz = 0.5d0*(m%bold(icelln,3,igridn) + m%bold(icelln,6,igridn))
       emag = 0.5d0*(bx**2 + by**2 + bz**2)
       e = e - emag/d
#endif
#if NENER>0
       ! Deal with non-thermal energies
       erad = 0.0d0
       do irad = 1, nener
          erad = erad + m%uold(icelln,5+irad,igridn)
       end do
       e = e - erad/d
#endif
       ! Get the weight for this cell based on the B-spline interpolation
       weight = vol(j)

       ! Get accreted mass for this cell
       d_acc = dMBH_overdt * g%dtnew(ilevel) * weight / vol_loc
       if (r%mass_weighting) d_acc = d_acc * d / rho_gas

       ! Ensure that the accreted amount is positive
       d_acc = max(d_acc, 0.0d0)

       ! Accrete from the cell
       m%unew(icelln,1,igridn)          = m%unew(icelln,1,igridn)          - d_acc
       m%unew(icelln,2:(ndim+1),igridn) = m%unew(icelln,2:(ndim+1),igridn) - d_acc * vv(1:ndim)
       m%unew(icelln,5,igridn)          = m%unew(icelln,5,igridn)          - d_acc * e

       ! Accrete passive scalars
#if NVAR>5+NENER
       do ivar = 6 + nener, nvar
          m%unew(icelln,ivar,igridn) = m%unew(icelln,ivar,igridn) - d_acc*m%uold(icelln,ivar,igridn)/d
       end do
#endif
       ! Proceed with accretion

       ! Relative velocity
       v_rel(1:ndim) = vv(1:ndim) - p%vp(ipart,1:ndim)

       ! Accreted mass
       m_acc = m_acc + d_acc * vol_loc

       ! Accreted energy
       e_acc = e_acc + d_acc * e * vol_loc

       if(r%momentum_conserving)then
          ! Accreted relative center of mass
          x_acc(1:ndim) = x_acc(1:ndim) + d_acc * x_rel(1:ndim) * vol_loc * dx_loc

          ! Accreted relative momentum
          p_acc(1:ndim) = p_acc(1:ndim) + d_acc * v_rel(1:ndim) * vol_loc

          ! Accreted relative angular momentum
          l_acc(1:ndim) = l_acc(1:ndim) + d_acc * cross(x_rel(1:ndim), v_rel(1:ndim)) * vol_loc * dx_loc
       endif

       ! Passive scalars
#if NVAR>NENER+6
       if(r%agn)then
          do ivar = 6 + nener, nvar
             passive_acc(ivar) = passive_acc(ivar) + d_acc * m%uold(icelln,ivar,igridn) / d * vol_loc
          end do
       end if
#endif
    end do ! End loop over j

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! Add accreted quantities to sink
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    ! Add accreted properties to sink variables
    if(.not.p%static)p%xp(ipart,1:ndim) = p%xp(ipart,1:ndim) + x_acc(1:ndim) / ( p%mp(ipart) + m_acc )
    p%vp(ipart,1:ndim)                  = p%vp(ipart,1:ndim) + p_acc(1:ndim) / ( p%mp(ipart) + m_acc )
    p%jp(ipart,1:ndim)                  = ( p%mp(ipart) * p%jp(ipart,1:ndim) + l_acc(1:ndim) ) / ( p%mp(ipart) + m_acc )
    if(.not.r%fix_sink_mass)p%mp(ipart) = p%mp(ipart) + m_acc

    ! Save accreted mass to total
    macc_loc = macc_loc + m_acc

    end associate
#endif
#endif
  contains
    ! Routine to return alpha, defined as rho/rho_inf, for a critical
    ! Bondi accretion solution. The argument is x = r / r_Bondi.
    ! This is from Krumholz et al. 2004 (AJC)
    REAL(kind=8) function bondi_alpha(x)
      implicit none
      REAL(kind=8) x
      REAL(kind=8), PARAMETER :: XMIN=0.01d0, XMAX=2.0d0
      INTEGER, PARAMETER :: NTABLE=51
      REAL(kind=8) lambda_c, xtable, xtablep1, alpha_exp
      integer idx
      !     Table of alpha values. These correspond to x values that run from
      !     0.01 to 2.0 with uniform logarithmic spacing. The reason for
      !     this choice of range is that the asymptotic expressions are
      !     accurate to better than 2% outside this range.
      REAL(kind=8), PARAMETER, DIMENSION(NTABLE) :: alphatable = (/ &
           820.254, 701.882, 600.752, 514.341, 440.497, 377.381, 323.427, &
           277.295, 237.845, 204.1, 175.23, 150.524, 129.377, 111.27, 95.7613, &
           82.4745, 71.0869, 61.3237, 52.9498, 45.7644, 39.5963, 34.2989, &
           29.7471, 25.8338, 22.4676, 19.5705, 17.0755, 14.9254, 13.0714, &
           11.4717, 10.0903, 8.89675, 7.86467, 6.97159, 6.19825, 5.52812, &
           4.94699, 4.44279, 4.00497, 3.6246, 3.29395, 3.00637, 2.75612, &
           2.53827, 2.34854, 2.18322, 2.03912, 1.91344, 1.80378, 1.70804, &
           1.62439 /)
      !     Define a constant that appears in these formulae
      lambda_c    = exp(1.5d0) / 4
      !     Deal with the off-the-table cases
      if (x .le. XMIN) then
         bondi_alpha = lambda_c / sqrt(2d0 * x**ndim)
      else if (x .ge. XMAX) then
         bondi_alpha = exp(1d0/x)
      else
         !     We are on the table
         idx = floor ((NTABLE-1) * log(x/XMIN) / log(XMAX/XMIN))
         xtable = exp(log(XMIN) + idx*log(XMAX/XMIN)/(NTABLE-1))
         xtablep1 = exp(log(XMIN) + (idx+1)*log(XMAX/XMIN)/(NTABLE-1d0))
         alpha_exp = log(x/xtable) / log(xtablep1/xtable)
         !     Note the extra +1s below because of fortran 1 offset arrays
         bondi_alpha = alphatable(idx+1) * (alphatable(idx+2)/alphatable(idx+1))**alpha_exp
      end if
    end function bondi_alpha
  end subroutine sink_accretion
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine AGN_feedback(s,p,ilevel,ipart,dx_loc,vol_loc,nBH_fb_nei,scale_v, &
       &                  dMBH_overdt,dMED_overdt,tan_theta,fbk_mom_agn,fbk_ener_agn)
    use constants
    use amr_parameters, only: ndim, twotondim
    use hydro_parameters, only: nvar, nener
    use ramses_commons, only: ramses_t
    use pm_commons, only: part_t, cross
    use params_module
    use nbors_utils
    implicit none
    type(ramses_t)::s
    type(part_t)::p
    integer::ilevel,ipart,nBH_fb_nei
    real(kind=8)::dx_loc,vol_loc,scale_v
    real(kind=8)::dMBH_overdt,dMED_overdt,tan_theta
    real(kind=8)::fbk_mom_agn,fbk_ener_agn
    !==================================================================
    ! This is the RAMSES routine for AGN feedback
    ! For now, it is focused on a simple two-regime model to deploy quasar and radio mode feedback
    ! depending on the accretion rate.
    ! Written by Nicholas Choustikov (Apr 2025)
    !==================================================================
    ! Local variables
    real(kind=8)::rr,x,y,z,rrad
    real(kind=8),dimension(1:ndim,1:nBH_fb_nei)::xBH_fb_nei
    integer,dimension(1:ndim,1:nBH_fb_nei)::ckey_fb_nei
    real(kind=8),dimension(1:nBH_fb_nei)::weight_fb_nei
    real(kind=8),dimension(1:ndim)::xcen,xnei,x_rel
    integer(kind=8),dimension(0:ndim)::hash_nbor
    integer::i,j,k,ii,jj,kk,icelln,igridn,ind,idim,ivar,iBHnei
    real(kind=8)::d,e,ethermal,r_rel,rho_gas_fb,energy_agn
    real(kind=8),dimension(1:ndim)::vv
    logical::ok,ok_blast_agn
    real(kind=8)::acc_ratio,jet_mass,local_weight,total_weight,jet_speed
    real(kind=8)::fbk_mass_agn_loc,fbk_mom_agn_loc,fbk_ener_agn_loc
    real(kind=8),dimension(1:ndim)::jet_direction
    real(kind=8)::cone_dist,orth_dist,weight
    real(kind=8),dimension(1:ndim,1:twotondim)::xCIC
    integer,dimension(1:ndim,1:twotondim)::ckeyCIC
    real(kind=8),dimension(1:twotondim)::volCIC
#ifdef MHD
    real(kind=8)::bx,by,bz,emag
#endif

#ifdef HYDRO
#if NDIM==3
    associate(r=>s%r,g=>s%g,m=>s%m)

    if(r%verbose)write(*,*)'Entering sink_accretion...'
    if(.not.r%agn)return

    hash_nbor(0) = ilevel+1
    xBH_fb_nei=0d0; ckey_fb_nei=0d0; weight_fb_nei=0d0

    ! Black hole position
    xcen(1:ndim) = (p%xp(ipart,1:ndim)+m%skip(1:ndim)) / dx_loc

    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    ! AGN Feedback: Set everything up
    !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    ! Check if feedback should go off
    ! For now, always let feedback go off
    ! TODO: Think about whether we need some criteria. Some T_min based on old ramses?
    ok_blast_agn = .true.

    rho_gas_fb=0d0; fbk_mom_agn=0d0; fbk_ener_agn=0d0
    if(ok_blast_agn)then

       ! Compute chi (fraction of Eddington)
       acc_ratio = dMBH_overdt/dMED_overdt
       acc_ratio = max(acc_ratio, 0.0d0)

       ! Compute the jet direction
       jet_direction(1:ndim) = p%jp(ipart,1:ndim) / (norm2(p%jp(ipart,:)) + tiny(0.0d0))

       ! Compute all of the necessary weights
       ! Loop over all possible cells within the feedback region
       iBHnei = 0
       weight_fb_nei = 0d0; xBH_fb_nei = 0d0; total_weight=0d0
       do kk=-r%agn_feedback_radius,r%agn_feedback_radius
          x_rel(3)=dble(kk)
          do jj=-r%agn_feedback_radius,r%agn_feedback_radius
             x_rel(2)=dble(jj)
             do ii=-r%agn_feedback_radius,r%agn_feedback_radius
                x_rel(1)=dble(ii)
                r_rel=norm2(x_rel(:))
                if(r_rel.lt.dble(r%agn_feedback_radius))then
                   iBHnei = iBHnei + 1

                   ! Collect all of the necessary positions and cartesian keys
                   do idim=1,ndim
                      xBH_fb_nei(idim,iBHnei) = x_rel(idim) + xcen(idim)
                      ckey_fb_nei(idim,iBHnei) = int(xBH_fb_nei(idim,iBHnei))
                   end do

                   ! Compute the weight of the cell in question
                   ok=.false.
                   if(acc_ratio.gt.r%agn_fbk_mode_switch_threshold)then
                      ok=.true.
                   else
                      cone_dist = dot_product(x_rel(1:ndim),jet_direction(1:ndim))
                      orth_dist = norm2((x_rel(1:ndim) - cone_dist*jet_direction(1:ndim)))
                      if(orth_dist.le.abs(cone_dist)*tan_theta)ok=.true.

                      if(r_rel.lt.1)ok=.false. ! Exclude the central cell in jet mode
                   end if

                   if(ok)then
                      call psy_function(acc_ratio.gt.r%agn_fbk_mode_switch_threshold,r_rel,local_weight)
                      weight_fb_nei(iBHnei) = local_weight
                      total_weight = total_weight + local_weight

                      hash_nbor(1:ndim)  = ckey_fb_nei(1:ndim,iBHnei)
                      call get_parent_cell(s,hash_nbor,igridn,icelln,flush_cache=.true.,fetch_cache=.true.)

                      ! If missing cycle
                      if(igridn==0)cycle

                      d = max(dble(m%uold(icelln,1,igridn)), r%smallr)
                      rho_gas_fb = rho_gas_fb + d * local_weight
                   end if

                end if
             end do ! End loop over ii
          end do ! End loop over jj
       end do ! End loop over kk

       ! Normalise the weights
       if(total_weight==0.0d0)write(*,*)'PROBLEM: total weight 0'
       weight_fb_nei = weight_fb_nei / total_weight
       rho_gas_fb = rho_gas_fb / total_weight

       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
       ! Administer the AGN Feedback
       !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

       ! Quasar mode energy
       energy_agn = r%epsilon_rad * dMBH_overdt * (c_cgs/scale_v)**2 * g%dtnew(ilevel)
       fbk_ener_agn = r%epsilon_quasar * energy_agn

       ! Radio mode momentum
       fbk_mom_agn  = r%epsilon_radio * r%momentum_boost * energy_agn / (c_cgs/scale_v)

       ! Loop over the affected cells
       do iBHnei=1,nBH_fb_nei

          ! Skip cells with zero weight
          if(weight_fb_nei(iBHnei)==0.0d0)cycle

          call sink_B_spline_weights_CIC(s,xBH_fb_nei(1:ndim,iBHnei),xCIC,ckeyCIC,volCIC,ilevel)

          do j=1,twotondim
             ! Compute neighbouring cell coordinates
             ! Note, periodic BCs for xCIC are already enforced in sink_B_spline_weights_CIC
             xnei(1:ndim) = xCIC(1:ndim,j)
             x_rel(1:ndim) = xnei(1:ndim) - xcen(1:ndim)
             r_rel = norm2(x_rel(:))

             ! Get neighboring cell at current level
             hash_nbor(1:ndim)  = ckeyCIC(1:ndim,j)
             call get_parent_cell(s,hash_nbor,igridn,icelln,flush_cache=.true.,fetch_cache=.true.)

             ! If missing cycle
             if(igridn==0)cycle

             ! Get the local gas properties
             d = max(dble(m%uold(icelln,1,igridn)),r%smallr)
             vv(1:ndim) = m%uold(icelln,2:ndim+1,igridn)/d

             ! Get the weight
             weight = volCIC(j) * weight_fb_nei(iBHnei)
             if (r%agn_use_mass_weighting) weight = weight * d / rho_gas_fb

             ! Proceed with the feedback
             if(acc_ratio.gt.r%agn_fbk_mode_switch_threshold)then

                ! Quasar mode (energy)
                ! Get the local feedback quantities (accounting for weightings)
                fbk_ener_agn_loc = fbk_ener_agn * weight / vol_loc

                ! Now we inject the actual feedback
                m%unew(icelln,5,igridn) = m%unew(icelln,5,igridn) + fbk_ener_agn_loc

             else
                ! Radio mode (momentum and associated work)
                ! Get the local feedback quantities (accounting for weightings)
                fbk_mom_agn_loc  = fbk_mom_agn  * weight / vol_loc

                ! Now we inject the actual feedback (note, all energy is due to work done)
                m%unew(icelln,2:4,igridn) = m%unew(icelln,2:4,igridn) + fbk_mom_agn_loc*dot_product(jet_direction(:),x_rel(:))*jet_direction(1:ndim)/(r_rel+tiny(0.0d0))
                m%unew(icelln,5,igridn)   = m%unew(icelln,5,igridn)   + fbk_mom_agn_loc*dot_product(jet_direction(:),x_rel(:)/(r_rel+tiny(0.0d0)))*dot_product(jet_direction(1:ndim), vv(1:ndim))

             end if

             ! All of the RT stuff can come here.

          end do ! End loop over j
       end do ! End loop over nBH_fb_nei

    end if ! End if ok_blast_agn

    end associate
#endif
#endif
  end subroutine AGN_feedback
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine dynamical_friction(s,p,ilevel,ipart,rho_av_all,cs_gas,vel_gas,factG)
    use constants
    use amr_parameters, only: ndim, twotondim
    use hydro_parameters, only: nvar
    use ramses_commons, only: ramses_t
    use pm_commons, only: part_t, cross
    use params_module
    implicit none
    type(ramses_t)::s
    type(part_t)::p
    integer::ipart,ilevel
    real(kind=8)::rho_av_all,cs_gas,factG
    real(kind=8),dimension(1:ndim)::vel_gas
    !==================================================================
    ! This is the RAMSES routine for dynamical friction, with the goal
    ! of informing the dynamics of the black hole.
    ! Here, we follow the approach of Ostriker 1999.
    ! Written by Nicholas Choustikov (Apr 2025)

    ! See also Beckmann+2018 for a similar approach.
    !==================================================================
    real(kind=8)::mach,vel_gas_mag,I,drag_force
    real(kind=8),dimension(1:ndim)::vel_gas_direction

    if(.not.s%r%drag_sink)return

    ! Calculate the mach number in the local gas
    vel_gas_mag = norm2(vel_gas)
    mach = vel_gas_mag/cs_gas

    ! Calculate the gas velocity direction
    vel_gas_direction(1:ndim) = vel_gas(1:ndim) / (vel_gas_mag + tiny(0.0d0))

    ! Compute the drag force
    if(mach.lt.0.01)then
       I = mach/3.0d0
    else if(abs(mach-1).lt.0.01)then
       I = 0.5*(0.5d0*log((mach+0.01)**2 - 1.0d0 + tiny(0.0d0)) &
            & + 4.0d0 +  0.5d0*log((1.0d0+mach-0.01)/(1.0d0-mach+0.01+tiny(0.0d0))) - mach+0.01)
    else
       ! Value for log(Lambda) taken from Beckmann+2018
       I = 0.5d0*log(mach**2 - 1.0d0 + tiny(0.0d0)) + 4.0d0
    end if
    drag_force = -I * 4.0d0*pi *factG**2 * p%mp(ipart)**2 * rho_av_all / vel_gas_mag**2

    ! Update the velocity of the sink due to the gas
    p%vp(ipart,1:ndim) = p%vp(ipart,1:ndim) - drag_force * vel_gas_direction(1:ndim) * s%g%dtnew(ilevel)

  end subroutine dynamical_friction
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine sink_B_spline_weights_PCS(s,x,xnei,ckey,vol,ilevel)
    use amr_parameters, only: ndim, fourtondim
    use ramses_commons, only: ramses_t
    implicit none
    type(ramses_t)::s
    real(kind=8),dimension(1:ndim)::x
    real(kind=8),dimension(1:ndim,1:fourtondim)::xnei
    integer,dimension(1:ndim,1:fourtondim)::ckey
    real(kind=8),dimension(1:fourtondim)::vol
    integer::ilevel
    !==================================================================
    ! Simple routine to compute B-spline cells and weights for a given sink
    ! Nicholas Choustikov
    !==================================================================
    integer::idim,j
    integer,dimension(1:ndim)::crr,cr,cl,cll
    real(kind=8)::xrr,xr,xl,xll
    real(kind=8),dimension(1:ndim)::wrr,wr,wl,wll

    associate(r=>s%r,m=>s%m)

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

    ! Compute cartesian keys
    ckey = pcs_index(cll,cl,cr,crr)

    ! Compute neighbour positions
    do j = 1,fourtondim
       do idim = 1,ndim
          xnei(idim,j) = dble(ckey(idim,j)) + 0.5d0
       end do
    end do

    end associate

  end subroutine sink_B_spline_weights_PCS
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine sink_B_spline_weights_TSC(s,x,xnei,ckey,vol,ilevel)
    use amr_parameters, only: ndim, threetondim
    use ramses_commons, only: ramses_t
    implicit none
    type(ramses_t)::s
    real(kind=8),dimension(1:ndim)::x
    real(kind=8),dimension(1:ndim,1:threetondim)::xnei
    integer,dimension(1:ndim,1:threetondim)::ckey
    real(kind=8),dimension(1:threetondim)::vol
    integer::ilevel
    !==================================================================
    ! Simple routine to compute B-spline cells and weights for a given sink
    ! Nicholas Choustikov
    !==================================================================
    integer::idim,j
    integer,dimension(1:ndim)::cr,cl,cc
    real(kind=8)::xr,xl,xc
    real(kind=8),dimension(1:ndim)::wr,wl,wc

    associate(r=>s%r,m=>s%m)

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

    ! Compute cartesian keys
    ckey = tsc_index(cl,cc,cr)

    ! Compute neighbour positions
    do j = 1,threetondim
       do idim = 1,ndim
          xnei(idim,j) = dble(ckey(idim,j)) + 0.5d0
       end do
    end do

    end associate

  end subroutine sink_B_spline_weights_TSC
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine sink_B_spline_weights_CIC(s,x,xnei,ckey,vol,ilevel)
    use amr_parameters, only: ndim, twotondim
    use ramses_commons, only: ramses_t
    implicit none
    type(ramses_t)::s
    real(kind=8),dimension(1:ndim)::x
    real(kind=8),dimension(1:ndim,1:twotondim)::xnei
    integer,dimension(1:ndim,1:twotondim)::ckey
    real(kind=8),dimension(1:twotondim)::vol
    integer::ilevel
    !==================================================================
    ! Simple routine to compute B-spline cells and weights for a given sink
    ! Nicholas Choustikov
    !==================================================================
    integer::idim,j
    integer,dimension(1:ndim)::ir,il
    real(kind=8),dimension(1:ndim)::dr,dl

    associate(r=>s%r,m=>s%m)

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

    ! Compute cartesian keys
    ckey = cic_index(il,ir)

    ! Compute neighbour positions
    do j = 1,twotondim
       do idim = 1,ndim
          xnei(idim,j) = dble(ckey(idim,j)) + 0.5d0
       end do
    end do

    end associate

  end subroutine sink_B_spline_weights_CIC
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine psy_function(mode,r,psy)
    use amr_commons
    implicit none

    real(kind=8)::r,psy
    logical::mode

    if(mode)then
       ! Quasar mode
       psy = 1.0d0
    else
       ! Radio mode
       psy = 1.0d0
    end if

  end subroutine psy_function
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine dump_sink_data_fine(s,p,ipart,ilevel,scale_l,scale_t,scale_d, &
       &                         dMBH_overdt,dMED_overdt,rho_inf,cs_gas)
    use amr_parameters, only: ndim
    use ramses_commons, only: ramses_t
    use pm_commons, only: part_t
    use mdl_module, only: mdl_mkdir
    implicit none
    type(ramses_t)::s
    type(part_t)::p
    integer::ilevel,ipart
    real(kind=8)::dMBH_overdt,dMED_overdt,rho_inf,cs_gas
    real(kind=8)::scale_l,scale_t,scale_d
    !==================================================================
    ! Simple routine to dump sink data to a CSV on every fine time step
    ! Nicholas Choustikov
    !==================================================================
    character(LEN=80)::filename
    integer::id_sink_loc,unit
    character(LEN=5)::nchar
    logical::file_exist
    real(kind=8)::scale_m,scale_v
    real(kind=8)::unit_amu,unit_pc,unit_msun,unit_dotM,unit_yr

    associate(r=>s%r, g=>s%g, mdl=>s%mdl)

    if(r%verbose_sink)write(*,*)'Entering output_sink_csv'

    ! Computing extra units and constants
    scale_m = scale_d * scale_l**ndim
    scale_v = scale_l / scale_t
    unit_amu = 1.660538921e-24
    unit_pc = 3.08567758096d18
    unit_msun = 1.98841586d33
    unit_dotM = (scale_m/scale_t)/unit_msun*3600*24*365.25
    unit_yr = 3600*24*365.25

    ! Check if the sinklog folder exists
    filename=TRIM('sinklog')
    inquire(file=filename, exist=file_exist)
    if(.not.file_exist)call mdl_mkdir(mdl,filename)

    ! Get the filename for this sink
    call title(p%idp(ipart),nchar)
    filename=TRIM('sinklog/sink_'//TRIM(nchar)//'.csv')

    ! If this is a new sink, then we need to make the file associated with that sink
    inquire(file=filename, exist=file_exist)
    unit = 10
    if(.not.file_exist)then
       if(r%verbose_sink)write(*,*)'Creating file: ',filename
       open(unit=unit,file=filename,form='formatted')
       write(unit,'(A)')'nstep,time,dt,mass,dMBHdt,dMEDdt,rho_inf,cs_gas,x,y,z,vx,vy,vz,jx,jy,jz'
       close(unit)
    end if

    ! Open the sink file
    open(unit=unit,file=filename,form='formatted',status='unknown',position='append')

    ! Write data to the sink file
    write(unit,'(I10,21(A1,ES21.10),A1,I10)')p%step_counter,',',g%t*scale_t/unit_yr, &
         & ',',g%dtnew(ilevel)*scale_t/unit_yr,',',p%mp(ipart)*scale_m/unit_msun,&
         & ',',dMBH_overdt*unit_dotM,',',dMED_overdt*unit_dotM,&
         & ',',rho_inf*scale_d,',',cs_gas*scale_v,&
         & ',',p%xp(ipart,1),',',p%xp(ipart,2),',',p%xp(ipart,3),&
         & ',',p%vp(ipart,1),',',p%vp(ipart,2),',',p%vp(ipart,3),&
         & ',',p%jp(ipart,1),',',p%jp(ipart,2),',',p%jp(ipart,3)

    ! Close the sink file
    close(unit)

    end associate
  end subroutine dump_sink_data_fine
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
  subroutine dump_sink_data_fine_AGN(s,p,ipart,ilevel,scale_l,scale_t,scale_d, &
       &                             dMBH_overdt,dMED_overdt,rho_inf,cs_gas, &
       &                             fbk_mom_agn,fbk_ener_agn)
    use amr_parameters, only: ndim
    use ramses_commons, only: ramses_t
    use pm_commons, only: part_t
    use mdl_module, only: mdl_mkdir
    implicit none
    type(ramses_t)::s
    type(part_t)::p
    integer::ilevel,ipart
    real(kind=8)::dMBH_overdt,dMED_overdt,rho_inf,cs_gas
    real(kind=8)::fbk_mom_agn,fbk_ener_agn
    real(kind=8)::scale_l,scale_t,scale_d
    !==================================================================
    ! Simple routine to dump sink data to a CSV on every fine time step
    ! Nicholas Choustikov
    !==================================================================
    character(LEN=80)::filename
    integer::id_sink_loc,unit
    character(LEN=5)::nchar
    logical::file_exist
    real(kind=8)::scale_m,scale_v,scale_E
    real(kind=8)::unit_amu,unit_pc,unit_msun,unit_dotM,unit_yr

    associate(r=>s%r, g=>s%g, mdl=>s%mdl)

    if(r%verbose_sink)write(*,*)'Entering output_sink_csv'

    ! Computing extra units and constants
    scale_m = scale_d * scale_l**ndim
    scale_v = scale_l / scale_t
    scale_E = scale_m * scale_v**2
    unit_amu = 1.660538921e-24
    unit_pc = 3.08567758096d18
    unit_msun = 1.98841586d33
    unit_dotM = (scale_m/scale_t)/unit_msun*3600*24*365.25
    unit_yr = 3600*24*365.25

    ! Check if the sinklog folder exists
    filename=TRIM('sinklog')
    inquire(file=filename, exist=file_exist)
    if(.not.file_exist)call mdl_mkdir(mdl,filename)

    ! Get the filename for this sink
    call title(p%idp(ipart),nchar)
    filename=TRIM('sinklog/sink_'//TRIM(nchar)//'.csv')

    ! If this is a new sink, then we need to make the file associated with that sink
    inquire(file=filename, exist=file_exist)
    unit = 10 !+g%myid
    if(.not.file_exist)then
       if(r%verbose_sink)write(*,*)'Creating file: ',filename
       open(unit=unit,file=filename,form='formatted')
       write(unit,'(A)')'nstep,time,dt,mass,dMBHdt,dMEDdt,rho_inf,cs_gas,x,y,z,vx,vy,vz,jx,jy,jz,p_fb,E_fb'
       close(unit)
    end if

    ! Open the sink file
    open(unit=unit,file=filename,form='formatted',status='unknown',position='append')

    ! Write data to the sink file
    write(unit,'(I10,21(A1,ES21.10),A1,I10)')p%step_counter,',',g%t*scale_t/unit_yr,&
         & ',',g%dtnew(ilevel)*scale_t/unit_yr,',',p%mp(ipart)*scale_m/unit_msun,&
         & ',',dMBH_overdt*unit_dotM,',',dMED_overdt*unit_dotM,&
         & ',',rho_inf*scale_d,',',cs_gas*scale_v,&
         & ',',p%xp(ipart,1),',',p%xp(ipart,2),',',p%xp(ipart,3),&
         & ',',p%vp(ipart,1),',',p%vp(ipart,2),',',p%vp(ipart,3),&
         & ',',p%jp(ipart,1),',',p%jp(ipart,2),',',p%jp(ipart,3),&
         & ',',fbk_mom_agn,',',fbk_ener_agn

    ! Close the sink file
    close(unit)

    end associate
  end subroutine dump_sink_data_fine_AGN
  !##############################################################################
  !##############################################################################
  !##############################################################################
  !##############################################################################
end module sink_evolution_module

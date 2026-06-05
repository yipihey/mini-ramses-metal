!============================================================================
! metal_bridge_iface.f90 — ISO_C_BINDING interfaces to the Obj-C++ Metal bridge
! (metal_bridge.mm).  bind(C, name=...) pins the symbol names so there is no
! gfortran/clang name-mangling mismatch.  The Fortran gpu_*.f90 modules call
! these; the big arrays live as MTLBuffers on the bridge side and are reached
! from Fortran via the mtl_ptr_* accessors + c_f_pointer (unified memory).
!============================================================================
module metal_bridge_iface
  use iso_c_binding
  implicit none

  interface
     function mtl_init(path) bind(C, name="mtl_init") result(ierr)
       import :: c_char, c_int
       character(kind=c_char), dimension(*), intent(in) :: path
       integer(c_int) :: ierr
     end function mtl_init

     subroutine mtl_alloc_buffers(ncell, npartmax, hash_size, nlevelmax) &
          bind(C, name="mtl_alloc_buffers")
       import :: c_int
       integer(c_int), value :: ncell, npartmax, hash_size, nlevelmax
     end subroutine mtl_alloc_buffers

     subroutine mtl_sort_part(ilevel, head_idx, num_parts, npartmax) &
          bind(C, name="mtl_sort_part")
       import :: c_int
       integer(c_int), value :: ilevel, head_idx, num_parts, npartmax
     end subroutine mtl_sort_part

     ! --- CUDA-style per-leaf multigrid ops (driven by a Fortran mirror of the
     !     CPU multigrid() driver; see m_metal_multigrid in metal_gravity.f90) ---
     subroutine mtl_mg_build(ilevel, ifine, head_idx, n_fine, mg_cap, box_min, box_max, &
          per0, per1, per2, is_base, dx_fine, has_coarse, tfrac) bind(C, name="mtl_mg_build")
       import :: c_int, c_float
       integer(c_int), value :: ilevel, ifine, head_idx, n_fine, mg_cap, per0, per1, per2, is_base, has_coarse
       integer(c_int), dimension(*), intent(in) :: box_min, box_max
       real(c_float), value :: dx_fine, tfrac
     end subroutine mtl_mg_build

     subroutine mtl_mg_make_mask(ilevel) bind(C, name="mtl_mg_make_mask")
       import :: c_int
       integer(c_int), value :: ilevel
     end subroutine mtl_mg_make_mask

     subroutine mtl_mg_make_initial_phi(ilevel, dx_fine, tfrac, has_coarse) &
          bind(C, name="mtl_mg_make_initial_phi")
       import :: c_int, c_float
       integer(c_int), value :: ilevel, has_coarse
       real(c_float), value :: dx_fine, tfrac
     end subroutine mtl_mg_make_initial_phi

     subroutine mtl_mg_make_rhs(ilevel, fourpi, offset, vol_loc, dx_fine, has_coarse, tfrac) &
          bind(C, name="mtl_mg_make_rhs")
       import :: c_int, c_float
       integer(c_int), value :: ilevel, has_coarse
       real(c_float), value :: fourpi, offset, vol_loc, dx_fine, tfrac
     end subroutine mtl_mg_make_rhs

     function mtl_mg_restrict_mask(ilevel, ifine) bind(C, name="mtl_mg_restrict_mask") result(allmasked)
       import :: c_int
       integer(c_int), value :: ilevel, ifine
       integer(c_int) :: allmasked
     end function mtl_mg_restrict_mask

     subroutine mtl_mg_gauss_seidel(ilevel, ifine, safe, redstep) bind(C, name="mtl_mg_gauss_seidel")
       import :: c_int
       integer(c_int), value :: ilevel, ifine, safe, redstep
     end subroutine mtl_mg_gauss_seidel

     subroutine mtl_mg_smooth(ilevel, ifine, safe, nsweep) bind(C, name="mtl_mg_smooth")
       import :: c_int
       integer(c_int), value :: ilevel, ifine, safe, nsweep
     end subroutine mtl_mg_smooth

     subroutine mtl_mg_cmp_residual(ilevel, ifine) bind(C, name="mtl_mg_cmp_residual")
       import :: c_int
       integer(c_int), value :: ilevel, ifine
     end subroutine mtl_mg_cmp_residual

     subroutine mtl_mg_restrict_residual(ilevel, ifine) bind(C, name="mtl_mg_restrict_residual")
       import :: c_int
       integer(c_int), value :: ilevel, ifine
     end subroutine mtl_mg_restrict_residual

     subroutine mtl_mg_reset_corr(ilevel, ifine) bind(C, name="mtl_mg_reset_corr")
       import :: c_int
       integer(c_int), value :: ilevel, ifine
     end subroutine mtl_mg_reset_corr

     subroutine mtl_mg_interpolate_correct(ilevel, ifine) bind(C, name="mtl_mg_interpolate_correct")
       import :: c_int
       integer(c_int), value :: ilevel, ifine
     end subroutine mtl_mg_interpolate_correct

     function mtl_mg_residual_norm2(ilevel) bind(C, name="mtl_mg_residual_norm2") result(norm2)
       import :: c_int, c_double
       integer(c_int), value :: ilevel
       real(c_double) :: norm2
     end function mtl_mg_residual_norm2

     function mtl_mg_norm_at(ilevel, ifine) bind(C, name="mtl_mg_norm_at") result(norm2)
       import :: c_int, c_double
       integer(c_int), value :: ilevel, ifine
       real(c_double) :: norm2
     end function mtl_mg_norm_at

     subroutine mtl_mg_gauge_pin(ilevel) bind(C, name="mtl_mg_gauge_pin")
       import :: c_int
       integer(c_int), value :: ilevel
     end subroutine mtl_mg_gauge_pin

     subroutine mtl_mg_zeromean_rhs(ilevel, ifine) bind(C, name="mtl_mg_zeromean_rhs")
       import :: c_int
       integer(c_int), value :: ilevel, ifine
     end subroutine mtl_mg_zeromean_rhs

     subroutine mtl_cic_part(ilevel, head_idx, num_parts, npartmax, hash_size, &
          ckey_max, key_off, m_refine, mass_cut, refine_on, per0, per1, per2) &
          bind(C, name="mtl_cic_part")
       import :: c_int, c_long, c_float
       integer(c_int),  value :: ilevel, head_idx, num_parts, npartmax, hash_size
       integer(c_int),  value :: ckey_max, refine_on, per0, per1, per2
       integer(c_long), value :: key_off
       real(c_float),   value :: m_refine, mass_cut
     end subroutine mtl_cic_part

     subroutine mtl_gpu_sort_part(ilevel, head_idx, num_parts) bind(C, name="mtl_gpu_sort_part")
       import :: c_int
       integer(c_int), value :: ilevel, head_idx, num_parts
     end subroutine mtl_gpu_sort_part

     function mtl_gpu_split_part(ilevel, head_idx, num_parts, hash_size, ckey_max, key_off) &
          bind(C, name="mtl_gpu_split_part") result(n_stay)
       import :: c_int, c_long
       integer(c_int), value :: ilevel, head_idx, num_parts, hash_size, ckey_max
       integer(c_long), value :: key_off
       integer(c_int) :: n_stay
     end function mtl_gpu_split_part

     subroutine mtl_flag(head, num, head1, num1, ngridmax, m_refine, nexpand, do_rules) &
          bind(C, name="mtl_flag")
       import :: c_int, c_float
       integer(c_int), value :: head, num, head1, num1, ngridmax, nexpand, do_rules
       real(c_float),  value :: m_refine
     end subroutine mtl_flag

     function mtl_ptr_flag1() bind(C, name="mtl_ptr_flag1") result(p)
       import :: c_ptr; type(c_ptr) :: p
     end function mtl_ptr_flag1

     subroutine mtl_cic_zero(cell_base, ncells) bind(C, name="mtl_cic_zero")
       import :: c_int
       integer(c_int), value :: cell_base, ncells
     end subroutine mtl_cic_zero

     subroutine mtl_cic_finalize(cell_base, ncells) bind(C, name="mtl_cic_finalize")
       import :: c_int
       integer(c_int), value :: cell_base, ncells
     end subroutine mtl_cic_finalize

     subroutine mtl_cic_deposit(ilevel, head_idx, num_parts, npartmax, hash_size, &
          ckey_max, key_off, m_refine, mass_cut, refine_on, per0, per1, per2) &
          bind(C, name="mtl_cic_deposit")
       import :: c_int, c_long, c_float
       integer(c_int),  value :: ilevel, head_idx, num_parts, npartmax, hash_size
       integer(c_int),  value :: ckey_max, refine_on, per0, per1, per2
       integer(c_long), value :: key_off
       real(c_float),   value :: m_refine, mass_cut
     end subroutine mtl_cic_deposit

     subroutine mtl_gauss_seidel(head_idx, num_octs, nsweep, safe) bind(C, name="mtl_gauss_seidel")
       import :: c_int
       integer(c_int), value :: head_idx, num_octs, nsweep, safe
     end subroutine mtl_gauss_seidel

     subroutine mtl_gradient_phi(head_idx, num_octs, dx, tfrac) bind(C, name="mtl_gradient_phi")
       import :: c_int, c_float
       integer(c_int), value :: head_idx, num_octs
       real(c_float),  value :: dx, tfrac
     end subroutine mtl_gradient_phi
     subroutine mtl_save_phi_old(head_idx, num_octs) bind(C, name="mtl_save_phi_old")
       import :: c_int
       integer(c_int), value :: head_idx, num_octs
     end subroutine mtl_save_phi_old

     subroutine mtl_kick_drift_part(ilevel, head_idx, num_parts, npartmax, hash_size, &
          action_part, dtnew_arr, dtold_arr, nlev, box0, box1, box2, per0, per1, per2) &
          bind(C, name="mtl_kick_drift_part")
       import :: c_int, c_float
       integer(c_int), value :: ilevel, head_idx, num_parts, npartmax, hash_size, action_part, nlev
       integer(c_int), value :: per0, per1, per2
       real(c_float),  dimension(*), intent(in) :: dtnew_arr, dtold_arr   ! per-level dt (1..nlev)
       real(c_float),  value :: box0, box1, box2
     end subroutine mtl_kick_drift_part

     subroutine mtl_finalize() bind(C, name="mtl_finalize")
     end subroutine mtl_finalize

     ! H1 connectivity: build flat hash/father/nbor from B.grid (already filled).
     subroutine mtl_build_connectivity(num_octs, levelmin, nlevelmax, &
          box_min, box_max, per0, per1, per2) bind(C, name="mtl_build_connectivity")
       import :: c_int
       integer(c_int), value :: num_octs, levelmin, nlevelmax, per0, per1, per2
       integer(c_int), intent(in) :: box_min(*), box_max(*)
     end subroutine mtl_build_connectivity

     ! Full GPU hash rebuild (parallel atomicCAS insert) + box bounds (no nbor).
     subroutine mtl_conn_rebuild_hash(num_octs, nlevelmax, box_min, box_max) &
          bind(C, name="mtl_conn_rebuild_hash")
       import :: c_int
       integer(c_int), value :: num_octs, nlevelmax
       integer(c_int), intent(in) :: box_min(*), box_max(*)
     end subroutine mtl_conn_rebuild_hash

     ! Rebuild nbor for octs [nbor_head, nbor_head+nbor_num) and father for octs
     ! [father_head, father_head+father_num) on the GPU (per-level, cheap).
     subroutine mtl_conn_build_range(nbor_head, nbor_num, father_head, father_num, nlevelmax) &
          bind(C, name="mtl_conn_build_range")
       import :: c_int
       integer(c_int), value :: nbor_head, nbor_num, father_head, father_num, nlevelmax
     end subroutine mtl_conn_build_range

     ! Declare the real-oct bound: octs 1..ngridmax real, (ngridmax,ncell] = cache region.
     subroutine mtl_set_cache_region(ngridmax) bind(C, name="mtl_set_cache_region")
       import :: c_int
       integer(c_int), value :: ngridmax
     end subroutine mtl_set_cache_region

     ! Materialise coarse-fine boundary cache (ghost) octs for the octs
     ! [head_idx, head_idx+num_octs) at level ilevel; returns #cache octs created.
     function mtl_make_cache(ilevel, head_idx, num_octs, nlevelmax, per0, per1, per2, tfrac, faces_edges_only) &
          bind(C, name="mtl_make_cache") result(ncache)
       import :: c_int, c_float
       integer(c_int), value :: ilevel, head_idx, num_octs, nlevelmax, per0, per1, per2, faces_edges_only
       real(c_float), value :: tfrac
       integer(c_int) :: ncache
     end function mtl_make_cache

     ! Raw byte copy of host type(oct) array into B.grid at 1-based dst_head.
     subroutine mtl_copy_grid_in(host_grid, dst_head, n) bind(C, name="mtl_copy_grid_in")
       import :: c_ptr, c_int
       type(c_ptr), value    :: host_grid
       integer(c_int), value :: dst_head, n
     end subroutine mtl_copy_grid_in

     ! Block until all submitted GPU work completes (call before a host read of a
     ! GPU buffer, or a host write to a buffer an in-flight cmd buffer may use).
     subroutine mtl_drain() bind(C, name="mtl_drain")
     end subroutine mtl_drain

     subroutine mtl_newdt_part(ilevel, head, num, npartmax, vmax, ekin) bind(C, name="mtl_newdt_part")
       import :: c_int, c_double
       integer(c_int), value :: ilevel, head, num, npartmax
       real(c_double) :: vmax, ekin
     end subroutine mtl_newdt_part

     subroutine mtl_get_mg_times(h, v) bind(C, name="mtl_get_mg_times")
       import :: c_double
       real(c_double) :: h, v
     end subroutine mtl_get_mg_times

     subroutine mtl_copy_grid_out(host_grid, src_head, n) bind(C, name="mtl_copy_grid_out")
       import :: c_ptr, c_int
       type(c_ptr), value    :: host_grid
       integer(c_int), value :: src_head, n
     end subroutine mtl_copy_grid_out

     ! GPU AMR refine: create/derefine/compact + rebuild connectivity.  head/noct
     ! are the host per-level arrays (m%head(levelmin)/m%noct(levelmin)); updated
     ! in place.  noct_used/ifree updated.  box_* are the per-level ckey bounds.
     subroutine mtl_refine(ilevel, levelmin, nlevelmax, head, noct, noct_used, ifree, &
          box_min, box_max, ncreate, nkill) bind(C, name="mtl_refine")
       import :: c_int
       integer(c_int), value :: ilevel, levelmin, nlevelmax
       integer(c_int) :: head(*), noct(*), noct_used, ifree, ncreate, nkill
       integer(c_int), intent(in) :: box_min(*), box_max(*)
     end subroutine mtl_refine

     function mtl_ptr_father()  bind(C, name="mtl_ptr_father")  result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_nbor()    bind(C, name="mtl_ptr_nbor")    result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_nref()    bind(C, name="mtl_ptr_nref")    result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_father_mg() bind(C, name="mtl_ptr_father_mg") result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_nbor_mg()   bind(C, name="mtl_ptr_nbor_mg")   result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_phi_old()   bind(C, name="mtl_ptr_phi_old")   result(p); import::c_ptr; type(c_ptr)::p; end function

     ! Buffer pointer accessors (return raw contents pointers for c_f_pointer).
     function mtl_ptr_ipos()      bind(C, name="mtl_ptr_ipos")      result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_vp()        bind(C, name="mtl_ptr_vp")        result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_mp()        bind(C, name="mtl_ptr_mp")        result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_levelp()    bind(C, name="mtl_ptr_levelp")    result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_idp()       bind(C, name="mtl_ptr_idp")       result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_sortp()     bind(C, name="mtl_ptr_sortp")     result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_hkey_part() bind(C, name="mtl_ptr_hkey_part") result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_grid()      bind(C, name="mtl_ptr_grid")      result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_hash_key()  bind(C, name="mtl_ptr_hash_key")  result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_hash_val()  bind(C, name="mtl_ptr_hash_val")  result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_ckey_max()  bind(C, name="mtl_ptr_ckey_max")  result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_key_off()   bind(C, name="mtl_ptr_key_off")   result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_rho()       bind(C, name="mtl_ptr_rho")       result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_phi()       bind(C, name="mtl_ptr_phi")       result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_f()         bind(C, name="mtl_ptr_f")         result(p); import::c_ptr; type(c_ptr)::p; end function
     subroutine mtl_gas_deposit(head, num, vol_loc, fp_scale) bind(C, name="mtl_gas_deposit")
       import :: c_int, c_double; integer(c_int), value :: head, num; real(c_double), value :: vol_loc, fp_scale
     end subroutine mtl_gas_deposit
     function mtl_epot(head, num, fp_scale) bind(C, name="mtl_epot") result(s)
       import :: c_int, c_double; integer(c_int), value :: head, num; real(c_double), value :: fp_scale; real(c_double) :: s
     end function mtl_epot
     subroutine mtl_hydro_fill_boundary(head, num, gamma) bind(C, name="mtl_hydro_fill_boundary")
       import :: c_int, c_double; integer(c_int), value :: head, num; real(c_double), value :: gamma
     end subroutine mtl_hydro_fill_boundary
     subroutine mtl_set_boundary(nbound, per0, per1, per2, btype, bdir, bshift, &
          bconst, bckmin, bckmax, nlevp1) bind(C, name="mtl_set_boundary")
       import :: c_int, c_float
       integer(c_int), value :: nbound, per0, per1, per2, nlevp1
       integer(c_int) :: btype(*), bdir(*), bshift(*), bckmin(*), bckmax(*)
       real(c_float)  :: bconst(*)
     end subroutine mtl_set_boundary
     function mtl_ptr_uold()      bind(C, name="mtl_ptr_uold")      result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_unew()      bind(C, name="mtl_ptr_unew")      result(p); import::c_ptr; type(c_ptr)::p; end function

     ! Hydro orchestration (gpu_hydro.cuf port).
     subroutine mtl_godunov_fine(ilevel, head, num, levelmin, levelmax, &
          gamma, dt, dx, slope, riemann, courant, fp_scale) bind(C, name="mtl_godunov_fine")
       import :: c_int, c_double
       integer(c_int), value :: ilevel, head, num, levelmin, levelmax, slope, riemann
       real(c_double), value :: gamma, dt, dx, courant, fp_scale
     end subroutine mtl_godunov_fine
     subroutine mtl_hydro_set_unew(head, num) bind(C, name="mtl_hydro_set_unew")
       import :: c_int; integer(c_int), value :: head, num
     end subroutine mtl_hydro_set_unew
     subroutine mtl_hydro_set_uold(head, num) bind(C, name="mtl_hydro_set_uold")
       import :: c_int; integer(c_int), value :: head, num
     end subroutine mtl_hydro_set_uold
     subroutine mtl_hydro_upload(head, num) bind(C, name="mtl_hydro_upload")
       import :: c_int; integer(c_int), value :: head, num
     end subroutine mtl_hydro_upload
     subroutine mtl_hydro_godunov_only(ilevel, head, num, levelmin, levelmax, &
          gamma, dt, dx, slope, riemann, courant, fp_scale) bind(C, name="mtl_hydro_godunov_only")
       import :: c_int, c_double
       integer(c_int), value :: ilevel, head, num, levelmin, levelmax, slope, riemann
       real(c_double), value :: gamma, dt, dx, courant, fp_scale
     end subroutine mtl_hydro_godunov_only
     subroutine mtl_hydro_fill_cache(head, num, interpol_var, interpol_type, smallr) &
          bind(C, name="mtl_hydro_fill_cache")
       import :: c_int, c_double
       integer(c_int), value :: head, num, interpol_var, interpol_type
       real(c_double), value :: smallr
     end subroutine mtl_hydro_fill_cache
     subroutine mtl_hydro_reflux_zero(head, num) bind(C, name="mtl_hydro_reflux_zero")
       import :: c_int; integer(c_int), value :: head, num
     end subroutine mtl_hydro_reflux_zero
     subroutine mtl_hydro_grav(head, num, gamma, dt) bind(C, name="mtl_hydro_grav")
       import :: c_int, c_double; integer(c_int), value :: head, num; real(c_double), value :: gamma, dt
     end subroutine mtl_hydro_grav
     function mtl_ptr_reflux_lo() bind(C, name="mtl_ptr_reflux_lo") result(p); import::c_ptr; type(c_ptr)::p; end function
     function mtl_ptr_reflux_hi() bind(C, name="mtl_ptr_reflux_hi") result(p); import::c_ptr; type(c_ptr)::p; end function
     subroutine mtl_hydro_reflux_finalize(head, num, fp_scale) bind(C, name="mtl_hydro_reflux_finalize")
       import :: c_int, c_double; integer(c_int), value :: head, num; real(c_double), value :: fp_scale
     end subroutine mtl_hydro_reflux_finalize
     function mtl_hydro_cmpdt(head, num, gamma, dx, courant, fp_scale, mass, ekin, eint) &
          bind(C, name="mtl_hydro_cmpdt") result(dt)
       import :: c_int, c_double
       integer(c_int), value :: head, num
       real(c_double), value :: gamma, dx, courant, fp_scale
       real(c_double), intent(out) :: mass, ekin, eint
       real(c_double) :: dt
     end function mtl_hydro_cmpdt
     subroutine mtl_hydro_flag(head, num, gamma, err_grad_d, err_grad_p, floor_d, floor_p) &
          bind(C, name="mtl_hydro_flag")
       import :: c_int, c_double
       integer(c_int), value :: head, num
       real(c_double), value :: gamma, err_grad_d, err_grad_p, floor_d, floor_p
     end subroutine mtl_hydro_flag
  end interface

end module metal_bridge_iface

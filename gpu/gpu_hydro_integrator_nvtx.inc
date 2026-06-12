#ifdef INTEGRATOR_NVTX
  ! Phased integrator launches for nsys (INTEGRATOR_NVTX=1). Host ranges in
  ! gpu_godunov follow the gpu_cic_part_large:multipole naming pattern; shared
  ! state spills to integrator_ws between launches.

  attributes(device) subroutine integrator_ws_save(sgid, ws, &
       & local_subgrid, &
#if NVAR>5+NENER
       & scalar_subgrid, &
#endif
#ifdef MHD
       & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
       & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
       & left_interfaces_z, right_interfaces_z)
    integer, value :: sgid
    type(integrator_subgrid_ws_t), device, intent(inout) :: ws(:)
    type(subgrid_6x6x6cell_primitive), intent(in) :: local_subgrid
#if NVAR>5+NENER
    real(kind=dp), intent(in) :: scalar_subgrid(0:2*nsubgrid+3, 0:2*nsubgrid+3, 0:2*nsubgrid+3, 1:nscalar)
#endif
#ifdef MHD
    type(subgrid_face_b_t), intent(in) :: bf
    type(subgrid_2x3x3edge_primitive), intent(in) :: emfx
    type(subgrid_3x2x3edge_primitive), intent(in) :: emfy
    type(subgrid_3x3x2edge_primitive), intent(in) :: emfz
    type(subgrid_2x3x3edge_corner), intent(in) :: corner_x
    type(subgrid_3x2x3edge_corner), intent(in) :: corner_y
    type(subgrid_3x3x2edge_corner), intent(in) :: corner_z
#endif
    type(subgrid_3x2x2cell_primitive), intent(in) :: left_interfaces_x, right_interfaces_x
    type(subgrid_2x3x2cell_primitive), intent(in) :: left_interfaces_y, right_interfaces_y
    type(subgrid_2x2x3cell_primitive), intent(in) :: left_interfaces_z, right_interfaces_z
    type(integrator_subgrid_ws_t) :: snap

    if (threadIdx%x == 1) then
       snap%local_subgrid = local_subgrid
#if NVAR>5+NENER
       snap%scalar_subgrid = scalar_subgrid
#endif
       snap%left_interfaces_x = left_interfaces_x
       snap%right_interfaces_x = right_interfaces_x
       snap%left_interfaces_y = left_interfaces_y
       snap%right_interfaces_y = right_interfaces_y
       snap%left_interfaces_z = left_interfaces_z
       snap%right_interfaces_z = right_interfaces_z
#ifdef MHD
       snap%bf = bf
       snap%emfx = emfx
       snap%emfy = emfy
       snap%emfz = emfz
       snap%corner_x = corner_x
       snap%corner_y = corner_y
       snap%corner_z = corner_z
#endif
       ws(sgid) = snap
    endif
  end subroutine integrator_ws_save

  attributes(device) subroutine integrator_ws_load(sgid, ws, &
       & local_subgrid, &
#if NVAR>5+NENER
       & scalar_subgrid, &
#endif
#ifdef MHD
       & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
       & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
       & left_interfaces_z, right_interfaces_z)
    integer, value :: sgid
    type(integrator_subgrid_ws_t), device, intent(in) :: ws(:)
    type(subgrid_6x6x6cell_primitive), intent(out) :: local_subgrid
#if NVAR>5+NENER
    real(kind=dp), intent(out) :: scalar_subgrid(0:2*nsubgrid+3, 0:2*nsubgrid+3, 0:2*nsubgrid+3, 1:nscalar)
#endif
#ifdef MHD
    type(subgrid_face_b_t), intent(out) :: bf
    type(subgrid_2x3x3edge_primitive), intent(out) :: emfx
    type(subgrid_3x2x3edge_primitive), intent(out) :: emfy
    type(subgrid_3x3x2edge_primitive), intent(out) :: emfz
    type(subgrid_2x3x3edge_corner), intent(out) :: corner_x
    type(subgrid_3x2x3edge_corner), intent(out) :: corner_y
    type(subgrid_3x3x2edge_corner), intent(out) :: corner_z
#endif
    type(subgrid_3x2x2cell_primitive), intent(out) :: left_interfaces_x, right_interfaces_x
    type(subgrid_2x3x2cell_primitive), intent(out) :: left_interfaces_y, right_interfaces_y
    type(subgrid_2x2x3cell_primitive), intent(out) :: left_interfaces_z, right_interfaces_z
    type(integrator_subgrid_ws_t) :: snap

    if (threadIdx%x == 1) snap = ws(sgid)
    call syncthreads()
    if (threadIdx%x == 1) then
       local_subgrid = snap%local_subgrid
#if NVAR>5+NENER
       scalar_subgrid = snap%scalar_subgrid
#endif
       left_interfaces_x = snap%left_interfaces_x
       right_interfaces_x = snap%right_interfaces_x
       left_interfaces_y = snap%left_interfaces_y
       right_interfaces_y = snap%right_interfaces_y
       left_interfaces_z = snap%left_interfaces_z
       right_interfaces_z = snap%right_interfaces_z
#ifdef MHD
       bf = snap%bf
       emfx = snap%emfx
       emfy = snap%emfy
       emfz = snap%emfz
       corner_x = snap%corner_x
       corner_y = snap%corner_y
       corner_z = snap%corner_z
#endif
    endif
    call syncthreads()
  end subroutine integrator_ws_load

  attributes(global) subroutine hydro_integrator_phase_kernel(grid, uold, unew, bold, bnew, &
       & f, father, nbor, ws, head_idx, num_subgrids, ngridmax, ilevel, levelmin, levelmax, &
       & gamma, smallr, smallc, dt, dx, slope, slope_mag, riemann, riemann2d, &
       & switch_llf_dmin, switch_llf_pmin, induction, constant_gravity, phase)
    type(oct), device :: grid(*)
    integer, device :: father(*)
    integer, device :: nbor(1:subgridsize,*)
    real(kind=8), device :: constant_gravity(1:3)
    real(kind=dp), device :: f(1:twotondim,1:ndim,*)
    real(kind=dp), device :: uold(1:twotondim,1:nvar,*)
    real(kind=dp), device :: unew(1:twotondim,1:nvar,*)
    real(kind=dp), device :: bold(1:twotondim,1:6,*)
    real(kind=dp), device :: bnew(1:twotondim,1:6,*)
    type(integrator_subgrid_ws_t), device, intent(inout) :: ws(:)
    integer(kind=4), intent(in), value :: head_idx, num_subgrids, ngridmax
    integer, intent(in), value :: ilevel, levelmin, levelmax
    real(kind=dp), intent(in), value :: gamma, smallr, smallc, dt, dx
    integer, intent(in), value :: slope, slope_mag, riemann, riemann2d
    real(kind=dp), intent(in), value :: switch_llf_dmin, switch_llf_pmin
    logical, intent(in), value :: induction
    integer, intent(in), value :: phase
#include "hydro_integrator_shared_decl.inc"
    integer :: block_idx, thread_idx, subgrid_idx
    real(kind=dp) :: dtdx

    block_idx = blockIdx%x - 1
    thread_idx = threadIdx%x - 1
    if (block_idx >= num_subgrids) return
    subgrid_idx = head_idx + block_idx
    dtdx = dt/dx

    select case (phase)
    case (hydro_int_cons2prim)
       call subgrid_conserved_2_primitive(grid, uold, bold, f, constant_gravity, nbor, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, &
#endif
            & head_idx, block_idx, thread_idx, gamma, smallr, smallc, dt, local_subgrid)
       call syncthreads()
       call integrator_ws_save(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)

    case (hydro_int_trace_3d)
       call integrator_ws_load(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
       call trace_3d(local_subgrid, thread_idx, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, corner_x, corner_y, corner_z, &
            & smallr, smallc, slope_mag, induction, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z, gamma, dtdx, slope)
       call syncthreads()
       call integrator_ws_save(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)

    case (hydro_int_flux)
       call integrator_ws_load(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
       call riemann_driver(left_interfaces_x, right_interfaces_x, left_interfaces_y, &
            & right_interfaces_y, left_interfaces_z, right_interfaces_z, thread_idx, &
            & gamma, smallr, smallc, riemann)
       call syncthreads()
       call integrator_ws_save(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)

#ifdef MHD
    case (hydro_int_zero_euler)
       call integrator_ws_load(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
       call zero_euler_fluxes_mhd(left_interfaces_x, left_interfaces_y, left_interfaces_z, thread_idx)
       call syncthreads()
       call integrator_ws_save(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)

    case (hydro_int_emf)
       call integrator_ws_load(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
       call cmp_mag_flx_mhd(corner_x, corner_y, corner_z, emfx, emfy, emfz, thread_idx, &
            & gamma, smallr, smallc, switch_llf_dmin, switch_llf_pmin, riemann2d)
       call syncthreads()
       call integrator_ws_save(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
#endif

    case (hydro_int_zero_fine)
       call integrator_ws_load(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
       call zero_fine_fluxes(local_subgrid, left_interfaces_x, left_interfaces_y, left_interfaces_z, thread_idx)
#ifdef MHD
       call zero_fine_emf(local_subgrid, emfx, emfy, emfz, thread_idx)
#endif
       call syncthreads()
       call integrator_ws_save(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)

    case (hydro_int_update)
       call integrator_ws_load(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
       call conservative_update(unew, bnew, nbor, left_interfaces_x, left_interfaces_y, left_interfaces_z, &
#ifdef MHD
            & emfx, emfy, emfz, &
#endif
            & head_idx, block_idx, thread_idx, dtdx)

    case (hydro_int_coarse)
       call integrator_ws_load(subgrid_idx, ws, local_subgrid, &
#if NVAR>5+NENER
            & scalar_subgrid, &
#endif
#ifdef MHD
            & bf, emfx, emfy, emfz, corner_x, corner_y, corner_z, &
#endif
            & left_interfaces_x, right_interfaces_x, left_interfaces_y, right_interfaces_y, &
            & left_interfaces_z, right_interfaces_z)
       call coarse_cell_update(grid, unew, bnew, father, nbor, left_interfaces_x, left_interfaces_y, left_interfaces_z, &
#ifdef MHD
            & emfx, emfy, emfz, &
#endif
            & head_idx, block_idx, thread_idx, ngridmax, dtdx)
    end select
  end subroutine hydro_integrator_phase_kernel
#endif

!=============================================================================
!  cooling_test_gpu
!=============================================================================
!  Standalone consistency test for the *equilibrium* cooling solver, comparing
!  the CPU implementation (cooling_module::solve_cooling) against the GPU
!  implementation (cooling_device::solve_cooling_gpu) cell-by-cell.
!
!  Both paths share ONE cooling table (built once on the host with set_table
!  and uploaded verbatim to the device with gpu_upload_cooling_table), so any
!  difference in the output reflects only CPU-vs-GPU arithmetic in the ODE
!  integrator itself -- not a different table, model, or set of inputs.
!
!  Build (from bin/, NVHPC is required for the GPU kernels):
!    Add this target to bin/Makefile (mirrors the existing test_cooling rule):
!
!      cooling_test_gpu: $(MODOBJ) cooling_test_gpu.o
!      	$(F90) $(FFLAGS) $(MODOBJ) cooling_test_gpu.o -o cooling_test_gpu $(LIBS)
!
!    then:
!      cd bin
!      make clean
!      make cooling_test_gpu NDIM=3 HYDRO=1 COMPILER=NVHPC
!
!  Run (from the repository root or bin/, no namelist needed):
!      bin/cooling_test_gpu        # or  ./cooling_test_gpu  from bin/
!
!  The program prints, for several timesteps, the maximum and mean relative
!  difference between the CPU and GPU delta(T/mu), the worst-offending cell,
!  and a final PASS/FAIL verdict.
!=============================================================================

module cooling_test_kernel
  use amr_parameters, only: dp
  use cooling_device, only: solve_cooling_gpu
  implicit none
contains

  !---------------------------------------------------------------------------
  ! One GPU thread per cell. Each thread runs the device-side equilibrium
  ! cooling ODE integrator to completion and stores the resulting delta(T/mu).
  !---------------------------------------------------------------------------
  attributes(global) subroutine test_cooling_kernel( &
       & nH, T2, zsolar, boost, dt, X_frac, deltaT2, ncell)
    implicit none
    integer, value :: ncell
    real(dp)       :: nH(ncell), T2(ncell), zsolar(ncell), boost(ncell)
    real(dp)       :: deltaT2(ncell)
    real(dp), value :: dt, X_frac

    integer  :: i
    real(dp) :: dT2

    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (i > ncell) return

    ! Same call signature the production cooling_kernel uses internally.
    call solve_cooling_gpu(nH(i), T2(i), zsolar(i), boost(i), dt, X_frac, dT2)
    deltaT2(i) = dT2

  end subroutine test_cooling_kernel

end module cooling_test_kernel

!=============================================================================
program cooling_test_gpu
!=============================================================================
  use amr_parameters,   only: dp
  use constants,        only: Myr2sec
  use cooling_module,   only: cooling_t, set_table, solve_cooling
  use cooling_device,   only: gpu_upload_cooling_table
  use cooling_test_kernel, only: test_cooling_kernel
  use cudafor
  implicit none

  ! ---- Problem setup -------------------------------------------------------
  type(cooling_t) :: c

  integer, parameter :: nnH = 16          ! number of densities
  integer, parameter :: nT  = 16          ! number of temperatures
  integer, parameter :: ncell = nnH * nT  ! total cells solved at once

  ! Sweep ranges (CGS): nH in H/cc, T2 = T/mu in K.
  real(dp), parameter :: nH_lo = 1.0d-4, nH_hi = 1.0d4
  real(dp), parameter :: T2_lo = 1.0d1,  T2_hi = 1.0d10   ! top points -> bremsstrahlung (>1e9)
  real(dp), parameter :: Zsolar_test = 1.0d0   ! solar metallicity (exercises metal term)
  real(dp), parameter :: boost_test  = 1.0d0   ! no self-shielding

  ! Timesteps to test, in Myr.
  integer, parameter :: ndt = 6
  real(dp), parameter :: dt_Myr(ndt) = (/ 1.0d-3, 1.0d-1, 1.0d0, 1.0d1, 1.0d2, 1.0d3 /)

  ! Verdict tolerance on the relative difference. CPU and GPU use identical
  ! arithmetic but differ in transcendental-function rounding (log10/10**/sqrt)
  ! and FMA contraction, so exact bit-agreement is not expected.
  real(dp), parameter :: tol = 1.0d-6

  ! ---- Host arrays ---------------------------------------------------------
  real(dp), allocatable :: nH(:), T2(:), zsolar(:), boost(:)
  real(dp), allocatable :: dT2_cpu(:), dT2_gpu(:)

  ! ---- Device arrays -------------------------------------------------------
  real(dp), allocatable, device :: nH_d(:), T2_d(:), zsolar_d(:), boost_d(:)
  real(dp), allocatable, device :: dT2_d(:)

  ! ---- Locals --------------------------------------------------------------
  real(dp) :: dlognH, dlogT, dt, relerr, abserr, denom
  real(dp) :: max_rel, sum_rel, worst_nH, worst_T2, worst_cpu, worst_gpu
  real(dp) :: overall_max_rel
  integer  :: i, j, idx, idt, istat, nbad
  integer  :: nthreads, nblocks
  logical  :: passed

  ! -------------------------------------------------------------------------
  ! 1. Build the equilibrium cooling table once on the host (z = 0, i.e.
  !    aexp = 1). The default cooling_t selects the Courty UV-background model,
  !    exactly as a non-cosmological RAMSES run would.
  ! -------------------------------------------------------------------------
  write(*,'(A)') '==================================================================='
  write(*,'(A)') ' Equilibrium cooling: CPU (solve_cooling) vs GPU (solve_cooling_gpu)'
  write(*,'(A)') '==================================================================='
  write(*,'(A)') ' Building cooling table on host (set_table, aexp = 1) ...'
  call set_table(c, 1.0d0)
  write(*,'(A,I0,A,I0)') '   table dimensions: n1(nH) = ', c%table%n1, &
       & ',  n2(T2) = ', c%table%n2
  write(*,'(A,F6.4)')    '   hydrogen mass fraction X = ', c%X

  ! -------------------------------------------------------------------------
  ! 2. Upload the SAME table to the device module arrays used by
  !    solve_cooling_gpu. After this both solvers see identical rates.
  ! -------------------------------------------------------------------------
  write(*,'(A)') ' Uploading the identical table to the GPU ...'
  call gpu_upload_cooling_table(c)

  ! -------------------------------------------------------------------------
  ! 3. Build the cell batch: a 2-D log-spaced sweep over (nH, T2).
  ! -------------------------------------------------------------------------
  allocate(nH(ncell), T2(ncell), zsolar(ncell), boost(ncell))
  allocate(dT2_cpu(ncell), dT2_gpu(ncell))
  allocate(nH_d(ncell), T2_d(ncell), zsolar_d(ncell), boost_d(ncell), dT2_d(ncell))

  dlognH = (log10(nH_hi) - log10(nH_lo)) / dble(nnH - 1)
  dlogT  = (log10(T2_hi) - log10(T2_lo)) / dble(nT  - 1)
  idx = 0
  do i = 1, nnH
     do j = 1, nT
        idx = idx + 1
        nH(idx)     = 10.0d0**(log10(nH_lo) + dlognH * dble(i - 1))
        T2(idx)     = 10.0d0**(log10(T2_lo) + dlogT  * dble(j - 1))
        zsolar(idx) = Zsolar_test
        boost(idx)  = boost_test
     end do
  end do

  write(*,'(A,I0,A)') ' Solving ', ncell, ' cells per timestep on both devices.'
  write(*,'(A)') ''

  ! Copy the (constant) inputs to the device once; only dt changes per loop.
  nH_d     = nH
  T2_d     = T2
  zsolar_d = zsolar
  boost_d  = boost

  nthreads = 128
  nblocks  = (ncell + nthreads - 1) / nthreads

  ! -------------------------------------------------------------------------
  ! 4. Loop over timesteps: solve on CPU and GPU, then compare.
  ! -------------------------------------------------------------------------
  write(*,'(A)') '   dt [Myr]      max rel err      mean rel err    worst cell (nH, T2 in)'
  write(*,'(A)') '   -------------------------------------------------------------------------'

  overall_max_rel = 0.0d0
  passed = .true.

  do idt = 1, ndt
     dt = dt_Myr(idt) * Myr2sec   ! seconds (both solvers expect CGS seconds)

     ! --- CPU reference ----------------------------------------------------
     ! solve_cooling reads nH/T2/zsolar/boost and writes only dT2_cpu.
     call solve_cooling(c, nH, T2, zsolar, boost, dt, dT2_cpu, ncell)

     ! --- GPU under test ---------------------------------------------------
     call test_cooling_kernel<<<nblocks, nthreads>>>( &
          & nH_d, T2_d, zsolar_d, boost_d, dt, c%X, dT2_d, ncell)
     istat = cudaDeviceSynchronize()
     if (istat /= 0) then
        write(*,'(A,A)') ' CUDA error after kernel: ', cudaGetErrorString(istat)
        stop 1
     end if
     dT2_gpu = dT2_d   ! D -> H

     ! --- Compare ----------------------------------------------------------
     max_rel  = 0.0d0
     sum_rel  = 0.0d0
     worst_nH = 0.0d0; worst_T2 = 0.0d0; worst_cpu = 0.0d0; worst_gpu = 0.0d0
     do idx = 1, ncell
        abserr = abs(dT2_gpu(idx) - dT2_cpu(idx))
        ! Relative to the CPU change, with a small absolute floor so that
        ! cells with a near-zero change do not produce a meaningless ratio.
        denom  = max(abs(dT2_cpu(idx)), 1.0d-6 * T2(idx))
        relerr = abserr / denom
        sum_rel = sum_rel + relerr
        if (relerr > max_rel) then
           max_rel   = relerr
           worst_nH  = nH(idx)
           worst_T2  = T2(idx)
           worst_cpu = dT2_cpu(idx)
           worst_gpu = dT2_gpu(idx)
        end if
     end do

     overall_max_rel = max(overall_max_rel, max_rel)
     if (max_rel > tol) passed = .false.

     write(*,'(2X,1PE10.3,3X,1PE12.4,5X,1PE12.4,5X,A,1PE9.2,A,1PE9.2,A)') &
          & dt_Myr(idt), max_rel, sum_rel / dble(ncell), &
          & '(', worst_nH, ', ', worst_T2, ')'
  end do

  ! -------------------------------------------------------------------------
  ! 5. Detailed dump for the largest timestep (most subcycles -> most chance
  !    for divergence), for a handful of representative cells.
  ! -------------------------------------------------------------------------
  write(*,'(A)') ''
  write(*,'(A,F8.2,A)') ' Sample cells at dt = ', dt_Myr(ndt), ' Myr:'
  write(*,'(A)') '       nH [H/cc]     T2_in [K]    dT2_cpu [K]    dT2_gpu [K]     rel err'
  write(*,'(A)') '   -------------------------------------------------------------------------'
  dt = dt_Myr(ndt) * Myr2sec
  call solve_cooling(c, nH, T2, zsolar, boost, dt, dT2_cpu, ncell)
  call test_cooling_kernel<<<nblocks, nthreads>>>( &
       & nH_d, T2_d, zsolar_d, boost_d, dt, c%X, dT2_d, ncell)
  istat = cudaDeviceSynchronize()
  dT2_gpu = dT2_d
  ! Print every (nnH/4 * nT + nT/2)-th cell so the sample spans the grid.
  do idx = 1, ncell, max(1, ncell / 12)
     denom  = max(abs(dT2_cpu(idx)), 1.0d-6 * T2(idx))
     relerr = abs(dT2_gpu(idx) - dT2_cpu(idx)) / denom
     write(*,'(3X,1PE12.4,1X,1PE12.4,2X,1PE13.5,2X,1PE13.5,2X,1PE10.2)') &
          & nH(idx), T2(idx), dT2_cpu(idx), dT2_gpu(idx), relerr
  end do

  ! -------------------------------------------------------------------------
  ! 6. Verdict.
  ! -------------------------------------------------------------------------
  nbad = 0
  do idx = 1, ncell
     denom = max(abs(dT2_cpu(idx)), 1.0d-6 * T2(idx))
     if (abs(dT2_gpu(idx) - dT2_cpu(idx)) / denom > tol) nbad = nbad + 1
  end do

  write(*,'(A)') ''
  write(*,'(A)') '==================================================================='
  write(*,'(A,1PE10.3)') ' Overall max relative error : ', overall_max_rel
  write(*,'(A,1PE10.3)') ' Tolerance                  : ', tol
  if (passed) then
     write(*,'(A)') ' RESULT: PASS  (CPU and GPU agree within tolerance)'
  else
     write(*,'(A,I0,A)') ' RESULT: FAIL  (', nbad, &
          & ' cells exceed tolerance at dt = largest; see table above)'
  end if
  write(*,'(A)') '==================================================================='

  deallocate(nH, T2, zsolar, boost, dT2_cpu, dT2_gpu)
  deallocate(nH_d, T2_d, zsolar_d, boost_d, dT2_d)

end program cooling_test_gpu

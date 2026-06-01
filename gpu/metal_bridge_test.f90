!============================================================================
! metal_bridge_test.f90 — standalone end-to-end test of the Fortran<->Metal
! bridge: a Fortran program fills a fixed-point position buffer, calls the Metal
! Hilbert radix sort through bind(C), reads back the permutation, and verifies
! the Hilbert keys are non-decreasing in sorted order.  Proves the full
! gfortran -> bind(C) -> Obj-C++ -> metallib path on real hardware.
!============================================================================
program metal_bridge_test
  use iso_c_binding
  use metal_bridge_iface
  implicit none

  integer, parameter :: nbits = 48, ndim = 3
  integer :: ierr, n, npartmax, i, d, ilevel, nbad
  integer(c_int),     pointer :: sortp(:)
  integer(c_int64_t), pointer :: ipos(:), hkey(:)
  type(c_ptr) :: cp
  real(c_double) :: u, scale
  integer(c_int64_t) :: prevkey
  character(len=512) :: libpath

  call get_command_argument(1, libpath)
  ierr = mtl_init(trim(libpath)//c_null_char)
  if (ierr /= 0) then
     write(*,*) 'mtl_init failed, code', ierr; stop 1
  end if

  n = 200000; npartmax = n; ilevel = 8
  scale = 2.0_c_double ** nbits
  call mtl_alloc_buffers(8, npartmax, 2*npartmax + 1031, 20)

  ! Fill fixed-point positions (column-major npartmax x ndim).
  call c_f_pointer(mtl_ptr_ipos(), ipos, [npartmax*ndim])
  call random_seed()
  do i = 1, n
     do d = 1, ndim
        call random_number(u)
        ipos((d-1)*npartmax + i) = int(u * scale, c_int64_t)
     end do
  end do

  call mtl_sort_part(ilevel, 1, n, npartmax)

  call c_f_pointer(mtl_ptr_sortp(),     sortp, [npartmax])
  call c_f_pointer(mtl_ptr_hkey_part(), hkey,  [npartmax])

  nbad = 0
  prevkey = -1_c_int64_t
  do i = 1, n
     if (hkey(sortp(i)) < prevkey) nbad = nbad + 1
     prevkey = hkey(sortp(i))
  end do

  if (nbad == 0) then
     write(*,'(A,I0,A)') ' BRIDGE SORT: PASS  (', n, ' particles sorted Fortran->Metal->Fortran)'
  else
     write(*,'(A,I0,A)') ' BRIDGE SORT: FAIL  (', nbad, ' key inversions)'
  end if

  call mtl_finalize()
  if (nbad /= 0) stop 2
end program metal_bridge_test

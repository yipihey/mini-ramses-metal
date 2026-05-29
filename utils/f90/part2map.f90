program part2map
  !--------------------------------------------------------------------------
  ! This code projects RAMSES data onto a map.
  ! This is based in the MINI-RAMSES prototype.
  ! R. Teyssier, Princeton, February 2nd 2023
  !--------------------------------------------------------------------------
  implicit none

  integer,parameter::flen=200
  integer,parameter::DEP_CIC=1, DEP_TSC=2, DEP_PCS=3

  integer::npart,nfile
  integer::nx_sample=128,ny_sample=128,nx,ny
  integer::i,j,idim,jdim,icpu
  integer::npart_file, ndim_file, npart_actual
  integer(kind=8)::ipos
  integer::dep_scheme=DEP_CIC

  real(KIND=8)::xmin=-1,xmax=-1,ymin=-1,ymax=-1,zmin=-1,zmax=-1
  real(KIND=8)::xxmin,xxmax,yymin,yymax,zzmin,zzmax
  real(KIND=8)::dx,dy,xx,yy,zz,mtot
  real(KIND=8)::jxin=0, jyin=0, jzin=0
  real(KIND=8)::xcenter, ycenter, zcenter, z_coord, y_coord, x_coord
  real(kind=8)::jx, jy, jz, kx, ky, kz, lx, ly, lz, ddx, ddy

  character(LEN=1)::proj='z'
  character(LEN=5)::ncharcpu
  character(LEN=flen)::nomfich,repository,prefix='part',outfich
  character(LEN=flen)::filetype='bin'
  character(LEN=flen)::file_part
  character(LEN=8)::depname

  logical::ok_part,backup_file=.false.,check_ramses_exist
  logical::rotation=.false., periodic=.false., sideon=.false.

  real(KIND=8),dimension(:,:),allocatable::map
  real(KIND=4),dimension(:,:),allocatable::toto
  real(KIND=4),dimension(:,:),allocatable::x
  real(KIND=4),dimension(:),allocatable::m

  type params
     integer::ndim
     integer::ncpu
     integer::nvar
     integer::levelmin
     integer::nlevelmax
     real(kind=8)::boxlen
     real(kind=8)::t
     real(kind=8)::texp
     real(kind=8)::aexp
     real(kind=8)::gamma
     real(kind=8)::unit_d
     real(kind=8)::unit_l
     real(kind=8)::unit_t
  end type params
  type(params)::p

  
  !------------------------------
  ! Read parameter and info files
  !------------------------------

  ! Read part2map parameters
  call read_params

  ! Check that all files exist
  if(.NOT. check_ramses_exist(repository,prefix))then
     write(*,*)'Repository '//TRIM(repository)//' incomplete.'
     write(*,*)'Stopping.'
     stop
  endif
  if(index(repository,'output')==0)backup_file=.true.

  ! Read RAMSES params
  call read_ramses_params
  write(*,*)'time =',real(p%t,kind=4)

  ! Read RAMSES info
  call read_info
  call read_part_header
  write(*,*)'npart=',npart
  write(*,*)'nfile=',nfile

  !-----------------
  ! Set up geometry 
  !-----------------
  if(xmin<0)xmin=0
  if(xmax<0)xmax=p%boxlen
  if(ymin<0)ymin=0
  if(ymax<0)ymax=p%boxlen
  if(zmin<0)zmin=0
  if(zmax<0)zmax=p%boxlen
  if(abs(jxin)+abs(jyin)+abs(jzin)==0)then
     if (proj=='x')then
        idim=2
        jdim=3
        xxmin=ymin ; xxmax=ymax
        yymin=zmin ; yymax=zmax
        zzmin=xmin ; zzmax=xmax
     else if (proj=='y') then
        idim=1
        jdim=3
        xxmin=xmin ; xxmax=xmax
        yymin=zmin ; yymax=zmax
        zzmin=ymin ; zzmax=ymax
     else
        idim=1
        jdim=2
        xxmin=xmin ; xxmax=xmax
        yymin=ymin ; yymax=ymax
        zzmin=zmin ; zzmax=zmax
     end if
     rotation=.false.
  else
     xcenter=0.5*(xmin+xmax)
     ycenter=0.5*(ymin+ymax)
     zcenter=0.5*(zmin+zmax)
     jx=jxin/sqrt(jxin**2+jyin**2+jzin**2)
     jy=jyin/sqrt(jxin**2+jyin**2+jzin**2)
     jz=jzin/sqrt(jxin**2+jyin**2+jzin**2)
     kx=0
     ky=-jz
     kz=jy
     lx=jy*kz-jz*ky
     ly=jz*kx-jx*kz
     lz=jx*ky-jy*kx
     rotation=.true.
  endif
  dx=(xxmax-xxmin)/dble(nx)
  dy=(yymax-yymin)/dble(ny)

  ! Map allocation
  npart_actual=0
  mtot=0.0d0
  allocate(map(0:nx,0:ny))
  map=0.0d0

  do icpu=1,nfile
     call title(icpu,ncharcpu)
     file_part=TRIM(repository)//'/'//TRIM(prefix)//'.'//TRIM(ncharcpu)
     open(unit=10,file=file_part,access="stream",action="read",form='unformatted')
     ipos=1
     read(10,POS=ipos)ndim_file
     ipos=5
     read(10,POS=ipos)npart_file
     allocate(x(1:npart_file,1:ndim_file))
     allocate(m(1:npart_file))
     ipos=9
     read(10,POS=ipos)x
     ipos=9+4*int(npart_file,kind=8)*2*int(ndim_file,kind=8)
     read(10,POS=ipos)m

     do i=1,npart_file
        ok_part=(x(i,1)>=xmin.and.x(i,1)<=xmax.and.&
     &           x(i,2)>=ymin.and.x(i,2)<=ymax.and.&
     &           x(i,3)>=zmin.and.x(i,3)<=zmax)
        if(.not.ok_part)cycle
        npart_actual=npart_actual+1

        if(rotation)then
           xx=x(i,1)-xcenter
           yy=x(i,2)-ycenter
           zz=x(i,3)-zcenter
           z_coord=xx*jx+yy*jy+zz*jz+zcenter
           y_coord=xx*kx+yy*ky+zz*kz+ycenter
           x_coord=xx*lx+yy*ly+zz*lz+xcenter
           ddx=(x_coord-xmin)/dx
           ddy=(y_coord-ymin)/dy
           if(sideon)then
              ddx=(x_coord-xmin)/dx
              ddy=(z_coord-zmin)/dy
           endif
        else
           ddx=(x(i,idim)-xxmin)/dx
           ddy=(x(i,jdim)-yymin)/dy
        endif

        select case(dep_scheme)
        case(DEP_CIC)
           call deposit_cic(map,nx,ny,ddx,ddy,dble(m(i)),periodic)
           mtot=mtot+dble(m(i))
        case(DEP_TSC)
           call deposit_tsc(map,nx,ny,ddx,ddy,dble(m(i)),periodic)
           mtot=mtot+dble(m(i))
        case(DEP_PCS)
           call deposit_pcs(map,nx,ny,ddx,ddy,dble(m(i)),periodic)
           mtot=mtot+dble(m(i))
        end select
     end do

     deallocate(x,m)
     close(10)
  end do

  write(*,*)'Data read and projected.'
  write(*,*)'Total number of part=',npart_actual
  write(*,*)'Total deposited mass=',mtot

  nomfich=TRIM(outfich)
  write(*,*)'Writing data to '//TRIM(nomfich)
  if(periodic)then
     allocate(toto(0:nx-1,0:ny-1))
     toto=real(map(0:nx-1,0:ny-1),kind=4)
  else
     allocate(toto(0:nx,0:ny))
     toto=real(map(0:nx,0:ny),kind=4)
  endif
  if(TRIM(filetype).eq.'bin')then
     open(unit=10,file=nomfich,form='unformatted')
     if(periodic)then
        write(10)p%t,xmax-xmin,ymax-ymin,zmax-zmin
        write(10)nx,ny
        write(10)toto
        write(10)xmin,xmax
        write(10)ymin,ymax
     else
        write(10)p%t,xmax-xmin,ymax-ymin,zmax-zmin
        write(10)nx+1,ny+1
        write(10)toto
        write(10)xmin,xmax
        write(10)ymin,ymax
     endif
     close(10)
  endif
  if(TRIM(filetype).eq.'ascii')then
     open(unit=10,file=nomfich,form='formatted')
     if(periodic)then
        do j=0,ny-1
           do i=0,nx-1
              xx=xmin+(dble(i)+0.5)/dble(nx)*(xmax-xmin)
              yy=ymin+(dble(j)+0.5)/dble(ny)*(ymax-ymin)
              write(10,*)xx,yy,toto(i,j)
           end do
           write(10,*) " "
        end do
     else
        do j=0,ny
           do i=0,nx
              xx=xmin+dble(i)/dble(nx)*(xmax-xmin)
              yy=ymin+dble(j)/dble(ny)*(ymax-ymin)
              write(10,*)xx,yy,toto(i,j)
           end do
           write(10,*) " "
        end do
     endif
     close(10)
  endif

contains

  subroutine read_params
    implicit none
    integer       :: i,n
    integer       :: iargc
    character(len=4)   :: opt
    character(len=128) :: arg
    real(kind=8)  :: xcen=0.5,ycen=0.5,zcen=0.5,rad=0

    n = iargc()
    if (n < 4) then
       print *, 'usage: part2map -inp  input_dir'
       print *, '                -out  output_file'
       print *, '               [-dir axis]'
       print *, '               [-dep depname]'
       print *, '               [-jx  jxin]'
       print *, '               [-jy  jyin]'
       print *, '               [-jz  jzin]'
       print *, '               [-xce xcen]'
       print *, '               [-yce ycen]'
       print *, '               [-zce zcen]'
       print *, '               [-rad rad]'
       print *, '               [-xmi xmin]'
       print *, '               [-xma xmax]'
       print *, '               [-ymi ymin]'
       print *, '               [-yma ymax]'
       print *, '               [-zmi zmin]'
       print *, '               [-zma zmax]'
       print *, '               [-nx  nx]'
       print *, '               [-ny  ny]'
       print *, '               [-fil filetype]'
       print *, '               [-pre prefix]'
       print *, '               [-per periodic]'
       print *, 'example:       part2map -inp output_00001 -out map.dat -dir z -dep TSC'
       print *, 'prefix:        part or star or sink (prefix of particle file name)'
       print *, 'depname:       CIC or TSC or PCS (name of mass deposition scheme)'
       print *, 'periodic:      .false. or .true. (periodic boundary conditions)'
       print *, 'filetype:      bin or ascii (output map file type)'
       stop
    end if

    do i = 1,n,2
       call getarg(i,opt)
       if (i == n) then
          print '("option ",a2," has no argument")', opt
          stop 2
       end if
       call getarg(i+1,arg)
       select case (opt)
       case ('-inp')
          repository = trim(arg)
       case ('-out')
          outfich = trim(arg)
       case ('-dir')
          proj = trim(arg)
       case ('-dep')
          depname=trim(arg)
          if(depname=='CIC' .or. depname=='cic' .or. depname=='1')then
             dep_scheme=DEP_CIC
          elseif(depname=='TSC' .or. depname=='tsc' .or. depname=='2')then
             dep_scheme=DEP_TSC
          elseif(depname=='PCS' .or. depname=='pcs' .or. depname=='3')then
             dep_scheme=DEP_PCS
          else
             write(*,*)'Unknown deposition: ',trim(depname),', defaulting to CIC'
             dep_scheme=DEP_CIC
          endif
       case ('-xce')
          read (arg,*) xcen
       case ('-yce')
          read (arg,*) ycen
       case ('-zce')
          read (arg,*) zcen
       case ('-rad')
          read (arg,*) rad
       case ('-xmi')
          read (arg,*) xmin
       case ('-xma')
          read (arg,*) xmax
       case ('-ymi')
          read (arg,*) ymin
       case ('-yma')
          read (arg,*) ymax
       case ('-zmi')
          read (arg,*) zmin
       case ('-zma')
          read (arg,*) zmax
       case ('-nx')
          read (arg,*) nx_sample
       case ('-ny')
          read (arg,*) ny_sample
       case ('-fil')
          read (arg,*) filetype
       case ('-pre')
          read (arg,*) prefix
       case ('-per')
          read (arg,*) periodic
       case default
          print '("unknown option ",a2," ignored")', opt
       end select
    end do

    if(rad>0)then
       xmin=xcen-rad
       xmax=xcen+rad
       ymin=ycen-rad
       ymax=ycen+rad
       zmin=zcen-rad
       zmax=zcen+rad
    endif

    if((abs(jxin)+abs(jyin)+abs(jzin))>0.and.rad==0)then
       write(*,*)'rotations only allowed with spherical selection'
       stop
    endif

    nx=nx_sample
    ny=ny_sample
  end subroutine read_params

  subroutine read_ramses_params
    !-----------------------------------------------
    ! Read RAMSES parameters file
    !-----------------------------------------------
    character(LEN=128)::nomfich
    integer::ilun,noutput,skip
    integer::nfile1,ncpu1
    nomfich=TRIM(repository)//'/params.bin'
    ilun=10
    open(unit=ilun,file=nomfich,access="stream",action="read",form='unformatted')
    read(ilun,POS=1)nfile1
    read(ilun,POS=5)ncpu1
    read(ilun,POS=9)p%ndim
    read(ilun,POS=13)p%levelmin
    read(ilun,POS=17)p%nlevelmax
    read(ilun,POS=21)p%boxlen
    read(ilun,POS=29)noutput
    skip=4*(11+4*noutput)+1
    read(ilun,POS=skip)p%t
    skip=skip+4*(2+4*p%nlevelmax+2+2*17)
    read(ilun,POS=skip)p%gamma
    close(ilun)

    if(backup_file)then
      p%ncpu=ncpu1
    else
      p%ncpu=nfile1
    endif

  end subroutine read_ramses_params

  subroutine read_part_header
    !-----------------------------------------------
    ! Read part header file
    !-----------------------------------------------
    character(LEN=128)::nomfich,fields
    integer::ilun
    nomfich=TRIM(repository)//'/'//TRIM(prefix)//'_header.txt'
    ilun=10
    open(unit=ilun,file=nomfich,form='formatted')
    read(ilun,*)
    read(ilun,*)npart
    read(ilun,*)
    read(ilun,*)nfile
    read(ilun,*)
    read(ilun,*)fields
  end subroutine read_part_header

  subroutine read_info
    !-----------------------------------------------
    ! Read ramses info file
    !-----------------------------------------------
    character(LEN=128)::nomfich
    character(LEN=80)::GMGM
    integer::ilun,nfile1,ncpu1,ndim1,levelmin1,levelmax1
    real(kind=8)::boxlen,t,omega_m,omega_b,omega_l,omega_k,gamma,h0
    nomfich=TRIM(repository)//'/info.txt'
    ilun=10
    open(unit=ilun,file=nomfich,form='formatted')
    read(ilun,'(A13,I11)')GMGM,nfile1
    read(ilun,'(A13,I11)')GMGM,ncpu1
    read(ilun,'(A13,I11)')GMGM,ndim1
    read(ilun,'(A13,I11)')GMGM,levelmin1
    read(ilun,'(A13,I11)')GMGM,levelmax1
    read(ilun,*)
    read(ilun,*)
    read(ilun,*)
    read(ilun,'(A13,E23.15)')GMGM,boxlen
    read(ilun,'(A13,E23.15)')GMGM,t
    read(ilun,'(A13,E23.15)')GMGM,p%texp
    read(ilun,'(A13,E23.15)')GMGM,p%aexp
    read(ilun,'(A13,E23.15)')GMGM,h0
    read(ilun,'(A13,E23.15)')GMGM,omega_m
    read(ilun,'(A13,E23.15)')GMGM,omega_l
    read(ilun,'(A13,E23.15)')GMGM,omega_k
    read(ilun,'(A13,E23.15)')GMGM,omega_b
    read(ilun,'(A13,E23.15)')GMGM,gamma
    read(ilun,'(A13,E23.15)')GMGM,p%unit_l
    read(ilun,'(A13,E23.15)')GMGM,p%unit_d
    read(ilun,'(A13,E23.15)')GMGM,p%unit_t
    read(ilun,*)
    close(ilun)
  end subroutine read_info

  subroutine deposit_cic(map,nx,ny,ddx,ddy,mpart,periodic)
    !-----------------------------------------------
    ! Deposit particles using CIC scheme
    !-----------------------------------------------
    real(kind=8),dimension(0:nx,0:ny)::map
    integer::nx,ny
    real(kind=8)::ddx,ddy,mpart
    logical::periodic
    integer::ix,iy,ixp1,iyp1
    real(kind=8)::fx,fy
    ix=int(ddx)
    iy=int(ddy)
    fx=ddx-dble(ix)
    fy=ddy-dble(iy)
    if(periodic)then
       if(ix<0)ix=ix+nx
       if(ix>=nx)ix=ix-nx
       if(iy<0)iy=iy+ny
       if(iy>=ny)iy=iy-ny
    endif
    ixp1=ix+1
    iyp1=iy+1
    if(periodic)then
       if(ixp1<0)ixp1=ixp1+nx
       if(ixp1>=nx)ixp1=ixp1-nx
       if(iyp1<0)iyp1=iyp1+ny
       if(iyp1>=ny)iyp1=iyp1-ny
    endif
    if(ix>=0.and.ix<nx.and.iy>=0.and.iy<ny.and.fx>=0.and.fy>=0)then
       map(ix  ,iy  )=map(ix  ,iy  )+mpart*(1.0d0-fx)*(1.0d0-fy)
       map(ix  ,iyp1)=map(ix  ,iyp1)+mpart*(1.0d0-fx)*fy
       map(ixp1,iy  )=map(ixp1,iy  )+mpart*fx*(1.0d0-fy)
       map(ixp1,iyp1)=map(ixp1,iyp1)+mpart*fx*fy
    endif
  end subroutine deposit_cic

  subroutine deposit_tsc(map,nx,ny,ddx,ddy,mpart,periodic)
    !-----------------------------------------------
    ! Deposit particles using TSC scheme
    !-----------------------------------------------
    real(kind=8),dimension(0:nx,0:ny)::map
    integer::nx,ny
    real(kind=8)::ddx,ddy,mpart
    logical::periodic
    integer::ixc,iyc,ix,iy,dxi,dyi
    real(kind=8)::xc,yc,wx(3),wy(3),dxrel,dyrel
    integer::ix_idx,iy_idx

    ! Central cell containing the particle
    ixc=int(ddx)
    iyc=int(ddy)
    xc=dble(ixc)+0.5d0
    yc=dble(iyc)+0.5d0

    ! Relative offset from central cell center
    dxrel=ddx-xc
    dyrel=ddy-yc

    ! 1D TSC weights for x (left, center, right)
    wx(1)=0.5d0*(1.5d0-abs(dxrel+1.0d0))**2  ! left (xc-1)
    wx(2)=0.75d0-        (dxrel        )**2  ! center (xc)
    wx(3)=0.5d0*(1.5d0-abs(dxrel-1.0d0))**2  ! right (xc+1)

    ! 1D TSC weights for y (bottom, center, top)
    wy(1)=0.5d0*(1.5d0-abs(dyrel+1.0d0))**2
    wy(2)=0.75d0-        (dyrel        )**2
    wy(3)=0.5d0*(1.5d0-abs(dyrel-1.0d0))**2

    do dxi=-1,1
       do dyi=-1,1
          ix=ixc+dxi
          iy=iyc+dyi
          ix_idx=dxi+2
          iy_idx=dyi+2
          if(periodic)then
             if(ix<0)ix=ix+nx
             if(ix>=nx)ix=ix-nx
             if(iy<0)iy=iy+ny
             if(iy>=ny)iy=iy-ny
             if(ix>=0.and.ix<nx.and.iy>=0.and.iy<ny)then
                map(ix,iy)=map(ix,iy)+mpart*wx(ix_idx)*wy(iy_idx)
             endif
          else
             if(ix>=0.and.ix<=nx.and.iy>=0.and.iy<=ny)then
                map(ix,iy)=map(ix,iy)+mpart*wx(ix_idx)*wy(iy_idx)
             endif
          endif
       end do
    end do
  end subroutine deposit_tsc

  subroutine deposit_pcs(map,nx,ny,ddx,ddy,mpart,periodic)
    !-----------------------------------------------
    ! Deposit particles using PCS scheme
    !-----------------------------------------------
    ! Piecewise Cubic Spline (PCS) deposition: 4x4 nodes using B-spline weights
    real(kind=8),dimension(0:nx,0:ny)::map
    integer::nx,ny
    real(kind=8)::ddx,ddy,mpart
    logical::periodic
    integer::ixc,iyc,ix,iy,dxi,dyi
    real(kind=8)::wx(4),wy(4)
    real(kind=8)::dxrel,dyrel
    integer::ix_idx,iy_idx

    ! Central cell index (nearest integer)
    ixc=int(ddx)
    iyc=int(ddy)
    dxrel=ddx-(dble(ixc)+0.5d0)
    dyrel=ddy-(dble(iyc)+0.5d0)

    ! 1D PCS weights following move_fine (wll, wl, wr, wrr)
    wx(1)=(2d0-abs(dxrel+1.5d0))**3/6d0
    wx(2)=(4d0-6d0*(dxrel+0.5d0)**2+3d0*abs(dxrel+0.5d0)**3)/6d0
    wx(3)=(4d0-6d0*(dxrel-0.5d0)**2+3d0*abs(dxrel-0.5d0)**3)/6d0
    wx(4)=(2d0-abs(dxrel-1.5d0))**3/6d0

    wy(1)=(2d0-abs(dyrel+1.5d0))**3/6d0
    wy(2)=(4d0-6d0*(dyrel+0.5d0)**2+3d0*abs(dyrel+0.5d0)**3)/6d0
    wy(3)=(4d0-6d0*(dyrel-0.5d0)**2+3d0*abs(dyrel-0.5d0)**3)/6d0
    wy(4)=(2d0-abs(dyrel-1.5d0))**3/6d0

    do dxi=-2,1
       do dyi=-2,1
          ix=ixc+dxi
          iy=iyc+dyi
          ix_idx=dxi+3  ! maps -2,-1,0,1 -> 1,2,3,4
          iy_idx=dyi+3
          if(periodic)then
             if(ix<0)ix=ix+nx
             if(ix>=nx)ix=ix-nx
             if(iy<0)iy=iy+ny
             if(iy>=ny)iy=iy-ny
             if(ix>=0.and.ix<nx.and.iy>=0.and.iy<ny)then
                map(ix,iy)=map(ix,iy)+mpart*wx(ix_idx)*wy(iy_idx)
             endif
          else
             if(ix>=0.and.ix<=nx.and.iy>=0.and.iy<=ny)then
                map(ix,iy)=map(ix,iy)+mpart*wx(ix_idx)*wy(iy_idx)
             endif
          endif
       end do
    end do
  end subroutine deposit_pcs
  !================================================================
  !================================================================
  !================================================================
  !================================================================
end program part2map

function check_ramses_exist(repository,prefix)
  !-----------------------------------------------
  ! Check that RAMSES files are there
  !-----------------------------------------------
  logical::check_ramses_exist
  character(len=80)::repository,prefix
  character(LEN=128)::nomfich_part
  check_ramses_exist=.true.
  nomfich_part=TRIM(repository)//'/'//TRIM(prefix)//'.00001'
  inquire(file=nomfich_part, exist=check_ramses_exist)
end function check_ramses_exist
!================================================================
!================================================================
!================================================================
!================================================================

!=======================================================================
subroutine title(n,nchar)
!=======================================================================
  implicit none
  integer::n
  character*5::nchar
  character*1::nchar1
  character*2::nchar2
  character*3::nchar3
  character*4::nchar4
  character*5::nchar5
  if(n.ge.10000)then
     write(nchar5,'(i5)') n
     nchar = nchar5
  elseif(n.ge.1000)then
     write(nchar4,'(i4)') n
     nchar = '0'//nchar4
  elseif(n.ge.100)then
     write(nchar3,'(i3)') n
     nchar = '00'//nchar3
  elseif(n.ge.10)then
     write(nchar2,'(i2)') n
     nchar = '000'//nchar2
  else
     write(nchar1,'(i1)') n
     nchar = '0000'//nchar1
  endif
end subroutine title

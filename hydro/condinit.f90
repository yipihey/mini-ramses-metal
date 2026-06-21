!================================================================
!================================================================
!================================================================
!================================================================
subroutine condinit(r,g,x,q,dx,nn)
  use amr_parameters, only: ndim, nvector
  use hydro_parameters, only: nvar, nener
  use amr_commons, only: run_t, global_t
  use input_hydro_condinit_module, only: region_condinit
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::nn                            ! Number of cells
  real(kind=8)::dx                            ! Cell size
#ifdef MHD
  real(kind=8),dimension(1:nvector,1:nvar+3-ndim)::q ! Primitive variables
#else
  real(kind=8),dimension(1:nvector,1:nvar)::q ! Primitive variables
#endif
  real(kind=8),dimension(1:nvector,1:ndim)::x ! Cell center position.
  !================================================================
  ! This routine generates initial conditions for RAMSES.
  ! Positions are in user (aka code) units:
  ! x(i,1:ndim) are in [0,box_size]**ndim.
  ! Q is the primitive variable vector. Conventions are here:
  ! Q(i,1): d, Q(i,2:4):u,v,w and Q(i,5): P.
  ! If nvar >= 6, remaining variables are treated as passive
  ! scalars or non-thermal energies in the hydro solver.
  ! For 1D MHD, Q(i,nvar+1) is By and Q(i,nvar+2) is Bz.
  ! For 2D MHD, Q(i,nvar+1) is Bz.
  ! Q(:,:) are in user (aka code) units.
  !================================================================
#define COEUR 1
#define INSTA 2
#define DOUBLEMACH 3
#define OT 4
#define PONO 5
#define ABC 6
#define CURRENTSHEET 7
#define RTZEQM 8
#define PANCAKE 9
#define ALFVENWAVE 10
#define BRIOWU 11
#define TURB 12
#define CPALFVEN 13

  integer::i
#if INIT==COEUR
  real(kind=8)::r2,rx,ry,rz,d,p,vx,vy,vz,r_trunc,r2_trunc,c2
  real(kind=8)::omega_code,AU,Msol,pi,M,sigma,r_min,r2_min,omega_const,r_vortex,invr2_vortex
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v,scale_m
#elif INIT==INSTA
  integer::id,iu,iv,iw,ip,ix,iy
  real(kind=8)::x0,lambday,ky,lambdaz,kz,rho1,rho2,p0,v0,v1,v2
#elif INIT==DOUBLEMACH
  integer::id,iu,iv,iw,ip
  real(kind=8)::pi,xp
#elif INIT==OT
  real(kind=8)::pi,xc,yc
#elif INIT==PONO
  real(kind=8)::xx,yy,zz,vx,vy,vz,rr,tt,omega,R0,twopi
#elif INIT==ABC
  real(kind=8)::xx,yy,zz,vx,vy,vz,A0,twopi
#elif INIT==CURRENTSHEET
  real(kind=8)::pi,xc,yc,beta,v0
#elif INIT==RTZEQM
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v,scale_m
#elif INIT==PANCAKE
  real(kind=8)::pi,del_ini
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v,scale_m
#elif INIT==ALFVENWAVE
  real(kind=8)::pi,del_ini
  real(kind=8)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v,scale_m
#elif INIT==TURB
  integer::nx,ny,nz,kk
  real(kind=8)::twopi,cs,rho0,p0,b0,norm,sumsq
  real(kind=8)::ax,ay,az,cx,cy,cz,csq,amp,ph,phase,vx,vy,vz
  real(kind=8)::hx,hy,hz,hp
  ! Precomputed forcing-mode constants (were recomputed PER CELL: ~16 modes x 5 sin
  ! over every one of the 343M cells = the entire init_flow_fine cost). |k| in [1,2]
  ! => at most ~16 hemisphere modes; 32 is a safe bound.
  integer::nmode,im
  integer,dimension(1:32)::mkk
  real(kind=8)::cosp
  real(kind=8),dimension(1:32)::mkx,mky,mkz,macx,macy,macz,mph
#elif INIT==CPALFVEN
  real(kind=8)::twopi,kx,bperp
#else
  ! Call built-in initial condition generator
  call region_condinit(r,g,x,q,dx,nn)
#ifdef GLMMHD
  ! GLM cell-centred B from the uniform-field namelist seeds (A_ave/B_ave/C_ave =
  ! uniform Bx/By/Bz). region_condinit leaves B=0 for GLM (the bold/vector-potential
  ! path is MHD-only), so set it here. Defaults to 0 -> unchanged for non-magnetised
  ! region ICs (e.g. Sod). Used by the magnetised driven-turbulence benchmark.
  q(1:nn,6) = r%A_ave
  q(1:nn,7) = r%B_ave
  q(1:nn,8) = r%C_ave
  q(1:nn,9) = 0.0d0
#endif
#endif
  
  ! Add here, if you wish, some user-defined initial conditions
  ! ........

#if INIT==COEUR
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  scale_m=scale_d*scale_l**3
  ! constants
  AU=1.49598d13
  Msol= 1.98892d33
  pi=3.14159
  ! mass, radius, and ratio of rotational to gravitational energy
  r_trunc=25*4000.*AU/scale_l
  r_min=10.*AU/scale_l
  r_vortex=4000.*AU/scale_l
  M=100.*Msol/scale_m
  sigma=M/(4*pi*r_trunc)
  r2_trunc=r_trunc**2
  r2_min=r_min**2
  invr2_vortex=1./r_vortex**2
  omega_const=0.1*sqrt(1./r2_trunc+invr2_vortex)*sqrt(M/r_trunc)
  c2=(18939.2/(scale_l/scale_t))**2
  do i=1,nn
     rx=x(i,1)-r%box_size(1)/2.
     ry=x(i,2)-r%box_size(2)/2.
     rz=x(i,3)-r%box_size(3)/2.
     !density
     r2=rx**2+ry**2+rz**2
     d=sigma/(r2+r2_min)
     omega_code=omega_const/sqrt(1.+invr2_vortex*r2)
     if (r2>=r2_trunc)then
        d=d*1.e-4
        omega_code=omega_code/sqrt(r2)*exp(10.*(r2_trunc-r2))
     end if
     !pressure
     p=d*c2
     !velocity
     vx=-omega_code*ry
     vy=omega_code*rx
     vz=0.
     ! primitive variables
     q(i,1)=d
     q(i,2)=vx
     q(i,3)=vy
     q(i,4)=vz
     q(i,5)=p
  end do
#endif

#if INIT==INSTA
  id=1; iu=2; iv=3; iw=4; ip=5
  x0=r%x_center(1)
  if(r%constant_gravity(2) .ne. 0)then
     ix=2
     iy=1
     iu=3
     iv=2
  else
     ix=1
     iy=2
     iu=2
     iv=3
  endif

  lambday=0.25
  ky=2.*acos(-1.0d0)/lambday
  lambdaz=0.25
  kz=2.*acos(-1.0d0)/lambdaz
  rho1=r%d_region(1)
  rho2=r%d_region(2)
  v1=r%v_region(1)
  v2=r%v_region(2)
  v0=0.1
  p0=10.
  do i=1,nn
     if(x(i,ix) < x0)then
        q(i,id)=rho1
        q(i,iu)=0.0
        q(i,iu)=v0*cos(ky*(x(i,iy)-lambday/2.))*exp(+ky*(x(i,ix)-x0))
        q(i,iv)=v1
        q(i,iw)=0.0D0
        q(i,ip)=p0+rho1*r%constant_gravity(ix)*x(i,ix)
     else
        q(i,id)=rho2
        q(i,iu)=0.0
        q(i,iu)=v0*cos(ky*(x(i,iy)-lambday/2.))*exp(-ky*(x(i,ix)-x0))
        q(i,iv)=v2
        q(i,iw)=0.0D0
        q(i,ip)=p0+rho1*r%constant_gravity(ix)*x0+rho2*r%constant_gravity(ix)*(x(i,ix)-x0)
     endif
  end do
#endif

#if INIT==DOUBLEMACH
  id=1; iu=2; iv=3; iw=4; ip=5
  pi=acos(-1.0d0)
  do i=1,nn
     xp=x(i,1)-x(i,2)/tan(pi/3.0)-10./sin(pi/3.0)*g%t
     if(xp<1./6.)then
        q(i,id)=8.
        q(i,iu)=7.145
        q(i,iv)=-4.125
        q(i,iw)=0
        q(i,ip)=116.5
     else
        q(i,id)=r%gamma
        q(i,iu)=0.0
        q(i,iv)=0.0
        q(i,iw)=0.0
        q(i,ip)=1.0
     endif
  end do
#endif

#if INIT==OT
  pi=acos(-1.0d0)
  do i=1,nn
     xc=x(i,1)
     yc=x(i,2)
     q(i,1)=25.0/(36.0*pi)
     q(i,2)=-sin(2.0*pi*yc)
     q(i,3)=+sin(2.0*pi*xc)
     q(i,4)=0.0
     q(i,5)=5.0/(12.0*pi)
#ifdef GLMMHD
     ! Dedner cell-centered B = curl(A_z), A_z = B0(cos4pix/4pi + cos2piy/2pi),
     ! B0 = 1/sqrt(4pi): Bx=-B0 sin(2pi y), By=B0 sin(4pi x), Bz=0, psi=0.
     q(i,6)=-sin(2.0*pi*yc)/sqrt(4.0*pi)
     q(i,7)= sin(4.0*pi*xc)/sqrt(4.0*pi)
     q(i,8)=0.0
     q(i,9)=0.0
#else
     q(i,nvar+1)=0.0 ! Bz
#endif
  end do
#endif

#if INIT==BRIOWU
  ! Brio-Wu (Wu) MHD shock tube (gamma=2). Discontinuity at x=boxlen/2.
  ! Left: rho=1, p=1, By=1 ; Right: rho=0.125, p=0.1, By=-1 ; Bx=0.75 both ; psi=0.
  do i=1,nn
     if (x(i,1) < 0.5d0*r%box_size(1)) then
        q(i,1)=1.0d0;   q(i,5)=1.0d0;   q(i,7)= 1.0d0
     else
        q(i,1)=0.125d0; q(i,5)=0.1d0;   q(i,7)=-1.0d0
     end if
     q(i,2)=0.0d0; q(i,3)=0.0d0; q(i,4)=0.0d0
     q(i,6)=0.75d0    ! Bx (continuous normal field)
     q(i,8)=0.0d0     ! Bz
     q(i,9)=0.0d0     ! psi
  end do
#endif

#if INIT==PONO
  R0=1.0
  twopi=2d0*ACOS(-1d0)
  do i=1,nn
     q(i,1)=1.0
     q(i,5)=1.0*(r%gamma-1.0)
     xx=x(i,1)-r%box_size(1)/2.
     yy=x(i,2)-r%box_size(2)/2.
     rr = SQRT(xx**2+yy**2)
     if(rr < 1.0)then
        omega=0.609711
        vz=0.792624
     else
        omega=0.0
        vz=0.0
     endif
     if(rr > 0.0)then
        if(yy > 0.0)then
           tt=acos(xx/rr)
        else
           tt=-acos(xx/rr)+twopi
        endif
        vx=-sin(tt)*rr*omega
        vy=+cos(tt)*rr*omega
     else
        vx=0.0
        vy=0.0
     endif
     q(i,2)=vx
     q(i,3)=vy
     q(i,4)=vz
  end do
#endif

#if INIT==ABC
  A0=1.0
  twopi=2d0*ACOS(-1d0)
  do i=1,nn
     q(i,1)=1.0
     q(i,5)=1.0*(r%gamma-1.0)
     xx=x(i,1)-r%box_size(1)/2.
     yy=x(i,2)-r%box_size(2)/2.
     zz=x(i,3)-r%box_size(3)/2.
     vx=A0*(cos(twopi*yy)+sin(twopi*zz))
     vy=A0*(sin(twopi*xx)+cos(twopi*zz))
     vz=A0*(cos(twopi*xx)+sin(twopi*yy))
     q(i,2)=vx
     q(i,3)=vy
     q(i,4)=vz
  end do
#endif

#if INIT==CURRENTSHEET
  pi = acos(-1.0d0)
  beta = 0.1
  v0 = 0.1
  do i = 1,nn
     xc = x(i,1)
     yc = x(i,2)
     q(i,1) = 1.0
     q(i,2) = v0*sin(pi*yc)
     q(i,3) = 0.0
     q(i,4) = 0.0
     q(i,5) = 0.5*beta
     q(i,nvar+1) = 0.0 ! Bz
  end do
#endif

#if INIT==RTZEQM
  ! get the units
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  ! Smoothly interpolate gas density between
  ! 1e-3 and 1e5, fix T to 10^4, and convert everything to code units
  ! note that this assumes a boxsize of 1 and Nx = Ny and unigrid
  do i = 1,nn
     q(i,1) = (10.d0**(x(i,1) * 8.d0 - 3.d0)) / scale_nH
     q(i,2) = 0.0 ! Vx
     q(i,3) = 0.0 ! Vy
     q(i,4) = 0.0 ! Vz
     q(i,5) = 1.d4 / scale_T2 ! Temperature is 10^4 K
  end do
#endif

#if INIT==PANCAKE
  pi = acos(-1.0d0)
  del_ini = 0.1
  ! get cgs units
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  do i = 1,nn
     q(i,1) = g%omega_b/g%omega_m/(1+del_ini*COS(2.0d0*pi*x(i,1)))
     q(i,2) = del_ini*g%vfact(1)*SIN(2.0d0*pi*x(i,1))/(2.0d0*pi)
     q(i,3) = 0.0 ! Vy
     q(i,4) = 0.0 ! Vz
     q(i,5) = 100./scale_T2 ! Temperature is 10^2 K
  end do
#endif

#if INIT==ALFVENWAVE
  pi = acos(-1.0d0)
  del_ini = 0.1
  ! get cgs units
  call units(r,g,scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  do i = 1,nn
     q(i,1) = 1 ! rho
     q(i,2) = 0 ! Vx
     q(i,3) = 0.1*SIN(2.0d0*pi*x(i,1)) ! Vy
     q(i,4) = 0.1*COS(2.0d0*pi*x(i,1)) ! Vz
     q(i,5) = 0.1 ! Pressure
     q(i,6) = 0.1*SIN(2.0d0*pi*x(i,1)) ! By
     q(i,7) = 0.1*COS(2.0d0*pi*x(i,1)) ! Bz
  end do
#endif

#if INIT==TURB
  ! Reproducible decaying-turbulence IC. Solenoidal velocity from a fixed set of
  ! transverse Fourier modes (|k| in [1,2]*2pi/L), normalized to v_rms = cs (Mach~1).
  ! Velocity/rho/p are byte-identical for HD and MHD builds; GLMMHD adds a uniform
  ! guide field with Alfven speed vA = cs (plasma beta ~ 1).
  twopi=2.0d0*acos(-1.0d0)
  rho0=1.0d0
  cs=1.0d0
  p0=rho0*cs*cs/r%gamma                  ! so cs = sqrt(gamma p/rho) = 1
  b0=cs*sqrt(rho0)                        ! vA = |B|/sqrt(rho) = cs
  ! normalization so that <v^2> = sum_hemisphere amp^2 |c|^2 / 2 = cs^2.
  ! Precompute the (cell-independent) mode constants ONCE here -- the original code
  ! recomputed ax/ay/az/ph/cx/cy/cz/amp inside the per-cell loop (~16 modes x 5 sin
  ! per cell x 343M cells). Bit-identical: same modes, same order, same accumulation.
  sumsq=0.0d0; nmode=0
  do nx=-2,2
   do ny=-2,2
    do nz=-2,2
     kk=nx*nx+ny*ny+nz*nz
     if(kk<1.or.kk>4)cycle
     if(.not.(nz>0.or.(nz==0.and.ny>0).or.(nz==0.and.ny==0.and.nx>0)))cycle
     hp=dble(100*nx+17*ny+3*nz+211)
     ax=2.0d0*(sin(hp+1.0d0)*43758.5453d0-floor(sin(hp+1.0d0)*43758.5453d0))-1.0d0
     ay=2.0d0*(sin(hp+2.0d0)*43758.5453d0-floor(sin(hp+2.0d0)*43758.5453d0))-1.0d0
     az=2.0d0*(sin(hp+3.0d0)*43758.5453d0-floor(sin(hp+3.0d0)*43758.5453d0))-1.0d0
     cx=dble(ny)*az-dble(nz)*ay; cy=dble(nz)*ax-dble(nx)*az; cz=dble(nx)*ay-dble(ny)*ax
     amp=1.0d0/sqrt(dble(kk))
     sumsq=sumsq+amp*amp*(cx*cx+cy*cy+cz*cz)*0.5d0
     nmode=nmode+1; mkk(nmode)=kk
     mkx(nmode)=dble(nx); mky(nmode)=dble(ny); mkz(nmode)=dble(nz)
     macx(nmode)=cx; macy(nmode)=cy; macz(nmode)=cz               ! raw c; amp applied below
     mph(nmode)=twopi*(sin(hp+4.0d0)*43758.5453d0-floor(sin(hp+4.0d0)*43758.5453d0))
    enddo
   enddo
  enddo
  norm=cs/sqrt(sumsq)
  do im=1,nmode
     amp=norm/sqrt(dble(mkk(im)))                                 ! identical to the old per-cell amp
     macx(im)=amp*macx(im); macy(im)=amp*macy(im); macz(im)=amp*macz(im)
  enddo
  do i=1,nn
     vx=0.0d0; vy=0.0d0; vz=0.0d0
     do im=1,nmode
        phase=twopi*(mkx(im)*x(i,1)+mky(im)*x(i,2)+mkz(im)*x(i,3))/r%box_size(1)+mph(im)
        cosp=cos(phase)
        vx=vx+macx(im)*cosp
        vy=vy+macy(im)*cosp
        vz=vz+macz(im)*cosp
     enddo
     q(i,1)=rho0
     q(i,2)=vx
     q(i,3)=vy
     q(i,4)=vz
     q(i,5)=p0
#ifdef GLMMHD
     q(i,6)=0.0d0      ! Bx
     q(i,7)=0.0d0      ! By
     q(i,8)=b0         ! Bz: uniform guide field, vA = cs
     q(i,9)=0.0d0      ! psi
#endif
  end do
#endif

#if INIT==CPALFVEN
  ! Circularly-polarized Alfven wave (Toth 2000) along x, +x traveling at v_A=Bx/sqrt(rho).
  ! Exact standing solution: |B_perp| and |v_perp| stay constant; returns to IC each
  ! period (t = L/v_A). The definitive test of the Alfven eigenvectors (waves 2,6).
  twopi=2.0d0*acos(-1.0d0)
  bperp=0.1d0
  do i=1,nn
     kx=twopi*x(i,1)/r%box_size(1)
     q(i,1)=1.0d0           ! rho
     q(i,5)=0.1d0           ! p
     q(i,2)=0.0d0           ! vx
     q(i,3)=-bperp*sin(kx)  ! vy = -By/sqrt(rho)  (+x traveling)
     q(i,4)=-bperp*cos(kx)  ! vz = -Bz/sqrt(rho)
#ifdef GLMMHD
     q(i,6)=1.0d0           ! Bx (guide field, v_A = 1)
     q(i,7)=bperp*sin(kx)   ! By
     q(i,8)=bperp*cos(kx)   ! Bz
     q(i,9)=0.0d0           ! psi
#endif
  end do
#endif

  ! Compute entropy if needed
  if(r%entropy)then
     q(1:nn,r%ientropy)=q(1:nn,5)/q(1:nn,1)**r%gamma
  endif

  ! Compute metallicity if needed
  if(r%metal)then
     q(1:nn,r%imetal)=r%z_ave*0.02
  endif

end subroutine condinit

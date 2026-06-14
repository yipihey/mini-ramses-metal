module input_part_zoom_module

  type::out_part_zoom_t
     integer(kind=8)::npart_tot
     integer(kind=4)::npart_max
  end type out_part_zoom_t

contains
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine m_input_part_zoom(pst)
  use ramses_commons, only: pst_t
  implicit none
  type(pst_t)::pst
  !--------------------------------------------------------------------
  ! This routine is the master procedure to read and dispatch particles
  ! from a Ramses restart file.
  !--------------------------------------------------------------------
  integer::dummy
  type(out_part_zoom_t)::output
  
  if(pst%s%r%verbose)write(*,*)'Entering input_part_zoom'
  if(TRIM(pst%s%r%filetype).NE.'grafic_zoom')return
  if(pst%s%r%nrestart>0)return

  ! Call recursive slave routine
  call r_input_part_zoom(pst,dummy,1,output,3)
  pst%s%p%npart_tot=output%npart_tot
  pst%s%p%npart_max=output%npart_max
  write(*,'(A,I0)')'Found npart_tot= ',pst%s%p%npart_tot

end subroutine m_input_part_zoom
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
recursive subroutine r_input_part_zoom(pst,dummy,input_size,output,output_size)
  use mdl_module
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  integer::dummy
  type(out_part_zoom_t)::output,next_output
  !--------------------------------------------------------------------
  ! This routine is the recursive slave procedure to create particles
  ! from a zoom-in AMR grid and grafic files.
  !--------------------------------------------------------------------
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_INPUT_PART_ZOOM,pst%iUpper+1,input_size,output_size,dummy)
     call r_input_part_zoom(pst%pLower,dummy,input_size,output,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size,next_output)
     output%npart_tot=output%npart_tot+next_output%npart_tot
     output%npart_max=MAX(output%npart_max,next_output%npart_max)
  else
     call input_part_zoom(pst%s%r,pst%s%g,pst%s%p,pst%s%m)
     output%npart_tot=int(pst%s%p%npart,kind=8)
     output%npart_max=pst%s%p%npart
  endif

end subroutine r_input_part_zoom
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
subroutine input_part_zoom(r,g,p,m)
  use amr_parameters, only: ndim, twotondim
  use amr_commons, only: run_t, global_t, mesh_t
  use pm_commons, only: part_t
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(part_t)::p
  type(mesh_t)::m
  !----------------------------------------------------
  ! Reading initial conditions from single GRAFIC file
  !----------------------------------------------------
  integer::icpu,ipart,idim,igrid,ilevel,ind
  integer::i1,i2,i3,i1_min,i1_max,i2_min,i2_max,i3_min,i3_max
  integer::plane_size
  real(kind=8)::dx,xx1,xx2,xx3
  real(kind=8)::dispmax=0.0
  integer(kind=8)::ipart_old
  integer(kind=8),dimension(1:g%ncpu+1)::start_ind
  real(kind=8),allocatable,dimension(:,:,:)::init_array
  real(kind=8),allocatable,dimension(:,:,:)::init_array_x
  real(kind=4),dimension(:,:),allocatable::init_plane
  character(LEN=80)::filename,filename_x
  character(LEN=5)::nchar
  logical::ok,error,keep_part,read_pos=.false.

#if NDIM>2
  
  p%npart=0
  do ilevel=r%levelmin,g%nlevelmax_part

     if(m%noct(ilevel)==0)cycle
        
     ! Mesh size at level ilevel in normalised units
     dx=0.5D0**ilevel
     
     !-------------------------------------------------------------------------
     ! First step: compute level boundaries in terms of initial condition array
     ! and initialize particle position and mass in leaf cells only.
     !-------------------------------------------------------------------------
     i1_min=g%n1(ilevel)+1; i1_max=0
     i2_min=g%n2(ilevel)+1; i2_max=0
     i3_min=g%n3(ilevel)+1; i3_max=0
     ipart_old=p%npart
     do igrid=m%head(ilevel),m%tail(ilevel)
        do ind=1,twotondim
           ! Coordinates in normalised units (between 0 and 1)
           xx1=(2*m%grid(igrid)%ckey(1)+MOD((ind-1)  ,2)+0.5)*dx
           xx2=(2*m%grid(igrid)%ckey(2)+MOD((ind-1)/2,2)+0.5)*dx
           xx3=(2*m%grid(igrid)%ckey(3)+MOD((ind-1)/4,2)+0.5)*dx
           ! Add a particle if we have a leaf cell
           if(.not. m%grid(igrid)%refined(ind))then
              p%npart=p%npart+1
              if(p%npart>r%npartmax)then
                 write(*,*)'Maximum number of particles reached'
                 write(*,*)'npartmax should be greater than',p%npart
                 stop
              endif
              if(ndim>0)p%xp(p%npart,1)=xx1
              if(ndim>1)p%xp(p%npart,2)=xx2
              if(ndim>2)p%xp(p%npart,3)=xx3
              p%mp(p%npart)=0.5d0**(3*ilevel)*(1.0d0-g%omega_b/g%omega_m)
           endif
           ! Scale to integer coordinates in the frame of the file
           xx1=(xx1*(g%dxini(ilevel)/dx)-g%xoff1(ilevel))/g%dxini(ilevel)
           xx2=(xx2*(g%dxini(ilevel)/dx)-g%xoff2(ilevel))/g%dxini(ilevel)
           xx3=(xx3*(g%dxini(ilevel)/dx)-g%xoff3(ilevel))/g%dxini(ilevel)
           ! Compute min and max
           i1_min=MIN(i1_min,int(xx1)+1); i1_max=MAX(i1_max,int(xx1)+1)
           i2_min=MIN(i2_min,int(xx2)+1); i2_max=MAX(i2_max,int(xx2)+1)
           i3_min=MIN(i3_min,int(xx3)+1); i3_max=MAX(i3_max,int(xx3)+1)
        end do
     end do

     ! Check that all grids are within initial condition region
     error=.false.
     if(i1_min<1.or.i1_max>g%n1(ilevel))error=.true.
     if(i2_min<1.or.i2_max>g%n2(ilevel))error=.true.
     if(i3_min<1.or.i3_max>g%n3(ilevel))error=.true.
     if(error) then
        write(*,*)'Some grid are outside initial conditions sub-volume'
        write(*,*)'for ilevel=',ilevel
        write(*,*)'and processor=',g%myid
        write(*,*)i1_min,i1_max
        write(*,*)i2_min,i2_max
        write(*,*)i3_min,i3_max
        write(*,*)g%n1(ilevel),g%n2(ilevel),g%n3(ilevel)
        stop
     end if

     !------------------------------------------
     ! Second step: read initial condition files
     !------------------------------------------
     filename_x=TRIM(r%initfile(ilevel))//'/ic_poscx'
     INQUIRE(file=filename_x,exist=ok)
     read_pos=.false.
     if(ok)then
        read_pos=.true.
     else
        if(g%myid==1)write(*,*)'File '//TRIM(filename_x)//' not found.'
     endif

     ! Allocate initial conditions array
     allocate(init_array(i1_min:i1_max,i2_min:i2_max,i3_min:i3_max))
     init_array=0
     if(read_pos)then
        allocate(init_array_x(i1_min:i1_max,i2_min:i2_max,i3_min:i3_max))
        init_array_x=0
     endif
     allocate(init_plane(1:g%n1(ilevel),1:g%n2(ilevel)))
     
     ! Loop over dimensions
     do idim=1,ndim
        
        ! Read dark matter initial displacement field
        if(idim==1)filename=TRIM(r%initfile(ilevel))//'/ic_velcx'
        if(idim==2)filename=TRIM(r%initfile(ilevel))//'/ic_velcy'
        if(idim==3)filename=TRIM(r%initfile(ilevel))//'/ic_velcz'

        if(g%myid==1)write(*,*)'Reading file '//TRIM(filename)           
        open(10,file=filename,form='unformatted')
        rewind 10
        read(10) ! skip first line
        do i3=1,i3_min-1
           read(10)
        end do
        do i3=i3_min,i3_max
           read(10) ((init_plane(i1,i2),i1=1,g%n1(ilevel)),i2=1,g%n2(ilevel))
           init_array(i1_min:i1_max,i2_min:i2_max,i3) = init_plane(i1_min:i1_max,i2_min:i2_max)
        end do
        close(10)
                
        ! Reading dark matter initial position field
        if(read_pos)then
           if(idim==1)filename_x=TRIM(r%initfile(ilevel))//'/ic_poscx'
           if(idim==2)filename_x=TRIM(r%initfile(ilevel))//'/ic_poscy'
           if(idim==3)filename_x=TRIM(r%initfile(ilevel))//'/ic_poscz'
           
           ! Reading the displacement file
           if(g%myid==1)write(*,*)'Reading file '//TRIM(filename_x)
           open(10,file=filename_x,form='unformatted')
           rewind 10
           read(10) ! skip first line
           do i3=1,i3_min-1
              read(10)
           end do
           do i3=i3_min,i3_max
              read(10) ((init_plane(i1,i2),i1=1,g%n1(ilevel)),i2=1,g%n2(ilevel))
              init_array_x(i1_min:i1_max,i2_min:i2_max,i3) = init_plane(i1_min:i1_max,i2_min:i2_max)
           end do
           close(10)
        end if
        
        ! Rescale initial displacement field to code units
        init_array=g%dfact(ilevel)*dx/g%dxini(ilevel)*init_array/g%vfact(ilevel)
        if(read_pos)then
           init_array_x = init_array_x/g%boxlen_ini
        endif

        ipart=ipart_old
        do igrid=m%head(ilevel),m%tail(ilevel)
           do ind=1,twotondim
              ! Coordinates in grafic file units (between 1 and n1, n2, n3)
              xx1=(2*m%grid(igrid)%ckey(1)+MOD((ind-1)  ,2)+0.5)*dx
              xx2=(2*m%grid(igrid)%ckey(2)+MOD((ind-1)/2,2)+0.5)*dx
              xx3=(2*m%grid(igrid)%ckey(3)+MOD((ind-1)/4,2)+0.5)*dx
              xx1=(xx1*(g%dxini(ilevel)/dx)-g%xoff1(ilevel))/g%dxini(ilevel)
              xx2=(xx2*(g%dxini(ilevel)/dx)-g%xoff2(ilevel))/g%dxini(ilevel)
              xx3=(xx3*(g%dxini(ilevel)/dx)-g%xoff3(ilevel))/g%dxini(ilevel)
              i1=int(xx1)+1
              i2=int(xx2)+1
              i3=int(xx3)+1
              ! Add a particle if we have a leaf cell
              if(.not. m%grid(igrid)%refined(ind))then
                 ipart=ipart+1
                 p%vp(ipart,idim)=init_array(i1,i2,i3)
                 if(read_pos)then
                    p%xp(ipart,idim)=p%xp(ipart,idim)+init_array_x(i1,i2,i3)
                 endif
                 p%idp(ipart)=i1+(i2-1)*g%n1(ilevel)+(i3-1)*g%n1(ilevel)*g%n2(ilevel)
              endif
           end do
           ! End loop over cells
        end do
        ! End loop over grids
     end do
     ! End loop over dimensions

     ! Deallocate temporary array
     deallocate(init_plane,init_array)
     if(read_pos)deallocate(init_array_x)

  end do
  ! End loop over levels

  ! Move particle according to Zeldovich approximation
  if(.not. read_pos)then
     do ipart=1,p%npart
        p%xp(ipart,1:ndim)=p%xp(ipart,1:ndim)+p%vp(ipart,1:ndim)
     enddo
  endif

  ! Scale displacement to velocity
  do ipart=1,p%npart
     p%vp(ipart,1:ndim)=g%vfact(1)*p%vp(ipart,1:ndim)
  end do

  ! Periodic box
  do ipart=1,p%npart
     do idim=1,ndim
        if(r%periodic(idim))then
           if(p%xp(ipart,idim)< 0.0d0           )p%xp(ipart,idim)=p%xp(ipart,idim)+r%box_size(idim)
           if(p%xp(ipart,idim)>=r%box_size(idim))p%xp(ipart,idim)=p%xp(ipart,idim)-r%box_size(idim)
        end if
     end do
  end do

  ! Compute particle initial level
  do ipart=1,p%npart
     p%levelp(ipart)=r%levelmin
  end do

  ! Put all particles in levelmin
  p%headp=p%npart+1
  p%tailp=p%npart
  p%headp(r%levelmin)=1
  p%tailp(r%levelmin)=p%npart
  if(ANY(.not.r%periodic(1:ndim)))then
     p%headp(r%levelmin-1)=1
     p%tailp(r%levelmin-1)=0
  endif

#endif
  
end subroutine input_part_zoom
!#########################################################################
!#########################################################################
!#########################################################################
!#########################################################################
end module input_part_zoom_module

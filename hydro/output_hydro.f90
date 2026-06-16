module output_hydro_module
contains
!###################################################
!###################################################
!###################################################
!###################################################
recursive subroutine r_output_hydro(pst,input_array,input_size,output_array,output_size)
  use mdl_module
  use amr_parameters, only: flen
  use ramses_commons, only: pst_t
  use mdl_parameters
  implicit none
  type(pst_t)::pst
  integer,VALUE::input_size
  integer::output_size
  integer,dimension(1:input_size)::input_array
  integer,dimension(1:output_size)::output_array
  
  character(LEN=flen)::filename
  integer::rID

  if(pst%nLower>0)then
     rID = mdl_send_request(pst%s%mdl,MDL_OUTPUT_HYDRO,pst%iUpper+1,input_size,output_size,input_array)
     call r_output_hydro(pst%pLower,input_array,input_size,output_array,output_size)
     call mdl_get_reply(pst%s%mdl,rID,output_size)
  else
     filename=transfer(input_array,filename)
     if(index(filename,'output')==0)then
        call backup_hydro(pst%s%r,pst%s%g,pst%s%m,pst%s%mdl,filename)
     else
        call output_hydro(pst%s,filename)
     endif
  endif

end subroutine r_output_hydro
!###################################################
!###################################################
!###################################################
!###################################################
subroutine output_hydro(s,filename)
  use amr_parameters, only: ndim, twotondim, flen
  use hydro_parameters, only: nvar, nprim, nener,ie
  use ramses_commons, only: ramses_t, open_file, close_file
  implicit none
  type(ramses_t)::s
  character(LEN=flen)::filename
  !-----------------------------------
  ! Output hydro data in file
  !-----------------------------------
  integer::ilevel,igrid,ilun,irad,n,ind
  integer(kind=8),dimension(s%r%levelmin:s%r%nlevelmax)::nskip
  real(kind=4),dimension(1:twotondim,1:nprim)::qout
  real(kind=8),dimension(1:twotondim,1:nprim)::qold
  real(kind=8),dimension(1:twotondim,1:nvar)::uold
  real(kind=8),dimension(1:twotondim,1:6)::bold
  real(kind=8)::vx,vy,vz,bx,by,bz
  real(kind=8)::etot,ekin,emag,erad,dd,pp

  associate(r=>s%r,g=>s%g,m=>s%m,mdl=>s%mdl)

#ifdef HYDRO

  call open_file(s,filename,nskip,ilun)

  do ilevel=r%levelmin,r%nlevelmax

     write(ilun,POS=nskip(ilevel))

     do igrid=m%head(ilevel),m%tail(ilevel)

        uold=m%uold(:,:,igrid)
#ifdef MHD
        bold=m%bold(:,:,igrid)
#endif
        do ind=1,twotondim

           ! Compute density
           dd=uold(ind,1)

           ! Compute velocity
           vx=uold(ind,2)/dd
           vy=uold(ind,3)/dd
           vz=uold(ind,4)/dd

           ! Compute kinetic energy
           ekin=0.5*dd*(vx**2+vy**2+vz**2)
           emag=0.0
#ifdef MHD
           ! Compute cell-centered magnetic field
           bx=0.5*(bold(ind,1)+bold(ind,4))
           by=0.5*(bold(ind,2)+bold(ind,5))
           bz=0.5*(bold(ind,3)+bold(ind,6))
           emag=0.5*(bx**2+by**2+bz**2)
#endif
           erad=0.0
#if NENER>0
           ! Compute non-thermal energy
           do irad=1,nener
              erad=erad+uold(ind,5+irad)
           end do
#endif
           ! Compute pressure
           etot=uold(ind,5)
           pp=(r%gamma-1)*(etot-ekin-emag-erad)

           ! Store primitive variables
           qold(ind,1)=dd
           qold(ind,2)=vx
           qold(ind,3)=vy
           qold(ind,4)=vz
           qold(ind,5)=pp
#ifdef MHD
           qold(ind,6)=bx
           qold(ind,7)=by
           qold(ind,8)=bz
           ! If one want to output instead the divergence in 2D
           ! qold(ind,8)=abs(bold(ind,5)-bold(ind,2)+bold(ind,4)-bold(ind,1))/r%boxlen/0.5**ilevel
#endif
#if NENER>0
           do irad=1,nener
              qold(ind,ie+irad)=(r%gamma_rad(irad)-1)*uold(ind,5+irad)
           end do
#endif
#if NVAR>5+NENER
           ! Compute passive scalars
           do n=1,nvar-5-nener
              qold(ind,ie+nener+n)=uold(ind,5+nener+n)/dd
           end do
#endif
        end do
        qout=real(qold,kind=4)
        write(ilun)qout

     end do
  enddo

  call close_file(s,filename,nskip,ilun)

#endif

  end associate

end subroutine output_hydro
!###################################################
!###################################################
!###################################################
!###################################################
subroutine backup_hydro(r,g,m,mdl,filename)
  use amr_parameters, only: ndim, flen
  use hydro_parameters, only: nvar
  use amr_commons, only: run_t, global_t, mesh_t
  use mdl_module
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  type(mdl_t)::mdl
  character(LEN=flen)::filename

  integer::ilevel,igrid,ilun,ierr
  character(LEN=5)::nchar
  character(LEN=flen)::fileloc
  logical::file_exist

#ifdef HYDRO
  ilun=10+mdl_core(mdl)
  call title(g%myid,nchar)
  fileloc=TRIM(filename)//TRIM(nchar)
  inquire(file=fileloc,exist=file_exist)
  if (file_exist) then
     open(unit=ilun,file=fileloc,iostat=ierr)
     close(ilun,status="delete")
  end if
  open(unit=ilun,file=fileloc,access="stream",action="write",form='unformatted')
  write(ilun)ndim
#ifdef MHD
  write(ilun)nvar+6
#else
  write(ilun)nvar
#endif
  write(ilun)r%levelmin
  write(ilun)r%nlevelmax
  do ilevel=r%levelmin,r%nlevelmax
     write(ilun)m%noct(ilevel)
  enddo
  do ilevel=r%levelmin,r%nlevelmax
     do igrid=m%head(ilevel),m%tail(ilevel)
        write(ilun)m%uold(:,:,igrid)
#ifdef MHD
        write(ilun)m%bold(:,:,igrid)
#endif
     end do
  enddo
  close(ilun)

#endif

end subroutine backup_hydro
!###################################################
!###################################################
!###################################################
!###################################################
subroutine file_descriptor_hydro(r,filename,write_bkp_file)
  use amr_parameters, only: ndim, flen
  use hydro_parameters, only: nvar, nener, nprim, ie, nion
  use amr_commons, only: run_t
  implicit none
  type(run_t)::r
  character(LEN=flen)::filename
  logical::write_bkp_file
  
  character(LEN=flen)::fileloc
  integer::ivar,ilun

  if(r%verbose)write(*,*)'Entering file_descriptor_hydro'

  ilun=11

  ! Open file
  fileloc=TRIM(filename)
  open(unit=ilun,file=fileloc,form='formatted')

  if(write_bkp_file)then
     ! Write variable names in backup file
#ifdef MHD
     write(ilun,'("nvar        =",I11)')nvar+6
#else
     write(ilun,'("nvar        =",I11)')nvar
#endif
     ivar=1
     write(ilun,'("variable #",I2,": density")')ivar
     ivar=2
     write(ilun,'("variable #",I2,": momentum_x")')ivar
     ivar=3
     write(ilun,'("variable #",I2,": momentum_y")')ivar
     ivar=4
     write(ilun,'("variable #",I2,": momentum_z")')ivar
     ivar=5
     write(ilun,'("variable #",I2,": total_energy")')ivar
#if NENER>0
     ! Non-thermal pressures
     do ivar=6,5+nener
        write(ilun,'("variable #",I2,": non_thermal_energy_",I1)')ivar,ivar-5
     end do
#endif
#if NVAR>5+NENER
     ! Passive scalars
     do ivar=6+nener,nvar
        write(ilun,'("variable #",I2,": density_scalar_",I1)')ivar,ivar-5-nener
     end do
#endif
#ifdef MHD
     ivar=nvar+1
     write(ilun,'("variable #",I2,": left_magnetic_field_x")')ivar
     ivar=nvar+2
     write(ilun,'("variable #",I2,": left_magnetic_field_y")')ivar
     ivar=nvar+3
     write(ilun,'("variable #",I2,": left_magnetic_field_z")')ivar
     ivar=nvar+4
     write(ilun,'("variable #",I2,": right_magnetic_field_x")')ivar
     ivar=nvar+5
     write(ilun,'("variable #",I2,": right_magnetic_field_y")')ivar
     ivar=nvar+6
     write(ilun,'("variable #",I2,": right_magnetic_field_z")')ivar
#endif
  else
     ! Write variable names in output file
     write(ilun,'("nvar        =",I11)')nprim
     ivar=1
     write(ilun,'("variable #",I2,": density")')ivar
     ivar=2
     write(ilun,'("variable #",I2,": velocity_x")')ivar
     ivar=3
     write(ilun,'("variable #",I2,": velocity_y")')ivar
     ivar=4
     write(ilun,'("variable #",I2,": velocity_z")')ivar
     ivar=5
     write(ilun,'("variable #",I2,": thermal_pressure")')ivar
#ifdef MHD
     ivar=6
     write(ilun,'("variable #",I2,": magnetic_field_x")')ivar
     ivar=7
     write(ilun,'("variable #",I2,": magnetic_field_y")')ivar
     ivar=8
     write(ilun,'("variable #",I2,": magnetic_field_z")')ivar
#endif
#if NENER>0
     ! Non-thermal pressures
     do ivar=ie+1,ie+nener
        write(ilun,'("variable #",I2,": non_thermal_pressure_",I1)')ivar,ivar-ie
     end do
#endif
#if NVAR>5+NENER
     ! Passive scalars
     ivar=ie+1+nener
     do while(ivar.le.nprim)
        if(r%entropy.and.ivar+5-ie.eq.r%ientropy)then
           write(ilun,'("variable #",I2,": entropy")')ivar
           ivar=ivar+1
        else if (r%metal.and.ivar+5-ie.eq.r%imetal)then
           write(ilun,'("variable #",I2,": metal_mass_fraction")')ivar
           ivar=ivar+1
        else if (r%sgs_turb.and.ivar+5-ie.eq.r%iturb)then
           write(ilun,'("variable #",I2,": turb_kinetic_energy")')ivar
           ivar=ivar+1
        else if(ivar+5-ie.ge.r%iions.and.ivar+5-ie.lt.r%iions+nion) then
          ! Special cases for ionisation fractions
          if(r%ixHI.gt.0) then
            write(ilun,'("variable #",I2,": xHI")')ivar
            ivar=ivar+1
          endif
          if(r%ixHII.gt.0) then
            write(ilun,'("variable #",I2,": xHII")')ivar
            ivar=ivar+1
          endif
          if(r%ixHeII.gt.0) then
            write(ilun,'("variable #",I2,": xHeII")')ivar
            ivar=ivar+1
          endif
          if(r%ixHeIII.gt.0) then
            write(ilun,'("variable #",I2,": xHeIII")')ivar
            ivar=ivar+1
          endif
          do while(ivar.lt.r%iions+nion)
            write(ilun,'("variable #",I2,": x_unused")')ivar
            ivar=ivar+1
          end do
        else
          write(ilun,'("variable #",I2,": scalar_",I1)')ivar,ivar-ie-nener
          ivar=ivar+1
        endif
     end do
#endif
  endif
  close(ilun)

end subroutine file_descriptor_hydro
end module output_hydro_module

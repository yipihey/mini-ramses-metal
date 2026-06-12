#############################################################################
# If you have problems with this makefile, contact Romain.Teyssier@gmail.com
#############################################################################
# Compilation time parameters
COMPILER = GNU
CUDA_ARCH = sm_80
NDIM = 3
MPI = 0
HYDRO = 1
GRAV = 0
MHD = 0
RT = 0
RTZ = 0
INIT =
UNITS =
NENER = 0
NMETAL = 0
TURB = 0
TRCFLX = 0
N_HYDROGEN_IONS = 0
N_H2 = 0
N_HELIUM_IONS = 0
N_CARBON_IONS = 0
N_NITROGEN_IONS = 0
N_OXYGEN_IONS = 0
N_NEON_IONS = 0
N_MAGNESIUM_IONS = 0
N_SILICON_IONS = 0
N_SULFUR_IONS = 0
N_IRON_IONS = 0
NION = 0
NPSCAL = 0
NRTGRP = 0
NHILBERT = 1
NVECTOR = 32
NPRE = 8
PATCH =
CUDA_PROFILE = 0
PAPER = 0
CUB_SORT_PART = 1
# Warp-cooperative fine-force gather in the GPU particle kick (default OFF).
KICK_COOP_GATHER = 0
INTEGRATOR_NVTX = 0
CUB_SCAN_REFINE = 1
FASTMATH = 0
DEBUG = 0
HDF5 = 0
MPIF90 = mpif90
EXEC = ramses
#############################################################################
# Compute NVAR
# MHD=1 raises the conserved baseline to 8 (the documented MHD build,
# hydro_parameters.f90: ie=8, nprim=NVAR+3): slots are rho, momenta, E(5)
# + 3 passive-scalar slots; B lives in the separate bold/bnew face arrays,
# NOT in uold. NENER/NPSCAL/... add on top.
ifeq ($(MHD),1)
NHYDRO = 8
else
NHYDRO = 5
endif
ifeq ($(RTZ),1)
NION := $(N_CARBON_IONS)+$(N_NITROGEN_IONS)+$(N_OXYGEN_IONS)
NION := $(NION)+$(N_NEON_IONS)+$(N_MAGNESIUM_IONS)+$(N_SILICON_IONS)
NION := $(NION)+$(N_SULFUR_IONS)+$(N_IRON_IONS)
NION := $(NION)+$(N_HYDROGEN_IONS)+$(N_H2)+$(N_HELIUM_IONS)
endif
NVAR := $(NHYDRO)+$(NENER)+$(NPSCAL)+$(NMETAL)+$(NION)
DEFINES = -DNDIM=$(NDIM) -DNVAR=$(NVAR)
#############################################################################
# Collect git version and Makefile options
GITBRANCH = $(shell git rev-parse --abbrev-ref HEAD)
GITHASH = $(shell git log --pretty=format:'%H' -n 1)
GITREPO = $(shell git config --get remote.origin.url)
BUILDDATE = $(shell date +"%D-%T")
BUILDCOMMAND = ${MAKE} ${MAKEFLAGS}
#############################################################################
# RTZ options
ifeq ($(RTZ),1)
DEFINES += -DN_HYDROGEN_IONS=$(N_HYDROGEN_IONS)
DEFINES += -DN_HELIUM_IONS=$(N_HELIUM_IONS)
DEFINES += -DN_CARBON_IONS=$(N_CARBON_IONS)
DEFINES += -DN_NITROGEN_IONS=$(N_NITROGEN_IONS)
DEFINES += -DN_OXYGEN_IONS=$(N_OXYGEN_IONS)
DEFINES += -DN_NEON_IONS=$(N_NEON_IONS)
DEFINES += -DN_MAGNESIUM_IONS=$(N_MAGNESIUM_IONS)
DEFINES += -DN_SILICON_IONS=$(N_SILICON_IONS)
DEFINES += -DN_SULFUR_IONS=$(N_SULFUR_IONS)
DEFINES += -DN_IRON_IONS=$(N_IRON_IONS)
DEFINES += -DN_H2=$(N_H2)
endif
#############################################################################
# Main compilation options
ifneq ($(NION),0)
   DEFINES += -DNION=$(NION)
endif
ifeq ($(HYDRO),1)
   DEFINES += -DHYDRO
endif
ifeq ($(MHD),1)
   DEFINES += -DMHD
endif
ifeq ($(GRAV),1)
   DEFINES += -DGRAV
endif
ifeq ($(TURB),1)
   DEFINES += -DTURB
endif
ifeq ($(TRCFLX),1)
   DEFINES += -DTRCFLX
endif
ifneq ($(NENER),0)
   DEFINES += -DNENER=$(NENER)
endif
ifeq ($(RT),1)
   DEFINES += -DRT -DNRTGRP=$(NRTGRP)
endif
ifeq ($(RTZ),1)
   DEFINES += -DRTZ
endif
ifneq ($(NVECTOR),32)
   DEFINES += -DNVECTOR=$(NVECTOR)
endif
ifneq ($(NPRE),8)
   DEFINES += -DNPRE=$(NPRE)
endif
ifneq ($(NHILBERT),1)
   DEFINES += -DNHILBERT=$(NHILBERT)
endif
#############################################################################
# Set proper units and initial conditions
ifneq ($(INIT),)
   DEFINES += -DINIT=$(INIT)
endif
ifneq ($(UNITS),)
   DEFINES += -DUNITS=$(UNITS)
endif
#############################################################################
# GNU compiler (gfortran)
ifeq ($(COMPILER),GNU)
   FFLAGS = -cpp $(DEFINES)
   ifeq ($(MPI),1)
      F90 = $(MPIF90)
      FC = gfortran
   else
      F90 = gfortran
      FC = gfortran
      FFLAGS += -DWITHOUTMPI
   endif
   F90 += -ffree-line-length-none -fimplicit-none
   FC += -ffree-line-length-none -fimplicit-none
   ifeq ($(DEBUG),1)
      F90 += -O0 -fcheck=bounds
      FFLAGS += -g -fbacktrace -ffpe-trap=zero,overflow,invalid -finit-real=nan
      F90 += -Wuninitialized -Wall
   else
      F90 += -O3
   endif
endif
#############################################################################
# Intel compiler
ifeq ($(COMPILER),INTEL)
   FFLAGS = -cpp $(DEFINES)
   ifeq ($(MPI),1)
      F90 = $(MPIF90)
      FC = ifort
   else
      F90 = ifort
      FC = ifort
      FFLAGS += -DWITHOUTMPI
   endif
   F90 += -fp-model source
   FC += -fp-model source
   ifeq ($(DEBUG),1)
      F90 += -warn all -O0 -g -traceback -check bounds
      FFLAGS += -fpe0 -ftrapuv -init=zero -init=snan -init=arrays
   else
      F90 += -O3 -diag-disable=10448
   endif
endif
#############################################################################
# NVHPC compiler (optional -gpu=$(CUDA_ARCH),fastmath via FASTMATH=1)
ifeq ($(COMPILER),NVHPC)
   ifeq ($(PAPER),1)
      DEFINES += -DPAPER
   endif
   ifneq ($(CUB_SORT_PART),0)
      DEFINES += -DCUB_SORT_PART
   endif
   ifneq ($(CUB_SCAN_REFINE),0)
      DEFINES += -DCUB_SCAN_REFINE
   endif
   ifeq ($(KICK_COOP_GATHER),1)
      DEFINES += -DKICK_COOP_GATHER
   endif
   ifeq ($(INTEGRATOR_NVTX),1)
      DEFINES += -DINTEGRATOR_NVTX
   endif
ifeq ($(FASTMATH),1)
   GPU_ARCH_OPTS = $(CUDA_ARCH),fastmath
else
   GPU_ARCH_OPTS = $(CUDA_ARCH)
endif
   FFLAGS = -cpp -gpu=$(GPU_ARCH_OPTS) -cuda -cudalib=nvtx $(DEFINES)
   LIBS = -gpu=$(GPU_ARCH_OPTS) -cuda -cudalib=nvtx -lstdc++
   GPU_PATH = ../gpu
   ifeq ($(CUDA_PROFILE),1)
      FFLAGS += -gpu=lineinfo,ptxinfo -Minstrument -traceback -gopt
      LIBS += -gpu=lineinfo,ptxinfo -Minstrument -traceback -gopt
   endif
   ifeq ($(MPI),1)
      F90 = $(MPIF90)
      FC = nvfortran
   else
      F90 = nvfortran
      FC = nvfortran
      FFLAGS += -DWITHOUTMPI
   endif
   F90 += -Mfreeform
   FC += -Mfreeform
   ifeq ($(DEBUG),1)
      F90 += -O0 -traceback -g -gpu=debug,lineinfo -Mprof=dwarf
      FFLAGS += -O0 -ffpe-trap=zero,overflow,invalid -Minit-real=snan -g -gpu=debug,lineinfo -Mprof=dwarf
   else
      F90 += -O4
   endif
endif
#############################################################################
C = ifort
CFLAGS = -g -O3
#############################################################################
MOD = mod
#############################################################################
# MPI librairies if needed
LIBMPI =
#LIBMPI = -lfmpi -lmpi -lelan
ifeq ($(HDF5),1)
   LIBHDF5 = -lhdf5 -lhdf5_hl -lhdf5_fortran -lhdf5hl_fortran
endif
ifeq ($(TURB),1)
   FFTW = /usr/local/fftw/gcc/3.3.9
   LIBFFTW = -L$(FFTW)/lib -lfftw3
   FFLAGS += -I$(FFTW)/include
endif
LIBS += $(LIBMPI) $(LIBFFTW)
#############################################################################
# Sources directories are searched in this exact order
VPATH = $(PATCH):$(GPU_PATH):../feedback:../sink:../star:../cooling:../turb:../hydro:../pm:../poisson:../amr:../clump:../mdl1:../rt:../lightcone
#############################################################################
# Modules
MODOBJ = amr_parameters.o hydro_parameters.o rt_parameters.o oct_commons.o \
	 hash.o pm_parameters.o pm_commons.o hilbert.o domain_hilbert.o \
	 hydro_commons.o rt_commons.o amr_commons.o poisson_parameters.o rngstream.o \
	 constants.o cooling_module.o cooling_module_ism.o init_cooling.o \
	 coolrates_module.o \
	 collisional_ionization_module.o recombination_module.o \
	 cosmic_ray_ionization_module.o photoionization_UVB_module.o dust_recombination_module.o \
	 molecules_module.o charge_exchange_module.o rtz_module.o
ifeq ($(RTZ),1)
   MODOBJ += rtz_coolrates_module.o rtz_cooling_module.o
else
   MODOBJ += neq_cooling_module.o
endif
MODOBJ += init_neq_chem.o clfind_commons.o turb_commons.o mdl_commons.o mdl.o \
	 cross_sections_module.o rt_spectra.o  \
	 ramses_commons.o call_back.o cache_commons.o cache.o gadgetreadfile.o \
	 boundaries.o output_hydro.o output_poisson.o output_part.o output_rt.o output_amr.o movie.o \
	 sort.o nbors_utils.o hydro_flag.o marshal.o
ifeq ($(COMPILER),NVHPC)
   MODOBJ += cub_module_radix_sort_c.o cub_module_radix_sort_f.o \
      cub_inclusive_scan_c.o cub_inclusive_scan_f.o gpu_utils.o gpu_hydro.o \
      gpu_cooling.o gpu_rho.o gpu_mg.o gpu_refine.o gpu_flag.o gpu_runner.o \
      gpu_part.o gpu_manager.o
endif
MODOBJ +=  multigrid_fine_coarse.o rho_fine.o upload.o init_xion.o \
	 rt_flux_module.o rt_godunov_fine.o rt_input_condinit.o rt_init_flow_fine.o rt_units.o  \
	 rt_condinit.o rt_flag.o rt_upload.o interpol_rt.o rt_godunov_utils.o read_rt_params.o \
	 turb_hydro.o turb_driving.o turb_init.o turb_update.o \
	 lightcone_utils.o lightcone_buffer.o lightcone_io.o lightcone_test.o lightcone.o \
	 init_hydro.o init_part.o init_amr.o smooth.o flag_utils.o init_time.o read_params.o \
	 init_rt.o input_part_grafic.o input_part_zoom.o input_part_ascii.o \
	 input_part_restart.o input_part_ramses.o input_part_gadget.o input_part.o input_hydro_grafic.o \
	 interpol_hydro.o update_time.o timer.o init_refine_basegrid.o load_balance.o refine_utils.o \
	 grav_ana.o interpol_phi.o phi_fine_cg.o force_fine.o poisson_flag.o multigrid_fine_commons.o \
	 boundana.o godunov_utils.o write_screen.o courant_fine.o newdt_fine.o \
	 godunov_fine.o umuscl.o synchro_hydro_fine.o source_hydro_fine.o cooling_fine.o \
	 input_hydro_condinit.o input_hydro_gadget.o init_flow_fine.o \
	 condinit.o vecpotentialinit.o init_refine_adaptive.o init_refine_restart.o \
	 init_refine_ramses.o title.o units.o move_fine.o rho_ana.o \
	 star_formation.o feedback.o clump_merger.o output_clump.o clump_finder.o \
	 rt_star_feedback.o sink_formation.o tree_formation.o \
	 sink_evolution.o sink_merger.o task_manager.o rt_step.o amr_step.o adaptive_loop.o \
	 write_gitinfo.o
#############################################################################
ramses:	$(MODOBJ) ramses.o
	$(F90) $(FFLAGS) $(MODOBJ) ramses.o -o $(EXEC)$(NDIM)d $(LIBS)
#############################################################################
test_cooling:	$(MODOBJ) test_cooling.o
	$(F90) $(FFLAGS) $(MODOBJ) test_cooling.o -o test_cooling $(LIBS)
#############################################################################
write_gitinfo.o: FORCE
	$(FC) -O0 -cpp -DPATCH=\'"$(PATCH)"\' -DGITBRANCH=\'"$(GITBRANCH)"\' -DGITHASH=\'"$(GITHASH)"\' \
                -DGITREPO=\'"$(GITREPO)"\' -DBUILDDATE=\'"$(BUILDDATE)"\' -DBUILDCOMMAND=\'"$(BUILDCOMMAND)"\' -c ../amr/write_gitinfo.f90 -o $@
#############################################################################
cub_inclusive_scan_c.o: cub_inclusive_scan.h cub_inclusive_scan.cu
	nvcc -arch=$(CUDA_ARCH) -O3 -lineinfo -c $(GPU_PATH)/cub_inclusive_scan.cu -o cub_inclusive_scan_c.o
#############################################################################
cub_inclusive_scan_f.o: cub_inclusive_scan.cuf cub_inclusive_scan_c.o
	nvfortran $(FFLAGS) -c cub_inclusive_scan_c.o $(GPU_PATH)/cub_inclusive_scan.cuf -o cub_inclusive_scan_f.o
#############################################################################
cub_module_radix_sort_c.o: cub_module_radix_sort.h cub_module_radix_sort.cu
	nvcc -arch=$(CUDA_ARCH) -O3 -lineinfo -c $(GPU_PATH)/cub_module_radix_sort.cu -o cub_module_radix_sort_c.o
#############################################################################
cub_module_radix_sort_f.o: cub_module_radix_sort.cuf cub_module_radix_sort_c.o
	nvfortran $(FFLAGS) -c cub_module_radix_sort_c.o $(GPU_PATH)/cub_module_radix_sort.cuf -o cub_module_radix_sort_f.o

%.o:%.cuf
	$(F90) $(FFLAGS) -c $< -o $@ $(LIBS)
%.o:%.f90
	$(F90) $(FFLAGS) -c $< -o $@ $(LIBS)
%.o:%.c
	$(C) $(CFLAGS) -c $< -o $@
FORCE:
#############################################################################
-include deps.mk
#############################################################################
clean :
	rm -f *.o *.$(MOD)

clobber: clean
	-find . -type d -name "NDIM*" -prune -exec rm -rf '{}' \;
#############################################################################

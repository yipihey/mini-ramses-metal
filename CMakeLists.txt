cmake_minimum_required(VERSION 3.20)
cmake_policy(VERSION 3.20...3.28)

include(CMakePrintHelpers)

# =============================================================================
# Compiler selection
# Set COMPILER=GNU|INTEL|NVHPC before the first cmake run (or via FC env var).
# Example: cmake -DCOMPILER=NVHPC -DCMAKE_Fortran_COMPILER=nvfortran ..
#      or: FC=nvfortran cmake -DCOMPILER=NVHPC ..
# Changing COMPILER after the initial configure requires a fresh build dir.
# =============================================================================
set(COMPILER "GNU" CACHE STRING "Compiler toolchain: GNU, INTEL, NVHPC")

# Map COMPILER to the right executable on a fresh configure only.
if(NOT CMAKE_Fortran_COMPILER)
    if(COMPILER STREQUAL "NVHPC")
        set(CMAKE_Fortran_COMPILER nvfortran)
    elseif(COMPILER STREQUAL "INTEL")
        set(CMAKE_Fortran_COMPILER ifort)
    endif()
endif()

# Register .cuf as a Fortran extension before project() so that CMake's
# dependency scanner tracks MODULE / USE relationships inside .cuf files.
# Without this, parallel builds (make -j / ninja) fail because gpu_runner.mod
# is not yet present when poisson/*.f90 files that USE it start compiling.
# Must come before project() / enable_language(Fortran); it replaces (not
# augments) the built-in list, so all default extensions must be included.
if(COMPILER STREQUAL "NVHPC")
    set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS
        f F fpp FPP f77 F77 f90 F90 for For FOR f95 F95 f03 F03 f08 F08
        cuf CUF)
endif()

project(ramses LANGUAGES Fortran)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# =============================================================================
# Simulation parameters
# =============================================================================
set(NDIM     3    CACHE STRING "Number of dimensions (1, 2, or 3)")
set(NVECTOR  32   CACHE STRING "SIMD vector length")
set(NHILBERT 1    CACHE STRING "Hilbert key length exponent")
set(NPRE     8    CACHE STRING "Floating-point precision in bytes")
set(NENER    0    CACHE STRING "Number of non-thermal energy variables")
set(NMETAL   0    CACHE STRING "Number of metal species")
set(NPSCAL   0    CACHE STRING "Number of passive scalar tracers")
set(NION     0    CACHE STRING "Number of ionisation species (auto-computed when RTZ=ON)")
set(NRTGRP   0    CACHE STRING "Number of RT photon groups (used when RT=ON)")
set(UNITS    ""   CACHE STRING "Unit system (e.g. COSMO, COEUR; empty = default)")
set(INIT     ""   CACHE STRING "Initial condition module (e.g. LOOP, COEUR; empty = default)")

# RTZ per-species ion counts (only relevant when RTZ=ON)
set(N_HYDROGEN_IONS  0  CACHE STRING "RTZ: number of hydrogen  ion states")
set(N_H2             0  CACHE STRING "RTZ: number of H2 states")
set(N_HELIUM_IONS    0  CACHE STRING "RTZ: number of helium    ion states")
set(N_CARBON_IONS    0  CACHE STRING "RTZ: number of carbon    ion states")
set(N_NITROGEN_IONS  0  CACHE STRING "RTZ: number of nitrogen  ion states")
set(N_OXYGEN_IONS    0  CACHE STRING "RTZ: number of oxygen    ion states")
set(N_NEON_IONS      0  CACHE STRING "RTZ: number of neon      ion states")
set(N_MAGNESIUM_IONS 0  CACHE STRING "RTZ: number of magnesium ion states")
set(N_SILICON_IONS   0  CACHE STRING "RTZ: number of silicon   ion states")
set(N_SULFUR_IONS    0  CACHE STRING "RTZ: number of sulfur    ion states")
set(N_IRON_IONS      0  CACHE STRING "RTZ: number of iron      ion states")

# Physics toggles
option(HYDRO  "Activate hydrodynamics solver"          ON)
option(GRAV   "Activate self-gravity / Poisson solver" OFF)
option(MHD    "Activate MHD solver"                    OFF)
option(RT     "Activate radiative transfer"            OFF)
option(RTZ    "Activate RTZ (metal-line RT) solver"    OFF)
option(TURB   "Activate turbulence driving"            OFF)
option(TRCFLX "Activate tracer-flux advection"         OFF)

# Build options
option(DEBUG        "Enable debug compilation flags"               OFF)
option(MPI          "Enable MPI parallelism"                       OFF)
option(HDF5         "Enable HDF5 output"                           OFF)
option(CUDA_PROFILE "Enable NVTX profiling instrumentation"        OFF)
option(PAPER        "Enable PAPER timing mode (NVHPC only)"        OFF)
option(CUB_SORT_PART   "Use CUB radix sort for particles (NVHPC)" ON)
option(CUB_SCAN_REFINE "Use CUB inclusive scan for refine (NVHPC)" ON)

set(CUDA_ARCH "sm_80" CACHE STRING "CUDA target architecture (e.g. sm_80, sm_90)")

set(EXEC ramses${NDIM}d)

# =============================================================================
# Derived variables
# =============================================================================

# Compute NION from per-species counts when RTZ is on
if(RTZ)
    math(EXPR NION "${N_HYDROGEN_IONS} + ${N_H2} + ${N_HELIUM_IONS} + ${N_CARBON_IONS} + ${N_NITROGEN_IONS} + ${N_OXYGEN_IONS} + ${N_NEON_IONS} + ${N_MAGNESIUM_IONS} + ${N_SILICON_IONS} + ${N_SULFUR_IONS} + ${N_IRON_IONS}")
endif()

# NVAR = 5 (hydro) + NENER + NPSCAL + NMETAL + NION
math(EXPR NVAR "5 + ${NENER} + ${NPSCAL} + ${NMETAL} + ${NION}")

cmake_print_variables(COMPILER NDIM NVAR NENER NION NRTGRP UNITS INIT HYDRO GRAV MHD RT RTZ TURB)

# =============================================================================
# Preprocessor defines
# =============================================================================
set(defines
    -DNDIM=${NDIM}
    -DNVAR=${NVAR}
    -DNVECTOR=${NVECTOR}
    -DNPRE=${NPRE}
    -DNHILBERT=${NHILBERT}
)

if(NENER GREATER 0)
    list(APPEND defines -DNENER=${NENER})
endif()
if(NION GREATER 0)
    list(APPEND defines -DNION=${NION})
endif()
if(NOT UNITS STREQUAL "")
    list(APPEND defines -DUNITS=${UNITS})
endif()
if(NOT INIT STREQUAL "")
    list(APPEND defines -DINIT=${INIT})
endif()
if(HYDRO)
    list(APPEND defines -DHYDRO)
endif()
if(GRAV)
    list(APPEND defines -DGRAV)
endif()
if(MHD)
    list(APPEND defines -DMHD)
endif()
if(RT)
    list(APPEND defines -DRT -DNRTGRP=${NRTGRP})
endif()
if(RTZ)
    list(APPEND defines -DRTZ
        -DN_HYDROGEN_IONS=${N_HYDROGEN_IONS}
        -DN_H2=${N_H2}
        -DN_HELIUM_IONS=${N_HELIUM_IONS}
        -DN_CARBON_IONS=${N_CARBON_IONS}
        -DN_NITROGEN_IONS=${N_NITROGEN_IONS}
        -DN_OXYGEN_IONS=${N_OXYGEN_IONS}
        -DN_NEON_IONS=${N_NEON_IONS}
        -DN_MAGNESIUM_IONS=${N_MAGNESIUM_IONS}
        -DN_SILICON_IONS=${N_SILICON_IONS}
        -DN_SULFUR_IONS=${N_SULFUR_IONS}
        -DN_IRON_IONS=${N_IRON_IONS}
    )
endif()
if(TURB)
    list(APPEND defines -DTURB)
endif()
if(TRCFLX)
    list(APPEND defines -DTRCFLX)
endif()
if(NOT MPI)
    list(APPEND defines -DWITHOUTMPI)
endif()

# =============================================================================
# Executable
# =============================================================================
add_executable(${EXEC} "")

# =============================================================================
# MPI
# =============================================================================
if(MPI)
    find_package(MPI REQUIRED Fortran)
    target_include_directories(${EXEC} PRIVATE ${MPI_Fortran_INCLUDE_PATH})
    target_link_libraries(${EXEC} PRIVATE ${MPI_Fortran_LIBRARIES})
    separate_arguments(mpi_compile_flags NATIVE_COMMAND "${MPI_Fortran_COMPILE_FLAGS}")
    target_compile_options(${EXEC} PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:${mpi_compile_flags}>)
endif()

# =============================================================================
# Compiler flags
# =============================================================================
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(fort_flags -cpp ${defines} -ffree-line-length-none -fimplicit-none -frecord-marker=4)
    if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        list(APPEND fort_flags -fallow-argument-mismatch)
    endif()
    if(DEBUG)
        list(APPEND fort_flags
            -O0 -g -fcheck=bounds -fbacktrace
            -ffpe-trap=zero,overflow,invalid -finit-real=nan
            -Wuninitialized -Wall)
    else()
        list(APPEND fort_flags -O3)
    endif()
    if(APPLE)
        target_link_options(${EXEC} PRIVATE -Wl,-no_compact_unwind)
    endif()

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(fort_flags -cpp ${defines} -recursive -fp-model source)
    if(DEBUG)
        list(APPEND fort_flags
            -warn all -O0 -g -traceback -check bounds
            -fpe0 -ftrapuv -init=zero -init=snan -init=arrays)
    else()
        list(APPEND fort_flags -O3 -diag-disable=10448)
    endif()

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    list(APPEND defines -DCUB_SORT_PART -DCUB_SCAN_REFINE)
    if(NOT CUB_SORT_PART)
        list(REMOVE_ITEM defines -DCUB_SORT_PART)
    endif()
    if(NOT CUB_SCAN_REFINE)
        list(REMOVE_ITEM defines -DCUB_SCAN_REFINE)
    endif()
    if(PAPER)
        list(APPEND defines -DPAPER)
    endif()
    set(fort_flags -cpp ${defines} -gpu=${CUDA_ARCH},fastmath -cuda -cudalib=nvtx -Mfreeform)
    if(DEBUG)
        list(APPEND fort_flags
            -O0 -traceback -g
            -gpu=debug,lineinfo -Mprof=dwarf
            -ffpe-trap=zero,overflow,invalid -Minit-real=snan)
    else()
        list(APPEND fort_flags -O4)
    endif()
    if(CUDA_PROFILE)
        list(APPEND fort_flags -gpu=lineinfo,ptxinfo -Minstrument -traceback -gopt)
    endif()
    # Link flags mirror LIBS in the Makefile
    set(nvhpc_link_flags -gpu=${CUDA_ARCH},fastmath -cuda -cudalib=nvtx -lstdc++)
    if(CUDA_PROFILE)
        list(APPEND nvhpc_link_flags -gpu=lineinfo,ptxinfo -Minstrument -traceback -gopt)
    endif()
    target_link_options(${EXEC} PRIVATE ${nvhpc_link_flags})

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
    set(fort_flags -e ZR ${defines})
endif()

target_compile_options(${EXEC} PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:${fort_flags}>)

# =============================================================================
# Generated source files (git info, build command, patch)
# =============================================================================
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/write_gitinfo.f90
    COMMAND ${CMAKE_SOURCE_DIR}/utils/scripts/cr_write_gitinfo.sh
            ${CMAKE_BINARY_DIR}/write_gitinfo.f90
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/write_makefile.f90
    COMMAND ${CMAKE_SOURCE_DIR}/utils/scripts/cr_write_makefile.sh
            ${CMAKE_SOURCE_DIR}/CMakeLists.txt
            ${CMAKE_BINARY_DIR}/write_makefile.f90
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/write_patch.f90
    COMMAND ${CMAKE_SOURCE_DIR}/utils/scripts/cr_write_patch.sh
            "" ${CMAKE_BINARY_DIR}/write_patch.f90
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)

# =============================================================================
# Fortran source files
# =============================================================================
set(fortran_files
  # AMR core
  amr/amr_parameters.f90
  amr/amr_commons.f90
  amr/oct_commons.f90
  amr/mdl_commons.f90
  amr/ramses_commons.f90
  amr/cache_commons.f90
  amr/hilbert.f90
  amr/domain_hilbert.f90
  amr/hash.f90
  amr/rngstream.f90
  amr/call_back.f90
  amr/boundaries.f90
  amr/nbors_utils.f90
  amr/flag_utils.f90
  amr/refine_utils.f90
  amr/smooth.f90
  amr/sort.f90
  amr/marshal.f90
  amr/cache.f90
  amr/timer.f90
  amr/title.f90
  amr/units.f90
  amr/output_amr.f90
  amr/movie.f90
  amr/load_balance.f90
  amr/read_params.f90
  amr/init_amr.f90
  amr/init_time.f90
  amr/init_refine_basegrid.f90
  amr/init_refine_adaptive.f90
  amr/init_refine_restart.f90
  amr/init_refine_ramses.f90
  amr/update_time.f90
  amr/amr_step.f90
  amr/adaptive_loop.f90
  amr/task_manager.f90
  amr/ramses.f90

  # Hydro
  hydro/hydro_parameters.f90
  hydro/hydro_commons.f90
  hydro/hydro_flag.f90
  hydro/boundana.f90
  hydro/godunov_utils.f90
  hydro/write_screen.f90
  hydro/output_hydro.f90
  hydro/interpol_hydro.f90
  hydro/init_hydro.f90
  hydro/init_flow_fine.f90
  hydro/input_hydro_condinit.f90
  hydro/input_hydro_grafic.f90
  hydro/input_hydro_gadget.f90
  hydro/courant_fine.f90
  hydro/umuscl.f90
  hydro/godunov_fine.f90
  hydro/synchro_hydro_fine.f90
  hydro/source_hydro_fine.f90
  hydro/upload.f90
  hydro/vecpotentialinit.f90
  hydro/condinit.f90

  # Particle mesh
  pm/pm_parameters.f90
  pm/pm_commons.f90
  pm/gadgetreadfile.f90
  pm/output_part.f90
  pm/init_part.f90
  pm/input_part.f90
  pm/input_part_restart.f90
  pm/input_part_ascii.f90
  pm/input_part_grafic.f90
  pm/input_part_gadget.f90
  pm/input_part_ramses.f90
  pm/input_part_zoom.f90
  pm/newdt_fine.f90
  pm/rho_fine.f90
  pm/move_fine.f90

  # Poisson / gravity
  poisson/poisson_parameters.f90
  poisson/multigrid_fine_commons.f90
  poisson/output_poisson.f90
  poisson/poisson_flag.f90
  poisson/rho_ana.f90
  poisson/grav_ana.f90
  poisson/interpol_phi.f90
  poisson/phi_fine_cg.f90
  poisson/force_fine.f90
  poisson/multigrid_fine_coarse.f90

  # Cooling
  cooling/constants.f90
  cooling/cooling_module.f90
  cooling/cooling_module_ism.f90
  cooling/coolrates_module.f90
  cooling/collisional_ionization_module.f90
  cooling/recombination_module.f90
  cooling/cosmic_ray_ionization_module.f90
  cooling/photoionization_UVB_module.f90
  cooling/dust_recombination_module.f90
  cooling/molecules_module.f90
  cooling/charge_exchange_module.f90
  cooling/rtz_module.f90
  cooling/init_cooling.f90
  cooling/init_xion.f90
  cooling/init_neq_chem.f90
  cooling/cooling_fine.f90

  # Radiative transfer
  rt/rt_parameters.f90
  rt/rt_commons.f90
  rt/cross_sections_module.f90
  rt/rt_spectra.f90
  rt/rt_flux_module.f90
  rt/rt_godunov_utils.f90
  rt/rt_flag.f90
  rt/rt_units.f90
  rt/output_rt.f90
  rt/interpol_rt.f90
  rt/rt_init_flow_fine.f90
  rt/rt_upload.f90
  rt/rt_condinit.f90
  rt/rt_input_condinit.f90
  rt/read_rt_params.f90
  rt/rt_godunov_fine.f90
  rt/rt_step.f90
  rt/init_rt.f90

  # Turbulence driving
  turb/turb_commons.f90
  turb/turb_hydro.f90
  turb/turb_driving.f90
  turb/turb_init.f90
  turb/turb_update.f90

  # Clump finder
  clump/clfind_commons.f90
  clump/output_clump.f90
  clump/tree_formation.f90
  clump/clump_merger.f90
  clump/clump_finder.f90

  # Stars, sinks, feedback
  star/star_formation.f90
  sink/sink_formation.f90
  sink/sink_evolution.f90
  sink/sink_merger.f90
  feedback/feedback.f90
  feedback/rt_star_feedback.f90

  # Lightcone
  lightcone/lightcone_utils.f90
  lightcone/lightcone_buffer.f90
  lightcone/lightcone_io.f90
  lightcone/lightcone_test.f90
  lightcone/lightcone.f90

  # MDL (replaced by mdl2/ when MDL2=ON)
  mdl1/mdl.f90

  # Generated build-info stubs
  ${CMAKE_BINARY_DIR}/write_gitinfo.f90
  ${CMAKE_BINARY_DIR}/write_makefile.f90
  ${CMAKE_BINARY_DIR}/write_patch.f90
)

# RTZ-conditional cooling modules (neq vs rtz variants)
if(RTZ)
    list(APPEND fortran_files
        cooling/rtz_coolrates_module.f90
        cooling/rtz_cooling_module.f90
    )
else()
    list(APPEND fortran_files cooling/neq_cooling_module.f90)
endif()

# =============================================================================
# NVHPC / GPU sources
# =============================================================================
if(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")

    # Regular CUDA Fortran sources – compiled with nvfortran + -cuda -gpu=...
    # Order matches bin/Makefile MODOBJ: gpu_runner must precede gpu_part.
    set(gpu_cuf_files
        gpu/gpu_utils.cuf
        gpu/gpu_hydro.cuf
        gpu/gpu_cooling.cuf
        gpu/gpu_star.cuf
        gpu/gpu_rho.cuf
        gpu/gpu_mg.cuf

        gpu/gpu_refine.cuf
        gpu/gpu_flag.cuf
        gpu/gpu_runner.cuf
        gpu/gpu_part.cuf
    )
    list(APPEND fortran_files ${gpu_cuf_files} gpu/gpu_manager.f90)

    # -------------------------------------------------------------------------
    # CUB wrappers: .cu files compiled with nvcc, then .cuf wrappers compiled
    # by nvfortran receiving the .o as input (for GPU device-code merging).
    # This mirrors the hand-written rules in bin/Makefile.
    # -------------------------------------------------------------------------
    set(cub_c_radix ${CMAKE_BINARY_DIR}/cub_module_radix_sort_c.o)
    set(cub_c_scan  ${CMAKE_BINARY_DIR}/cub_inclusive_scan_c.o)
    set(cub_f_radix ${CMAKE_BINARY_DIR}/cub_module_radix_sort_f.o)
    set(cub_f_scan  ${CMAKE_BINARY_DIR}/cub_inclusive_scan_f.o)


    add_custom_command(
        OUTPUT  ${cub_c_radix}
        COMMAND nvcc -arch=${CUDA_ARCH} -O3 -lineinfo
                -c ${CMAKE_SOURCE_DIR}/gpu/cub_module_radix_sort.cu
                -o ${cub_c_radix}
        DEPENDS ${CMAKE_SOURCE_DIR}/gpu/cub_module_radix_sort.cu
                ${CMAKE_SOURCE_DIR}/gpu/cub_module_radix_sort.h
        COMMENT "Compiling cub_module_radix_sort.cu (nvcc)"
        VERBATIM
    )
    add_custom_command(
        OUTPUT  ${cub_c_scan}
        COMMAND nvcc -arch=${CUDA_ARCH} -O3 -lineinfo
                -c ${CMAKE_SOURCE_DIR}/gpu/cub_inclusive_scan.cu
                -o ${cub_c_scan}
        DEPENDS ${CMAKE_SOURCE_DIR}/gpu/cub_inclusive_scan.cu
                ${CMAKE_SOURCE_DIR}/gpu/cub_inclusive_scan.h
        COMMENT "Compiling cub_inclusive_scan.cu (nvcc)"
        VERBATIM
    )
    add_custom_command(
        OUTPUT  ${cub_f_radix}
        COMMAND nvfortran ${fort_flags} -c
                ${cub_c_radix}
                ${CMAKE_SOURCE_DIR}/gpu/cub_module_radix_sort.cuf
                -o ${cub_f_radix}
        DEPENDS ${cub_c_radix}
                ${CMAKE_SOURCE_DIR}/gpu/cub_module_radix_sort.cuf
        BYPRODUCTS ${CMAKE_BINARY_DIR}/cub_module.mod
        COMMENT "Compiling cub_module_radix_sort.cuf (nvfortran)"
        VERBATIM
    )
    add_custom_command(
        OUTPUT  ${cub_f_scan}
        COMMAND nvfortran ${fort_flags} -c
                ${cub_c_scan}
                ${CMAKE_SOURCE_DIR}/gpu/cub_inclusive_scan.cuf
                -o ${cub_f_scan}
        DEPENDS ${cub_c_scan}
                ${CMAKE_SOURCE_DIR}/gpu/cub_inclusive_scan.cuf
        BYPRODUCTS ${CMAKE_BINARY_DIR}/cub_inclusive_scan_mod.mod
        COMMENT "Compiling cub_inclusive_scan.cuf (nvfortran)"
        VERBATIM
    )

    # Mark as pre-built external objects so CMake passes them straight to the linker
    set(cub_objects ${cub_c_radix} ${cub_c_scan} ${cub_f_radix} ${cub_f_scan})
    set_source_files_properties(${cub_objects}
        PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_sources(${EXEC} PRIVATE ${cub_objects})

endif()

# =============================================================================
# MDL2 (optional, needs mdl2/ sub-tree present)
# =============================================================================
if(EXISTS ${CMAKE_SOURCE_DIR}/mdl2)
    set(MDL2 ON  CACHE BOOL "Enable the MDL2 version of RAMSES")
else()
    set(MDL2 OFF CACHE BOOL "Enable the MDL2 version of RAMSES")
endif()

if(MDL2)
    enable_language(CXX)
    set_target_properties(${EXEC} PROPERTIES
        CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
    add_subdirectory(mdl2)
    # Replace mdl1 with mdl2
    list(REMOVE_ITEM fortran_files mdl1/mdl.f90)
    list(APPEND fortran_files mdl2/mdl.f90 amr/mdlcache.f90)
    target_sources(${EXEC} PRIVATE amr/mdlcache.cxx)
    target_link_libraries(${EXEC} PRIVATE mdl2)
    target_compile_definitions(${EXEC} PRIVATE MDL2)
endif()

# =============================================================================
# NVHPC: enforce GPU module compilation order via explicit OBJECT_DEPENDS.
#
# CMake's Fortran dependency scanner does not process .cuf files, so it
# cannot infer that (e.g.) multigrid_fine_commons.f90 must wait for
# gpu_runner.cuf to be compiled first.  We fix this in two steps:
#
#  1. Chain the GPU .cuf files sequentially (each waits for the previous
#     object file), mirroring the MODOBJ order in bin/Makefile.
#
#  2. Scan every non-GPU source file at configure time for "USE gpu_*".
#     Any file that USEs a GPU module is made to depend on the last GPU
#     object in the chain, which transitively ensures all GPU modules exist.
# =============================================================================
if(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    set(_gpu_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${EXEC}.dir/gpu")

    # Step 1 – sequential chain across gpu_cuf_files
    set(_prev_gpu_obj "")
    foreach(_cuf IN LISTS gpu_cuf_files)
        get_filename_component(_cuf_name "${_cuf}" NAME)
        set(_cuf_obj "${_gpu_obj_dir}/${_cuf_name}.o")
        if(_prev_gpu_obj)
            set_source_files_properties("${_cuf}"
                PROPERTIES OBJECT_DEPENDS "${_prev_gpu_obj}")
        endif()
        set(_prev_gpu_obj "${_cuf_obj}")
    endforeach()
    # gpu_manager.f90 also waits for the tail of the .cuf chain
    set_source_files_properties(gpu/gpu_manager.f90
        PROPERTIES OBJECT_DEPENDS "${_prev_gpu_obj}")
    set(_last_gpu_obj "${_prev_gpu_obj}")

    # Step 2 – scan non-GPU sources for GPU module USE statements
    foreach(_src IN LISTS fortran_files)
        if(IS_ABSOLUTE "${_src}")
            continue()  # skip generated files (absolute paths)
        endif()
        if("${_src}" MATCHES "^gpu/")
            continue()  # skip GPU sources themselves
        endif()
        set(_full "${CMAKE_SOURCE_DIR}/${_src}")
        if(NOT EXISTS "${_full}")
            continue()
        endif()
        file(STRINGS "${_full}" _use_lines REGEX "[Uu][Ss][Ee][ \t]+gpu_")
        if(_use_lines)
            set_source_files_properties("${_src}"
                PROPERTIES OBJECT_DEPENDS "${_last_gpu_obj}")
        endif()
    endforeach()
endif()

# =============================================================================
# Attach all Fortran sources
# =============================================================================
target_sources(${EXEC} PRIVATE ${fortran_files})

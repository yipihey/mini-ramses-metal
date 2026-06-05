# Auto-generated Fortran module dependencies
# Regenerate: python3 gen_deps.py  (from repo root, output to bin/deps.mk)

adaptive_loop.o: amr_step.o clump_finder.o gpu_manager.o init_amr.o init_hydro.o init_part.o init_refine_adaptive.o init_refine_basegrid.o init_refine_ramses.o init_refine_restart.o init_rt.o init_time.o init_xion.o input_part.o load_balance.o mdl.o ramses_commons.o read_params.o turb_init.o update_time.o
amr_commons.o: amr_parameters.o domain_hilbert.o hash.o hydro_commons.o hydro_parameters.o oct_commons.o rt_commons.o rt_parameters.o
amr_step.o: clump_finder.o cooling_fine.o feedback.o flag_utils.o godunov_fine.o gpu_manager.o interpol_phi.o lightcone.o move_fine.o movie.o newdt_fine.o output_amr.o pm_parameters.o ramses_commons.o refine_utils.o rho_fine.o rt_godunov_fine.o rt_step.o sink_evolution.o sink_formation.o sink_merger.o source_hydro_fine.o star_formation.o synchro_hydro_fine.o tree_formation.o turb_driving.o turb_hydro.o update_time.o upload.o
boundana.o: amr_commons.o amr_parameters.o hydro_parameters.o
boundaries.o: amr_commons.o amr_parameters.o hydro_parameters.o ramses_commons.o rt_parameters.o
cache.o: amr_commons.o amr_parameters.o cache_commons.o clfind_commons.o hash.o mdl.o
cache_commons.o: amr_parameters.o call_back.o hydro_parameters.o rt_parameters.o
call_back.o: amr_commons.o amr_parameters.o clfind_commons.o ramses_commons.o
clfind_commons.o: hash.o
clump_finder.o: amr_parameters.o boundaries.o cache.o cache_commons.o clump_merger.o marshal.o mdl.o mdl_commons.o multigrid_fine_coarse.o nbors_utils.o output_clump.o ramses_commons.o
clump_merger.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o clfind_commons.o hash.o hilbert.o marshal.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o ramses_commons.o
condinit.o: amr_commons.o amr_parameters.o hydro_parameters.o input_hydro_condinit.o
cooling_fine.o: amr_commons.o amr_parameters.o constants.o cooling_module.o coolrates_module.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o rt_parameters.o
cooling_module.o: constants.o
cooling_module_ism.o: constants.o
coolrates_module.o: amr_commons.o amr_parameters.o constants.o hydro_parameters.o rt_parameters.o
courant_fine.o: amr_commons.o amr_parameters.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o
cross_sections_module.o: constants.o
domain_hilbert.o: amr_parameters.o hilbert.o
feedback.o: amr_parameters.o boundaries.o cache.o cache_commons.o godunov_fine.o hilbert.o hydro_parameters.o marshal.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o ramses_commons.o
flag_utils.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o hilbert.o hydro_flag.o marshal.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o ramses_commons.o rt_flag.o smooth.o
force_fine.o: multigrid_fine_coarse.o
godunov_fine.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o hydro_commons.o hydro_parameters.o marshal.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o
godunov_utils.o: amr_commons.o amr_parameters.o hydro_parameters.o
gpu_manager.o: mdl.o mdl_commons.o ramses_commons.o
grav_ana.o: amr_commons.o amr_parameters.o
hash.o: amr_parameters.o
hilbert.o: amr_parameters.o
hydro_commons.o: amr_parameters.o hydro_parameters.o rt_parameters.o
hydro_flag.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o hydro_parameters.o nbors_utils.o ramses_commons.o
hydro_parameters.o: amr_parameters.o
init_amr.o: amr_commons.o amr_parameters.o hash.o hilbert.o hydro_parameters.o mdl.o mdl_commons.o output_amr.o ramses_commons.o rt_parameters.o
init_cooling.o: amr_commons.o cooling_module.o
init_flow_fine.o: input_hydro_condinit.o input_hydro_gadget.o input_hydro_grafic.o ramses_commons.o
init_hydro.o: amr_commons.o mdl.o mdl_commons.o ramses_commons.o
init_neq_chem.o: amr_commons.o coolrates_module.o
init_part.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o pm_commons.o pm_parameters.o ramses_commons.o
init_refine_adaptive.o: flag_utils.o init_part.o input_hydro_grafic.o input_part.o input_part_zoom.o ramses_commons.o refine_utils.o rt_upload.o upload.o
init_refine_basegrid.o: amr_parameters.o flag_utils.o hash.o hilbert.o input_hydro_grafic.o mdl.o mdl_commons.o ramses_commons.o
init_refine_ramses.o: amr_parameters.o hash.o hilbert.o hydro_parameters.o init_refine_basegrid.o input_hydro_condinit.o load_balance.o mdl.o mdl_commons.o output_amr.o ramses_commons.o read_params.o
init_refine_restart.o: amr_parameters.o hash.o hilbert.o hydro_parameters.o init_refine_basegrid.o load_balance.o mdl.o mdl_commons.o output_amr.o ramses_commons.o read_params.o rt_parameters.o
init_rt.o: amr_commons.o mdl.o mdl_commons.o ramses_commons.o
init_time.o: amr_commons.o amr_parameters.o cooling_module.o coolrates_module.o gadgetreadfile.o init_cooling.o init_neq_chem.o mdl.o mdl_commons.o ramses_commons.o rt_spectra.o
init_xion.o: amr_commons.o amr_parameters.o constants.o cooling_module.o coolrates_module.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o upload.o
input_hydro_condinit.o: amr_commons.o amr_parameters.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o
input_hydro_gadget.o: amr_parameters.o boundaries.o cache.o cache_commons.o godunov_fine.o hydro_parameters.o input_hydro_condinit.o marshal.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o
input_hydro_grafic.o: amr_commons.o amr_parameters.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o
input_part.o: input_part_ascii.o input_part_gadget.o input_part_grafic.o input_part_ramses.o input_part_restart.o mdl.o mdl_commons.o ramses_commons.o rt_spectra.o
input_part_ascii.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o
input_part_gadget.o: amr_commons.o amr_parameters.o gadgetreadfile.o init_part.o mdl.o mdl_commons.o output_amr.o pm_commons.o ramses_commons.o
input_part_grafic.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o
input_part_ramses.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o output_amr.o pm_commons.o pm_parameters.o ramses_commons.o
input_part_restart.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o output_amr.o pm_commons.o pm_parameters.o ramses_commons.o
input_part_zoom.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o
interpol_hydro.o: amr_parameters.o hydro_parameters.o
interpol_phi.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o ramses_commons.o
interpol_rt.o: amr_parameters.o rt_parameters.o
lightcone.o: amr_parameters.o lightcone_buffer.o lightcone_io.o lightcone_utils.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o
lightcone_buffer.o: amr_parameters.o
lightcone_io.o: amr_parameters.o lightcone_buffer.o ramses_commons.o
lightcone_test.o: lightcone_io.o lightcone_utils.o ramses_commons.o
lightcone_utils.o: amr_commons.o
load_balance.o: amr_commons.o amr_parameters.o cache.o cache_commons.o domain_hilbert.o hash.o hilbert.o hydro_parameters.o init_refine_basegrid.o marshal.o mdl.o mdl_commons.o nbors_utils.o oct_commons.o pm_commons.o pm_parameters.o ramses_commons.o rho_fine.o rt_parameters.o
marshal.o: amr_commons.o amr_parameters.o cache_commons.o hydro_parameters.o rt_parameters.o
mdl.o: amr_commons.o amr_parameters.o clfind_commons.o mdl_commons.o
move_fine.o: amr_commons.o amr_parameters.o cache.o cache_commons.o hydro_parameters.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o pm_parameters.o ramses_commons.o rho_fine.o rngstream.o rt_parameters.o
movie.o: amr_commons.o amr_parameters.o hilbert.o hydro_parameters.o mdl.o mdl_commons.o output_amr.o pm_commons.o ramses_commons.o rt_parameters.o
nbors_utils.o: amr_commons.o amr_parameters.o boundaries.o cache_commons.o hash.o hilbert.o hydro_parameters.o mdl.o ramses_commons.o
neq_cooling_module.o: amr_commons.o amr_parameters.o constants.o coolrates_module.o hydro_parameters.o rt_parameters.o
newdt_fine.o: amr_commons.o amr_parameters.o constants.o courant_fine.o mdl.o mdl_commons.o pm_commons.o pm_parameters.o ramses_commons.o
oct_commons.o: amr_parameters.o
output_amr.o: amr_commons.o amr_parameters.o cooling_module.o gadgetreadfile.o hydro_commons.o mdl.o mdl_commons.o output_hydro.o output_part.o output_poisson.o output_rt.o pm_commons.o ramses_commons.o turb_commons.o
output_clump.o: amr_commons.o amr_parameters.o clfind_commons.o clump_merger.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o
output_hydro.o: amr_commons.o amr_parameters.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o
output_part.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o
output_poisson.o: amr_commons.o amr_parameters.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o
output_rt.o: amr_commons.o amr_parameters.o constants.o hydro_parameters.o mdl.o mdl_commons.o ramses_commons.o rt_parameters.o
phi_fine_cg.o: multigrid_fine_coarse.o
pm_commons.o: amr_parameters.o
poisson_flag.o: amr_commons.o amr_parameters.o hash.o hydro_parameters.o ramses_commons.o
ramses_commons.o: amr_commons.o amr_parameters.o clfind_commons.o cooling_module.o coolrates_module.o hydro_parameters.o mdl.o pm_commons.o rt_parameters.o rt_spectra.o turb_commons.o
read_params.o: amr_commons.o amr_parameters.o constants.o hydro_parameters.o mdl.o mdl_commons.o movie.o ramses_commons.o read_rt_params.o rt_parameters.o
read_rt_params.o: amr_parameters.o constants.o hydro_parameters.o mdl.o movie.o ramses_commons.o rt_parameters.o
refine_utils.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o call_back.o hash.o hilbert.o hydro_parameters.o init_refine_basegrid.o load_balance.o marshal.o mdl.o mdl_commons.o nbors_utils.o oct_commons.o ramses_commons.o rt_parameters.o
rho_ana.o: amr_parameters.o
rho_fine.o: amr_commons.o amr_parameters.o cache.o cache_commons.o hilbert.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o pm_parameters.o ramses_commons.o
rt_commons.o: amr_parameters.o oct_commons.o rt_parameters.o
rt_condinit.o: amr_commons.o amr_parameters.o rt_input_condinit.o rt_parameters.o
rt_flag.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o nbors_utils.o ramses_commons.o rt_parameters.o
rt_flux_module.o: amr_parameters.o rt_parameters.o
rt_godunov_fine.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o hash.o marshal.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o rt_commons.o rt_flux_module.o rt_parameters.o
rt_godunov_utils.o: amr_commons.o amr_parameters.o rt_parameters.o
rt_init_flow_fine.o: amr_commons.o amr_parameters.o coolrates_module.o mdl.o mdl_commons.o output_rt.o ramses_commons.o rt_input_condinit.o rt_parameters.o rt_spectra.o
rt_input_condinit.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o ramses_commons.o rt_parameters.o
rt_parameters.o: amr_parameters.o
rt_spectra.o: amr_commons.o constants.o cross_sections_module.o hydro_parameters.o pm_commons.o rt_parameters.o
rt_star_feedback.o: amr_commons.o amr_parameters.o cache.o cache_commons.o constants.o hilbert.o hydro_parameters.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o ramses_commons.o rt_parameters.o rt_spectra.o
rt_step.o: amr_parameters.o cooling_fine.o mdl.o newdt_fine.o ramses_commons.o rt_godunov_fine.o rt_input_condinit.o rt_star_feedback.o rt_upload.o update_time.o
rt_units.o: amr_commons.o amr_parameters.o
rt_upload.o: amr_commons.o amr_parameters.o cache.o cache_commons.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o rt_flag.o rt_parameters.o
rtz_cooling_module.o: amr_commons.o amr_parameters.o charge_exchange_module.o collisional_ionization_module.o constants.o coolrates_module.o cosmic_ray_ionization_module.o dust_recombination_module.o hydro_parameters.o molecules_module.o photoionization_UVB_module.o recombination_module.o rt_parameters.o rtz_coolrates_module.o rtz_module.o
rtz_coolrates_module.o: amr_commons.o charge_exchange_module.o coolrates_module.o cosmic_ray_ionization_module.o molecules_module.o photoionization_UVB_module.o rt_parameters.o rtz_module.o
sink_evolution.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o constants.o godunov_fine.o hilbert.o hydro_parameters.o marshal.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o ramses_commons.o read_params.o rho_fine.o
sink_formation.o: amr_commons.o amr_parameters.o cache.o cache_commons.o clfind_commons.o clump_finder.o clump_merger.o constants.o hydro_parameters.o mdl.o mdl_commons.o output_amr.o output_clump.o output_part.o output_poisson.o pm_commons.o ramses_commons.o rngstream.o
sink_merger.o: amr_commons.o amr_parameters.o cache.o cache_commons.o constants.o flag_utils.o hilbert.o marshal.o mdl.o mdl_commons.o nbors_utils.o pm_commons.o pm_parameters.o ramses_commons.o sink_evolution.o
smooth.o: amr_parameters.o boundaries.o cache.o cache_commons.o marshal.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o
sort.o: amr_parameters.o
source_hydro_fine.o: amr_parameters.o boundaries.o cache.o cache_commons.o hydro_flag.o hydro_parameters.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o
star_formation.o: amr_commons.o amr_parameters.o constants.o hydro_parameters.o input_hydro_condinit.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o rngstream.o rt_spectra.o
synchro_hydro_fine.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o ramses_commons.o
task_manager.o: amr_commons.o amr_parameters.o cache.o cache_commons.o call_back.o clfind_commons.o clump_finder.o clump_merger.o cooling_fine.o courant_fine.o feedback.o flag_utils.o godunov_fine.o gpu_manager.o hash.o hilbert.o init_amr.o init_hydro.o init_part.o init_refine_basegrid.o init_refine_ramses.o init_refine_restart.o init_rt.o init_time.o init_xion.o input_hydro_condinit.o input_hydro_gadget.o input_hydro_grafic.o input_part.o input_part_ascii.o input_part_gadget.o input_part_grafic.o input_part_ramses.o input_part_restart.o input_part_zoom.o interpol_phi.o lightcone.o load_balance.o mdl.o mdl_commons.o move_fine.o movie.o newdt_fine.o output_amr.o output_clump.o output_hydro.o output_part.o output_poisson.o output_rt.o ramses_commons.o read_params.o refine_utils.o rho_fine.o rt_godunov_fine.o rt_init_flow_fine.o rt_input_condinit.o rt_star_feedback.o rt_upload.o sink_evolution.o sink_formation.o sink_merger.o smooth.o source_hydro_fine.o star_formation.o synchro_hydro_fine.o tree_formation.o turb_driving.o turb_hydro.o turb_init.o turb_update.o update_time.o upload.o
test_cooling.o: amr_commons.o amr_parameters.o constants.o cooling_module.o coolrates_module.o hydro_parameters.o init_neq_chem.o neq_cooling_module.o rt_parameters.o
timer.o: amr_parameters.o mdl.o
tree_formation.o: amr_commons.o amr_parameters.o cache.o cache_commons.o clfind_commons.o clump_finder.o clump_merger.o constants.o hydro_parameters.o mdl.o mdl_commons.o pm_commons.o ramses_commons.o rngstream.o
turb_commons.o: amr_commons.o amr_parameters.o
turb_driving.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o ramses_commons.o turb_commons.o
turb_hydro.o: amr_commons.o amr_parameters.o mdl.o mdl_commons.o ramses_commons.o
turb_init.o: amr_commons.o mdl.o mdl_commons.o ramses_commons.o turb_commons.o
turb_update.o: amr_commons.o mdl.o mdl_commons.o ramses_commons.o turb_commons.o
umuscl.o: amr_commons.o amr_parameters.o hydro_parameters.o
units.o: amr_commons.o amr_parameters.o constants.o
update_time.o: amr_parameters.o hash.o mdl.o mdl_commons.o ramses_commons.o rt_init_flow_fine.o turb_update.o
upload.o: amr_commons.o amr_parameters.o cache.o cache_commons.o hydro_parameters.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o
vecpotentialinit.o: amr_commons.o amr_parameters.o
write_gitinfo.o: amr_parameters.o
write_screen.o: amr_commons.o amr_parameters.o hydro_parameters.o

ifeq ($(COMPILER),NVHPC)
cooling_fine.o: gpu_runner.o
courant_fine.o: gpu_runner.o
flag_utils.o: gpu_runner.o gpu_utils.o
force_fine.o: gpu_runner.o
godunov_fine.o: gpu_runner.o
gpu_cooling.o: amr_parameters.o hydro_parameters.o oct_commons.o
gpu_flag.o: amr_parameters.o gpu_utils.o hydro_parameters.o oct_commons.o
gpu_hilbert.o: amr_parameters.o hilbert.o
gpu_hydro.o: amr_parameters.o gpu_utils.o hydro_parameters.o oct_commons.o
gpu_manager.o: gpu_refine.o gpu_runner.o gpu_utils.o
gpu_mg.o: amr_parameters.o gpu_utils.o hydro_parameters.o oct_commons.o
gpu_mpi.o: amr_parameters.o cache.o cache_commons.o gpu_hydro.o gpu_utils.o hash.o hydro_commons.o hydro_parameters.o marshal.o mdl.o nbors_utils.o oct_commons.o ramses_commons.o
gpu_nbor.o: amr_parameters.o gpu_utils.o oct_commons.o
gpu_part.o: amr_parameters.o cub_module_radix_sort_f.o gpu_refine.o gpu_runner.o gpu_utils.o oct_commons.o ramses_commons.o
gpu_refine.o: amr_parameters.o gpu_utils.o hydro_parameters.o oct_commons.o
gpu_rho.o: amr_parameters.o gpu_utils.o hydro_parameters.o oct_commons.o
gpu_runner.o: amr_parameters.o cache.o cache_commons.o cub_inclusive_scan_f.o gpu_cooling.o gpu_flag.o gpu_hydro.o gpu_mg.o gpu_refine.o gpu_rho.o gpu_utils.o hydro_parameters.o oct_commons.o ramses_commons.o
gpu_scan.o: amr_parameters.o
gpu_utils.o: amr_parameters.o oct_commons.o
init_amr.o: gpu_runner.o gpu_utils.o
init_part.o: gpu_part.o gpu_runner.o
interpol_phi.o: gpu_runner.o
move_fine.o: gpu_part.o
multigrid_fine_coarse.o: gpu_runner.o
multigrid_fine_commons.o: gpu_runner.o
newdt_fine.o: gpu_part.o
phi_fine_cg.o: gpu_runner.o
refine_utils.o: gpu_runner.o
rho_fine.o: gpu_part.o gpu_runner.o
smooth.o: gpu_runner.o
synchro_hydro_fine.o: gpu_runner.o
upload.o: gpu_runner.o
endif

ifeq ($(GRAV),1)
amr_step.o: force_fine.o multigrid_fine_commons.o phi_fine_cg.o
clump_finder.o: rho_fine.o
force_fine.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o interpol_phi.o mdl.o mdl_commons.o nbors_utils.o phi_fine_cg.o ramses_commons.o
init_refine_adaptive.o: rho_fine.o
init_refine_basegrid.o: rho_fine.o
init_refine_ramses.o: rho_fine.o
init_refine_restart.o: rho_fine.o
multigrid_fine_coarse.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o hash.o hilbert.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o
multigrid_fine_commons.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o hash.o hilbert.o interpol_phi.o mdl.o mdl_commons.o multigrid_fine_coarse.o nbors_utils.o phi_fine_cg.o poisson_parameters.o ramses_commons.o
phi_fine_cg.o: amr_commons.o amr_parameters.o boundaries.o cache.o cache_commons.o call_back.o interpol_phi.o mdl.o mdl_commons.o nbors_utils.o ramses_commons.o
rho_fine.o: amr_commons.o amr_parameters.o cache.o cache_commons.o hilbert.o mdl.o mdl_commons.o multigrid_fine_coarse.o nbors_utils.o pm_commons.o pm_parameters.o ramses_commons.o
sink_formation.o: rho_fine.o
task_manager.o: force_fine.o multigrid_fine_coarse.o multigrid_fine_commons.o phi_fine_cg.o rho_fine.o
tree_formation.o: rho_fine.o
endif

ifeq ($(MHD),1)
godunov_utils.o: amr_commons.o amr_parameters.o hydro_parameters.o
umuscl.o: amr_commons.o amr_parameters.o hydro_parameters.o
endif

ifeq ($(RTZ),0)
cooling_fine.o: neq_cooling_module.o
init_neq_chem.o: neq_cooling_module.o
init_xion.o: amr_commons.o amr_parameters.o hydro_parameters.o neq_cooling_module.o ramses_commons.o rt_parameters.o
refine_utils.o: init_xion.o
rt_init_flow_fine.o: neq_cooling_module.o
endif

ifeq ($(RTZ),1)
cooling_fine.o: rtz_cooling_module.o rtz_module.o
init_neq_chem.o: charge_exchange_module.o cosmic_ray_ionization_module.o photoionization_UVB_module.o rtz_cooling_module.o rtz_coolrates_module.o
init_xion.o: rtz_module.o
input_hydro_condinit.o: rtz_module.o
output_rt.o: rtz_module.o
read_params.o: rtz_module.o
read_rt_params.o: cross_sections_module.o rt_spectra.o
rt_init_flow_fine.o: rtz_cooling_module.o
rt_spectra.o: constants.o rt_parameters.o rtz_module.o
task_manager.o: rtz_module.o
endif


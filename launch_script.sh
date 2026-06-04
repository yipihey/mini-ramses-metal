## Recompile
cd bin_sedov3d_fast_math && make NDIM=3 COMPILER=NVHPC HYDRO=1 NPRE=4 && cd ..

## Launch without profiling
# exec=./bin_sedov3d_fast_math/ramses3d 
# ${exec} namelist/sedov3d_amr.nml 

# ## Launch with Nsight System.
# # nsys profile --trace=cuda,nvtx --stats=true --force-overwrite true -o nsys/nsys_sedov_3d_amr ${exec} namelist/sedov3d_amr.nml 
# # nsys profile --trace=cuda,nvtx --stats=true --force-overwrite true -o nsys/nsys_sedov_3d_amr_large ${exec} namelist/sedov3d_amr.nml 
# # nsys profile --trace=cuda,nvtx --stats=true --force-overwrite true -o nsys/nsys_sedov_3d_amr_large_lmin8_lmax10_nstepmax50 ${exec} namelist/sedov3d_amr.nml 

## Launch with Nsight Compute.
# sudo nvidia-smi -lgc `nvidia-smi -q -d SUPPORTED_CLOCKS | grep Graphics | head -1 | cut -d: -f2 | cut -d' ' -f2` --mode 1

# for kernel in hydro_integrator_kernel
# do
#   set -x
#   ncu --clock-control none --cache-control none \
#     --set full --import-source on \
#     --kernel-name regex:${kernel} --launch-count 16 --launch-skip 100 \
#     -f -o ./ncu/ncu_sedov_3d_${kernel}_large_lmin8_lmax10_fp32_fast_math_blockDim64 \
#     ${exec} ./namelist/sedov3d_amr.nml
#   set +x
# done


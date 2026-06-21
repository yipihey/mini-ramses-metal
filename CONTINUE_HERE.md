# CONTINUE_HERE — GLM-MHD robustness at large Mach numbers

Working handoff note (branch `cuda-dedner-mhd`). Next task: **make the GPU GLM-MHD
solvers robust in strongly supersonic flow.** They currently blow up (positivity
failures) at high Mach. Delete this file when the work lands.

## The problem (observed in the prior session)
256³, isothermal, β=2 driven turbulence (vary `turb_rms` to set Mach):
- **PLM** (`slope_type=2`): stable to ~Mach 3.7; NaN by ~Mach 5.6.
- **parabolic PPM** (`slope_type=11`): stable to ~Mach 2.9; dies ~Mach 3.7.
- **characteristic PPM** (`slope_type=10`): very fragile — blew up even at **Mach 1.2**.
  Effectively unusable supersonic; the 7-wave Roe-Balsara projection produces unphysical
  states the fallback doesn't catch.
- Mechanism: **positivity failures** (negative ρ/p) in strong rarefactions / low-density
  voids — NOT a CFL/dt problem (smaller dt still NaNs). **Higher resolution = less stable**
  (more resolved extrema).
- The forcing **ramp** (linear 0→full over the first crossing time, in `gpu_turb_interp`,
  `t_ramp=0.3`) fixed the cold-start transient but not the steady high-Mach blow-up.
- The **c_h fix** (`glm_ch_scale=0.25`, just landed) reduced the over-driven cleaning and
  helped div·B + a little stability, but did not solve the high-Mach blow-up.

## Most promising lead (untried)
**Per-cell LLF fallback for unsafe states.** The upstream CT solver switches its Riemann
flux to LLF where ρ or p drop below thresholds (`switch_llf_dmin`, `switch_llf_pmin`, used
in `riemann2d_hlld_emf` in `../mini-ramses-upstream/gpu/gpu_hydro.cuf`). **Our GLM HLLD has
no such hybrid switch** — it always runs the full 5-state HLLD, whose intermediate states
go negative in strong rarefactions. Those two namelist params already exist in
`hydro_params` (read into `run_t`) but are unused by the GLM path. Adding "use the LLF flux
(`llf_mhd_fluxes`, the robust Rusanov path already in the file) when
`min(ρ_L,ρ_R) < switch_llf_dmin` or `min(p_L,p_R) < switch_llf_pmin`, else HLLD" is the
natural first robustification. Also consider: detect non-positive HLLD star states and fall
back to HLL/LLF for just that face.

Other directions: (2) make char-PPM (`slope_type=10`) fall back to parabolic/PLM near
strong shocks or limit the characteristic amplitudes; (3) positivity-preserving floors
consistent with the isothermal EOS.

## Where the code is (`gpu/gpu_hydro.cuf` unless noted)
- Riemann solvers: `hlld_mhd_fluxes` (~485), `hll_mhd_fluxes` (~462), `llf_mhd_fluxes`
  (~441, Rusanov — most robust), dispatch `mhd_riemann_fluxes` (~2244). Default
  `riemann='llf'`; the turbulence runs use `hlld`.
- Positivity fallbacks (reduce to 1st order): half-step reversion `mh=m0` (~2670/2736/2792);
  face clipping in `store_mhd_x/y/z` (~2405); char clip reverting all 7 vars (~2596).
- `strong_pressure_jump` (~766, threshold 2.0) → PPM→PLM (fires in only ~0.5% of cells, so
  it is NOT the robustness lever).
- char eigensystem: `mhd_char_trace_x` (~2470), `mhd_char_faces`.
- Dispatch by `slope_type` + `c_h` computation: `gpu/gpu_runner.cuf` (~358, ~203).
- Params: `amr/read_params.f90` + `amr/amr_commons.f90` (`run_t`) — `smallr`, `smallc`,
  `switch_llf_dmin`, `switch_llf_pmin`, `glm_ch_scale` (0.25), `glm_cp_coef` (0.18).

## Build + test harness
```
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH   # nvfortran/ncu
cd bin_glm && make COMPILER=NVHPC GLM=1 HYDRO=1 GRAV=1 TURB=1 NDIM=3 NPRE=4 \
    CUDA_ARCH=sm_86 FFTW=/usr LIBFFTW="-L/usr/lib64 -lfftw3" ramses
```
- **High-Mach turbulence test:** copy a turb namelist (β=2, ramp; the prior one lived at
  `/tmp/cmp/glm_plm/turb.nml` — `/tmp` is ephemeral, may need regenerating). Bump
  `turb_rms` until each `slope_type` NaNs; record the failure Mach per scheme. Compare to
  upstream CT (`~/Projects/mini-ramses-upstream/bin`, built `MHD=1`) at the same forcing.
- **Add dedicated high-Mach tests:** a Mach-10+ MHD shock tube and/or colliding flows,
  beyond the existing OT / Brio-Wu / CP-Alfvén.
- **Validation after any change:** `INIT=OT` build + `runs/mhd_cuda/validate_mhd.py`
  (div·B + positivity); re-check Brio-Wu and the Mach-3 turbulence spectra didn't regress
  (the comparison report: `docs/glm_vs_ct_turbulence/`).
- Output reader / 256³ oct→cell map were in `/tmp` (`read_mr.py`, `octmap.npy`) — ephemeral;
  regenerate from the C-API (`ramses_get_hydro` returns cell ijk) if gone.

## Recently landed on this branch (for context)
c_h fix `73df102a` (tunable `glm_ch_scale`/`glm_cp_coef`); nsubgrid=2 / launch_bounds /
fastmath / char lazy-predictor (perf); `A_ave/B_ave/C_ave` uniform-B seed; forcing ramp;
turbulence comparison report (`docs/glm_vs_ct_turbulence/`, live on GitHub Pages).

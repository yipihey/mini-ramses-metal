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

## LANDED 2026-06-20 (uncommitted on this branch) — two changes

**(1) HLLD was never actually selected in GLM builds — fixed.** The riemann
string→int mapping in `amr/read_params.f90` (~1410) was guarded by `#ifdef MHD`,
but GLM uses `-DGLMMHD` (a *different* macro). So `riemann='hlld'` matched
nothing → `s%r%riemann` stayed at its run_t default **0** → the `else`/LLF branch
of `mhd_riemann_fluxes`. **Every prior GLM "hlld" run (incl. the turbulence
sweep + `docs/glm_vs_ct_turbulence/` report + the `runs/mhd_cuda` README table)
was actually running LLF.** Fix: guard is now `#if defined(MHD) || defined(GLMMHD)`.
Real HLLD is now active and is sharper/less diffusive: OT (5/3, 128³, t=0.5) real
HLLD = max|divB|≈5.9, ρ_min≈0.089 vs LLF 2.30 / 0.098. **Likely more `#ifdef MHD`
vs GLMMHD gaps elsewhere — grep for them.**

**(2) Per-cell LLF fallback for unsafe states — implemented + verified.**
`mhd_riemann_fluxes` now swaps HLLD/HLL→`llf_mhd_fluxes` per-face when
`min(ρ_L,ρ_R) < switch_llf_dmin` or `min(p_L,p_R) < switch_llf_pmin`
(both namelist `&HYDRO_PARAMS`, default −1 = OFF, so HD/existing tests unchanged).
**Threaded as kernel value args** through `mhd_integrator_kernel{,_ppm,_par}` →
`riemann_driver_mhd` → `mhd_riemann_fluxes` (NOT a module device var — host→device
module-scalar assignment silently no-ops in this nvfortran; the fork threads every
solver param as a value arg, e.g. `ch`/`riemann`). Verified on OT: forcing it on
(`switch_llf_dmin=10`) reproduces the pure-LLF field bit-for-bit; OFF is
bit-identical to HLLD. Build/test = `bin_ot/` (`INIT=OT libramses`) + `runs/mhd_cuda/validate_mhd.py`.

## LANDED 2026-06-21 (uncommitted) — GPU init pipeline: 700³ init 342 s → 8.6 s (~40×)

Large GPU turbulence runs were paying a huge serial CPU init. Now grid + ICs are built on
the device. Stages (each validated: 48³ mass-conservation + step0 bit-identical to the CPU
IC, OT non-turb regression PASS, 700³ smoke PASS):
- **IC mode-hoist** (`hydro/condinit.f90`): `INIT==TURB` recomputed ~16 forcing-mode
  constants (~5 `sin` each) *inside* the per-cell loop → hoisted out. init_flow_fine
  309 s→34 s. Bit-identical.
- **GPU IC** (`gpu/gpu_turb.cuf` `turb_ic_kernel`, `gpu/gpu_runner.cuf` `gpu_init_flow_turb`,
  called from `r_set_grid_device`; CPU `condinit` skipped in `hydro/init_flow_fine.f90` for
  `_CUDA+GLMMHD+TURB+r%turb`): synthesise the solenoidal IC + write conserved `uold` on the
  device. 66 s→33.8 s. Enumerate octs like the hash insert (`head_idx=1, num_octs=ifree-1`,
  skip `lev<=0`).
- **init_amr alloc** (`amr/init_amr.f90`): for `_CUDA` skip zeroing the host hydro arrays
  that are never uploaded (`m%unew`+GRAV always; `m%uold` for `r%turb` since the device IC
  fills the zeroed device `uold`); skip the `m%uold` H2D upload for turb (`gpu_manager.f90`).
  host-zero 11.2 s→0; 33.8 s→21.7 s.
- **GPU base-grid build** (`gpu/gpu_refine.cuf` `build_basegrid_kernel`, `gpu/gpu_runner.cuf`
  `gpu_build_basegrid`, gated in `amr/init_refine_basegrid.f90` `gpu_bg = turb & nlevelmax==
  levelmin & .not.poisson & even-box`; `rho_fine`/`flag_fine` skipped; grid H2D upload
  skipped via `m%grid_on_device`): replaces the serial 134M-key CPU Hilbert walk. **Octs
  MUST be in 2×2×2-block (Morton-8) order** — each `nsubgridtondim=8` consecutive octs is a
  spatial block for the subgrid gather (Cartesian order broke conservation: rel_dmass 2e-4
  → block-order 8e-11). 21.7 s→8.6 s.

**Gotchas / limits:** the GPU build leaves **host `m%grid` unpopulated** (device is source of
truth) → output/restart needs a D2H grid copy (not yet added; smoke tests don't output).
Block-order requires **even** box dims (else CPU fallback). Remaining 8.6 s init ≈ ~7 s
`init_amr` (the host `m%grid` alloc + `lev=0` loop, still wasted for the GPU-build case) +
~1 s device. The CUDA-700 "blocker" hunted early was a **stale incremental-build artifact**
— always `make clean` after touching CUDA-Fortran modules.

### fp32 turbulence forcing (uncommitted, 2026-06-21) — `turb_drive_apply` 3.8× faster
The fused forcing kernel `turb_drive_apply_kernel` (`gpu/gpu_turb.cuf`) ran its trilinear
afield interp AND its source-term math (KE removal / momentum kick / energy recompute) in
**fp64** — catastrophic on the A6000 (FP64 is 1:64). Converted to fp32:
- `afield_*_d` split: `afield_last_d/next_d` stay fp64 (host-uploaded), `afield_now_d` → fp32
  (`gpu/gpu_runner.cuf`; `turb_interp_field_kernel` blends fp64→stores fp32; `init_amr.f90`
  alloc/zero converts cleanly).
- trilinear weights `w1/w2` → fp32; source-term `rho/ener` → `dp` (=fp32, was needlessly
  `kind=8`); `0.5_dp` literals + fp32 `dteff_s`. Position/`floor` (`x, rr`) kept fp64.
- **Measured (nsys 128³, integrator kernel as invariant 1.598 ms/step ruler):**
  `turb_drive_apply` 1.84 → 1.25 (weights) → **0.479 ms/step** (source term) = −74%. The
  fp64 source term, not the interp, was the bottleneck. The godunov integrator is now cleanly
  the #1 kernel (38%); forcing is 11.5%.
- **Validated:** 48³ mass-conservation rel_dmass 2.5e-10 PASS; momentum trajectory
  bit-identical to the fp32-weights-only run (fp32 math did not perturb physics); 128³ smoke
  PASS (ρ finite, ρ_min 0.974).

### fp32 cmpdt + glm fusion + the buffer-swap wall (uncommitted, 2026-06-21)
Continued the 512³ step-time attack (10-step `trunc512.nml`, level 9, 134.2M cells; nsys, the
unchanged integrator kernel = 102.6 ms/step is the invariant ruler):
- **cmpdt fp32** (`gpu/gpu_hydro.cuf` `cmpdt_kernel_mhd`): the dt-path scalars `ctot/grav/dt_s`
  were `kind=8` → the per-cell fast-speed sum + `sqrt`/divide ran fp64 (1:64). Made them `dp`
  (conservation accumulators `mass/ekin/eint/emag` stay fp64); cast back to fp64 `dt_loc` only
  for the reduction. **cmpdt 46.7 → 24.4 ms/step (−48%)**; conservation/dt unchanged.
- **glm_damp fused into set_uold** (`set_uold_kernel` takes `glm_fac`; `gpu_set_uold` computes
  `fac=exp(-(ch²/cp2)·dt)` on the host and the separate `glm_damp_psi_kernel` launch is gone):
  the damping is one multiply on a uniform per-step scalar, so a 16.8 ms full-grid pass was pure
  waste. **glm 16.8 → ~0**; `set_uold` grew 22.46→22.62 ms; conservation byte-identical.
- **Net 512³ step: 240 → 201.6 ms (godunov 42.8% → 50.7%, full-step 558 → 666 Mcell/s).** The
  integrator alone is 1314 Mcell/s.

**Buffer-swap WALL (blocks >90% godunov-dominated step):** the remaining non-godunov cost is
mandatory bookkeeping — `cmpdt` (dt), `set_unew`/`set_uold` (the uold↔unew double-buffer copies,
~45 ms together), `turb_drive`. The double-buffer exists because the integrator reads `uold`
neighbours while writing `unew` (and AMR does `atomicadd(unew(...,father))` reflux — never taken
single-level). To make the step godunov-dominated the copy-back must become a pointer SWAP, but
**both swap mechanisms fail in this nvfortran/CUDA-Fortran setup**: `move_alloc(uold,…)` on
`device, allocatable` → CUDA 700 next step; `device, pointer` uold/unew with target buffers →
716 misaligned at init. Root cause: `uold/unew` are module globals consumed by ~30 kernels via
assumed-size dummies; nvfortran doesn't propagate a swapped descriptor/pointer through that. A
real >90% step needs the buffers threaded as explicit, parity-swapped *arguments* through every
kernel + the `ramses_get_hydro`/output "current buffer" contract — a large, invasive refactor.
Both swap experiments were reverted; the cmpdt-fp32 + glm-fusion wins are kept and validated.

### uold/unew device-pointer double-buffer — set_uold copy → free swap (uncommitted, 2026-06-21)
The buffer-swap WALL (above) is BROKEN. Root cause of the prior failures: pointer/descriptor
ops in a `.f90` unit. Fix: keep all swap machinery in `.cuf`.
- **Mechanism** (`gpu/gpu_runner.cuf`): `uold/unew` are now `device, pointer` handles over two
  STABLE `device, allocatable, target` buffers `ubuf_a/ubuf_b`. Decisive fact — every kernel
  dummy is assumed-size (`real(dp),device::u(...,*)`), so a pointer actual passes only the base
  address; pointer-vs-allocatable is identical at the ~25 launch sites → **zero kernel/launch
  changes**. Invariant: **`uold` always names the current valid buffer.**
- New `.cuf` routines (all pointer ops confined here): `gpu_alloc_hydro_buffers(n)` (alloc +
  associate + zero; replaces `init_amr.f90` alloc), and `gpu_upload_uold/gpu_download_uold/
  gpu_uold_devptr` helpers that `gpu_manager.f90` (H2D/D2H @39/41/122/150) and
  `julia/ramses_capi.f90` (@757) now call — **no `.f90` unit names the pointer** (the 716 fix).
- **Swap** in `gpu_set_uold`, gated `levelmin==nlevelmax .and. static_mesh`: `tmp=>uold;
  uold=>unew; unew=>tmp` (the integrator already wrote the full new state into unew). AMR/cosmo
  keep the `set_uold_kernel` copy (global buffers span levels → per-level swap would desync).
- **GLM damping folded into the integrator** so the swap stays a net win and keeps psi-damping:
  `glm_fac=exp(-(ch²/cp2)·dt)` computed in `gpu_godunov`, threaded through `mhd_integrator_kernel*`
  → `mhd_conservative_update`, applied as `unew(9)=(unew(9)+flux)*glm_fac`. `set_uold_kernel`
  reverted to pure copy; `glm_damp_psi_kernel` deleted. (Commutes with gravity/source — they
  don't touch psi — so identical on both paths.)
- **Validated:** Phase-0 micro-test (probe 1.0→2.0 across a swap, no CUDA err); 48³ turb
  conservation rel_dmass 2.8e-10, momentum trajectory `(-0.2803,1.584,1.605)` **bit-identical**
  to the pre-swap run, stable over 10 steps/swaps; **OT 2D GLMMHD PASS** (divB cleaned, max|psi|
  0.063 bounded — proves relocated damping); **Brio-Wu PASS**. 512³ step: `set_uold` + `glm_damp`
  GONE → **201.6 → 180.8 ms** (godunov 50.7% → **56.5%**, full-step **666 → 742 Mcell/s**).
### Phase 2a — set_unew eliminated via integrator base-write (uncommitted, 2026-06-21)
`mhd_conservative_update` now takes `uold` + a `base_write` logical. Single-level static
(`base_write=.true.`, gated `levelmin==nlevelmax .and. static_mesh` in `gpu_godunov`): each
cell is written exactly once (verified: each subgrid block writes only its own 2×2×2 octs via
`ind_nbor` i_s/j_s/k_s∈{1,2}, no cross-block reflux in the MHD path), so it writes
`unew = uold + fluxdiv` directly and `gpu_set_unew` is **skipped** (gated `#ifdef GLMMHD` +
single-level). AMR/cosmo keep the accumulate (pre-seeded unew). **Reformulate note:** the
naive dual-branch (18 stores) cost the integrator +10 ms (occupancy edge — PLM is ~100 regs vs
the 128 cap at launch_bounds(256,2)); collapsing to one branch that selects the base into
locals + a single 9-store block restored it to 102 ms. Validated bit-identical: 48³ conservation
1.7e-10, OT divB/psi identical, momentum trajectory unchanged.
- **512³ cumulative (swap + set_unew): 201.6 → 158.7 ms/step, godunov 50.7% → 64.2%,
  666 → 846 Mcell/s.** Per-step now = integrator 101.9 (64%) + turb_drive 31.5 (20%) +
  cmpdt 25.3 (16%).

**Honest >90% assessment (STOPPED here):** the *bookkeeping* overhead (set_unew/set_uold/glm
copies) is now fully gone — that was the real waste. The remaining ~36% is **genuine per-cell
physics**, not bookkeeping: `turb_drive` (forcing synthesis: afield trilinear interp + KE/energy
recompute) and `cmpdt` (wave-speed + dt reduction). Neither fuses cleanly:
- **turb fusion** moves the forcing *compute* (~25 ms, not memory) into the integrator; it
  doesn't vanish, and the interp's ~20 extra registers would very likely drop the PLM kernel
  2→1 block/SM (the +10 ms from just 9 store regs already showed the edge) — best case save
  ~6 ms, worst case lose ~100 ms. Poor trade.
- **cmpdt fold** only helps via **lagged dt** (use the integrator's wave-speed reduction for the
  *next* step) — a physics change (CFL), not bit-identical; conflicts with the robustness focus.
### Phase 2b — turb forcing fused into the integrator (uncommitted, 2026-06-21)
Tried it despite the occupancy worry; it's a NET WIN. The forcing source term (afield trilinear
interp + KE/energy recompute) is applied to the freshly-computed state inside
`mhd_conservative_update` on the base_write path (gated `do_turb = base_write .and. r%turb`),
replacing the separate `turb_drive_apply` pass (`gpu_turb_hydro` returns early for single-level).
Threaded `grid, afield_now_d, d_skip, dx, boxlen, turb_min_rho, smallr, smallc2, dt, do_turb`
(all `#ifdef TURB`) through the 3 mhd integrator kernels. `hydro_device` now `use turb_commons`
for TURB_GS/turb_gs_real.
- **512³: 158.7 → 152.2 ms/step.** Integrator 101.9 → 126.6 ms (+24.7, the forcing compute),
  turb_drive 31.5 → 0 → net −6.8 ms. **Occupancy held** (integrator +24%, NOT the feared 2×).
  godunov "share" 64.2% → **83.1%** (forcing is now part of the integrator); **882 Mcell/s**.
- Bit-identical: 48³ momentum trajectory unchanged; OT PASS (turb off → do_turb=false, forcing
  skipped); 740³ smoke PASS.
- **Remaining = cmpdt (25.7 ms, ~17%).** Only folds via lagged dt (a CFL/physics change). That's
  the last step to >90% and needs an explicit robustness decision.

Plan file: `~/.claude/plans/frolicking-bubbling-pretzel.md`.

### fp64 hot-path cleanup in the f32 GPU build (uncommitted, 2026-06-21)
The `boxlen`/`turb_min_rho` kind=8 args (cast at the integrator launch) surfaced a broader issue:
stray fp64 in the f32 (NPRE=4 → dp=fp32) device hot path runs at 1:64 on the A6000. Audited
the GLMMHD+TURB per-step routines (most "fp64" the survey flagged were actually `_dp` =
**fp32** already; the real offenders are `d0` literals and `kind=8` casts). Genuine hot-path fixes:
- `slope_moncen` (`gpu/gpu_hydro.cuf:253`): `factor = real(slope, kind=8)` → `real(slope, kind=dp)`.
  `factor` is already dp but the cast made an fp64 temp + fp64 `factor*slope_minmod` **per call**,
  and slope_moncen runs ~8–24×/cell in reconstruction (`mhd_slope`/trace). **This alone cut the
  integrator ~11 ms** (126.6 → 115.6 ms at 512³).
- HLLD degenerate-state guard (`gpu/gpu_hydro.cuf` ~521/532): `1d-4*A*A` → `1e-4_dp*A*A` (per face).
- Validated value-equivalent: 48³ conservation 2.84e-10, OT divB/psi identical, Brio-Wu identical.
- **KEPT as necessary fp64** (do NOT convert): global position/`floor` math (`d_skip`, `ckey`,
  `tx/trr`), conservation reduction accumulators + atomics (`mass/ekin/eint/emag/dt_loc`,
  `data_out`), `constant_gravity` interface, the one-shot `turb_ic_kernel`, and all `i8b`/kind=8
  **integer** keys. Compile-time `real(kind=dp),parameter :: x=1d-10` are fp32 at compile (fine).
- **512³ cumulative now: 141.8 ms/step, godunov 81.5%, 947 Mcell/s** (integrator 115.6 +
  cmpdt 26.2). Remaining f32-build fp64 lives in non-turb subsystems (HD `cmpdt_kernel`,
  particles/cooling/MG/poisson) — much of it necessary; sweep those per-build with their own tests.

### "kept" fp64 → fp32: forcing position + d_skip (uncommitted, 2026-06-21)
Pushed further: converted the per-step fp64 I'd deliberately kept — the turbulent driving
**position/floor** math (`tx/trr` in `mhd_conservative_update`, `x/rr` in
`turb_drive_apply_kernel`, incl. the `turb_gs_real` + `floor` casts) and **`d_skip`** (module
decl in `gpu_runner`, host fill `d_skip=m%skip`, and all 6 kernel dummies: gpu_hydro ×4,
gpu_turb, gpu_star) → fp32 (`dp`).
- **768³ × 10 steps, fp64-kept vs all-fp32: BYTE-IDENTICAL** (mass, all 3 momentum comps, Etot,
  rho[min,max,mean], sample checksum match to all 12 printed digits).
- **AND ~11 ms faster:** the forcing position (`tx/trr` + floor, 3 dims) runs *per cell* in the
  fused integrator, so its fp64 was a real 1:64 cost (not negligible as first assumed). 512³
  integrator 115.6 → **104.1 ms**; per-step **141.8 → 131.2 ms**, **1023 Mcell/s** full-step
  (godunov share 81.5%→79.4% — lower only because the integrator got faster while cmpdt held).
  Certified: 48³ conservation 2.84e-10, OT divB/psi identical, Brio-Wu PASS.
- Why exact here: for `boxlen=1` at level 10, `dx=2⁻¹⁰` (power of two) and `(2·ckey+cellbit+0.5)`
  is exactly representable in fp32, so the position is bit-exact. **But precision is irrelevant
  anyway:** the driving is a *synthetic* field (random solenoidal modes on a made-up 64³ grid),
  so the forcing position **never needs fp64 at any box size** — sampling a made-up field at an
  fp32-rounded location just reads it at a trivially different made-up point. Same for the IC
  velocity synthesis (also made-up modes). fp64 would only matter for *physical* coordinate
  mappings (self-gravity, particle positions, output) — NOT the forcing/IC. So `tx/trr`, `x/rr`,
  and `d_skip` are unconditionally fine in fp32.
- LEFT fp64 (correctly): the one-shot `turb_ic_kernel` (kept identical → isolates the per-step
  effect), the cmpdt conservation accumulators (diagnostics only; comparison sums on host fp64;
  dt already fp32), and `constant_gravity` (dead in GRAV builds). These don't affect results.
- Reference lib snapshot: `/tmp/libramses3d_fp64kept.so`; harness `runs/mhd_cuda/compare_fields.py`,
  `trunc768.nml`.

### Driving field → FULLY fp32 (host + device + single-precision FFT) — DONE (2026-06-21)
The entire turbulent-driving pipeline is now fp32 for the GPU build: host `turb%afield_*`
(`real4`) + `turb_last/next` spectrum (`complex4`) + **single-precision FFT** (`sfftw_*`,
`turb/turb_commons.f90` FFT_1D/2D/3D + power_rms_norm; OU generation math stays fp64 in
registers, stored fp32) + device `afield_last/next_d` + `afield_now_d` (`real4`).
- **THE BUG was never fp32** (Tom was right): it was the **`.f90` H2D copy of a real4 device
  array**. `afield_*_d = pst%s%turb%afield_*` inside `gpu_manager.f90` (a `.f90` unit) mis-sizes
  the real4 device-array `cudaMemcpy` → garbage/NaN (real8 happened to work). **Fix:** route it
  through `.cuf` helpers `gpu_upload_afield_init`/`gpu_upload_afield_next` in `gpu_runner.cuf`
  (same `.f90`-can't-touch-device gotcha as the uold pointer / 716). Confirmed: host field is
  finite (`anynan=F`) — the NaN was purely the upload.
- The momentum-vs-fp64 metric I chased was a **hypersensitive near-cancelling residual** (a poor
  validity test for a made-up field; even two fp64 builds differ 25%). Correct metric:
  **determinism (run 5×) + finiteness + physical rho.** `power_rms_norm`/`turb_norm` are IDENTICAL
  fp32 vs fp64 (2.8529306e-5) — the math is fine in 32-bit.
- **Validated:** 48³ rel_dmass 2.856e-10 over **5 identical runs** (momentum matches the fp64
  reference); OT GLMMHD PASS (divB/psi identical); 768³ smoke PASS ×2 deterministic.
- **BUILD NOW NEEDS `-lfftw3f`** (single-precision FFTW): `LIBFFTW="-L/usr/lib64 -lfftw3 -lfftw3f"`.
- Perf: none expected from the field (the FFT is a one-shot CPU op; the snapshots feed a
  0.014 ms/step blend) — this is full-fp32 consistency.

### d_skip → fp32 (these runs have no particles) — DONE (2026-06-21)
`d_skip` (the box-corner offset) is now fp32 too, so the turb position math (`tx/rr` in
`mhd_conservative_update`, `x/rr` in `turb_drive_apply`) is fully fp32 (no fp64 promotion on the
subtract) — this is what gives the 512³ integrator **104 ms** (the ~11 ms position win).
- **A clean rebuild exposed a latent bug** the incremental builds had hidden: `d_skip` is shared
  with the **particle/star** kernels (`compute_hkey_part_kernel` gpu_part.cuf, `star_formation_kernel`
  gpu_star.cuf), whose `skip` dummies were `real(kind=8)` → kind mismatch. These runs are
  `pic=.false.` (no particles), so those kernels are **dead code**; just made their `skip`/`d_skip`
  dummies `real(dp)` to compile. (If a particle run ever needs fp64 box-corner precision, that's a
  separate concern — not relevant here.)
- The `d_skip = m%skip` H2D in `init_amr.f90` (a `.f90` unit) is the same real4-copy hazard → routed
  through `.cuf` `gpu_upload_d_skip` (fp32 host stage + direct copy).
- **Certified on a CLEAN build (EXIT 0):** 48³ rel_dmass 2.856e-10 ×5 identical; OT + Brio-Wu PASS;
  768³ smoke PASS; 512³ integrator 104.0 ms + cmpdt 27.0 ms ≈ 131 ms/step, ~1023 Mcell/s.
- **The whole GPU turb path is now fp32:** reconstruction (slope_moncen), HLLD, cmpdt dt-path,
  forcing field (host sfftw + spectrum + device), positions, d_skip. Only genuine fp64 left:
  conservation reduction accumulators (diagnostics) + `i8b` integer keys.

### Folded cmpdt → godunov-dominated step (uncommitted, 2026-06-21)
Overlapping cmpdt on a stream FAILED (0% overlap — the integrator's millions of blocks + heavy
shared mem saturate the GPU, no idle capacity). So instead **folded cmpdt's dt-reduction into the
integrator**, reusing the primitives it already builds (no separate uold read):
- `subgrid_conserved_2_primitive_mhd` (`gpu/gpu_hydro.cuf`): for owned leaf cells (gated
  `i,j,k∈[2..2*nsubgrid+1]`, `do_dt=base_write`) computes the CFL dt from the primitive `p` it
  just built (`mhd_fast_speed_n` ×3 + the cmpdt formula), `block_reduce_min` + `atomicmin` into a
  global `dt_reduce`. Threaded `dx, courant_factor, dt_out, do_dt` through the 3 mhd integrator
  kernels; `gpu_godunov` seeds `dt_reduce` and harvests it → `dt_lagged`.
- `gpu_cmpdt` returns `dt_lagged` for single-level (skips the kernel entirely); step 0 bootstraps
  with one synchronous cmpdt. **1-step-lagged dt** (the integrator's reduction is used next step).
- **512³: cmpdt GONE (only 1 bootstrap instance); integrator 104 → 118.9 ms (the dt reduction),
  but cmpdt's 27 ms eliminated → steady-state 131 → 118.9 ms/step (−9%), godunov-dominated,
  1023 → 1129 Mcell/s.**
- Validated: 48³ rel_dmass 5.18e-10 ×5 identical (deterministic — atomicmin is order-free);
  stable over 20 + 50 steps (turbulence develops, rho_min 0.86, finite); OT PASS; 768³ ×2 det.
- **TRADEOFFS (Tom to decide if worth keeping):** (1) lagged dt is a CFL/physics change (stable
  here; for violent high-Mach add a safety margin / dt-growth clamp). (2) The conservation
  diagnostics (mass/ekin/eint/emag) are NOT computed on the folded path — `gpu_cmpdt` returns 0
  for them, so the control print loses per-step conservation monitoring (`conserve_check.py`
  still works, summing on the host). Could restore via a periodic (every ncontrol) diagnostic-only
  cmpdt if wanted.

### CFL signal-speed aggregation toggle: `cfl_sqrt3` (uncommitted, 2026-06-21)
New `&HYDRO_PARAMS` logical `cfl_sqrt3` (default `.false.`) selecting how the per-cell CFL signal
speed `ctot` is aggregated over the 3 dimensions in the dt reduction:
- `.false.` (DEFAULT) — conservative **sum-over-dims** `ctot = Σ_dir (|v_dir| + c_fast,dir)`. This
  is the original form; the default path is byte-for-byte the pre-toggle baseline.
- `.true.` — **`ctot = √3·max_dir(|v_dir| + c_fast,dir)`**. Larger dt (smaller ctot) ⇒ **~1.25×
  fewer steps / faster to a fixed time**, at the edge of the unsplit scheme's stability
  (per-direction Courant ~0.4 vs sum's stricter bound). Validated stable at Mach 10.
- Threaded as `logical, value :: cfl_sqrt3` through both dt sites in `gpu/gpu_hydro.cuf`
  (folded `subgrid_conserved_2_primitive_mhd` + bootstrap `cmpdt_kernel_mhd`) and the 3 mhd
  integrator launches + `cmpdt_kernel_mhd` launch in `gpu/gpu_runner.cuf` (passes `sim%r%cfl_sqrt3`).
  Param declared in `amr/amr_commons.f90` run_t + read in `amr/read_params.f90` (default/namelist/assign).
- **Measured (256³ PLM GLM Mach-10 turb, 6 crossing times t=0.6):** sum 54,410 steps / 19.2 min
  (768³ est ~24 h); √3·max 43,559 steps / 15.4 min (768³ est ~19 h, 1.25×). **Pure max
  (no √3): UNSTABLE** (NaN in ~200 steps — per-dir Courant 0.7 exceeds the 3D limit); ruled out.
- Validated: 48³ default rel_dmass 5.18e-10 (matches pre-toggle sum baseline exactly); `.true.`
  rel_dmass 5.23e-11, finite, deterministic ×2. Both `CHECK`.

## NEXT STEP — re-validate high-Mach robustness with REAL HLLD now active
The handoff's Mach numbers below were measured with "hlld" = LLF. Re-run the
high-Mach turbulence sweep with **real** HLLD (sharper → expect it fails EARLIER
than the old "hlld"=LLF numbers), then turn the LLF fallback on
(`switch_llf_dmin`/`pmin` ~ a few × `smallr`/`smallc²` floor) and measure how much
higher the failure Mach goes. The β=2 ramp turb namelist must be regenerated (it
lived in `/tmp`). Also still worth: detect non-positive HLLD *star* states and
fall back for just that face (a second guard inside `hlld_mhd_fluxes`).

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

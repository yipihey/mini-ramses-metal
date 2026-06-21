# GPU fused single-level step for the CT-MHD solver

Optional, opt-in optimizations that make the single-level static GPU step on the
constrained-transport (CT) MHD solver **Godunov-bound** instead of orchestration-bound.
All are gated behind the namelist switch `gpu_fused_step` (default `.false.` = the classic
per-pass path, byte-for-byte unchanged). Enable with, in `&HYDRO_PARAMS`:

```
gpu_fused_step=.true.      ! activate the fused path (levelmin==nlevelmax & static_mesh only)
gpu_dt_nharvest=8          ! optional: harvest the folded dt every N steps (1=identical dt; >1
                           !           lets the host run ahead, at the cost of an N-step-stale dt)
```

## Result (256³ isothermal MHD turbulence, RTX A6000, fp32 / NPRE=4)

| config | full-step Mcell/s | notes |
|---|---|---|
| baseline (`gpu_fused_step=.false.`) | ~104 | classic per-pass step |
| fused, `gpu_dt_nharvest=1` | ~119 | identical dt trajectory to baseline |
| fused, `gpu_dt_nharvest=8–16` | **~160** | 1.5×; step is ~89% Godunov-bound |

Every step validated: **div·B stays at machine zero** (the CT guarantee — bit-for-bit preserved),
mass/energy conserved, deterministic, finite. With the toggle off the build is byte-unchanged.

## What it does (each commit, all behind the toggle)

1. **cmpdt fold** — the CFL `dt` is reduced from the primitives the integrator already builds
   (`block_reduce_min` + `atomicmin`); the separate cmpdt pass is skipped (1-step-lagged dt).
2. **base_write + set_unew elimination** — the integrator writes `unew = uold + flux` and
   `bnew = bold + curl(EMF)` directly (each cell/face once), skipping the set_unew seed copy.
   `bnew = bold + dflux` is algebraically identical to the seed+accumulate path → div·B exact.
3. **turb fusion** — the OU forcing apply is folded into the integrator write-back (the separate
   `turb_hydro` pass is skipped; `drive_turb` still fills `fturb`).
4. **GPU-residency** — the real ceiling was per-step host↔device serialization, not the GPU
   passes: `m_timer` did a `cudaDeviceSynchronize` on every pass boundary (~10/step). A
   `timer_gpu_sync` flag (set `.false.` on the fused path) stops that, and the dt readback is
   harvested only every `gpu_dt_nharvest` steps, so the host queues kernels and runs ahead.

A C-API harness (`julia/ramses_capi.f90`: `ramses_init`/`ramses_amr_step`/
`ramses_get_timer_godunov`/`ramses_get_divb_max`/`ramses_device_sync`) drives/validates it
headlessly. **Benchmark gotcha:** measure with `foutput=0`, else a full mesh is dumped to disk
every step (~90% of wall-clock) and swamps the GPU work.

## Deliberately NOT done, and why

- **Lean (3-component) face-B storage.** CT stores B face-staggered as 6 components/cell (each
  interior face twice), so global state is 22 reals/cell vs GLM's 18; de-duplicating to 3 (→ 16,
  would fit 512³ which currently OOMs) is a **~134-site core rewrite** (device kernels + the host
  CT solver + boundaries + IC + I/O). Worse, the redundancy is a deliberate **locality** trick:
  a lean layout makes a cell's high face a *neighbour's* low face, forcing neighbour lookups and a
  **larger integrator halo** — which would likely *slow* the (already integrator-bound) 256³ kernel.
  Verdict: only worth it as a hard 512³-capability requirement, never for speed.
- **Pointer-swap double buffer** (Phase 4, ~3 ms set_uold copy) and **nsubgrid=2** (Phase 6):
  low payoff / blocked. nsubgrid=2 is gated by the *shared-memory* corner-state arrays (not the
  global `bold` redundancy), which overflow 48 KB.

## The honest contrast: CT vs GLM-MHD PLM

GLM PLM runs ~5× faster at the kernel (1100 vs ~215 Mcell/s) and fits `nsubgrid=2`, because it
keeps **B cell-centred** (Dedner-cleaned ψ) and does one 1-D Riemann per face — no face-staggered
EMF, no corner-state shared arrays, no redundant storage, no neighbour lookups. CT's cost is the
2-D EMF + induction, **intrinsic to constrained transport**, not the byte layout. CT is the right
tool only when *exact face div·B = 0* is required; otherwise GLM PLM is the fast path. This work
made CT's **orchestration** GLM-class; the remaining ~5× is the scheme itself.

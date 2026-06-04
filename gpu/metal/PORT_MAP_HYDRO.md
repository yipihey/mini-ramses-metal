# CUDA/CPU → Metal hydro-port map

Companion to PORT_MAP.md (the DM-gravity port). Same methodology: faithful
kernel-by-kernel rewrite, unit-tested per kernel, then CPU-vs-Metal diffed end to
end through the RamsesNG.jl C-API hydro slice (`ramses_godunov_fine` etc., already
wired — see the hydro test `RamsesNG.jl/.../test/hydro_sedov.jl`).

## Sources of truth
- **GPU structure**: `gpu/gpu_hydro.cuf` (the CUDA-Fortran port) — one
  threadgroup-per-oct monolithic Godunov kernel with a 6×6×6 shared-memory
  subgrid. We mirror its kernel decomposition.
- **Numerics (the diff target)**: the CPU solver `hydro/umuscl.f90` +
  `hydro/godunov_utils.f90` (the C-API diffs against this). The CUDA device math
  is a faithful transliteration of it, so matching CUDA == matching CPU.

## The scheme (confirmed)
Second-order **unsplit MUSCL-Hancock**, **HLLC** Riemann solver (LLF/HLL also
available), slope limiters 0=1st-order / 1=minmod / 2=moncen. EOS: ideal gas,
`gamma=1.4` default. Conservative state `uold/unew(twotondim, nvar, noct)`,
variable order `nvar = [ρ, ρu_x, ρu_y, ρu_z, E]` (= 5; +nener/passive scalars
deferred). Per-oct stencil = the oct's 2×2×2 cells + a 2-cell halo gathered from
the 27 neighbour octs (a 6×6×6 subgrid in 3D) — the same `nbor`/hash/cache-oct
machinery the gravity port already built (nbor.h, hash.h, refine.metal cache octs).

## Hardware constraints (Apple GPU)
No fp64, no 64-bit float atomics. Decisions:
- Bulk hydro in **fp32** (the gravity port showed fp32 tracks the CPU to the
  chaos floor for the DM problem; revisit per-field if a parity test fails).
- The CFL `cmpdt` reduction (min dt + sum mass/ekin) reuses the gravity port's
  **fixed-point atomic** reduction (reduce.h) for bit-reproducibility.
- The coarse-fine `coarse_cell_update` flux correction uses atomics in CUDA;
  port via fixed-point atomics or a per-face serialized pass (TBD when reached).

## Kernel inventory (gpu_hydro.cuf → metal)  — order = port order
Device math (pure, fp32, unit-testable in isolation):
- [x] magnitude_squared, compute_pressure/energy, sound_speed           → hydro.h
- [x] slope_minmod, slope_moncen                                        → hydro.h
- [x] conserved_2_primitive, primitive_2_conserved                      → hydro.h
- [x] hll_flux, hll_fluxes, hllc_fluxes, riemann_fluxes                 → hydro.h
Simple whole-array kernels:
- [x] set_unew_kernel, set_uold_kernel  (uold<->unew copy)              → hydro.metal
- [x] upload_kernel       (restriction: avg 8 children -> parent cell)  → hydro.metal
The Godunov pipeline (mirrors hydro_integrator_kernel):
- [x] trace_3d            (moncen slopes + MUSCL-Hancock trace -> ±face states) :400
      -> hydro.h trace_cell_1d / trace_cell_3d (unit-tested vs double replica)
- [x] riemann_driver      (per-interface riemann_fluxes + velocity rotation)    :815
      -> hydro.h flux_x/flux_y/flux_z (the rotation), composed in godunov_oct_*
- [x] conservative_update (du = (F_L - F_R)*dt/dx, 3 directions)                :978
      -> hydro.h godunov_oct_1d / godunov_oct_3d (unit-tested, 1D+3D)
- [ ] subgrid_conserved_2_primitive  (load 27-nbor octs -> 6^NDIM subgrid,
      c2p + gravity half-step predictor) -- NEXT, needs the mesh nbor gather :304
- [ ] zero_fine_fluxes    (zero fluxes at faces touching a finer level)         :919
- [ ] coarse_cell_update  (atomic flux correction onto coarse parent)           :1061
- [ ] hydro_integrator_kernel (assembles: gather -> godunov_oct -> update)      :1350
Coupling / control / refine:
- [ ] cmpdt_kernel        (CFL dt + mass/ekin reductions)                       :1795
- [ ] grav_hydro / sync_hydro (gravity source half/full step)             :1731/:1673
- [ ] hydro_flag_kernel   (density/pressure-gradient refinement criterion)      :1466

## Bridge / buffers (deferred until kernels land)
The gravity bridge has grid/phi/phi_old/f/rho/nbor/father/flag1 device buffers.
Hydro adds **uold** and **unew** buffers (twotondim*nvar per oct); the bridge
copies rho/uold in, runs the per-level hydro sequence (set_unew → godunov →
[grav/source] → set_uold → upload), copies uold out. The C-API
`ramses_godunov_fine` etc. will route to these under the Metal library, exactly
like the gravity routines route to the GPU solve.

## Test strategy
Per-kernel `.metal`+`.mm` harness (build the kernel into a metallib, feed known
input, check vs a host double-precision replica of the SAME formula or an analytic
value) — NO full RAMSES runs to infer correctness. End-to-end parity comes from
the C-API Sedov/tube diff (CPU lib vs Metal lib) once the pipeline is assembled.
- [x] test_hydro: EOS round-trip, HLL/HLLC/LLF vs host-double replica +
      identical-state→physical-flux, slope limiters, set_unew/set_uold copy.

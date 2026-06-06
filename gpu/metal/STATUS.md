# Metal (Apple Silicon) gravity port — status

A faithful re-implementation of the mini-RAMSES particle-mesh **gravity** stack
(density deposit → multigrid Poisson V-cycle → 4th-order force gradient → kick),
plus GPU AMR refinement flagging and refine/derefine, running on Apple GPUs via
Metal. It mirrors the existing CUDA-Fortran port (`gpu/*.cuf`) one kernel at a
time; the CUDA `.cuf` files are the single source of truth.

- **Kernels**: `gpu/metal/*.metal` (+ shared `.h`). Compiled to one
  `ramses_kernels.metallib`.
- **Host bridge**: `gpu/metal_bridge.mm` (Obj-C++; device/queue/buffers/dispatch)
  with the `extern "C"` surface declared in `gpu/metal_bridge_iface.f90`.
- **In-loop driver**: `gpu/metal_gravity.f90` — per AMR level, replaces the CPU
  Poisson+force with the Metal pipeline. AMR step/move stays on the CPU; only
  `rho` crosses in and `phi`/`f` cross out (no per-step particle marshalling).
- **Detailed dev history / per-kernel parity notes**: `gpu/metal/PORT_MAP.md`.

## Current status: validated to the floating-point / chaos floor

The 3D DMO cosmological run (`levelmin=7`, `levelmax=15`, 2.1M particles, to
`a=1`) on the Metal GPU now matches the CPU reference to within the
floating-point + AMR-chaos floor — the GPU-vs-CPU difference is no larger than
the CPU-fp64-vs-CPU-fp32 difference.

Velocity distribution at `a≈1.0` (`runs/dmo/`, identical ICs & namelist):

| run | engine | steps | max level | σ_v | v(p99) | v(p99.9) | ekin |
|-----|--------|------:|----------:|----:|-------:|---------:|-----:|
| `cpu64_full`  | CPU fp64 | 419 | 13 | 0.0288 | 0.1466 | 0.2006 | 1.4226e-3 |
| `fresh_cpu32` | CPU fp32 | 419 | 13 | 0.0288 | 0.1468 | 0.2011 | 1.4233e-3 |
| `fresh_metal` | GPU      | 419 | 13 | 0.0288 | 0.1465 | 0.2009 | 1.4183e-3 |
| `batch_metal` | GPU      | 419 | 13 | 0.0288 | 0.1466 | 0.2012 | 1.4241e-3 |

The high-velocity tail (p99 / p99.9) is the meaningful test — total `ekin` is
mean-dominated and hides the physics. GPU↔CPU agree to <0.3% in the deep tail.

### The over-energization bug is fixed

Earlier the full-GPU 3D run over-energized: a runaway high-v tail, deep
over-refinement to levels 14–15 the CPU never reached, and ~14× too many steps
(runaway velocities → `dt` ~14× too small). Root cause and fix:

- **Root cause**: the pre-solve `phi → phi_old` snapshot was taken at the wrong
  point. Saving it *before* `make_cache` wrote `phi_old` into the wrong device
  oct slots, so a finer subcycle level's coarse-fine boundary time-extrapolation
  (`make_initial_phi`, `icount=2`, the `tfrac` term) read `phi_old ≈ 0` →
  a large spurious boundary kick at refined levels → over-energization.
- **Fix** (`gpu/metal_gravity.f90`): snapshot `phi → phi_old` **after**
  `make_cache` (device octs in the order the boundary read uses) and **before**
  the multigrid overwrites `B.phi`; plus seed resident `B.phi`/`B.phi_old` from
  the host on a fresh/restart cold start (commit `ffcf1b54`). See the comments at
  `metal_gravity.f90:374` and `:200`.

The three gross signatures (419 steps = CPU, max level 13 = CPU, matched v-tail)
confirm it. A related earlier fix — the GPU flag reading a **stale father/nbor
connectivity cache** at coarse-fine boundaries — is documented at the end of
`PORT_MAP.md` (forced resync in `m_metal_flag`).

## Build

Requires the macOS Metal toolchain (`xcrun metal`/`metallib`, ships with Xcode)
and `gfortran` for the host.

```sh
cd bin
# Shared library for the Julia / C-callable API (RamsesNG.jl):
make COMPILER=METAL GRAV=1 NDIM=3 libramses     # -> libramses3d_metal.dylib
# Standalone executable:
make COMPILER=METAL GRAV=1 NDIM=3 ramses        # -> ramses3d
```

This compiles `gpu/metal/{scan,primitives,sort,hash,nbor,reduce,rho,mg,part,flag,refine}.metal`
into `ramses_kernels.metallib` (built with `-fno-fast-math` — **required**, fast-math
collapses the df64 error-free transforms back to fp32 and breaks the periodic-base
parity), then the Fortran host + `metal_bridge.mm`.

> Stale-object footgun: the Makefile relinks against existing `.o`/binaries. To
> force a clean rebuild of the GPU path, `rm` the relevant `.o` and the binary
> first. A stale `ramses_kernels.metallib` silently ships old kernels.

## Run

The GPU path is selected at build time (`COMPILER=METAL`). On startup the run
prints a banner — confirm the GPU actually engaged:

```
[metal] init on Apple M5 Max
[METAL] in-loop GPU gravity ENABLED; gpu_flag=T mg_driver=F
```

> **NDIM / metallib footgun**: a 1D run will silently load a `NDIM=3` metallib by
> default and read stride-8 garbage. Set `RAMSES_METALLIB` to the matching
> library, or rely on the per-NDIM build output.

### Key environment toggles

| var | default | meaning |
|-----|---------|---------|
| `RAMSES_GPU_FLAG`   | on (`1`) | GPU refinement flagging (`0` = CPU flag, bit-reproducible vs historical runs) |
| `RAMSES_GPU_REFINE` | on       | GPU AMR create/derefine/compact (`0` = CPU refine) |
| `RAMSES_METAL_MG`   | on       | GPU multigrid driver |
| `RAMSES_METALLIB`   | —        | explicit path to the metallib (set this for non-default NDIM) |

Diagnostics (off by default): `RAMSES_*_DUMP`, `RAMSES_MG_VERBOSE`,
`RAMSES_FALLBACK_DIAG`, `RAMSES_TFRAC_DBG`, and the `[METAL-PROF]` per-level
multigrid timing. Solver knobs: `RAMSES_MG_{EPS,MAXITER}`,
`RAMSES_NCYC_{BASE,FINE}`, `RAMSES_MG_PLATEAU`. (`RAMSES_SKIP_SAVE_PHIOLD` is a
solve-isolation test hook — leave unset in production.)

## Tests

Per-kernel unit tests (known input → one kernel → check vs CPU/analytic
reference; no full RAMSES runs needed):

```sh
cd gpu/metal/tests && ./run_tests.sh
```

Covers df64, reductions/scan, the full MG V-cycle core (residual / Gauss-Seidel /
restrict / interpolate / mask / energy / norm), CIC deposit + gather/kick
(momentum-conserving), `newdt`, flag, cache-oct compaction/creation, an
integrated MG-solve convergence test (`test_mg_vcycle`), and the cache-oct host
orchestration (`test_cache_orch`).

## Known limitations / future work

- Forced flag connectivity resync each flag pass costs ~19% (correctness over
  speed); could rebuild only when the mesh changed since the last flag.
- Hard HW constraints (Apple GPU): no fp64 and no 64-bit atomics. Handled with
  df64 (double-single) only where fp32 breaks parity (periodic-base RHS mean /
  residual, global reductions) and a 32-bit-CAS hash. See `PORT_MAP.md`.
- Hydro is not ported (DM gravity / PIC only); `gpu_hydro.cuf` has no Metal twin.

# PORT: Fortran hydro integrator → CUDA-C kernel — CONTINUE HERE

**★ BREAKTHROUGH (the GRAV=0↔1 mystery is SOLVED):** the gap was **fp64 arithmetic in the
`constant_gravity` predictor** (`constant_gravity` is `real(kind=8)` → `v + cg*0.5_dp*dt` promotes to fp64
→ 1:64 throughput on the A6000), NOT an L1/TEX throttle (the prior ncu reading was fp64-ALU saturation
misread as a memory-pipe stall). Isolated with the C port's controllability: f-allocated-but-unread =
slow (not the 1.6GB); only the predictor reads matter; **broadcast vs divergent f-read are equally fast
(kills MLP/pacing theory)**; fp64-vs-fp32 predictor is the decider. **FIX (one line, fp32-safe): cast cg→dp.**
Applied to BOTH kernels: CUDA-C GRAV=0 2967→~4176 (≈GRAV=1); **Fortran GRAV=0 2864→~3631 (+27%, ≈Fortran
GRAV=1), conserve PASS** — a free production win. GRAV=0 is now the best config (GRAV=1 speed + saves 1.6GB).
TODO: same latent fp64 bug in the GLM-MHD predictors (gpu_hydro.cuf ~2499/3420/3931) — not yet fixed/tested.

**Status:** Stages 0+1+2 DONE + GRAV=0↔1 mystery solved. **S1 (faithful scalar-load translation) PASSED — and BEATS the Fortran
ceiling**: 480³ turb warm-interleaved GRAV=1 **~4254 vs ~3560 (+19%)**, GRAV=0 **~2951 vs ~2838 (+5%)**,
conserve_check turb 96³ rel_dmass=0/rel_dmom=0 PASS. **S2 (float4 wide loads) IMPLEMENTED + CORRECT but
FALSIFIED as a speedup**: the load path under `#ifdef WIDELOAD` emits 5 `LDG.E.128` / 0 scalar uold loads
(the exact thing nvfortran can't do), conserve PASS, but throughput GRAV=1 −5% / GRAV=0 +1% (noise) — the
nvcc kernel is NOT initial-load-instruction-throttled like the Fortran one (better scheduling +
fastmath), so the ≥15% gate is not met. The win is BANKED from nvcc codegen + `--use_fast_math` +
`__launch_bounds__(256,3)`, NOT wide loads. The GRAV=0↔1 gap persists as the genuine MLP/pacing↔capacity
tradeoff. Two non-obvious build levers BOTH required (missing either ≈ −50%): `--use_fast_math` on the
nvcc rule + `__launch_bounds__(256,3)` (macro `CUMINBLK`; `-maxrregcount` is IGNORED under a bare
launch_bounds). WIDELOAD kept flag-gated, default OFF (scalar is faster). **Next (optional, untested):**
(a) ncu on the nvcc kernel (sudo, Tom) to confirm whether tex_throttle is even present; (b) the only
lever targeting a different bottleneck = register-block trace_3d to cut short_scoreboard (shared
round-trips) — a larger rewrite. Files: `gpu/cuda_hydro_integrator.{cu,cuf}` (wired `#ifdef CUKERNEL` at
gpu_runner.cuf). Binaries: `bin_hydro/ramses3d_cuk_g{0,1}` (scalar), `ramses3d_cuk_g{0,1}_wl` (wide),
`libramses3d.so` (scalar GRAV=1).
**Branch:** `cuda-dedner-mhd` (fork `git@github.com:yipihey/mini-ramses-metal.git`).
**Env:** NVHPC 26.3 (`/opt/nvidia/hpc_sdk/Linux_x86_64/26.3`), CUDA 13.1, RTX A6000 (sm_86, 46 GB, 768 GB/s).
**Repo:** `~/Projects/mini-ramses-metal`. Runs/measurement dir: `runs/mhd_cuda`.

---

## 1. Why we're doing this (the proven goal)

The f16/nsubgrid=3 hydro godunov kernel (`hydro_integrator_kernel`) is fast, but **GRAV=0 is ~24% slower
than GRAV=1** (480³ turb godunov: GRAV=1 ≈ **3620**, GRAV=0 ≈ **2893** Mcell/s) even though GRAV=0 does
*less* work. We proved the cause with `ncu` (sudo): the kernel is **throttled on the L1/TEX request pipe**,
not bandwidth/latency/occupancy.

| ncu metric (480³, byte-identical kernels except the `f`-reads) | GRAV=1 | GRAV=0 |
|---|---|---|
| duration | 38.5 ms | 50.8 ms |
| SM (compute) SOL | 64.7% | **82.5%** |
| stall `tex_throttle` (L1TEX queue full) | 0.00 | **2.24** |
| stall `short_scoreboard` (shared/L1) | 0.76 | **2.88** |
| stall `long_scoreboard` (global) | 2.97 | 1.88 |
| occupancy | 49% | 49% (identical) |

**Mechanism:** GRAV=0 issues `uold` loads in tight bursts (5× scalar `LDG.32`/cell) → the LSUIN
instruction queue backs up → `tex_throttle`. GRAV=1's divergent `f`-reads (in the dt/predictor, interspersed
with compute) *pace* the L1 request issue, so the queue never fills. It's **request-pacing / pipe-balance**,
not latency hiding.

**The fix** is fewer/wider load instructions (e.g. one `LDG.128` for 4 cells instead of 4× `LDG.32`).
**We proved this is impossible in CUDA Fortran**: nvfortran 26.3 never emits `LDG.128` (it can't assert the
runtime alignment) and has no device `cp.async`. A from-Fortran WIDELOAD attempt *regressed* −5%/−11%
(more scalar loads), which itself re-confirmed the mechanism. **So we port the kernel to CUDA-C**, where
`reinterpret_cast<float4*>` + asserted alignment makes wide loads trivial.

Exhausted-and-rejected levers (don't re-try): software prefetch (+3%, already committed), occupancy/4-blocks
(negative — reg-cut spills to stack), cp.async (no-go — needs overlappable compute the load phase lacks).
Full detail in the memory file `project_glmmhd_metal_hll_f16.md`.

---

## 2. The staged plan (with kill gates)

| stage | what | gate | cost |
|---|---|---|---|
| **S0** ABI spike | verify CUDA-C can read the Fortran `oct`/`uold` device arrays | ✅ **PASSED** | done |
| **S1** faithful translation | full nsubgrid=3 kernel in CUDA-C, **scalar loads** | `conserve_check` PASS **and** throughput parity (≈3620/2893) | ~2–3 d |
| **S2** wide loads | `float4`/`reinterpret_cast` loads in the `.cu` | `ncu`: `tex_throttle`→0 **and** ≥15% on GRAV=0 | ~1–2 d |
| **S3** (optional) | cp.async / request pacing | — | — |

Rationale for the S1 parity gate: if a scalar-load CUDA-C kernel can't match the Fortran one, the bug is in
the translation/ABI — stop there (cheap) before investing in S2. The speedup only lands in S2.

---

## 3. Stage 0 result — VERIFIED ABI (reuse this verbatim)

A CUDA-C kernel reads the Fortran device arrays **directly, no marshalling**. Validated: Fortran filled a
device `oct`/`uold` with known values, a CUDA-C kernel read them back — `sizeof` 64==64, every field matched
across 8 octs, `uold` indexing matched.

```c
// mirror of amr/oct_commons.f90 for NDIM=3, nhilbert=1  (nvfortran default logical = 4 bytes)
struct Oct {
    long long hkey[1];    // integer(8) hkey(1)          offset 0
    int       ckey[3];    // integer(4) ckey(3)          offset 8
    int       refined[8]; // logical    refined(8)       offset 20   (4 bytes each; read !=0)
    int       lev;        // integer(4)                  offset 52
    int       superoct;   // integer(4)                  offset 56
};                        // sizeof == 64  (padded to i8-hkey alignment; matches Fortran stride)

// uold(cell,var,oct) Fortran col-major  ->  uold[(oct*nvar + var)*8 + cell]   (fp32; NPRE=4)
// f(cell,dim,oct)    same shape (twotondim=8, ndim=3, ngrid)  ->  f[(oct*3 + dim)*8 + cell]
// nbor(ind, subgrid) col-major  ->  nbor[ subgridsize*subgrid + ind ]   (int32)
```

Indexing is 0-based in C vs 1-based in Fortran (oct `i` in C == `i+1` in Fortran).

---

## 4. Stage 1 — exact task

### 4a. What to TRANSLATE (Fortran → CUDA-C), all in `gpu/gpu_hydro.cuf`
The nsubgrid=3 cooperative-shared-tile orchestration. 256 threads/block, `work_size = 2*nsubgrid+4 = 10`,
1000-cell f16 shared tile, **3 `__syncthreads`** (after load, after trace, after riemann). Anchors:

| Fortran (`gpu/gpu_hydro.cuf`) | line | role |
|---|---|---|
| `hydro_integrator_kernel` (the `__global__`) | **2232** | orchestration: load → trace → riemann → zero-fine-fluxes → update → cmpdt fold → turb |
| `subgrid_conserved_2_primitive` | **622** | cooperative tile LOAD + c2p + dt-fold + gravity predictor ← **the wide-load target** |
| `trace_3d` | **1023** | MUSCL-Hancock reconstruction over the shared tile |
| `riemann_driver` | **1580** | per-face HLLC/LLF over interface tiles |
| `zero_fine_fluxes` | **1684** | zero flux at refined-cell faces (uses tile `refined`) |
| `conservative_update` | **1743** | `unew = uold + dt/dx·ΣF` (+ turb fusion when `base_write`) |
| `coarse_cell_update` | **1929** | reflux for the coarse neighbours |

Shared tile types are in `gpu/gpu_utils.cuf`: `subgrid_6x6x6cell_primitive` (f16, the 5 prim vars + `refined`)
and the `subgrid_*_interfaces` face tiles. f16 tile = `__half` shared arrays. `tp` = TPRE kind = f16;
c2p/trace math stays fp32 (accuracy — the energy→pressure cancellation needs fp32).

### 4b. What to REUSE (CUDA-C, from `gpu/metal/hydro.h`, 1085 lines C++)
Already has the scheme-independent per-cell physics: `conserved_2_primitive`, `slope_moncen`,
`hllc_fluxes`/`hll_fluxes`/`riemann_fluxes`, `flux_x/y/z`, `trace_cell_3d`, `compute_pressure/energy`,
`sound_speed`, `strong_pressure_jump`. Adaptation = mechanical Metal→CUDA (drop `thread`/`device` address-
space qualifiers, `threadgroup`→`__shared__`). **Do NOT reuse `gpu/metal/hydro.metal`'s `hydro_godunov`** —
it's the **nsubgrid=1** old scheme (one thread per oct, 27-neighbour gather), wrong structure for our tiled
kernel. (`gpu/metal/PORT_MAP.md` documents the Metal port's nsubgrid=1 decision.)

### 4c. Integration wiring
- **Launch site:** `gpu/gpu_runner.cuf:569`, currently:
  ```fortran
  call hydro_integrator_kernel<<<num_blocks, threads_per_block>>>(grid, uold, unew, f, father, nbor, &
       & head_idx, num_subgrids, sim%m%ngridmax, ilevel, sim%r%levelmin, sim%r%nlevelmax, &
       & gamma, smallr, smallc2, dt, dx, slope, riemann, constant_gravity, base_write, &
       & real(sim%r%courant_factor,kind=dp), dt_reduce, sim%r%cfl_sqrt3 &
  #ifdef TURB
       & , afield_now_d, d_skip, real(sim%r%boxlen,kind=dp), real(sim%r%turb_min_rho,kind=dp), do_turb &
  #endif
       & )
  ```
  Behind a new `#ifdef CUKERNEL`, replace this with a `call launch_hydro_integrator(c_devloc(grid), …)` to
  the `extern "C"` launcher (pass device arrays via `c_devloc(...)`, scalars by value). Keep the Fortran
  `<<<>>>` as the `#else` default.
- **The `.cu` + `.cuf` pair** (mirror `gpu/cub_module_radix_sort.{cu,cuf}`):
  - `gpu/cuda_hydro_integrator.cu`: the `__global__` + `extern "C" void launch_hydro_integrator(...)` host
    wrapper that does the `<<<num_blocks, threads_per_block>>>`.
  - `gpu/cuda_hydro_integrator.cuf`: a module with the `bind(c)` interface. **`c_devptr` is from `cudafor`,
    not `iso_c_binding`.** Pass `c_devloc(arr)` by value to a C `void*`.
- **Build rules** (clone `bin/Makefile:359-369`, the cub pattern):
  ```make
  cuda_hydro_integrator_c.o: cuda_hydro_integrator.cu
  	nvcc -arch=$(CUDA_ARCH) -O3 -lineinfo -Xcompiler -fPIC -c $(GPU_PATH)/cuda_hydro_integrator.cu -o cuda_hydro_integrator_c.o
  cuda_hydro_integrator_f.o: cuda_hydro_integrator.cuf cuda_hydro_integrator_c.o
  	nvfortran $(FFLAGS) -c cuda_hydro_integrator_c.o $(GPU_PATH)/cuda_hydro_integrator.cuf -o cuda_hydro_integrator_f.o
  ```
  Add `cuda_hydro_integrator_c.o cuda_hydro_integrator_f.o` to **`MODOBJ`** (`bin/Makefile:296`, where the
  cub objects are listed). Link already pulls `-lstdc++` (LIBS). Gate the rules + MODOBJ entries on
  `ifdef CUKERNEL` so default builds are untouched.

---

## 5. Build & measure (copy-paste)

```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/bin:$PATH
cd ~/Projects/mini-ramses-metal/bin_hydro      # build dir (bin/ now also has NSUB/TPRE flags)
LF="-L/usr/local/fftw/gcc/3.3.9/lib -lfftw3 -L/usr/lib64 -lfftw3f"
# multi-pass -j -k to survive .mod-ordering races, then a final pass:
for p in 1 2 3; do make -j 32 -k CUDA_ARCH=sm_86 NPRE=4 NDIM=3 HYDRO=1 GLM=0 GRAV=0 TURB=1 \
  COMPILER=NVHPC NSUB=3 TPRE=2 CUKERNEL=1 "LIBFFTW=$LF" ramses; done
# GRAV=1 build: same with GRAV=1.  (GRAV=0 saves ~1.6GB but is the slow one we're trying to fix.)
```

**Throughput (480³ turb godunov-only — the fair metric; whole amr_step is host-overhead-bound):**
```bash
cd ~/Projects/mini-ramses-metal/runs/mhd_cuda
tmp=$(mktemp --suffix=.nml); sed -e 's/nstepmax=[0-9]*/nstepmax=40/' -e 's/tout=[0-9.]*/tout=1d30/' turb480_ns3.nml > "$tmp"
god=$(../../bin_hydro/ramses3d "$tmp" 2>&1 | grep "hydro - godunov" | tail -1 | awk '{print $1}'); rm -f "$tmp"
python3 -c "print(f'{480**3*40/$god/1e6:.0f} Mcell/s  ({$god:.3f}s)')"
# baselines to beat/match:  GRAV=1 ~3620,  GRAV=0 ~2893 Mcell/s
```

**Correctness — conserve_check (S1 gate). MUST use a TURB nml** (`turb_ns3_96.nml`), NOT sod (see gotchas):
```bash
# build the .so first:  make ... GRAV=0 CUKERNEL=1 ... libramses   (capi GRAV guards already committed)
cd ~/Projects/mini-ramses-metal/runs/mhd_cuda
python3 conserve_check.py ../../bin_hydro/libramses3d.so turb_ns3_96.nml 7 100
# PASS = rel_dmass=0.000e+00 rel_dmom=0.000e+00
```

**Full test ladder (before declaring S1 done):** Brio-Wu (`briowu*.nml`) → Orszag-Tang (`ot*.nml`) →
CP-Alfvén (`cpalfven*.nml`, `cp_alfven_test.py`) → forced turb (`turb_ns3_96.nml` conserve + `turb_test.py`).
These are MHD-leaning; for pure HYDRO the key ones are sod conservation (via turb path) + turb stats.

**Inspect SASS / occupancy (no sudo):** `cuobjdump -sass <bin> | grep LDG` (count `LDG.E.128` vs `LDG.E`),
`cuobjdump -res-usage <bin> | grep -A1 integrator` (REG/SHARED/STACK).

**ncu (S2, needs sudo — counters are perm-locked):** reuse the pattern in the memory file / prior
`/tmp/ncu_profile.sh`: sudo strips `LD_LIBRARY_PATH`, so the script must re-export it (NVHPC compilers/lib +
cuda/lib64 + math_libs/lib64 + fftw). `ncu --section WarpStateStats,... -k regex:hydro_integrator_kernel
-s 3 -c 1 -o /tmp/ncu_g0 <bin> <nml>`; analyze with `ncu -i <rep> --page raw | grep stall`.

---

## 6. Gotchas (hard-won — read before starting)

- **nvfortran emits NO `LDG.128` / no `cp.async`.** This is the whole reason for the port; don't try to fix
  the load in Fortran again (verified: even a `bind(c)` 16-byte struct read scalarizes).
- **conserve_check turb-gate.** The GPU-resident basegrid build only runs with `turb=.true.`
  (`init_refine_basegrid.f90` gate: `turb .and. nlevelmax==levelmin .and. .not.poisson`). A `turb=.false.`
  nml (sod) routes through the CPU Hilbert build → the C-API reads the wrong octs → **~30% mass loss / NaN
  that is NOT a kernel bug**. Always validate conservation with `turb_ns3_96.nml`.
- **nsubgrid=3 needs grids divisible by 6** (octs/dim divisible by 3 so `noct/27` is integer). 480³/96³ work;
  128³ (64 octs/dim) NaNs. Set via `&BOUNDARY_PARAMS` `box_xmax` (octs) + `bound_levelmin`.
- **`.mod` build races** with `make -j`: do multi-pass `make -j 32 -k` (repeat) then a final pass.
- **Link needs `-lstdc++`** for the `.cu` object; **`c_devptr` is from `cudafor`** (not `iso_c_binding`).
- **PATH** must include NVHPC `compilers/bin` (nvfortran) and `cuda/bin` (nvcc, cuobjdump, ncu).
- **Module ABI segfault:** changing `gpu_utils.cuf` (a module) requires a `make clean` full rebuild, else a
  derived-type ABI mismatch segfaults at startup.
- **GRAV=0 build** needs the `#ifdef GRAV` capi guards (already committed, f61e782f) for `libramses` to link.
- **f16 tile = `tp`/`TPRE` kind; c2p/trace MUST stay fp32** (energy→pressure catastrophic cancellation).
  `merge(real2,...)` is unsupported by nvfortran (blocks f16 in GLM/MHD, fine for NVAR=5 hydro).
- The `nbor` indirection gives **semi-local** scattered `uold` reads (block-contiguous grid → halo reuse in
  L2); a microbench with random scatter overstates latency and misleads (we hit this).

---

## 7. Current committed state (branch `cuda-dedner-mhd`, local only, not pushed)

```
fc9a2ced  build: track the NSUB/TPRE fast-tile opt-in flags in bin/Makefile
f61e782f  GPU HD: GRAV=0 build support — #ifdef GRAV guards
ae1b1406  GPU HD: software-prefetch load loop + f16/nsubgrid=3 fast tile (flag-gated)
```
- f16/nsubgrid=3 fast tile + software-prefetch load loop: behind `NSUB`/`TPRE` (default builds unchanged).
  Validated: 96³ turb conserve PASS; 480³ ~3620 (GRAV=1) / ~2893 (GRAV=0) Mcell/s.
- GRAV=0 buildability: `#ifdef GRAV` guards in `julia/ramses_capi.f90` + `amr/{amr_step,adaptive_loop}.f90`.
- NSUB/TPRE flags now in the tracked `bin/Makefile` (buildable from a clean clone).
- Binaries present: `bin_hydro/ramses3d_ns3` (GRAV=1), `ramses3d_ns3_grav0` (GRAV=0), `libramses3d.so`.

## 8. File index
- `gpu/gpu_hydro.cuf` — the kernel to translate (anchors in §4a).
- `gpu/gpu_utils.cuf` — shared tile types (`subgrid_6x6x6cell_primitive`, interface tiles), `nsubgrid`/`tp`.
- `gpu/gpu_runner.cuf:569` — the launch site to wire `#ifdef CUKERNEL`.
- `gpu/metal/hydro.h` — reusable C++ physics (§4b).
- `gpu/metal/PORT_MAP.md` — the CUDA→Metal port decisions (note: Metal is nsubgrid=1).
- `gpu/cub_module_radix_sort.{cu,cuf}` + `bin/Makefile:296,365-369` — the `.cu`+`bind(c)`+build template.
- `bin/Makefile` (tracked) / `bin_hydro/Makefile` (gitignored build copy) — `gpu_hydro.o` has
  `-gpu=fastmath` + `-gpu=maxregcount:72` (non-GLM); add `ifdef CUKERNEL` rules + MODOBJ entries.
- `runs/mhd_cuda/` — `turb480_ns3.nml` (480³ perf), `turb_ns3_96.nml` (96³ conserve), `conserve_check.py`,
  `cp_alfven_test.py`, `turb_test.py`, `validate_mhd.py`.
- Memory: `~/.claude/projects/-home-tabel-Projects/memory/project_glmmhd_metal_hll_f16.md` (full history).

## 9. First actions for the new session
1. Read this file + the memory note `project_glmmhd_metal_hll_f16.md`.
2. Confirm baselines still reproduce: build `bin_hydro` GRAV=0/1 (NSUB=3 TPRE=2), measure 480³ (≈3620/2893).
3. Begin S1: create `gpu/cuda_hydro_integrator.{cu,cuf}` (scalar loads first), adapt `hydro.h` physics, wire
   the `#ifdef CUKERNEL` launch + build rules, build, then hit the parity gate (conserve_check turb 96³ PASS
   + 480³ throughput ≈ Fortran). Only then move to S2 (float4 wide loads).

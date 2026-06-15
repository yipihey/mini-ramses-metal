# Plan: Generalize GPU MHD to `nsubgrid > 1`

**Goal.** Run GPU MHD with `nsubgrid > 1` to amortize ghost-cell work, at the
production precision **NPRE=8 (FP64)** — single precision is *not* a fallback
here, because at high AMR levels FP32 breaks the integrator, so the whole plan
is sized for FP64.

The work is split into two phases with very different cost/risk:

- **Phase I — `nsubgrid = 2` on the existing fused kernel** (`hydro_integrator_kernel`,
  [gpu/gpu_hydro.cuf:3745](gpu/gpu_hydro.cuf#L3745)). A small, surgical change:
  the only blocker is the 48 KB static-shared cap, fixed by moving the working
  set to dynamic shared memory. This is the near-term deliverable.
- **Phase II — `nsubgrid = 4` on a plane-tiled MHD kernel** built from the
  existing `PAPER` kernel. A substantial new kernel, because `nsubgrid=4` cannot
  fit the fused design on any GPU (§2). This is a follow-on project.

---

# PHASE I — `nsubgrid = 2`, fused kernel

## 1. Root cause: it is *only* the 48 KB static-shared cap

Almost everything in the integrator is already `nsubgrid`-generic. The evidence:

- **Shared buffers are all sized from `nsubgrid`** (compile-time parameter in
  [gpu/gpu_utils.cuf:9](gpu/gpu_utils.cuf#L9)). Every subgrid derived type is
  dimensioned `(0:2*nsubgrid+3, …)`, `(0:2*nsubgrid, …)`, etc.
- **Every work loop is a grid-stride loop over a `nsubgrid`-derived size**, e.g.
  `do work_idx = thread_idx, work_size**3 - 1, blockDim%x` with
  `work_size = 2*nsubgrid+4` ([gpu/gpu_hydro.cuf:1305](gpu/gpu_hydro.cuf#L1305)),
  and similarly with `interface_array_size = (2*nsubgrid+1)*(2*nsubgrid)**2`
  ([gpu/gpu_hydro.cuf:2356](gpu/gpu_hydro.cuf#L2356)) and
  `emf_size = (2*nsubgrid+1)**2*(2*nsubgrid)` ([gpu/gpu_hydro.cuf:3485](gpu/gpu_hydro.cuf#L3485)).
  Raising `nsubgrid` and the thread count just makes these loops iterate
  correctly — no index math changes.
- **The launch site already supports it**: `gpu_godunov` already has
  `if(nsubgrid == 2) threads_per_block = 256`
  ([gpu/gpu_runner.cuf:377-378](gpu/gpu_runner.cuf#L377)), and `head_idx` /
  `num_subgrids` are already computed from `nsubgridtondim`
  ([gpu/gpu_runner.cuf:359-360](gpu/gpu_runner.cuf#L359)).
- **The neighbour stencil is already general**: `nbor(1:subgridsize, *)` with
  `subgridsize = (nsubgrid+2)**ndim` ([gpu/gpu_utils.cuf:14](gpu/gpu_utils.cuf#L14)),
  and `update_nbor` is driven by `nsubgridtondim`
  ([gpu/gpu_runner.cuf:934-950](gpu/gpu_runner.cuf#L934)).
- **The non-MHD hydro path already runs at `nsubgrid=2`** (its footprint stays
  under the cap), which proves the surrounding decode/launch machinery is
  correct.

The one thing MHD adds is a large set of extra resident shared buffers (`bf`,
`emfx/y/z`, `corner_x/y/z`, plus 3 extra B fields in each of the 6 interface
arrays). The shared-memory budget block at [gpu/gpu_hydro.cuf:3677](gpu/gpu_hydro.cuf#L3677)
already calls this out:

> *"raising nsubgrid past 1 will NOT fit (the corner family alone grows ~5.6x at
> nsubgrid=2) — re-derive this budget before touching nsubgrid."*

So the entire Phase-I problem reduces to: **the MHD working set at `nsubgrid=2`
exceeds the 48 KB hardware cap on *static* shared memory.** Nothing else blocks
it.

## 2. The byte budget (re-derived as instructed)

The model below reproduces the documented `nsubgrid=1` figure (37,072 B FP64)
*exactly*, so the `nsubgrid=2` projection is reliable. Counts are in `real`s
(nvar=8 ⇒ nscalar=3, nener=0); `refined` is `logical(kind=1)` = 1 B.

| Buffer | reals @ n=1 | reals @ n=2 | reals @ n=4 |
|---|---|---|---|
| `local_subgrid` (5 prim fields, `(2n+4)³`) | 1080 | 2560 | 8640 |
| `scalar_subgrid` (3 × `(2n+4)³`) | 648 | 1536 | 5184 |
| `bf` (face-B, trimmed) | 540 | 1344 | ~3700 |
| `emfx/y/z` | 54 | 300 | ~1700 |
| **`corner_x/y/z`** (4 slots × 7 fields) | 1512 | **8400** | **54432** |
| 6 interface arrays (11 fields) | 792 | 5280 | 38016 |
| `refined` (bytes) | 64 B | 216 B | 1000 B |
| **TOTAL reals** | **4626** | **19420** | **~111700** |

**FP64 (NPRE=8 — production):**

| | bytes/block | A100 (163 KB max dyn) | H100 (227 KB max dyn) |
|---|---|---|---|
| n=1 | 37 KB | 4 blocks/SM (static) | — |
| **n=2** | **152 KB** | **1 block/SM (dynamic)** | 1 block/SM (dynamic) |
| n=4 | **~870 KB** | **does not fit** | does not fit |

(FP32 footprints are ~half — n=2 ≈ 76 KB — but FP32 is not the production target
and is not pursued.)

**Conclusions:**
- `nsubgrid=2` MHD at FP64 is **152 KB**, over the 48 KB static cap but inside
  the A100 dynamic opt-in carveout (163 KB) at **1 block/SM**. → Phase I.
- `nsubgrid=4` MHD at FP64 is **~870 KB**, ~5× the largest carveout, dominated by
  `corner_*` (5.6×/oct growth) and the interface arrays. The fused kernel cannot
  reach it on any GPU. → requires Phase II's plane-tiled design.

The single Phase-I mechanism is therefore **dynamic shared memory + the larger
carveout opt-in**.

## 3. The minimal change

### 3a. Lift the MHD kernel's shared working set out of the static cap (the only hard part)

Static shared memory is capped at 48 KB on all targeted architectures regardless
of any attribute; the extra capacity (up to 163/227 KB) is *only* reachable as
**dynamic** shared memory, which additionally requires opting in via
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes)`.
There is no way around this for a 152 KB working set.

Recommended approach — **one dynamic-shared pool, typed overlay, accesses
unchanged**:

1. Define a single derived type aggregating *all* of the integrator's resident
   shared buffers as components, next to the existing subgrid types:

   ```fortran
   type :: mhd_smem_t
      type(subgrid_6x6x6cell_primitive) :: local_subgrid
      real(kind=dp) :: scalar_subgrid(0:2*nsubgrid+3,0:2*nsubgrid+3,0:2*nsubgrid+3,1:nscalar)
      type(subgrid_face_b_t)             :: bf
      type(subgrid_2x3x3edge_primitive)  :: emfx
      type(subgrid_3x2x3edge_primitive)  :: emfy
      type(subgrid_3x3x2edge_primitive)  :: emfz
      type(subgrid_2x3x3edge_corner)     :: corner_x
      type(subgrid_3x2x3edge_corner)     :: corner_y
      type(subgrid_3x3x2edge_corner)     :: corner_z
      type(subgrid_3x2x2cell_primitive)  :: left_interfaces_x, right_interfaces_x
      type(subgrid_2x3x2cell_primitive)  :: left_interfaces_y, right_interfaces_y
      type(subgrid_2x2x3cell_primitive)  :: left_interfaces_z, right_interfaces_z
   end type
   ```

2. In the kernel, replace the ~13 individual `… , shared :: …` declarations
   ([gpu/gpu_hydro.cuf:3776-3795](gpu/gpu_hydro.cuf#L3776)) with one dynamic
   shared pool overlaid by this type, plus `associate` so the **rest of the
   ~1500-line kernel body is untouched**:

   ```fortran
   real(kind=dp), shared :: dyn_pool(*)        ! size = 3rd chevron arg
   type(mhd_smem_t), pointer, shared :: S
   call c_f_pointer(c_loc(dyn_pool), S)        ! overlay — spike this first
   associate(local_subgrid => S%local_subgrid, bf => S%bf, &
             corner_x => S%corner_x, corner_y => S%corner_y, corner_z => S%corner_z, &
             emfx => S%emfx, emfy => S%emfy, emfz => S%emfz, &
             left_interfaces_x => S%left_interfaces_x, … )
      …  ! existing kernel body, verbatim
   end associate
   ```

   The associated names match the existing variable names, so every downstream
   `call subgrid_conserved_2_primitive(…)`, `call trace_3d(…)`, etc. compiles
   unchanged.

   > **Spike first (the only genuine unknown).** Confirm the NVHPC CUDA-Fortran
   > idiom for overlaying a derived type onto a single dynamic-shared allocation
   > on the actual `nvfortran`/`sm_80` toolchain *before* editing the kernel.
   > Write a ~30-line standalone kernel: declare `mhd_smem_t` over a dynamic
   > pool, write/read a few fields, diff against a static-shared control. If the
   > overlay is unsupported, fall back to **splitting static/dynamic**: keep
   > everything except `corner_*` static and put only `corner_x/y/z` in the
   > dynamic pool (still needs the carveout opt-in; touches only 3 buffers).

3. **Opt into the carveout, once**, at GPU init (or guarded first entry to
   `gpu_godunov`):

   ```fortran
   istat = cudaFuncSetAttribute(hydro_integrator_kernel, &
             cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes)
   ```

4. **Pass the dynamic size at launch** (3rd chevron arg) at
   [gpu/gpu_runner.cuf:379](gpu/gpu_runner.cuf#L379):

   ```fortran
   call hydro_integrator_kernel<<<num_blocks, threads_per_block, shmem_bytes>>>(…)
   ```

   with `shmem_bytes = storage_size(S)/8` (one host parameter, derived from the
   same `mhd_smem_t` so it can't drift from the declaration).

### 3b. Hard-code `nsubgrid = 2`

Change `nsubgrid = 1 → 2` in [gpu/gpu_utils.cuf:9](gpu/gpu_utils.cuf#L9). This is
a hard-coded source constant, **not** a CMake/build option. All derived
parameters (`nsubgridsq`, `nsubgridtondim`, `nsubgridp2`, `subgridsize`)
recompute automatically. Because `nsubgrid` is now `2` everywhere, the MHD
integrator's dynamic-shared path becomes the only path it compiles (the static
declarations would exceed 48 KB and fail to build), so there is no in-tree
`nsubgrid=1` MHD build to gate against — the `nsubgrid=1` baseline for
regression lives on `develop` (§4).

### 3c. Confirm thread count covers the larger lattice

`threads_per_block = 256` for `nsubgrid=2` is already set
([gpu/gpu_runner.cuf:378](gpu/gpu_runner.cuf#L378)). 256 ≥ every fixed partition
requirement — the largest is the 6-way `coarse_cell_update` partition needing
`6 * nsubgrid**2 = 24` threads ([gpu/gpu_hydro.cuf:2775](gpu/gpu_hydro.cuf#L2775));
everything else is a `blockDim%x`-stride loop. Add an assertion/comment that
`threads_per_block ≥ 6*nsubgrid**2`.

## 4. Why this is performant at FP64 (not just "fits")

The FP64 occupancy arithmetic is the key result, and it is favorable:

- **Resident threads/SM are unchanged.** On A100 with the 163 KB carveout:
  - n=1 (37 KB, static): **4 blocks/SM × 64 threads = 256 threads/SM**
  - n=2 (152 KB, dynamic): **1 block/SM × 256 threads = 256 threads/SM**

  Same 8 warps/SM, i.e. **no loss of occupancy** moving to `nsubgrid=2` —
  shared memory is the limiter in both cases and it lands on the same
  thread count. (Registers are not the limiter at 1 block/SM: 65536 regs / 256
  threads = 256 regs/thread available vs the ~96–102 the kernel uses.)
- **But `nsubgrid=2` does far less wasted work per real cell.** Useful-work
  fraction = interior octs / loaded lattice = `(2n)³/(2n+4)³`:
  - n=1: 8/216 = **3.7%**
  - n=2: 64/512 = **12.5%** → **3.4× fewer redundant ghost loads, traces, and
    Riemann/EMF solves per real cell.**

  Since the kernel is shared-traffic bound (NCU already flagged shared-store
  wavefronts as its #1 signal, per the budget comment), trading equal occupancy
  for 3.4× better halo amortization should be a net win. This **must be
  measured** (§7), but the FP64 numbers make `nsubgrid=2` a strictly better
  occupancy/efficiency trade than the FP32 case would have been.

## 5. "Unigrid" simplifies validation, not the footprint

Starting unigrid (`levelmin == levelmax`) is the right bring-up order because:

- `zero_fine_fluxes` / `zero_fine_emf` are skipped (`ilevel < levelmax` false,
  [gpu/gpu_hydro.cuf:3869](gpu/gpu_hydro.cuf#L3869)) → the `refined` path is inert.
- `coarse_cell_update` is skipped (`ilevel > levelmin` false,
  [gpu/gpu_hydro.cuf:3889](gpu/gpu_hydro.cuf#L3889)) → cross-level atomics and
  the 6-way coarse-flux partition don't execute.

So unigrid removes AMR correctness concerns from the first bring-up. It does
**not** shrink the dominant shared buffers (`corner_*`, interfaces, prim), so the
dynamic-shared change in §3a is required regardless; AMR `nsubgrid=2` is enabled
after unigrid is bitwise-validated.

## 6. Phase-I validation & rollout

1. **Spike** the dynamic-shared overlay idiom in isolation (§3a step 2) — the
   only piece with toolchain uncertainty; do it before touching the kernel.
2. **`nsubgrid=1` baseline**: capture reference outputs from a `develop` build
   (FP64 MHD) for the regression diffs below.
3. **Unigrid `nsubgrid=2` correctness**: run a unigrid MHD test
   (a namelist under [namelist/](namelist/) with `levelmin==nlevelmax`); results
   must agree with the `nsubgrid=1` baseline to round-off, and **∇·B must stay at
   machine zero** — the most sensitive indicator that the EMF/corner/face-B
   tiling is correct on the larger lattice.
4. **AMR `nsubgrid=2`** once unigrid passes: re-enable refinement and verify the
   `zero_fine_*` and `coarse_cell_update` paths against the baseline.
5. **Performance**: NCU A/B of `develop` (n=1) vs this branch (n=2), FP64/A100 —
   confirm the halo-amortization win at equal occupancy; record blocks/SM,
   achieved occupancy, ∇·B, and per-step wall time. Keep registers ≤102/thread;
   if a change pushes past, cap with `-gpu=maxregcount` on `gpu_hydro.o`.

## 7. Phase-I risks

- **Dynamic-shared overlay syntax** — mitigated by the spike; fallback is the
  `corner_*`-only static/dynamic split.
- **1 block/SM at FP64** — note this is *equal* resident-thread occupancy to n=1,
  not a regression (§4); still latency-sensitive, so measure.
- **`shmem_bytes` drift** — derive it from `storage_size(mhd_smem_t)` and assert
  `shmem_bytes ≤ device max carveout` at runtime.
- **Re-derive the budget comment** at [gpu/gpu_hydro.cuf:3677](gpu/gpu_hydro.cuf#L3677):
  it currently asserts `nsubgrid>1` "will NOT fit"; update to "fits only as
  dynamic shared with carveout opt-in; see this plan."

## Summary of Phase-I edits

| File | Change | Size |
|---|---|---|
| [gpu/gpu_utils.cuf:9](gpu/gpu_utils.cuf#L9) | hard-code `nsubgrid` 1→2 | 1 line |
| `gpu_hydro`/`gpu_utils` | add `mhd_smem_t` aggregate type | ~15 lines |
| [gpu/gpu_hydro.cuf:3776-3795](gpu/gpu_hydro.cuf#L3776) | static shared decls → one dynamic pool + typed overlay + `associate` | ~20 lines, body unchanged |
| GPU init | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` | ~3 lines |
| [gpu/gpu_runner.cuf:379](gpu/gpu_runner.cuf#L379) | add 3rd chevron `shmem_bytes` arg | 1 line |
| budget comment | re-derive for `nsubgrid=2` dynamic-shared | doc |

Everything else — decode loops, neighbour stencil, thread counts, Riemann and CT
solvers — is already `nsubgrid`-generic and needs no change.

---

# PHASE II — `nsubgrid = 4` via a plane-tiled MHD kernel

`nsubgrid=4` MHD needs ~870 KB of resident shared memory in the fused design
(§2) — ~5× the largest GPU carveout, dominated by `corner_*` and the interface
arrays, both of which scale as the *volume* `(2n+4)³`. No carveout trick reaches
it. The footprint must be reduced from a volume to a **plane**.

## 8. The template already exists: the `PAPER` kernel

`hydro_integrator_paper_kernel` ([gpu/gpu_hydro.cuf:5057](gpu/gpu_hydro.cuf#L5057))
is a plane-tiled **HD** integrator: it sweeps `iz = 0 … m+3` keeping only a
rolling 3-plane window (`im`/`ic`/`ip` slots rotating through `q_storage`,
[gpu/gpu_hydro.cuf:5139-5145](gpu/gpu_hydro.cuf#L5139)), so its shared cost scales
as `(2n+4)²`, not `(2n+4)³`. Its own footprint table
([gpu/gpu_hydro.cuf:5106-5111](gpu/gpu_hydro.cuf#L5106)):

```
m = 2*nsubgrid       FP32     FP64
m = 8 (nsubgrid=4)   22 KB    44 KB
m =16 (nsubgrid=8)   68 KB   135 KB
```

`nsubgrid=4` HD fits comfortably even at FP64. And the kernel already carries the
matching FIXME ([gpu/gpu_hydro.cuf:5128-5131](gpu/gpu_hydro.cuf#L5128)):

> *"if nsubgrid is large (=>4) and this is an MHD kernel … tile the ix-iy plane
> as well … minimise shared memory usage."*

It is launched under `#ifdef PAPER` ([gpu/gpu_runner.cuf:369-374](gpu/gpu_runner.cuf#L369)),
with `threads_per_block = ((2*nsubgrid+4)² rounded to 32)` — i.e. one thread per
plane cell, each sweeping its z-column. The Phase-II task is to add CT/MHD to
this kernel under `#if defined(PAPER) && defined(MHD)` and run it at
`nsubgrid=4`.

## 9. What MHD adds to the plane sweep

The challenge is that constrained-transport couples the z-direction across
planes, so the rolling window must carry more than HD's 3 conserved-variable
planes. Per the CPU CT scheme already mirrored by the fused kernel:

1. **Resident planes.** Extend the rolling window to carry, for each of the few
   resident z-slots: the primitive plane, the face-centred B planes
   (`bf`), and the edge-EMF planes. Cell-centred B and the transverse face-B
   slopes are **recomputed inline** from the resident `bf` (exactly as the fused
   kernel does, [gpu/gpu_hydro.cuf:3692-3701](gpu/gpu_hydro.cuf#L3692)) — never
   stored — which is what keeps the plane footprint small.
2. **EMF z-coupling — the crux.** The CT update of each face-B is a curl of E:
   e.g. `∂Bx/∂t = ∂Ez/∂y − ∂Ey/∂z`. The `∂/∂z` terms couple plane `k` with plane
   `k−1`, so the sweep must **retain the previous plane's `Ey`/`Ex` edge values**
   (a 2-deep EMF history) to apply the update when it advances. `Ez` (edges
   parallel to z) is local to a plane column. Lay the window out so that when the
   sweep advances from `k−1` to `k`, both planes' transverse EMFs are available
   for the face-B curl on the face between them.
3. **2D corner Riemann → EMF**, per plane. Compute the four edge-corner states
   on the current plane's edges and feed `cmp_mag_flx_mhd`-style 2D solves to
   produce that plane's EMFs. Because corners are consumed immediately to make
   EMFs and not kept, the `corner_*` volume buffers (the 5.6×/oct grower that
   sinks the fused design) shrink to per-plane scratch — this is the entire
   reason plane-tiling reaches `nsubgrid=4`.
4. **Boundary stages.** `zero_fine_emf` and the coarse-level EMF/flux corrections
   must be applied plane-by-plane as the sweep crosses each level boundary;
   unigrid (Phase II bring-up) skips both, as in Phase I §5.

Estimated `nsubgrid=4` MHD FP64 footprint: HD's 44 KB plus the B/EMF/corner
plane scratch and history (order of HD again) → **~80–100 KB/block** — fits the
A100 carveout with room for ≥1 block/SM, and far better useful-work fraction than
n=2 (`8³/12³ = 38%`). To be confirmed once the buffer set is fixed.

## 10. Phase-II validation & rollout

1. **HD parity first.** With the MHD additions `#ifdef`-gated off, confirm the
   `PAPER` HD kernel still matches the fused kernel at `nsubgrid=2` (no
   regression to the existing plane-tiled path).
2. **MHD plane kernel at `nsubgrid=2`**, unigrid: must agree to round-off with
   the Phase-I fused result, **∇·B at machine zero**. This is the real
   correctness gate — same physics, two memory layouts.
3. **`nsubgrid=4`**, unigrid then AMR: agreement with `nsubgrid=2`, ∇·B zero,
   and NCU confirmation that the plane-tiled footprint stays within the carveout
   at the expected occupancy.
4. **Performance**: compare `nsubgrid=4` plane-tiled vs `nsubgrid=2` fused on the
   same problem; expect the higher useful-work fraction to win despite the extra
   inline recompute.

## 11. Phase-II risks

- **The z-coupled EMF/CT bookkeeping** (§9.2) is the hard, error-prone part;
  ∇·B = 0 is unforgiving of any window-indexing mistake. Budget the bulk of the
  effort here, and validate ∇·B at every step.
- **New kernel, large surface area** — this is a multi-week effort, not a
  surgical edit; keep it entirely behind `#if defined(PAPER) && defined(MHD)` so
  Phase I (and HD `PAPER`) are never perturbed.
- **Inline recompute cost** — plane-tiling trades shared memory for recomputing
  cell-B and face-B slopes every plane; at FP64 this is the right trade, but
  confirm it doesn't become compute-bound.

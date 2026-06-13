# MPI for the GPU path in mini-RAMSES

**Branch:** base off `gpu_mhd`, copy into a new `gpu_mpi` branch and do all work there.
**Goal:** make the GPU-resident code path (hydro/MHD, Poisson/multigrid, gravity, refine/flag, rho/CIC, particles, cooling, star formation) correct under `ncpu > 1`, mirroring the CPU MPI philosophy, while keeping the existing kernels as intact as possible.
**Thesis:** the GPU kernels are already *rank-oblivious by construction*. The entire job is to (a) populate the device's existing **cache-oct region** with real remote ("ghost") data before kernels run, (b) flush scatter/restriction contributions back to owners afterward, and (c) migrate particles between ranks. We do **not** rewrite the integrators.

---

## 1. Background: the CPU philosophy we are mirroring

mini-RAMSES has **no static ghost-zone arrays** and no `make_virtual_fine`/`make_reverse_fine`. It uses a **reactive software-cache (one-sided-RPC) model** built from two independent subsystems that both ride on MPI but do different jobs:

- **MDL layer** (`mdl1/mdl.f90`, `amr/task_manager.f90`, `amr/mdl_commons.f90`) — a master-driven RPC tree that runs a phase on every rank and reduces scalars on the way back up. `mdl_send_request(...,MDL_GODUNOV_FINE,...)` fans a phase down a binary tree of ranks; `mdl_get_reply` collects/reduces. This is "run this everywhere + reduce."
- **Cache layer** (`amr/cache.f90`, `amr/nbors_utils.f90:get_grid`, `amr/task_manager.f90:check_mail`/`destage`) — moves *boundary oct data* between ranks on demand. This is "fetch the neighbor octs I don't own."

**Domain decomposition** is a per-level Hilbert space-filling curve. Each level's ownership is `m%domain(ilevel)%b(1:nhilbert, 0:ncpu)`; rank `i` owns the half-open Hilbert interval `[b(:,i-1), b(:,i))`. Ownership queries: `domain_t%in_rank(key)` and `domain_t%get_rank(key)` (binary-search `hunt`, `amr/domain_hilbert.f90:32/56/269`). There is **no `cpu_map` array**. All key comparisons go through the multi-limb helpers `ge_keys/gt_keys/eq_keys` (`amr/hilbert.f90:405+`) — never compare Hilbert keys with raw `<`.

**Ghost vs halo (confirmed meanings in this codebase):**
- **Ghost oct** = a remote oct **fetched (read)** from its owner into your cache region. Direction owner → you. `get_grid(..., fetch_cache=.true.)`.
- **Halo oct** = a cache slot you **write contributions into**, flushed back to the owner who merges them with a combiner (`+=` or `MAX`). Direction you → owner. `get_grid(..., flush_cache=.true.)`, merged on receipt by `unpack_flush_*`.

The grid array is one contiguous block with two regions (`amr/amr_commons.f90:683-783`):

| Region | Index range | Meaning |
|---|---|---|
| Real / owned octs | `1 : ngridmax` | Octs this rank owns, packed per level via `head/tail/noct`. |
| **Cache octs** | `ngridmax+1 : ngridmax+ncachemax` | Transient remote octs (ghosts) **and** write-staging slots (halos). |

Cache slots are tracked by parallel arrays indexed `icache = igrid - ngridmax`: `occupied`, `dirty` (pending flush), `locked` (pinned for current stencil), and crucially **`parent_cpu(icache)`** = the owner rank (the flush target / fetch source). `clean_dirty` (`amr/refine_utils.f90:1004`) pre-classifies owned octs into `indx_clean` (all 26 neighbors local — no comm) vs **`indx_dirty`** (≥1 neighbor remote or at another level — needs cache). **`indx_dirty` is exactly the boundary-oct list our GPU map builder will iterate.**

**Two structural rules cover every physics module** (verified against the code):

> **Rule A — stencil operators FETCH ghosts** (read neighbors before compute): pass `pack`/`unpack`, use `fetch_cache=.true.`.
> **Rule B — scatter/restriction operators RETURN halos** (sum contributions to owner after compute): pass `init`/`flush`/`combine`, combiner is `+=` (or `MAX` for flags).

Per-module exchange sites on the CPU (for reference when wiring the GPU):

| Module / routine | Direction | Payload |
|---|---|---|
| Hydro Godunov `godfine1` (`opt/godunov_fine.f90:53`) | **read-write** | fetch `uold`/`bold` 3³ stencil → return coarse-fine reflux (`+=`) |
| `source_hydro_fine` (`hydro/source_hydro_fine.f90:93`) | fetch | neighbor `uold` (divu/pdV) |
| Hydro upload/restriction (`hydro/upload.f90:129`) | return halos | `uold`/`bold`/`mflux` fine→coarse (`+=`) |
| Poisson CG `cmp_residual`/`cmp_Ap` (`poisson/phi_fine_cg.f90:314/452`) | fetch | neighbor `phi`, search dir `p` |
| MG smooth/residual (`multigrid_fine_coarse.f90:293/535`) | fetch | neighbor `phi`+mask each sweep |
| MG restrict mask/residual (`…:104/761`) | return halos | mask & residual fine→coarse (`+=`) |
| MG `build_mg` (`multigrid_fine_commons.f90:431`) | return (CREATE) | build distributed coarse hash |
| Force `gradient_phi` (`poisson/force_fine.f90:260`) | fetch | neighbor `phi` → `f` |
| Refine create/destroy (`amr/refine_utils.f90:201/278`) | read-write | fetch parent state → return child data |
| Flag `init_flag` (`amr/flag_utils.f90:144`) | return halos | `flag1` fine→coarse (**MAX**) |
| Flag `smooth_fine` (`amr/smooth.f90:107`) | fetch | neighbor `flag1` (per dim) |
| Rho gas `cic_multipole` (`pm/rho_fine.f90:558`) | return halos | gas mass → `rho`/`nref` (`+=`) |
| Rho particles `cic_part` (`pm/rho_fine.f90:793`) | **return halos** | particle mass → `rho`/`nref` (`+=`) |
| Particle kick/drift `move_fine` (`pm/move_fine.f90:254`) | fetch | force `f`/`phi` |
| Particle **migration** `balance_part` (`amr/load_balance.f90:828`) | **all-to-all move** | every particle field; only every `nremap` steps |

---

## 2. Background: how the GPU path works today (and why it is single-rank)

- **Whole-run residency.** `r_set_grid_device` (`gpu/gpu_manager.f90:11`) copies the *entire local mesh* (`grid`, `uold`, `bold`, `flag1`, particles, stars) host→device once, builds the device hash (`insert_hash_kernel`), and seeds the neighbor array (`update_nbor_array`) for `levelmin`. The mesh then lives on-device for the whole run; `r_transfer_grid_host` (`:106`) copies back **only at I/O/restart** (`amr/amr_step.f90:113/125/134`). An MPI port must keep this residency — halo exchange must be device-side or device-staged, never a full mesh round-trip.
- **The "rock"/cube kernel is the consume point.** `hydro_integrator_kernel` (`gpu/gpu_hydro.cuf:3745`, launched from `gpu_godunov`, `gpu/gpu_runner.cuf:324/379`) runs one block per oct, builds a 6×6×6-cell shared subgrid (inner 2×2×2 real + 2-cell ghost layer) by gathering the **27-oct stencil** from the precomputed `nbor(1:27, oct)` array (center = index 14). The gather (`subgrid_conserved_2_primitive`, `gpu/gpu_hydro.cuf:1251-1455`, the `nbor(...)` read at `:1324`) does **no bounds check, no wrap, no off-rank logic**. All boundary semantics are *baked into the values stored in `nbor`* before launch. **We never touch the integrator.**
- **`nbor` is the produce point.** `update_nbor_array` (`gpu/gpu_refine.cuf:1933-2008`) fills each of the 27 slots: compute the neighbor Cartesian key, **periodic-wrap against the GLOBAL box** `box_ckey_min/max(idim,ilevel)` (`:1980-1985`), `hash_get` it (`:1996`); on a **miss (returns 0)** flag the slot, and `make_cache_octs` (`gpu/gpu_refine.cuf:2191`) allocates a cache oct at `ngridmax+…` and **D4-prolongs the coarse parent** into it. Today a miss can *only* mean "coarser here" (single rank), so it is always interpolated. Multi-level is fully handled — `gpu_refine` rebuilds `nbor` for all finer levels after each refinement (`gpu/gpu_refine.cuf:926-948`).
- **Scalar reductions already work.** GPU drivers run at MDL-tree leaves (e.g. `gpu_cmpdt`→`cmpdt_kernel` block-reduces into `data_out`, `gpu/gpu_hydro.cuf:4690-4702`); cross-rank min/sum happens in the MDL reply combination on the way up the tree (same path the CPU uses). Courant `dt`, `epot`, `rhomax`, residual norm, `mstar` should require **no new code** — *but verify this early* (Milestone 0).
- **Dead-but-reusable scaffolding.** `gpu/gpu_mpi.cuf` already sketches the right design — `setup_ghost_mapping` (`:283`, bins `m%locked`/`m%parent_cpu` cache slots per rank), `setup_halo_mapping` (`:362`, `MPI_ALLTOALL` of counts + key exchange), `halo_exchange_uold` (`:429`, `IRECV/ISEND/WAITALL` over flat host buffers), `full_cache_prefetch` (`:545`, front-loads all fetches over `indx_dirty`). It is gated behind `if(.false.)` (`:73`) and calls a stale kernel signature. The device pack/scatter kernels `gather_send_buffer`/`scatter_recv_buffer` (`gpu/gpu_hydro.cuf:4708/4743`) are generic and index-list-driven — **directly reusable**.

### Single-rank assumptions to remove (with locations)

1. **Periodic wrap is global-box, single-rank.** `box_ckey_min/max` + `periodic` are the *global* box, copied H→D once (`amr/init_amr.f90:540-541`). Every wrap site assumes the wrapped key resolves to a *local* oct: hydro `update_nbor_array` (`gpu/gpu_refine.cuf:1980`), `make_cache_octs` (`:2268`), refine prolong (`:585`), multigrid (`gpu/gpu_mg.cuf:1236`), particle CIC (`gpu/gpu_part.cuf`).
2. **Device hash holds only local octs.** `hash_get` miss ⇒ "coarser here," never "remote" (`gpu/gpu_refine.cuf:1996`).
3. **`father` assumed always present.** `gpu/gpu_refine.cuf:914` comment "they should always exist without MPI" — under MPI a father can be off-rank.
4. **Particle kernels silently drop off-rank work.** Force gather skips `igrid > ngridmax` (`gpu/gpu_part.cuf:2096-2099`); CIC deposit suppresses the atomic when `hash_get==0` (`:1048-1051,1116-1118,1163`). No particle migration exists anywhere on the GPU.
5. **Load balance is CPU/host-only and incompatible with residency.** `balance_part` (`amr/load_balance.f90:828`) permutes *host* arrays and never pushes back to device; the GPU has never exercised `mdl_threads>1`.

---

## 3. Core design: device-staged ghost/halo exchange

Mirror the CPU's `open_cache → full_cache_prefetch → compute → close_cache` cycle, but **batched** (the CPU's per-oct `get_grid`/`check_mail` polling is fatal on a GPU). Split the work into an **occasional topology/mapping step** and a **per-phase data refresh**, exactly the split `gpu_mpi.cuf` intended.

### 3a. Topology step — rebuild ghost/halo maps (after refine and after load balance only)

This is the GPU analogue of `full_cache_prefetch` + `setup_ghost_mapping` + `setup_halo_mapping`. Runs **per level**, on the host, using the CPU's existing domain-decomposition machinery (which is already correct and available):

1. Iterate the boundary octs (`m%indx_dirty(ilevel)` from `clean_dirty`). For each, compute its 26 neighbor keys (reusing the same wrap logic). For each neighbor key, `m%domain(ilevel)%get_rank(hk)`:
   - owner == me, exists locally → real oct, nothing to do;
   - owner != me → **ghost** to receive: record `(local cache-slot, key, owner rank)`. Bin into `n_recv(owner)`.
2. **`MPI_ALLTOALL(n_recv → n_send)`** to learn how many octs each peer needs *from me* (`setup_halo_mapping`); exchange the requested keys; resolve each to a local real-oct index via the device hash → `idx_send` (which of *my* octs to gather and send).
3. Allocate device cache slots for all ghost octs `[ngridmax+1 …]`, set `parent_cpu`, and **insert ghost keys into the device hash** (`insert_hash_kernel` on the ghost key list).
4. **Rebuild `nbor` (existing `update_nbor_array`, unchanged).** Because ghost keys are now in the hash, `hash_get` *finds* same-level remote neighbors as real cache-oct indices. The miss→interpolate path then fires **only** for genuine coarser neighbors — exactly the single-rank behavior, now correct under MPI. **This is the crucial "keep code intact" move: rank-awareness lives in the map builder, not in `update_nbor_array` or any kernel.**

Outputs (per level, persistent until next topology change): `idx_recv[]` (ghost cache-slot indices, grouped by source rank, counts `n_recv`), `idx_send[]` (local real-oct indices, grouped by dest rank, counts `n_send`), and the ghost octs registered in hash + `nbor`.

### 3b. Per-phase data refresh — the exchange itself

Field-agnostic, reusing `gather_send_buffer`/`scatter_recv_buffer` (generalized to arbitrary component count `ncomp` and to `bold`). Two directions, **same index maps**:

**FETCH ghosts** (Rule A — hydro, Poisson, force, flag-smooth):
```
gather (idx_send → send_buf)  [device kernel]
   D→H send_buf               [or CUDA-aware MPI: skip]
   IRECV/ISEND/WAITALL        [host, grouped by rank, counts n_send/n_recv]
   H→D recv_buf
scatter (recv_buf → idx_recv ghost slots, OVERWRITE)  [device kernel]
```

**RETURN halos** (Rule B — rho/CIC deposit, hydro reflux, upload restriction, flag MAX): the **dual** — roles of send/recv swap, and scatter uses a **combiner** instead of overwrite:
```
gather (idx_recv ghost/halo slots → send_buf)   [contributions I computed into remote octs]
   exchange (IRECV/ISEND/WAITALL, swapped)
scatter into idx_send real octs with atomicAdd  (or atomicMax for flag1)   [device kernel]
```

This `fetch = make_virtual` / `return = make_reverse` duality over one set of maps is the heart of the port. **Payload buffers carry whatever the phase needs**, parameterized by field + `ncomp`:
- hydro/MHD: `uold` (`twotondim*nvar`) **and** `bold` (`twotondim*6`) — mirror the host `#ifdef MHD` already in `gpu_manager.f90:35-39`; per oct = `twotondim*(nvar+6)` doubles under MHD (replace the hard-coded `40` in `gpu_mpi.cuf:325/396`).
- Poisson/force: `phi` (1/cell), then `f` (`ndim`/cell).
- rho deposit (return): `rho` + `nref`.
- flag (return, MAX): `flag1` (integer; `atomicMax` scatter).

Use one tag per field/direction. Prefer **CUDA-aware MPI** (pass device pointers straight to `MPI_Isend/Irecv`) to drop the D↔H copies; keep host-staging (as in `gpu_mpi.cuf`) as the portable fallback behind a macro. Respect `#if NPRE==4` for `MPI_REAL` vs `MPI_DOUBLE_PRECISION` (already in `gpu_mpi.cuf:456-459`) — note this couples to the single-precision+fastmath work.

### 3c. Wiring into the existing `r_*` drivers

Each `r_*` wrapper's GPU leaf gets a fetch *before* and/or a return *after* its kernel launch, e.g. in `r_godunov_fine`'s leaf (`opt/godunov_fine.f90:27`):
```
#ifdef _CUDA
   call gpu_halo_fetch(s, ilevel, FIELD_HYDRO)   ! uold+bold ghosts
   call gpu_godunov(s, ilevel)
   if (ilevel > levelmin) call gpu_halo_return(s, ilevel, FIELD_FLUX)  ! coarse-fine reflux
#endif
```
The MDL fan-out/reduce around it is untouched. The phase→direction→field mapping is precisely §1's table.

---

## 4. The hard part: particles

Particles **drift between rank domains**, so beyond the deposit/gather halos they need genuine **migration** (variable-count, all-fields). Split into two buckets.

### Bucket A — deposit/gather halos (same shape as hydro; do first)
- **CIC mass deposit (return halo).** Today the GPU drops off-rank atomics (`gpu/gpu_part.cuf:1163` etc.). Instead, when a deposit lands on a ghost/cache oct, accumulate it there (atomicAdd into the cache slot's `rho`/`nref`) and **return-halo** it to the owner with the `+=` combiner — the GPU analogue of `unpack_flush_rho` (`pm/rho_fine.f90:1185`).
- **Force/phi read (fetch halo).** `gather_cic_force_part` already *reads* ghost octs but skips them because their `f`/`phi` is stale (`gpu/gpu_part.cuf:2096`). Once §3 populates ghost-oct `f`/`phi`, **delete the skip** — no particle code moves.

### Bucket B — particle migration between domains (genuinely new)
GPU equivalent of `balance_part` (`amr/load_balance.f90:828-1903`): each particle belongs to the rank owning its **parent-grid Hilbert key** (`domain_part%get_rank`, `:1056`).

**Cadence decision (important).** The CPU migrates only every `nremap` (=10) coarse steps and tolerates far-drifted particles because its read/write cache can fetch *arbitrarily distant* octs on demand. **The GPU has no on-demand fetch — only a fixed 1-oct-deep halo.** A particle sitting in another rank's territory has a parent oct that is a ghost; its CIC stencil then reaches octs up to 2 deep, beyond the halo. Two ways out:

- **(Recommended) Migrate every coarse step.** Keep every particle's parent oct *locally owned*, so its CIC stencil stays within the standard 1-deep halo and Bucket A suffices unchanged. More frequent than the CPU, but it is the safest mirror, keeps the halo shallow, and the per-step cost is small relative to a full hydro+gravity step. Diverges from CPU cadence by design — document it.
- (Alternative) Keep `nremap` cadence but build a 2-deep particle halo (deeper ghost layer for the CIC region). More code, more memory, harder to get right. Defer.

**Migration mechanics** (mirror `balance_part`, on device where possible):
1. After kick-drift, compute each particle's destination rank from its parent-grid Hilbert key (device kernel).
2. Count per destination (device reduction) → `MPI_ALLTOALL` to recv counts.
3. **Compact** leaving particles to the array tail via a `gpu_scan` prefix-sum + gather/scatter of **all fields** (`xp,vp,mp,levelp,idp`, plus `tp,zp` for stars) — reuse the existing `gpu_split_part` swap machinery (`gpu/gpu_part.cuf:420-552`).
4. Variable-count `MPI_IRECV/ISEND/WAITALL` per field (CUDA-aware preferred; else host-staged, matching the per-field loop in `balance_part:1248-1873`).
5. Append received particles; rebuild host `headp/tailp/npart` and re-`gpu_sort_part`. (`headp/tailp` already live on the host in the hybrid model — `gpu/gpu_part.cuf`.)

---

## 5. Load balancing

`balance_part` and the grid load-balance (`amr/load_balance.f90`, recompute `b`, radix-sort + migrate octs) are host-array operations incompatible with GPU residency. **Phase-1 approach (simple, correct, amortized over `nremap`):** at a remap boundary, `r_transfer_grid_host` (D→H) → run the existing CPU `balance_part` + grid migration unchanged → `r_set_grid_device` (H→D) → rebuild hash/`nbor`/ghost-halo maps (§3a). This reuses *all* the battle-tested CPU balancing code at the cost of one D↔H round-trip every `nremap` steps. A device-native balance is a later optimization, not needed for correctness.

---

## 6. Milestones (each independently testable)

- **M0 — Reductions & harness.** Confirm scalar reductions (Courant `dt`, `epot`, `rhomax`, MG residual, `mstar`) already reduce correctly across ranks via MDL with GPU leaves. Stand up a 2-rank vs 1-rank comparison and CPU-vs-GPU bitwise/`tol` diff harness (reuse the existing validation harness from the NPRE=4/fastmath and kick-coop work). *Gate: a non-exchanging quantity matches across rank counts.*
- **M1 — Hydro ghost fetch (HD).** Implement §3a map build + §3b FETCH for `uold`; revive/rewire `gpu_mpi.cuf` (remove `if(.false.)`, fix kernel signature, generalize buffer to `twotondim*nvar`). *Gate: 2-rank HD sod/blast matches 1-rank.*
- **M2 — MHD.** Add `bold` (`twotondim*6`) to the hydro buffers + a parallel gather/scatter; add the coarse-fine **reflux return-halo** at rank+level boundaries. *Gate: 2-rank MHD matches 1-rank; ∇·B preserved across rank faces.*
- **M3 — Gravity.** FETCH `phi` (MG smooth/residual each sweep, prolongation), RETURN-halo MG restrict (mask+residual `+=`), FETCH `f` for `gradient_phi`. Handle `build_mg`'s distributed coarse hash (COMBINER_CREATE) — the one place octs are *created* by a flush. *Gate: 2-rank Poisson residual converges identically.*
- **M4 — Refine/flag.** RETURN-halo `flag1` (MAX), FETCH flag-smooth neighbors; make `gpu_refine`'s `father` rebuild rank-aware (`gpu/gpu_refine.cuf:914`); rebuild maps after every refine. *Gate: 2-rank refinement map == 1-rank.*
- **M5 — Particles A.** Ghost-oct `rho`/`nref` deposit return-halo + remove the force-gather skip. *Gate: 2-rank CIC density field == 1-rank.*
- **M6 — Particles B.** Per-step particle migration (§4 Bucket B). *Gate: N-body Zel'dovich/cosmo IC conserves momentum and matches 1-rank.*
- **M7 — Load balance.** D→H / balance / H→D + map rebuild (§5). *Gate: a multi-step cosmo run with `nremap` remaps matches the single-rank GPU run within tolerance, then the cluster A100 cosmo run from the NPRE=4 verification.*

---

## 7. File-by-file change map

- **`gpu/gpu_mpi.cuf`** — promote from dead scaffold to the real exchange module: remove `if(.false.)` (`:73`), fix the `hydro_integrator_kernel` call to the current 5-arg signature, generalize buffers (`nvar+6`, not `40`), add `bold`/`phi`/`f`/`rho`/`nref`/`flag1` variants and the FETCH/RETURN duality (§3b). Keep `setup_ghost_mapping`/`setup_halo_mapping`/`halo_exchange_*`/`full_cache_prefetch` as the backbone.
- **`gpu/gpu_hydro.cuf`** — generalize `gather_send_buffer`/`scatter_recv_buffer` (`:4708/4743`) to `ncomp` + add `bold` gather/scatter and an `atomicAdd`/`atomicMax` scatter for returns. Integrator untouched.
- **`gpu/gpu_refine.cuf`** — ensure ghost keys are inserted into the device hash *before* `update_nbor_array`; make the `father` rebuild rank-aware (`:914`). `update_nbor_array` itself stays as-is.
- **`gpu/gpu_part.cuf`** — ghost-oct deposit accumulation + return; remove force-gather skip (`:2096`); add migration (count/compact/exchange/rebuild) reusing `gpu_split_part`/`gpu_sort_part` machinery.
- **`gpu/gpu_runner.cuf`** — host-side exchange buffers/maps already declared (`:84-99`); wire `gpu_halo_fetch`/`gpu_halo_return` calls.
- **`gpu/gpu_manager.f90`** — already does whole-mesh H↔D; add map (re)build after H→D in the load-balance path.
- **`r_*` driver leaves** (`opt/godunov_fine.f90`, `poisson/*`, `amr/refine_utils.f90`, `amr/flag_utils.f90`, `pm/rho_fine.f90`, `pm/move_fine.f90`, `multigrid_fine_*.f90`) — add fetch-before/return-after around each `gpu_*` call per §1's table. MDL fan-out unchanged.
- **No change** to `hydro_integrator_kernel` or any compute kernel — the rock kernel stays rank-oblivious.

---

## 8. Risks & open questions

1. **Particle migration cadence** (§4) — the central correctness subtlety. Recommend per-step migration; validate that no particle drifts beyond the 1-deep halo within one coarse step.
2. **Coarse-fine *and* rank boundary coinciding** — reflux/restriction return-halos must compose with the existing in-kernel coarse-fine fixups (`coarse_cell_update`). Test a refined patch straddling a rank face explicitly (M2/M4).
3. **`build_mg` COMBINER_CREATE** — the only flush that *creates* octs on the receiver; needs a device-hash insert during a return. Isolated to M3.
4. **CUDA-aware MPI availability** on the target cluster — keep host-staging fallback behind a macro so M1 isn't blocked.
5. **Map-rebuild frequency** — rebuilding ghost/halo maps every refine could dominate if refinement is frequent; measure, and rebuild only affected levels.
6. **`nhilbert > 1`** (deep AMR) — all destination-rank math must use the multi-limb key helpers, not raw int64 compares.
7. **Reductions assumption (M0)** — if any GPU reduction is *not* already MDL-reduced across ranks, that is extra (small) scope surfaced before any boundary work.

---

## 9. One-paragraph summary

The GPU integrators already consume neighbor data exclusively through the precomputed `nbor` array and produce nothing rank-specific, so the MPI port never touches them. We mirror the CPU's cache layer in batched form: a per-level **topology step** (host-side, using the existing Hilbert `get_rank` + `indx_dirty` machinery) registers remote **ghost** octs into the device cache region and hash so the unchanged `update_nbor_array` resolves them as ordinary neighbors; a per-phase **data refresh** moves field payloads with a `fetch`/`return` duality over one set of index maps (FETCH overwrites ghosts before stencil ops; RETURN `atomicAdd`/`atomicMax` flushes contributions to owners after scatter/restriction ops), reviving the dead `gpu_mpi.cuf` plumbing and the generic `gather/scatter` kernels. Particles get the same deposit/gather halos plus genuine per-step **migration** (the one variable-count, all-fields exchange). Load balancing reuses the CPU path via a D↔H round-trip at remap boundaries. Scalar reductions already flow through MDL. The work is staged M0→M7, each gated by a 2-rank-vs-1-rank (and CPU-vs-GPU) comparison.

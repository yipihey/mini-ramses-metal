# mini-ramses-dev — agent guide

Single source of truth for Claude Code and Cursor. Symlink: `CLAUDE.md` → `AGENTS.md`.

Obsidian vault: `/Users/moseley/Documents/Obsidian Vault/` — when Cursor MCP is loaded, use **obsidian** MCP tools (`obsidian_read_note`, `obsidian_search_notes`, etc.) in addition to direct file reads.

---

## Session start (mandatory)

Before editing code or suggesting cluster runs:

1. Read [[Home]] — vault orientation and recent runs.
2. Read [[mini-ramses — hub]] — project purpose, tracks, key files.
3. Read [[mini-ramses — debug-playbook]] — **especially before any cluster submit.** Ranked failure modes; verify fixes in fresh `run.log`.
4. Read the target cluster runbook: [[runbook — Stellar]] (A100) or [[runbook — Marlowe]] (H100).
5. Skim [[Vault setup & AI wiring]] if vault/MCP access is unclear. With the Obsidian MCP server enabled in Cursor, use its tools (`obsidian_read_note`, `obsidian_search_notes`, etc.) to read vault notes directly.
6. Check latest [[session-handoff]] or `30-runs/` notes if continuing prior work.

Do not assume prior chat context. The vault holds failure modes that are not obvious from the repo alone.

---

## Project overview

**mini-ramses-dev** is a streamlined, **GPU-first fork of RAMSES** (upstream `rteyssie`). Hot paths (PM gravity, hydro/MHD) run on the GPU via CUDA Fortran (`.cuf`).

- **Runtime today:** single-rank GPU, compiled `WITHOUTMPI` (`MPI=0` in Makefiles).
- **MPI:** planned; design may live outside this tree (check for `mpi_gpu_plan.md` on active branches).
- **Upstream remote:** `rteyssie` → `develop`.

### Active physics tracks

| Track | Branch / area | Notes |
| --- | --- | --- |
| GPU constrained-transport MHD | `gpu_mhd` | ∇·B control; **`nsubgrid=1` only** (shared-memory limit) |
| Particle PM / CIC deposition | `gpu_part`, `develop` | `rho` / CIC dominates GPU time |
| CPU↔GPU DMO parity | harness `dmo-cpu` vs `dmo-gpu` | Cosmological dark-matter-only |
| GPU turbulence | `gpu_turb` | Host FFT + device interpolation (`gpu_turb.cuf`) |
| Tracer flux advection | `mini-ramses-dev-2` | `TRCFLX=1`; see [[tracers — hub]] |

---

## Repos and branches

| Path | Role |
| --- | --- |
| `~/ramses-development/mini-ramses-dev` | **This repo** — main GPU tree |
| `~/ramses-development/mini-ramses-dev-2` | Tracer-particle / `TRCFLX` work |

**Branches (local + origin):**

- **`gpu_mhd`** — GPU MHD (expected for Brio–Wu, Orszag–Tang, `ot-amr`)
- **`develop`** — integration / DMO+cosmo harness defaults
- **`gpu_turb`** — driven turbulence on GPU
- **`new_edits`** — in-flight edits
- Many feature branches (`gpu_part`, `kick_drift_edits`, tracer variants, …)

Harness expects `MINIRAM_EXPECTED_BRANCH=gpu_mhd` for MHD tests. Override with `MINIRAM=` if your checkout path differs from cluster default `~/mini-ramses-dev`.

---

## Clusters and harness (canonical)

Harness scripts live **outside the repo** (synced to home on each cluster). Source of truth on disk: `~/hackathon/` (Stellar) and `~/marlowe/hackathon/` (Marlowe).

| Cluster | GPU | Harness | Makefile | Scratch (typical) |
| --- | --- | --- | --- | --- |
| Stellar (`stellar.princeton.edu`) | A100 | `~/hackathon/` | `Makefile.a100` (`CUDA_ARCH=sm_80`) | `/scratch/gpfs/moseley/hackathon/` |
| Marlowe | H100 | `~/marlowe/hackathon/` | `Makefile.h100` (`CUDA_ARCH=sm_90`) | Marlowe scratch (mirror layout) |

Shared helpers: `hackathon_common.sh`, `hackathon_bootstrap.sh`, `CLUSTER.md`, `submit_profiles.sh`.

### Environment (both clusters)

```bash
module load nvhpc          # note exact version; must match r3d/nvomp build
source ~/hackathon/hackathon_common.sh          # Stellar
# source ~/marlowe/hackathon/hackathon_common.sh  # Marlowe
export MINIRAM=~/ramses-development/mini-ramses-dev   # if not ~/mini-ramses-dev
```

NVHPC module mismatch → link errors (`libr3d.a`, `__fd_sincos_1`). See debug-playbook #4.

### Build (via harness)

GPU builds use **`hackathon_build_binary`** (wraps serial `make -f Makefile.a100|.h100` — **never `-j`**; Makefile module order is fragile).

Harness env vars map to Makefile flags:

| Harness var | Makefile flag | Typical DMO | Typical MHD test |
| --- | --- | --- | --- |
| `GPU_GRAV=1` | `GRAV=1` | yes | no (`GPU_GRAV=0`) |
| `GPU_HYDRO=1` | `HYDRO=1` | cosmo only | yes |
| `GPU_MHD=1` | `MHD=1` | no | yes |
| `GPU_TURB=1` | `TURB=1` | no | `mhd-turb` only |
| `GPU_NPRE` | `NPRE=` | 4 (default) | 4 or 8 per case |
| `GPU_FASTMATH` | `FASTMATH=` | 0 (default) | 0 |
| `GPU_UNITS` | `UNITS=` | `COSMO` | empty (omit) |
| `CUB_SORT_PART` | `-DCUB_SORT_PART` | 1 | 1 |
| `CUB_SORT_REFINE` | `-DCUB_SORT_REFINE` | 1 | 1 |
| `CUB_SCAN_REFINE` | `-DCUB_SCAN_REFINE` | 1 | 1 |
| `GPU_DEBUG` | `DEBUG=` | 0 | 0 |
| `GPU_CUDA_ARCH` | `CUDA_ARCH=` | `sm_80` / `sm_90` | same |

Example **manual** Stellar build (equivalent to harness):

```bash
cd ~/ramses-development/mini-ramses-dev/bin
make -f ~/hackathon/Makefile.a100 clean
make -f ~/hackathon/Makefile.a100 \
  COMPILER=NVHPC CUDA_ARCH=sm_80 DEBUG=0 NHILBERT=1 \
  GRAV=1 HYDRO=0 MHD=0 NPRE=4 FASTMATH=0 \
  CUB_SORT_PART=1 CUB_SORT_REFINE=1 CUB_SCAN_REFINE=1 \
  KICK_COOP_GATHER=0 NDIM=3 UNITS=COSMO ramses
# → bin/ramses3d
```

MHD binary (harness sets `BIN_GPU=.../ramses3d.mhd`):

```bash
make -f ~/hackathon/Makefile.a100 \
  COMPILER=NVHPC CUDA_ARCH=sm_80 DEBUG=0 \
  GRAV=0 HYDRO=1 MHD=1 NPSCAL=0 NPRE=4 FASTMATH=0 \
  CUB_SORT_PART=1 NDIM=3 ramses
```

**Repo-local Makefiles** (no harness copy):

- `bin/Makefile` — generic NVHPC (`CUDA_ARCH=sm_80`), `CUB_SORT_*` toggles
- `bin/Makefile.gpu` — multi-arch `sm_80,sm_90`
- `bin/Makefile.a100` — in-repo A100 variant (harness copy may be newer; harness refreshes if newer)

CPU parity baseline (login node, OpenMPI):

```bash
BUILD_CPU=1 ./submit_profiles.sh dmo-cpu   # → ramses3d.cpu via Makefile.cpu
```

### Submit (harness)

From harness directory:

```bash
cd ~/hackathon    # or ~/marlowe/hackathon
./submit_profiles.sh test          # smoke: L7 + AMR validation
./submit_profiles.sh dmo-cpu       # DMO CPU (parity baseline)
./submit_profiles.sh dmo-gpu       # DMO GPU (parity target)
./submit_profiles.sh ncu           # Nsight Compute (kick + CIC kernels)
./submit_profiles.sh nsys-l7       # Nsight Systems timeline, level-7
./submit_profiles.sh brio-wu       # Brio–Wu MHD shock tube
./submit_profiles.sh orszag-tang   # Orszag–Tang vortex (MHD)
./submit_profiles.sh ot-amr        # Orszag–Tang + AMR
./submit_profiles.sh cosmo-gpu     # cosmological DM+gas (HYDRO=1)
```

Direct Slurm: `sbatch dmo_gpu.slurm`, `sbatch dmo_cpu.slurm`, etc.

Useful overrides: `BUILD_BINARIES=0` (reuse binary), `GPU_NPRE=8`, `PART_DEP_ALGO=2`, `ORSZAG_TANG_LEVEL=7` (smaller GPU).

---

## Build flags (critical)

Makefile / CMake compile-time parameters (non-exhaustive):

| Flag | Meaning |
| --- | --- |
| `NDIM` | 1/2/3 (always 3 for harness) |
| `HYDRO`, `MHD`, `GRAV` | Physics modules |
| `TURB` | Driven turbulence (+ FFTW) |
| `TRCFLX` | Tracer-flux advection (`mini-ramses-dev-2`) |
| `NPRE` | Real precision bytes (4=single, 8=double) |
| `NPSCAL` | Passive scalars → `NVAR = 5 + NPSCAL + …` |
| `PATCH` | Optional patch directory prepended to `VPATH` |
| `MPI` | 0 → `-DWITHOUTMPI` |
| `DEBUG` | 0 release / 1 debug (slow; breaks some sanitizer combos) |
| `CUB_SORT_PART` | CUB radix sort for particles |
| `CUB_SORT_REFINE`, `CUB_SCAN_REFINE` | CUB for AMR refine |
| `FASTMATH` | NVHPC `-gpu=fastmath` (harness `GPU_FASTMATH`) |
| `CUDA_ARCH` | `sm_80` (A100), `sm_90` (H100) |
| `KICK_COOP_GATHER` | Warp-cooperative particle kick gather |
| `PAPER`, `CUDA_PROFILE` | Timing / lineinfo instrumentation |

### Known-bad combination

**`NPRE=4` + `FASTMATH=1` → illegal memory access** (`copyout Memcpy FAILED: 700`).  
Do not combine. Harness defaults: `GPU_NPRE=4`, `GPU_FASTMATH=0`. Exception: `cosmo-zoom` sets `NPRE=8` + `FASTMATH=1` (safe at NPRE=8).

### Cross-cluster drift

Stellar harness `Makefile.a100` defaults **`NPRE=4`**; Marlowe `Makefile.h100` defaults **`NPRE=8`**. Also watch `levelmax`, plot scripts, and namelist caps. Do not copy namelists verbatim across clusters.

---

## Agent constraints

1. **Mac agent cannot compile NVHPC/CUDA or run on cluster.** State assumptions; give exact commands for the human or cluster session; suggest smoke tests (`test`, short `ot-amr`).
2. **Never change namelist defaults silently.** Show unified diff; RAMSES reads `character*80` paths — long `initfile` paths **truncate silently** (playbook #1). Stage short symlink `ic_grafic` in run dir via harness.
3. **Never assume a fix worked** until verified in fresh `run.log`: `Reading ic_grafic/ic_bxleft`, **`emag > 0` at step 1** for MHD ICs.
4. **Core physics (Fortran/CUDA):** prefer careful review, minimal diffs, no silent algorithm changes.
5. **Harness / plots / docs:** faster iteration OK; still match existing patterns.
6. **Serial `make` only** for GPU builds on cluster (parallel make races `.mod` files).

---

## Validation gates

See [[Validation & test problems MOC]] and [[GPU optimization MOC]] in vault.

| Case | Command | Pass criteria (minimum) |
| --- | --- | --- |
| Smoke | `./submit_profiles.sh test` | Job completes; `Run completed` in slurm out |
| DMO parity | `dmo-cpu` then `dmo-gpu` | GPU matches CPU within harness tolerance (`compare_dmo_outputs.py`) |
| Brio–Wu | `./submit_profiles.sh brio-wu` | Completes; sensible shock structure |
| Orszag–Tang | `./submit_profiles.sh orszag-tang` | `emag>0` step 1; non-degenerate plots (`plot_ot.py`) |
| OT + AMR | `./submit_profiles.sh ot-amr` | AMR refines; no zero-B garbage (IC symlink!) |
| CIC profile | `./submit_profiles.sh ncu` | NCU reports for kick/CIC kernels |
| Cosmo GPU | `./submit_profiles.sh cosmo-gpu` | Runs with `HYDRO=1`, `ramses3d.hydro` |

Checker scripts: `check_ot_snapshot.py` (assert physics, not just file existence). Stale ICs can false-PASS (playbook #5).

---

## Key files

| File | Role |
| --- | --- |
| `gpu/gpu_part.cuf` | Particle kernels: CIC deposition, kick-drift |
| `gpu/gpu_hydro.cuf` | GPU hydro / MHD integrator |
| `gpu/gpu_runner.cuf` | Top-level GPU orchestration, mesh state |
| `gpu/gpu_manager.f90` | Host-side GPU dispatch |
| `gpu/gpu_rho.cuf` | Density / multipole on device |
| `pm/move_fine.f90` | Particle leapfrog on fine levels |
| `pm/rho_fine.f90` | CPU/GPU rho dispatch |
| `amr/read_params.f90` | Namelist read (80-char path limit!) |
| `bin/Makefile`, `bin/Makefile.gpu`, `bin/Makefile.a100` | Build system |
| `CMakeLists.txt` | Alternate CMake build (`COMPILER=NVHPC`, same toggles) |
| `utils/py/miniramses.py` | Python snapshot I/O |
| `utils/py/grafic/*.py` | IC generators (brio_wu, orszag_tang, …) |

Harness plotting: `~/hackathon/plot_ot.py`, `plot_dmo_cpu_gpu.py`, `compare_dmo_outputs.py`.

---

## Session handoff

At end of long sessions, ask the user to fill Obsidian `templates/session-handoff.md` (or write a note under `30-runs/`) with:

- Commits / SHAs touched
- What was **verified** vs **assumed**
- Exact next command (e.g. `BUILD_BINARIES=0 ./submit_profiles.sh ot-amr`)
- Cluster, branch, build flags, job IDs
- Open blockers

---

## Links

**Vault (wikilinks — open as `.md` paths):**

- `/Users/moseley/Documents/Obsidian Vault/Home.md`
- `/Users/moseley/Documents/Obsidian Vault/20-projects/mini-ramses/mini-ramses — hub.md`
- `/Users/moseley/Documents/Obsidian Vault/20-projects/mini-ramses/mini-ramses — debug-playbook.md`
- `/Users/moseley/Documents/Obsidian Vault/20-projects/mini-ramses/runbook — Stellar.md`
- `/Users/moseley/Documents/Obsidian Vault/20-projects/mini-ramses/runbook — Marlowe.md`
- `/Users/moseley/Documents/Obsidian Vault/Vault setup & AI wiring.md`

**Repo:**

- Harness docs: `~/hackathon/CLUSTER.md`, `~/marlowe/hackathon/CLUSTER.md`
- This file: `AGENTS.md` (root)

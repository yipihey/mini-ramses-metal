# Warp-cooperative fine-force gather for the GPU kick — implementation note

Implements Phase 2 of `kick_gather_warp_coop_plan.md` (the cooperative fine
gather) and *only* that. No histogram (Phase 0), no per-thread corner dedup
(Phase 1), no cooperative coarse fallback (Phase 3), no test harness/namelist/
profiling scripts.

## What changed

- `gpu/gpu_part.cuf`
  - New `attributes(device) subroutine gather_cic_force_part_coop` (guarded by
    `#ifdef KICK_COOP_GATHER`), inserted between the unchanged
    `gather_cic_force_part` and `kick_drift_part_kernel`.
  - `kick_drift_part_kernel` now selects, at compile time, the cooperative fine
    gather (flag on) or the original scalar `gather_cic_force_part` (flag off,
    default). The coarse `x/2` fallback always calls the **unchanged** scalar
    `gather_cic_force_part`.
- `bin/Makefile.a100`, `bin/Makefile.gpu`: add `KICK_COOP_GATHER = 0` and wire
  `-DKICK_COOP_GATHER` when set to 1 (mirrors the existing `PAPER` toggle).

`gather_cic_force_part` and `cic_part_medium_kernel` are byte-for-byte unchanged.

## Segment key

A *segment* is a maximal run of adjacent warp lanes sharing the CIC node
`n(idim) = int(x_in(idim) + 0.5)` — i.e. the scalar gather's `ir(idim)` **before**
the periodic wrap, the same node the medium deposit groups by. By the AMR-safety
theorem (plan §3) the 8 corner octs, their hashes, `in_domain`, the ghost/miss
test, the `f_d` values, and therefore `ok_level` are all pure functions of
`(node, level)`. Level is constant per launch, so a whole segment stays fine or
falls back to coarse together — no intra-segment divergence, AMR-safe.

Segment detection is by **direct component comparison** (`node1/node2/node3` plus
a validity bit `vint`) shuffled with `__shfl_up`, not by packing a node key. This
is collision-free for any coordinate range and any `ndim`, avoiding the
level-dependent bit budget (and silent-collision risk) of an `integer(kind=8)`
pack. The packed-key form the plan mentions is the documented alternative; the
component form was chosen purely for collision-safety.

## Broadcast method

The segment head computes `(cok, cf1, cf2, cf3)` for each corner; all other lanes
hold zero. A **segmented inclusive Hillis-Steele scan with the `+` operator** —
structurally identical to `cic_part_medium_kernel:1506-1524`, with `+` acting as
a copy via the additive identity — propagates the head value down the segment:
`result = head + 0 + … + 0 = head`, exactly. `head_acc` (seeded from `is_head`,
combined with `ior`) stops the scan at the segment boundary so a lane never pulls
the previous segment's value. The head propagates from the lowest lane upward,
matching the `__shfl_up` (read-from-lower-lane) direction.

This reuses the kernel's already-proven `__shfl_up` segmented-scan primitive
instead of an arbitrary-source `__shfl(var, srcLane)`, sidestepping the 1-based/
0-based `srcLane` convention concern.

## Why tail lanes are masked, not returned

Warp shuffles require all 32 lanes live. The cooperative branch therefore drops
the scalar path's `if (idx >= num_parts) return` and instead computes
`valid_lane = (slot_global <= num_parts)` (deposit-kernel lane map, which yields
the *same* `idx -> ipart` mapping as the scalar raster, so A/B is apples-to-apples).
Invalid lanes carry `vint = 0`, forming a separate zero-segment that reads nothing;
all global writes (and the `xp_d` read) are guarded by `valid_lane`.

In the scalar (default) build the early return is preserved and `valid_lane` is a
compile-time-constant `.true.`, so the added `if (valid_lane)` guard folds away.
The guarded update block was intentionally **not** re-indented, to keep the diff
to "add a guard" rather than reflowing ~35 unchanged lines.

## Bit-for-bit equivalence (by construction)

Each lane computes the same `ff` as the scalar path: same `(bz,by,bx)` corner
order (`bx` = bit 0 fastest), same `dr/dl` weights, same `real(f_d,kind=8)`
promotion, same accumulation order. Only *who reads* shared global data changes,
never *what each lane computes*. Adding `0.0` in the broadcast scan is exact in
IEEE-754, so the broadcast value equals the head's read bit-for-bit. (The lone
theoretical wrinkle is a `-0.0` force becoming `+0.0`; the scalar path collapses
that on its first `ff +=` too, and real forces are never exactly `-0.0`.)

`seg_ok` equals the scalar `ok_level` (true iff every corner is `in_domain` and
hashes to a valid non-ghost oct), and `ff` is zeroed identically when false.

## What could NOT be verified locally (no nvfortran / no NVIDIA GPU)

- Does not compile here; no NVHPC/`nvfortran` and no GPU available.
- `__shfl_up` typing for `integer(kind=4)` (cok/hacc/node/vint) and
  `real(kind=8)` (cf1..cf3) is assumed from existing usage in
  `cic_part_medium_kernel` (real8 + int4 already shuffled there) but not compiled.
- The bit-for-bit A/B assertion (`max|ff_coop − ff_scalar| == 0`) and the
  AMR-boundary stress test (plan §7) must be run on the cluster.
- Register-pressure / occupancy impact and the actual Long-Scoreboard reduction
  need NCU re-profiling on the A100.

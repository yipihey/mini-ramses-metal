# Fable cherry-pick notes (post `7b22f641`)

**Baseline:** `fc782d7a` (256³ Orszag–Tang ~105 s on A100; includes the
`96775e20` resident-shared rewrite that removed global `mhd_corner_scratch`,
plus P1 inline `edge_e`, P0 pad at that time, and follow-on cleanups through
`6f309f5d` / `fc782d7a`).

**Full Fable follow-up (reverted here):** `7b22f641` — archived on branch
`fable-7b22f641-archive` and as
`docs/archive/0001-lexicographic-fill-refined_bytes-packed-map.patch`.

## Kept from `7b22f641` (NCU-supported wins)

| Change | File | Rationale |
|--------|------|-----------|
| Lexicographic `(i,j,k)` fill loop | `gpu/gpu_hydro.cuf` | Shared stores to `local_subgrid` / `bf` coalesced; fill lines went to 0 excessive shared wavefronts in job 2823875 source.csv |
| Merged `bf` face branches + `source_idx_face` selects | `gpu/gpu_hydro.cuf` | Same commit; fewer divergent stores on hottest shared lines |
| P0 revert — unpadded `subgrid_6x6x6cell_primitive` (stride-6 rows) | `gpu/gpu_utils.cuf` | Required for `work_idx == i+6j+36k` store coalescing; P0 pad measured no-op on job 2823239 |

## Reverted from `7b22f641` (NCU + 256³ wall time)

| Change | Rationale |
|--------|-----------|
| `pack_refined_kernel` + module `refined_bytes` | Full-grid sweep every `gpu_godunov`; 256³ ~113 s vs ~105 s baseline |
| `refined_bytes` integrator argument / `grid` dropped from fill | Restored in-fill `grid(source_idx)%refined(cell_idx)` gather (15% global excessive at baseline vs 52% with lex+pack combined) |

## Not in tree (future experiment)

- **`refined_bytes` without full-grid pack** — e.g. pack only level-active octs, or lazy per-stencil — preserved in archive patch for reuse.

## Verify after rebuild on Stellar

```bash
BUILD_BINARIES=1 ./submit_profiles.sh ncu-ot-localize   # 128³ NCU
# 256³ production OT timing vs 105 s baseline
```

Expected vs full `7b22f641`: shared fill wins retained; no pack kernel tax; global sector count closer to pre-Fable 2823239 than to 2823875.

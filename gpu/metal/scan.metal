//============================================================================
// scan.metal  —  device-wide integer prefix sum
//
// Transliteration of gpu_scan.cuf (block_scan / uniform_add).  The original
// relies on separate kernel launches for the multi-level scan because there is
// no global barrier inside one dispatch; Metal is identical, so the host
// (metal_bridge.mm) encodes the same block_scan -> spine block_scan ->
// uniform_add tree across separate dispatches in one command buffer.
//
// Semantics: INCLUSIVE prefix sum, computed in place over data[offset-1 ..].
//   - block_scan writes each block's inclusive total to partial_sums[bid]
//     (when provided) and the global inclusive total to total_sum (when the
//     last element falls in this block).
//   - uniform_add adds the (already-scanned) cumulative block offsets.
//
// 1-based Fortran index arithmetic is preserved; arrays are dereferenced 0-based.
//============================================================================
#include <metal_stdlib>
using namespace metal;

// Inclusive warp (SIMD-group) scan — faithful to warp_scan in gpu_scan.cuf.
// lane0 is the 0-based lane index; width is the SIMD width (32 on Apple).
inline int warp_scan_incl(int scan, uint lane0, uint width) {
    for (uint offset = 1; offset < width; offset <<= 1) {
        int num = simd_shuffle_up(scan, offset);
        if (lane0 >= offset) scan += num;
    }
    return scan;
}

kernel void block_scan(
    device int*        data         [[buffer(0)]],
    device int*        partial_sums [[buffer(1)]],   // may be unused (flag below)
    device int*        total_sum    [[buffer(2)]],   // may be unused (flag below)
    constant int&      offset       [[buffer(3)]],   // 1-based start of active range
    constant int&      size         [[buffer(4)]],   // # active elements
    constant int&      flags        [[buffer(5)]],   // bit0: write partial_sums, bit1: write total_sum
    uint  tid_tg   [[thread_position_in_threadgroup]],
    uint  bid      [[threadgroup_position_in_grid]],
    uint  ntg      [[threads_per_threadgroup]],
    uint  lane0    [[thread_index_in_simdgroup]],
    uint  wid0     [[simdgroup_index_in_threadgroup]],
    uint  width    [[threads_per_simdgroup]],
    uint  n_warps  [[simdgroups_per_threadgroup]])
{
    threadgroup int sums[32];

    uint gid = tid_tg + bid * ntg;            // 0-based global element index
    int  scan = 0;
    if (gid < (uint)size) scan = data[gid + offset - 1];

    // Each warp performs its own inclusive scan.
    scan = warp_scan_incl(scan, lane0, width);

    // Last lane of each warp writes its warp total to threadgroup memory.
    if (lane0 == width - 1) sums[wid0] = scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp scans the per-warp totals.
    int wscan = (tid_tg < n_warps) ? sums[lane0] : 0;
    if (wid0 == 0) {
        wscan = warp_scan_incl(wscan, lane0, width);
        sums[lane0] = wscan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Add the total of all preceding warps.
    if (wid0 > 0) scan += sums[wid0 - 1];

    // Store the inclusive scan back.
    if (gid < (uint)size) data[gid + offset - 1] = scan;

    // Block inclusive total (last thread of the block).
    if ((flags & 1) && tid_tg == ntg - 1) partial_sums[bid] = scan;
    // Global inclusive total (thread holding the last element).
    if ((flags & 2) && gid == (uint)(size - 1)) total_sum[0] = scan;
}

kernel void uniform_add(
    device int*        data         [[buffer(0)]],
    device int*        partial_sums [[buffer(1)]],   // cumulative (scanned) block sums
    device int*        total_sum    [[buffer(2)]],
    constant int&      offset       [[buffer(3)]],
    constant int&      size         [[buffer(4)]],
    constant int&      flags        [[buffer(5)]],   // bit1: write total_sum
    uint  tid_tg [[thread_position_in_threadgroup]],
    uint  bid    [[threadgroup_position_in_grid]],
    uint  ntg    [[threads_per_threadgroup]])
{
    threadgroup int buf;
    if (tid_tg == 0) buf = partial_sums[bid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gid = tid_tg + bid * ntg;
    if (gid < (uint)size) data[gid + offset - 1] = data[gid + offset - 1] + buf;

    if ((flags & 2) && gid == (uint)(size - 1)) total_sum[0] = data[gid + offset - 1];
}

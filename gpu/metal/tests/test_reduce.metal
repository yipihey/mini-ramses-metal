// Unit test for block_reduce_sum_df64 (reduce.h).  One threadgroup of 256
// threads reduces [1e7, 1, 1, ... ,1]; fp32 loses the ones (1e7 ulp ~ 1), df64
// keeps them.  Host checks df64 == 1e7+255 while fp32 == 1e7.
#include "../reduce.h"

kernel void test_reduce_df64(device const float* in  [[buffer(0)]],
                             device float*       out [[buffer(1)]],
                             uint gid   [[thread_position_in_grid]],
                             uint lane  [[thread_index_in_simdgroup]],
                             uint sg_id [[simdgroup_index_in_threadgroup]],
                             uint n_sg  [[simdgroups_per_threadgroup]])
{
    threadgroup df64  dscratch[32];
    threadgroup float fscratch[32];
    float v = in[gid];
    df64  rd = block_reduce_sum_df64(df64_from(v), dscratch, lane, sg_id, n_sg);
    float rf = block_reduce_sum(v, fscratch, lane, sg_id, n_sg);
    if (gid == 0) { out[0] = rd.hi; out[1] = rd.lo; out[2] = rf; }   // hi,lo (reconstruct in double), fp32
}

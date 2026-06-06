// Unit test for df64.h (double-single arithmetic).  One thread accumulates
// a*N two ways: df64 (error-compensated) and plain fp32 (drifts).  Host checks
// the df64 result matches the fp64 reference while the fp32 sum does not.
#include "../df64.h"

kernel void test_df64_sum(device const float* in  [[buffer(0)]],
                          device float*       out [[buffer(1)]],
                          constant int&       n   [[buffer(2)]],
                          uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    float a = in[0];
    df64  acc  = df64_from(0.0f);
    float facc = 0.0f;
    for (int i = 0; i < n; ++i) { acc = df64_add_f(acc, a); facc = facc + a; }
    out[0] = df64_value(acc);   // df64 accumulated sum
    out[1] = facc;              // fp32 naive sum
    // two_prod exactness: hi+lo must equal a*a to ~fp64
    df64 p = two_prod(a, a);
    out[2] = p.hi;
    out[3] = p.lo;
}

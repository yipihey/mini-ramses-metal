// test_hydro.metal — exercises the hydro.h numerical core for the .mm harness.
// One thread computes EOS round-trip, the three Riemann fluxes for a given L/R
// interface, an identical-state flux, and the two slope limiters; writes them to
// `out` for the host to check against a double-precision replica + analytics.
#include <metal_stdlib>
#include "../../ramses_metal.h"
#include "../hydro.h"
using namespace metal;

// in : [0]=gamma [1..5]=L(rho,u,v,w,P) [6..10]=R [11]=slopefactor [12..14]=l,m,r
// out: [0..4]=c2p(p2c(L))  [5..9]=HLLC  [10..14]=HLL  [15..19]=LLF
//      [20]=minmod [21]=moncen  [22..26]=HLLC(L,L) identical-state flux
kernel void hydro_core_test(device const float* in  [[buffer(0)]],
                            device       float* out [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    float gamma = in[0];

    HPrimitive L = { in[1], in[2], in[3], in[4], in[5] };
    HPrimitive R = { in[6], in[7], in[8], in[9], in[10] };

    // EOS round-trip
    HPrimitive rt = conserved_2_primitive(primitive_2_conserved(L, gamma), gamma);
    out[0]=rt.density; out[1]=rt.velocity_x; out[2]=rt.velocity_y; out[3]=rt.velocity_z; out[4]=rt.pressure;

    // Riemann fluxes (each gets its own clamped copies)
    { HPrimitive a=L, b=R; HConserved f=hllc_fluxes(a,b,gamma);
      out[5]=f.density; out[6]=f.momentum_x; out[7]=f.momentum_y; out[8]=f.momentum_z; out[9]=f.energy; }
    { HPrimitive a=L, b=R; HConserved f=hll_fluxes(a,b,gamma,false);
      out[10]=f.density; out[11]=f.momentum_x; out[12]=f.momentum_y; out[13]=f.momentum_z; out[14]=f.energy; }
    { HPrimitive a=L, b=R; HConserved f=hll_fluxes(a,b,gamma,true);
      out[15]=f.density; out[16]=f.momentum_x; out[17]=f.momentum_y; out[18]=f.momentum_z; out[19]=f.energy; }

    out[20] = slope_minmod(in[12], in[13], in[14]);
    out[21] = slope_moncen(in[12], in[13], in[14], (int)in[11]);

    // Identical-state HLLC must reduce to the physical flux of L.
    { HPrimitive a=L, b=L; HConserved f=hllc_fluxes(a,b,gamma);
      out[22]=f.density; out[23]=f.momentum_x; out[24]=f.momentum_y; out[25]=f.momentum_z; out[26]=f.energy; }
}

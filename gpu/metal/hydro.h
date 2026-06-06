//============================================================================
// hydro.h  <-  gpu_hydro.cuf  (device math: EOS, conversions, slopes, Riemann)
//
// Faithful fp32 transliteration of the CUDA-Fortran hydro device functions
// (gpu_hydro.cuf:84-799).  Pure per-cell / per-interface math, no threadgroup
// state — the numerical core the Godunov kernel (hydro.metal) and the unit test
// (tests/test_hydro.metal) both call.  Line refs are to gpu_hydro.cuf.
//
// Variable order matches the CPU/CUDA conserved_t/primitive_t:
//   conserved = [rho, rho*u_x, rho*u_y, rho*u_z, E_tot]
//   primitive = [rho, u_x, u_y, u_z, P]
// All reals fp32 (Metal has no double).
//============================================================================
#ifndef RAMSES_HYDRO_H
#define RAMSES_HYDRO_H

#include <metal_stdlib>
#include "../ramses_metal.h"   // SOLVER_*, NHVAR, geometry
using namespace metal;

// `scalar` carries the dual-energy entropy (ivar=NHVAR=6) when NHVAR>5; it is an
// unused register field for NHVAR==5 (pure hydro) so the output is byte-identical.
struct HConserved { float density, momentum_x, momentum_y, momentum_z, energy, scalar; };
struct HPrimitive { float density, velocity_x, velocity_y, velocity_z, pressure, scalar; };

// x^2 + (y^2 + z^2) — parenthesised for associativity (gpu_hydro.cuf:84).
inline float magnitude_squared(float x, float y, float z) {
    return x * x + (y * y + z * z);
}

// P = (gamma-1)(E - 0.5|p|^2/rho)                            (:101)
inline float compute_pressure(HConserved c, float gamma) {
    float m2 = magnitude_squared(c.momentum_x, c.momentum_y, c.momentum_z);
    return (gamma - 1.0f) * (c.energy - 0.5f * m2 / c.density);
}

// E = P/(gamma-1) + 0.5 rho |u|^2                            (:121)
inline float compute_energy(HPrimitive p, float gamma) {
    float v2 = magnitude_squared(p.velocity_x, p.velocity_y, p.velocity_z);
    return (p.pressure / (gamma - 1.0f)) + 0.5f * p.density * v2;
}

// c = sqrt(gamma P / rho)                                    (:141)
inline float sound_speed(HPrimitive p, float gamma) {
    return sqrt(gamma * p.pressure / p.density);
}

// minmod slope (:160)
inline float slope_minmod(float left, float middle, float right) {
    float sl = middle - left;
    float sr = right - middle;
    if (sl * sr <= 0.0f)      return 0.0f;
    else if (sl > 0.0f)       return min(sl, sr);
    else                      return max(sl, sr);
}

// moncen slope, `slope`: 0=1st order, 1=minmod, 2=moncen (:187)
inline float slope_moncen(float left, float middle, float right, int slope) {
    float sl = middle - left;
    float sr = right - middle;
    float sc = 0.5f * (sl + sr);
    float factor = (float)slope;
    if (sl * sr <= 0.0f)      return 0.0f;
    else if (sl > 0.0f)       return min(factor * min(sl, sr), sc);
    else                      return max(factor * max(sl, sr), sc);
}

inline HPrimitive conserved_2_primitive(HConserved c, float gamma) {   // (:222)
    HPrimitive p;
    p.density    = c.density;
    p.velocity_x = c.momentum_x / c.density;
    p.velocity_y = c.momentum_y / c.density;
    p.velocity_z = c.momentum_z / c.density;
    p.pressure   = compute_pressure(c, gamma);
#if NHVAR > 5
    p.scalar     = c.scalar / c.density;   // entropy density -> specific entropy (umuscl ctoprim)
#endif
    return p;
}

#if NHVAR > 5
// Dual-energy pressure recovery — the DEVICE half of the entropy/dual-energy
// formalism (the CPU half is source_hydro_fine.f90).  In a cold supersonic flow
// E ~= ekin, so eint = E - 0.5|p|^2/rho is a CATASTROPHIC fp32 cancellation; the
// CPU umuscl dodges it by computing the whole godunov in real(kind=8), which Metal
// (fp32, no fp64) cannot.  RAMSES advects the entropy s = P/rho^gamma precisely so
// the pressure can be recovered robustly: the CONSERVED slot carries s*rho =
// P/rho^(gamma-1), hence P_s = scalar * rho^(gamma-1) (== source_hydro_fine.f90:185,
// e_prim = unew(ientropy)*d^(gamma-1)/(gamma-1)).  Switch to P_s when the ROBUST
// thermal fraction eint_s/E < dual_energy: the test uses ONLY robust quantities
// (eint_s from the advected entropy, E the stored total) so the fp32-corrupted
// eint_cons can never self-mask the cold detection (the failure mode that defeated
// the host switch alone).  In shocks/hot gas eint_s/E is O(0.1) > dual_energy, so
// E-ekin (reliable there) is used and the host source_hydro_fine (fp64) resyncs the
// entropy each step — the device thus always reads a faithful s next step.
inline float dual_energy_pressure(HConserved c, float gamma, float dual_energy) {
    float p_cons = compute_pressure(c, gamma);                          // (gamma-1)(E-ekin), fp32
    float p_s    = c.scalar * pow(max(c.density, 1e-30f), gamma - 1.0f);// robust advected pressure
    float eint_s = p_s / (gamma - 1.0f);
    return (eint_s < dual_energy * c.energy) ? p_s : p_cons;
}
#endif

// conserved -> primitive with the dual-energy pressure switch (dual_energy>=0).
// Used by every device path that needs a faithful pressure in cold flow: the
// Godunov reconstruction (load_cell_prim), the gravity velocity-kick round-trips
// (sync_hydro/grav_hydro — without it the c->p->c round-trip recomputes eint from
// the corrupted E-ekin and re-corrupts E), and the dt/diagnostic (hydro_cmpdt).
inline HPrimitive conserved_2_primitive_de(HConserved c, float gamma, float dual_energy) {
    HPrimitive p = conserved_2_primitive(c, gamma);
#if NHVAR > 5
    if (dual_energy >= 0.0f) p.pressure = dual_energy_pressure(c, gamma, dual_energy);
#endif
    return p;
}

inline HConserved primitive_2_conserved(HPrimitive p, float gamma) {   // (:245)
    HConserved c;
    c.density    = p.density;
    c.momentum_x = p.velocity_x * p.density;
    c.momentum_y = p.velocity_y * p.density;
    c.momentum_z = p.velocity_z * p.density;
    c.energy     = compute_energy(p, gamma);
#if NHVAR > 5
    c.scalar     = p.scalar * p.density;
#endif
    return c;
}

// Single HLL flux component (:583)
inline float hll_flux(float speed_l, float speed_r,
                      float left_flux, float right_flux,
                      float left_c, float right_c) {
    return (speed_r * left_flux - speed_l * right_flux
            + speed_r * speed_l * (right_c - left_c)) / (speed_r - speed_l);
}

// HLL / LLF solver (:599).  left/right are clamped in place (CUDA intent(inout)).
inline HConserved hll_fluxes(thread HPrimitive& L, thread HPrimitive& R,
                             float gamma, bool llf) {
    const float smallr = 1e-10f;
    const float smallc_squared = 1e-10f * 1e-10f;
    float smallp = smallc_squared / gamma;
    L.density  = max(L.density,  smallr);
    R.density  = max(R.density,  smallr);
    L.pressure = max(L.pressure, smallp * L.density);
    R.pressure = max(R.pressure, smallp * R.density);

    float cl = sound_speed(L, gamma);
    float cr = sound_speed(R, gamma);
    float speed_l, speed_r;
    if (llf) {
        float smax = max(abs(L.velocity_x) + cl, abs(R.velocity_x) + cr);
        speed_l = -smax; speed_r = smax;
    } else {
        speed_l = min(min(L.velocity_x, R.velocity_x) - max(cl, cr), 0.0f);
        speed_r = max(max(L.velocity_x, R.velocity_x) + max(cl, cr), 0.0f);
    }

    HConserved Lc = primitive_2_conserved(L, gamma);
    HConserved Rc = primitive_2_conserved(R, gamma);

    HConserved lf, rf;
    lf.density    = Lc.momentum_x;
    lf.momentum_x = L.pressure + L.velocity_x * Lc.momentum_x;
    lf.momentum_y = L.velocity_x * Lc.momentum_y;
    lf.momentum_z = L.velocity_x * Lc.momentum_z;
    lf.energy     = L.velocity_x * (L.pressure + Lc.energy);
    rf.density    = Rc.momentum_x;
    rf.momentum_x = R.pressure + R.velocity_x * Rc.momentum_x;
    rf.momentum_y = R.velocity_x * Rc.momentum_y;
    rf.momentum_z = R.velocity_x * Rc.momentum_z;
    rf.energy     = R.velocity_x * (R.pressure + Rc.energy);
#if NHVAR > 5
    lf.scalar     = L.velocity_x * Lc.scalar;   // advective flux of the scalar density
    rf.scalar     = R.velocity_x * Rc.scalar;
#endif

    HConserved flux;
    flux.density    = hll_flux(speed_l, speed_r, lf.density,    rf.density,    Lc.density,    Rc.density);
    flux.momentum_x = hll_flux(speed_l, speed_r, lf.momentum_x, rf.momentum_x, Lc.momentum_x, Rc.momentum_x);
    flux.momentum_y = hll_flux(speed_l, speed_r, lf.momentum_y, rf.momentum_y, Lc.momentum_y, Rc.momentum_y);
    flux.momentum_z = hll_flux(speed_l, speed_r, lf.momentum_z, rf.momentum_z, Lc.momentum_z, Rc.momentum_z);
    flux.energy     = hll_flux(speed_l, speed_r, lf.energy,     rf.energy,     Lc.energy,     Rc.energy);
#if NHVAR > 5
    flux.scalar     = hll_flux(speed_l, speed_r, lf.scalar,     rf.scalar,     Lc.scalar,     Rc.scalar);
#endif
    return flux;
}

// HLLC solver (:669).  left/right clamped in place.
inline HConserved hllc_fluxes(thread HPrimitive& L, thread HPrimitive& R, float gamma) {
    const float smallr = 1e-10f;
    const float smallc_squared = 1e-10f * 1e-10f;
    float smallp = smallc_squared / gamma;
    L.density  = max(L.density,  smallr);
    R.density  = max(R.density,  smallr);
    L.pressure = max(L.pressure, smallp * L.density);
    R.pressure = max(R.pressure, smallp * R.density);

    float rl = L.density, rr = R.density;
    float ul = L.velocity_x, ur = R.velocity_x;
    float vl = L.velocity_y, vr = R.velocity_y;
    float wl = L.velocity_z, wr = R.velocity_z;
    float pl = L.pressure, pr = R.pressure;

    float entho = 1.0f / (gamma - 1.0f);
    float el = pl * entho, er = pr * entho;
    float ekinl = 0.5f * rl * magnitude_squared(ul, vl, wl);
    float ekinr = 0.5f * rr * magnitude_squared(ur, vr, wr);
    float etotl = el + ekinl, etotr = er + ekinr;

    float cl = sound_speed(L, gamma);
    float cr = sound_speed(R, gamma);
    float speed_l = min(ul, ur) - max(cl, cr);
    float speed_r = max(ul, ur) + max(cl, cr);

    float rcl = rl * (ul - speed_l);
    float rcr = rr * (speed_r - ur);
    float ustar = (rcr * ur + rcl * ul + (pl - pr)) / (rcr + rcl);
    float pstar = (rcr * pl + rcl * pr + rcl * rcr * (ul - ur)) / (rcr + rcl);

    float rstarl = rl * (speed_l - ul) / (speed_l - ustar);
    float rstarr = rr * (speed_r - ur) / (speed_r - ustar);
    float etotstarl = ((speed_l - ul) * etotl - pl * ul + pstar * ustar) / (speed_l - ustar);
    float etotstarr = ((speed_r - ur) * etotr - pr * ur + pstar * ustar) / (speed_r - ustar);

    float ro, uo, vo, wo, po, etoto;
    if (speed_l > 0.0f) {
        ro = rl; uo = ul; vo = vl; wo = wl; po = pl; etoto = etotl;
    } else if (ustar > 0.0f) {
        ro = rstarl; uo = ustar; vo = vl; wo = wl; po = pstar; etoto = etotstarl;
    } else if (speed_r > 0.0f) {
        ro = rstarr; uo = ustar; vo = vr; wo = wr; po = pstar; etoto = etotstarr;
    } else {
        ro = rr; uo = ur; vo = vr; wo = wr; po = pr; etoto = etotr;
    }

    HConserved flux;
    flux.density    = ro * uo;
    flux.momentum_x = ro * uo * uo + po;
    flux.momentum_y = ro * uo * vo;
    flux.momentum_z = ro * uo * wo;
    flux.energy     = (etoto + po) * uo;
#if NHVAR > 5
    // passively advected scalar (entropy): mass flux * upwind primitive scalar
    // (riemann_hllc: fgdnv = ro*uo*qleft/qright by sign of ustar).
    flux.scalar     = flux.density * (ustar > 0.0f ? L.scalar : R.scalar);
#endif
    return flux;
}

// Dispatch (:782).  Unknown -> LLF, as in CUDA.
inline HConserved riemann_fluxes(thread HPrimitive& L, thread HPrimitive& R,
                                 float gamma, int riemann) {
    if (riemann == SOLVER_HLL)       return hll_fluxes(L, R, gamma, false);
    else if (riemann == SOLVER_HLLC) return hllc_fluxes(L, R, gamma);
    else                             return hll_fluxes(L, R, gamma, true);  // LLF
}

//----------------------------------------------------------------------------
// MUSCL-Hancock reconstruction + 1D Godunov update.  The 3D trace_3d
// (gpu_hydro.cuf:400) reduces EXACTLY to this in 1D: the transverse (y,z) slopes
// vanish, leaving the x source terms below, and no velocity rotation is needed
// since the sweep normal is already x.  This is the unit-testable core of the
// integrator; the 3D directional sweeps + the 27-oct subgrid gather build on it.
//----------------------------------------------------------------------------

// MUSCL-Hancock trace of one cell `m` from its x-stencil (l,m,r).  Returns the
// two reconstructed interface states: qL = state at the cell's i+1/2 (right) face
// (= cell+slope), qR = state at its i-1/2 (left) face (= cell-slope), each after
// the half-step source prediction.  Faithful to trace_3d with y,z slopes = 0.
inline void trace_cell_1d(HPrimitive l, HPrimitive m, HPrimitive r,
                          float gamma, float dtdx, int slope,
                          thread HPrimitive& qL, thread HPrimitive& qR) {
    const float smallr = 1e-10f;
    const float smallp = 1e-10f * (1e-10f * 1e-10f);   // smallr*smallc_squared (trace_3d:417)

    HPrimitive s;   // 0.5 * limited x-slope
    s.density    = 0.5f * slope_moncen(l.density,    m.density,    r.density,    slope);
    s.velocity_x = 0.5f * slope_moncen(l.velocity_x, m.velocity_x, r.velocity_x, slope);
    s.velocity_y = 0.5f * slope_moncen(l.velocity_y, m.velocity_y, r.velocity_y, slope);
    s.velocity_z = 0.5f * slope_moncen(l.velocity_z, m.velocity_z, r.velocity_z, slope);
    s.pressure   = 0.5f * slope_moncen(l.pressure,   m.pressure,   r.pressure,   slope);
#if NHVAR > 5
    s.scalar     = 0.5f * slope_moncen(l.scalar,     m.scalar,     r.scalar,     slope);
#endif

    HPrimitive src;   // x-direction source terms (trace_3d:460-479, y,z dropped)
    src.density    = -m.velocity_x * s.density    - s.velocity_x * m.density;
    src.velocity_x = -m.velocity_x * s.velocity_x - s.pressure / m.density;
    src.velocity_y = -m.velocity_x * s.velocity_y;
    src.velocity_z = -m.velocity_x * s.velocity_z;
    src.pressure   = -m.velocity_x * s.pressure   - s.velocity_x * gamma * m.pressure;
#if NHVAR > 5
    src.scalar     = -m.velocity_x * s.scalar;   // passive advection (umuscl trace1d:461 sr0=-u*drx)
#endif

    HPrimitive p;   // half-step predicted cell-centered state
    p.density    = m.density    + dtdx * src.density;
    p.velocity_x = m.velocity_x + dtdx * src.velocity_x;
    p.velocity_y = m.velocity_y + dtdx * src.velocity_y;
    p.velocity_z = m.velocity_z + dtdx * src.velocity_z;
    p.pressure   = m.pressure   + dtdx * src.pressure;
#if NHVAR > 5
    p.scalar     = m.scalar     + dtdx * src.scalar;
#endif

    qL.density    = p.density    + s.density;
    qL.velocity_x = p.velocity_x + s.velocity_x;
    qL.velocity_y = p.velocity_y + s.velocity_y;
    qL.velocity_z = p.velocity_z + s.velocity_z;
    qL.pressure   = p.pressure   + s.pressure;
#if NHVAR > 5
    qL.scalar     = p.scalar     + s.scalar;
#endif
    if (qL.density  < smallr) qL.density  = m.density;    // 1st-order fallback (orig value)
    if (qL.pressure < smallp) qL.pressure = m.pressure;

    qR.density    = p.density    - s.density;
    qR.velocity_x = p.velocity_x - s.velocity_x;
    qR.velocity_y = p.velocity_y - s.velocity_y;
    qR.velocity_z = p.velocity_z - s.velocity_z;
    qR.pressure   = p.pressure   - s.pressure;
#if NHVAR > 5
    qR.scalar     = p.scalar     - s.scalar;
#endif
    if (qR.density  < smallr) qR.density  = m.density;
    if (qR.pressure < smallp) qR.pressure = m.pressure;
}

// One-direction Godunov update of a 2-cell oct given its 6-cell primitive
// subgrid sg[0..5] (central oct = sg[2], sg[3]; sg[0..1], sg[4..5] are halo from
// the neighbour octs).  Writes the conservative update du[0], du[1] for the two
// central cells (= unew increment, gpu_hydro.cuf conservative_update).  dtdx=dt/dx.
inline void godunov_oct_1d(thread const HPrimitive sg[6], float gamma, float dtdx,
                           int slope, int riemann, thread HConserved du[2]) {
    HPrimitive qL[4], qR[4];   // traces for subgrid cells 1..4 -> index c-1
    for (int c = 1; c <= 4; ++c)
        trace_cell_1d(sg[c-1], sg[c], sg[c+1], gamma, dtdx, slope, qL[c-1], qR[c-1]);

    // Flux at the face between cells (c, c+1): L = qL of c, R = qR of c+1.
    HPrimitive a, b;
    a = qL[0]; b = qR[1]; HConserved fa = riemann_fluxes(a, b, gamma, riemann); // face 1|2
    a = qL[1]; b = qR[2]; HConserved fb = riemann_fluxes(a, b, gamma, riemann); // face 2|3
    a = qL[2]; b = qR[3]; HConserved fc = riemann_fluxes(a, b, gamma, riemann); // face 3|4

    // Central cell 2 sees faces (1|2, 2|3); cell 3 sees (2|3, 3|4).
    du[0].density    = (fa.density    - fb.density)    * dtdx;
    du[0].momentum_x = (fa.momentum_x - fb.momentum_x) * dtdx;
    du[0].momentum_y = (fa.momentum_y - fb.momentum_y) * dtdx;
    du[0].momentum_z = (fa.momentum_z - fb.momentum_z) * dtdx;
    du[0].energy     = (fa.energy     - fb.energy)     * dtdx;
    du[1].density    = (fb.density    - fc.density)    * dtdx;
    du[1].momentum_x = (fb.momentum_x - fc.momentum_x) * dtdx;
    du[1].momentum_y = (fb.momentum_y - fc.momentum_y) * dtdx;
    du[1].momentum_z = (fb.momentum_z - fc.momentum_z) * dtdx;
    du[1].energy     = (fb.energy     - fc.energy)     * dtdx;
}

//----------------------------------------------------------------------------
// 3D MUSCL-Hancock Godunov (faithful to trace_3d + riemann_driver +
// conservative_update).  Operates on a 6x6x6 primitive subgrid sg[i+6j+36k]
// (central oct = the {2,3}^3 block) and writes du for the 8 central cells.
// The transverse sweeps reuse the x-normal Riemann solver by rotating the
// velocity triple (the riemann_driver permutation, gpu_hydro.cuf:858-909).
//----------------------------------------------------------------------------
struct HTrace { HPrimitive qLx, qRx, qLy, qRy, qLz, qRz; };

inline HPrimitive hp_add(HPrimitive a, HPrimitive b, float s) {   // a + s*b
    HPrimitive r;
    r.density=a.density+s*b.density; r.velocity_x=a.velocity_x+s*b.velocity_x;
    r.velocity_y=a.velocity_y+s*b.velocity_y; r.velocity_z=a.velocity_z+s*b.velocity_z;
    r.pressure=a.pressure+s*b.pressure;
#if NHVAR > 5
    r.scalar=a.scalar+s*b.scalar;
#endif
    return r;
}

// Trace one subgrid cell (i,j,k) (1<=i,j,k<=4): all 6 face interface states.
inline HTrace trace_cell_3d(thread const HPrimitive sg[216], int i, int j, int k,
                            float gamma, float dtdx, int slope) {
    const float smallr = 1e-10f, smallp = 1e-10f * (1e-10f * 1e-10f);
    #define SG3(a,b,c) sg[(a) + 6*(b) + 36*(c)]
    HPrimitive m = SG3(i,j,k);
    HPrimitive sx, sy, sz;
    sx.density   =0.5f*slope_moncen(SG3(i-1,j,k).density,   m.density,   SG3(i+1,j,k).density,   slope);
    sx.velocity_x=0.5f*slope_moncen(SG3(i-1,j,k).velocity_x,m.velocity_x,SG3(i+1,j,k).velocity_x,slope);
    sx.velocity_y=0.5f*slope_moncen(SG3(i-1,j,k).velocity_y,m.velocity_y,SG3(i+1,j,k).velocity_y,slope);
    sx.velocity_z=0.5f*slope_moncen(SG3(i-1,j,k).velocity_z,m.velocity_z,SG3(i+1,j,k).velocity_z,slope);
    sx.pressure  =0.5f*slope_moncen(SG3(i-1,j,k).pressure,  m.pressure,  SG3(i+1,j,k).pressure,  slope);
    sy.density   =0.5f*slope_moncen(SG3(i,j-1,k).density,   m.density,   SG3(i,j+1,k).density,   slope);
    sy.velocity_x=0.5f*slope_moncen(SG3(i,j-1,k).velocity_x,m.velocity_x,SG3(i,j+1,k).velocity_x,slope);
    sy.velocity_y=0.5f*slope_moncen(SG3(i,j-1,k).velocity_y,m.velocity_y,SG3(i,j+1,k).velocity_y,slope);
    sy.velocity_z=0.5f*slope_moncen(SG3(i,j-1,k).velocity_z,m.velocity_z,SG3(i,j+1,k).velocity_z,slope);
    sy.pressure  =0.5f*slope_moncen(SG3(i,j-1,k).pressure,  m.pressure,  SG3(i,j+1,k).pressure,  slope);
    sz.density   =0.5f*slope_moncen(SG3(i,j,k-1).density,   m.density,   SG3(i,j,k+1).density,   slope);
    sz.velocity_x=0.5f*slope_moncen(SG3(i,j,k-1).velocity_x,m.velocity_x,SG3(i,j,k+1).velocity_x,slope);
    sz.velocity_y=0.5f*slope_moncen(SG3(i,j,k-1).velocity_y,m.velocity_y,SG3(i,j,k+1).velocity_y,slope);
    sz.velocity_z=0.5f*slope_moncen(SG3(i,j,k-1).velocity_z,m.velocity_z,SG3(i,j,k+1).velocity_z,slope);
    sz.pressure  =0.5f*slope_moncen(SG3(i,j,k-1).pressure,  m.pressure,  SG3(i,j,k+1).pressure,  slope);
#if NHVAR > 5
    sx.scalar=0.5f*slope_moncen(SG3(i-1,j,k).scalar,m.scalar,SG3(i+1,j,k).scalar,slope);
    sy.scalar=0.5f*slope_moncen(SG3(i,j-1,k).scalar,m.scalar,SG3(i,j+1,k).scalar,slope);
    sz.scalar=0.5f*slope_moncen(SG3(i,j,k-1).scalar,m.scalar,SG3(i,j,k+1).scalar,slope);
#endif
    #undef SG3

    float divu = sx.velocity_x + sy.velocity_y + sz.velocity_z;
    HPrimitive src;
    src.density    = -m.velocity_x*sx.density   -m.velocity_y*sy.density   -m.velocity_z*sz.density   - divu*m.density;
    src.velocity_x = -m.velocity_x*sx.velocity_x-m.velocity_y*sy.velocity_x-m.velocity_z*sz.velocity_x- sx.pressure/m.density;
    src.velocity_y = -m.velocity_x*sx.velocity_y-m.velocity_y*sy.velocity_y-m.velocity_z*sz.velocity_y- sy.pressure/m.density;
    src.velocity_z = -m.velocity_x*sx.velocity_z-m.velocity_y*sy.velocity_z-m.velocity_z*sz.velocity_z- sz.pressure/m.density;
    src.pressure   = -m.velocity_x*sx.pressure  -m.velocity_y*sy.pressure  -m.velocity_z*sz.pressure  - divu*gamma*m.pressure;
#if NHVAR > 5
    src.scalar     = -m.velocity_x*sx.scalar    -m.velocity_y*sy.scalar    -m.velocity_z*sz.scalar;   // passive (no divu)
#endif

    HPrimitive p = hp_add(m, src, dtdx);   // half-step predicted state

    HTrace t;
    t.qLx=hp_add(p,sx, 1.0f); t.qRx=hp_add(p,sx,-1.0f);
    t.qLy=hp_add(p,sy, 1.0f); t.qRy=hp_add(p,sy,-1.0f);
    t.qLz=hp_add(p,sz, 1.0f); t.qRz=hp_add(p,sz,-1.0f);
    // first-order fallback to the ORIGINAL cell value (trace_3d:498)
    thread HPrimitive* q[6] = {&t.qLx,&t.qRx,&t.qLy,&t.qRy,&t.qLz,&t.qRz};
    for (int n=0;n<6;++n){ if(q[n]->density<smallr) q[n]->density=m.density; if(q[n]->pressure<smallp) q[n]->pressure=m.pressure; }
    return t;
}

inline HConserved flux_x(HPrimitive L, HPrimitive R, float g, int riem) {
    return riemann_fluxes(L, R, g, riem);
}
inline HConserved flux_y(HPrimitive L, HPrimitive R, float g, int riem) {   // rotate (u,v,w)->(v,w,u)
    HPrimitive Lr={L.density,L.velocity_y,L.velocity_z,L.velocity_x,L.pressure};
    HPrimitive Rr={R.density,R.velocity_y,R.velocity_z,R.velocity_x,R.pressure};
#if NHVAR > 5
    Lr.scalar=L.scalar; Rr.scalar=R.scalar;   // scalar is rotation-invariant
#endif
    HConserved f=riemann_fluxes(Lr,Rr,g,riem);
    HConserved o={f.density,f.momentum_z,f.momentum_x,f.momentum_y,f.energy};
#if NHVAR > 5
    o.scalar=f.scalar;
#endif
    return o;
}
inline HConserved flux_z(HPrimitive L, HPrimitive R, float g, int riem) {   // rotate (u,v,w)->(w,u,v)
    HPrimitive Lr={L.density,L.velocity_z,L.velocity_x,L.velocity_y,L.pressure};
    HPrimitive Rr={R.density,R.velocity_z,R.velocity_x,R.velocity_y,R.pressure};
#if NHVAR > 5
    Lr.scalar=L.scalar; Rr.scalar=R.scalar;
#endif
    HConserved f=riemann_fluxes(Lr,Rr,g,riem);
    HConserved o={f.density,f.momentum_y,f.momentum_z,f.momentum_x,f.energy};
#if NHVAR > 5
    o.scalar=f.scalar;
#endif
    return o;
}

inline HConserved cdiff(HConserved a, HConserved b, float s) {   // (a-b)*s
    HConserved r={ (a.density-b.density)*s, (a.momentum_x-b.momentum_x)*s,
                   (a.momentum_y-b.momentum_y)*s, (a.momentum_z-b.momentum_z)*s,
                   (a.energy-b.energy)*s };
#if NHVAR > 5
    r.scalar=(a.scalar-b.scalar)*s;
#endif
    return r;
}
inline HConserved cadd(HConserved a, HConserved b) {
    HConserved r={a.density+b.density,a.momentum_x+b.momentum_x,a.momentum_y+b.momentum_y,
                  a.momentum_z+b.momentum_z,a.energy+b.energy};
#if NHVAR > 5
    r.scalar=a.scalar+b.scalar;
#endif
    return r;
}

// du for the 8 central cells (cell index = 1 + ci + 2*cj + 4*ck, ci,cj,ck in {0,1}).
inline void godunov_oct_3d(thread const HPrimitive sg[216], float gamma, float dtdx,
                           int slope, int riemann, thread HConserved du[8]) {
    for (int ck=0; ck<2; ++ck) for (int cj=0; cj<2; ++cj) for (int ci=0; ci<2; ++ci) {
        int I=ci+2, J=cj+2, K=ck+2;
        HConserved fxl=flux_x(trace_cell_3d(sg,I-1,J,K,gamma,dtdx,slope).qLx, trace_cell_3d(sg,I,J,K,gamma,dtdx,slope).qRx, gamma,riemann);
        HConserved fxr=flux_x(trace_cell_3d(sg,I,J,K,gamma,dtdx,slope).qLx, trace_cell_3d(sg,I+1,J,K,gamma,dtdx,slope).qRx, gamma,riemann);
        HConserved fyl=flux_y(trace_cell_3d(sg,I,J-1,K,gamma,dtdx,slope).qLy, trace_cell_3d(sg,I,J,K,gamma,dtdx,slope).qRy, gamma,riemann);
        HConserved fyr=flux_y(trace_cell_3d(sg,I,J,K,gamma,dtdx,slope).qLy, trace_cell_3d(sg,I,J+1,K,gamma,dtdx,slope).qRy, gamma,riemann);
        HConserved fzl=flux_z(trace_cell_3d(sg,I,J,K-1,gamma,dtdx,slope).qLz, trace_cell_3d(sg,I,J,K,gamma,dtdx,slope).qRz, gamma,riemann);
        HConserved fzr=flux_z(trace_cell_3d(sg,I,J,K,gamma,dtdx,slope).qLz, trace_cell_3d(sg,I,J,K+1,gamma,dtdx,slope).qRz, gamma,riemann);
        HConserved d = cadd(cadd(cdiff(fxl,fxr,dtdx), cdiff(fyl,fyr,dtdx)), cdiff(fzl,fzr,dtdx));
        du[ci + 2*cj + 4*ck] = d;
    }
}

//----------------------------------------------------------------------------
// AMR-aware Godunov (zero_fine_fluxes + the boundary fluxes the coarse reflux
// needs).  These mirror godunov_oct_{1d,3d} but additionally:
//   (1) zero any interface flux that touches a refined subgrid cell -- the
//       finer level computes that flux, so the coarse cell must not (faithful
//       to zero_fine_fluxes, gpu_hydro.cuf:919); and
//   (2) return bnd[2*NDIM] = the boundary-face flux SUM over the transverse
//       central cells, ordered {-x,+x,-y,+y,-z,+z}, which coarse_cell_update
//       (gpu_hydro.cuf:1061) scatters (*dtdx/twotondim, signed) onto the coarse
//       parent of any coarser (cache-oct) neighbour.
// ref[] = refined flag per subgrid cell (true if that cell is refined).  With
// ref all-false and no cache-oct neighbours these reduce EXACTLY to the plain
// godunov_oct_{1d,3d} du (the boundary sums are then simply unused).
//----------------------------------------------------------------------------
inline HConserved czero() { HConserved z = {0.0f,0.0f,0.0f,0.0f,0.0f}; return z; }

inline void godunov_oct_1d_amr(thread const HPrimitive sg[6], thread const bool ref[6],
                               float gamma, float dtdx, int slope, int riemann,
                               thread HConserved du[2], thread HConserved bnd[2]) {
    HPrimitive qL[4], qR[4];
    for (int c = 1; c <= 4; ++c)
        trace_cell_1d(sg[c-1], sg[c], sg[c+1], gamma, dtdx, slope, qL[c-1], qR[c-1]);
    // fx[a] = flux at face between cells (a+1, a+2): a=0 -> -x bnd, a=1 interior, a=2 +x bnd
    HConserved fx[3];
    for (int a = 0; a < 3; ++a) {
        HPrimitive l = qL[a], r = qR[a+1];
        fx[a] = riemann_fluxes(l, r, gamma, riemann);
        if (ref[a+1] || ref[a+2]) fx[a] = czero();   // zero_fine_fluxes
    }
    du[0] = cdiff(fx[0], fx[1], dtdx);                // central cell 2 (sg index 2)
    du[1] = cdiff(fx[1], fx[2], dtdx);               // central cell 3 (sg index 3)
    bnd[0] = fx[0];                                   // -x boundary
    bnd[1] = fx[2];                                   // +x boundary
}

inline void godunov_oct_3d_amr(thread const HPrimitive sg[216], thread const bool ref[216],
                               float gamma, float dtdx, int slope, int riemann,
                               thread HConserved du[8], thread HConserved bnd[6]) {
    #define RIDX(a,b,c) ((a) + 6*(b) + 36*(c))
    HConserved fx[3][2][2], fy[2][3][2], fz[2][2][3];
    for (int ck=0; ck<2; ++ck) for (int cj=0; cj<2; ++cj) {
        int J=cj+2, K=ck+2;
        for (int a=0; a<3; ++a) {
            HConserved f = flux_x(trace_cell_3d(sg,a+1,J,K,gamma,dtdx,slope).qLx,
                                  trace_cell_3d(sg,a+2,J,K,gamma,dtdx,slope).qRx, gamma, riemann);
            if (ref[RIDX(a+1,J,K)] || ref[RIDX(a+2,J,K)]) f = czero();
            fx[a][cj][ck] = f;
        }
    }
    for (int ck=0; ck<2; ++ck) for (int ci=0; ci<2; ++ci) {
        int I=ci+2, K=ck+2;
        for (int b=0; b<3; ++b) {
            HConserved f = flux_y(trace_cell_3d(sg,I,b+1,K,gamma,dtdx,slope).qLy,
                                  trace_cell_3d(sg,I,b+2,K,gamma,dtdx,slope).qRy, gamma, riemann);
            if (ref[RIDX(I,b+1,K)] || ref[RIDX(I,b+2,K)]) f = czero();
            fy[ci][b][ck] = f;
        }
    }
    for (int cj=0; cj<2; ++cj) for (int ci=0; ci<2; ++ci) {
        int I=ci+2, J=cj+2;
        for (int c=0; c<3; ++c) {
            HConserved f = flux_z(trace_cell_3d(sg,I,J,c+1,gamma,dtdx,slope).qLz,
                                  trace_cell_3d(sg,I,J,c+2,gamma,dtdx,slope).qRz, gamma, riemann);
            if (ref[RIDX(I,J,c+1)] || ref[RIDX(I,J,c+2)]) f = czero();
            fz[ci][cj][c] = f;
        }
    }
    for (int ck=0; ck<2; ++ck) for (int cj=0; cj<2; ++cj) for (int ci=0; ci<2; ++ci) {
        HConserved d = cadd(cadd(cdiff(fx[ci][cj][ck], fx[ci+1][cj][ck], dtdx),
                                 cdiff(fy[ci][cj][ck], fy[ci][cj+1][ck], dtdx)),
                                 cdiff(fz[ci][cj][ck], fz[ci][cj][ck+1], dtdx));
        du[ci + 2*cj + 4*ck] = d;
    }
    // boundary-face flux sums over the 2x2 transverse central cells
    bnd[0]=czero(); bnd[1]=czero(); bnd[2]=czero(); bnd[3]=czero(); bnd[4]=czero(); bnd[5]=czero();
    for (int cj=0; cj<2; ++cj) for (int ck=0; ck<2; ++ck) { bnd[0]=cadd(bnd[0],fx[0][cj][ck]); bnd[1]=cadd(bnd[1],fx[2][cj][ck]); }
    for (int ci=0; ci<2; ++ci) for (int ck=0; ck<2; ++ck) { bnd[2]=cadd(bnd[2],fy[ci][0][ck]); bnd[3]=cadd(bnd[3],fy[ci][2][ck]); }
    for (int ci=0; ci<2; ++ci) for (int cj=0; cj<2; ++cj) { bnd[4]=cadd(bnd[4],fz[ci][cj][0]); bnd[5]=cadd(bnd[5],fz[ci][cj][2]); }
    #undef RIDX
}

//----------------------------------------------------------------------------
// interpol_hydro (interpol_hydro.f90): the coarse-fine ghost prolongation the
// Godunov reads at a refinement boundary (godfine1:534).  From a coarse stencil
// u1[0..2*NDIM] -- center (0) + the 2*NDIM face neighbours, ordering dir 2i-1 = -,
// 2i = + in dim i (the iii/hhh order) -- produce the TWOTONDIM fine sub-cell
// conserved states.  interpol_var: 0 interpolate conserved (rho,rhou,E); 1 with
// E as internal energy (E->eint, interpolate, eint->E).  interpol_type: 0 inject,
// 1 minmod (the RAMSES defaults).  moncen(2)/central(3) -- the multidimensional
// corner limiter -- are NOT yet ported (fall back to minmod; assert in tests).
//----------------------------------------------------------------------------
inline float il_slope(float center, float left, float right, int interpol_type) {
    if (interpol_type == 0) return 0.0f;                 // straight injection
    float dl = 0.5f * (right - center);                  // compute_limiter_minmod
    float dr = 0.5f * (center - left);
    if (dl * dr <= 0.0f) return 0.0f;
    return min(fabs(dl), fabs(dr)) * (dl / fabs(dl));
}

inline void interpol_hydro_oct(thread const HConserved u1[1 + 2*NDIM], int interpol_var,
                               int interpol_type, float smallr, float gamma, float dual_energy,
                               thread HConserved u2[TWOTONDIM]) {
    HConserved s[1 + 2*NDIM];
    for (int j = 0; j < 1 + 2*NDIM; ++j) s[j] = u1[j];
    if (interpol_var == 1)                               // total -> internal energy
        for (int j = 0; j < 1 + 2*NDIM; ++j) {
            float eint = s[j].energy - 0.5f * magnitude_squared(s[j].momentum_x, s[j].momentum_y, s[j].momentum_z)
                           / max(s[j].density, smallr);  // fp32 E-ekin (cancels in cold flow)
#if NHVAR > 5
            // Dual-energy: in cold flow recover eint robustly from the advected entropy
            // instead of the catastrophic fp32 E-ekin -- the coarse-fine-ghost analogue of
            // the godunov dual_energy_pressure fix (only interpol_var==1 hits the cancellation;
            // interpol_var==0 interpolates conserved E directly).  Faithful to the conserved
            // entropy slot s = P/rho^(gamma-1) -> eint_s = scalar*rho^(gamma-1)/(gamma-1).
            if (dual_energy >= 0.0f) {
                float p_s    = s[j].scalar * pow(max(s[j].density, 1e-30f), gamma - 1.0f);
                float eint_s = p_s / (gamma - 1.0f);
                if (eint_s < dual_energy * s[j].energy) eint = eint_s;
            }
#endif
            s[j].energy = eint;
        }
    for (int c = 1; c <= TWOTONDIM; ++c) {
        int b = c - 1;
        float xc[3] = { (float)(b & 1) - 0.5f, (float)((b >> 1) & 1) - 0.5f, (float)((b >> 2) & 1) - 0.5f };
        #define IL_COMP(field) { float a0 = s[0].field, v = a0;                          \
            for (int idim = 0; idim < NDIM; ++idim)                                       \
                v += il_slope(a0, s[2*idim+1].field, s[2*idim+2].field, interpol_type) * xc[idim]; \
            u2[b].field = v; }
        IL_COMP(density) IL_COMP(momentum_x) IL_COMP(momentum_y) IL_COMP(momentum_z) IL_COMP(energy)
#if NHVAR > 5
        IL_COMP(scalar)
#endif
        #undef IL_COMP
    }
    if (interpol_var == 1)                               // internal -> total energy
        for (int c = 0; c < TWOTONDIM; ++c)
            u2[c].energy += 0.5f * magnitude_squared(u2[c].momentum_x, u2[c].momentum_y, u2[c].momentum_z)
                            / max(u2[c].density, smallr);
}

#endif // RAMSES_HYDRO_H

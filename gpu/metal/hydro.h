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

struct HConserved { float density, momentum_x, momentum_y, momentum_z, energy; };
struct HPrimitive { float density, velocity_x, velocity_y, velocity_z, pressure; };

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
    return p;
}

inline HConserved primitive_2_conserved(HPrimitive p, float gamma) {   // (:245)
    HConserved c;
    c.density    = p.density;
    c.momentum_x = p.velocity_x * p.density;
    c.momentum_y = p.velocity_y * p.density;
    c.momentum_z = p.velocity_z * p.density;
    c.energy     = compute_energy(p, gamma);
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

    HConserved flux;
    flux.density    = hll_flux(speed_l, speed_r, lf.density,    rf.density,    Lc.density,    Rc.density);
    flux.momentum_x = hll_flux(speed_l, speed_r, lf.momentum_x, rf.momentum_x, Lc.momentum_x, Rc.momentum_x);
    flux.momentum_y = hll_flux(speed_l, speed_r, lf.momentum_y, rf.momentum_y, Lc.momentum_y, Rc.momentum_y);
    flux.momentum_z = hll_flux(speed_l, speed_r, lf.momentum_z, rf.momentum_z, Lc.momentum_z, Rc.momentum_z);
    flux.energy     = hll_flux(speed_l, speed_r, lf.energy,     rf.energy,     Lc.energy,     Rc.energy);
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
    return flux;
}

// Dispatch (:782).  Unknown -> LLF, as in CUDA.
inline HConserved riemann_fluxes(thread HPrimitive& L, thread HPrimitive& R,
                                 float gamma, int riemann) {
    if (riemann == SOLVER_HLL)       return hll_fluxes(L, R, gamma, false);
    else if (riemann == SOLVER_HLLC) return hllc_fluxes(L, R, gamma);
    else                             return hll_fluxes(L, R, gamma, true);  // LLF
}

#endif // RAMSES_HYDRO_H

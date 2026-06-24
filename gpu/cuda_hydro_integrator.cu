// ============================================================================
// cuda_hydro_integrator.cu
//
// Stage-1 faithful CUDA-C translation of hydro_integrator_kernel
// (gpu/gpu_hydro.cuf), the f16/nsubgrid=3 fast-tile godunov integrator.
//
// WHY THIS EXISTS: nvfortran 26.3 emits no LDG.128 / no cp.async, so the
// L1/TEX request-pipe throttle that caps the Fortran kernel (see
// PORT_METAL_TO_CUDA_CONTINUE_HERE.md and project_glmmhd_metal_hll_f16.md)
// cannot be relieved from CUDA Fortran. This .cu is the host-launched twin
// (cub_*.cu pattern: extern "C" + bind(c), nvcc -> .o, linked with -lstdc++).
//
// STAGE 1 = scalar loads only, must hit throughput PARITY with the Fortran
// kernel (~3567 GRAV=1 / ~2756 GRAV=0 Mcell/s at 480^3) + conserve_check PASS.
// Stage 2 then swaps the uold loads for reinterpret_cast<float4*> wide loads.
//
// Build flags mirror the Fortran build: NPRE=4 => dp=float (fp32 math),
// TPRE=2 => tp=__half (f16 shared tile), NSUB=3 => nsubgrid=3, NVAR=5 hydro,
// optional GRAV and TURB. The shared tiles are internal (not part of the ABI),
// so their layout is chosen freely; only uold/unew/f/nbor/grid/father/afield
// cross the Fortran<->C boundary and use the verified col-major flat indexing.
// ============================================================================

#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>

// ---- compile-time problem geometry (nsubgrid=3 fast tile) ------------------
#ifndef NSUB
#define NSUB 3
#endif
static constexpr int NS    = NSUB;           // nsubgrid
static constexpr int NDIM  = 3;
static constexpr int NVAR  = 5;              // hydro: rho, mom_x/y/z, E
static constexpr int TWOTONDIM = 8;
static constexpr int TD    = 2*NS + 4;       // tile per-dim extent (0..2*NS+3)
static constexpr int TILE  = TD*TD*TD;       // 1000 for NS=3
static constexpr int NSP2  = NS + 2;         // nsubgridp2
static constexpr int NSP2SQ = NSP2*NSP2;
static constexpr int NSTOND = NS*NS*NS;      // nsubgridtondim
static constexpr int SUBGRIDSIZE = NSP2*NSP2*NSP2;   // 125 for NS=3

// Interface (face) tile extents.
static constexpr int IX_NX = 2*NS+1, IX_NY = 2*NS,   IX_NZ = 2*NS;   // 7,6,6
static constexpr int IY_NX = 2*NS,   IY_NY = 2*NS+1, IY_NZ = 2*NS;   // 6,7,6
static constexpr int IZ_NX = 2*NS,   IZ_NY = 2*NS,   IZ_NZ = 2*NS+1; // 6,6,7
static constexpr int IX_SZ = IX_NX*IX_NY*IX_NZ;
static constexpr int IY_SZ = IY_NX*IY_NY*IY_NZ;
static constexpr int IZ_SZ = IZ_NX*IZ_NY*IZ_NZ;

// Riemann solver ids (hydro/hydro_parameters.f90).
static constexpr int SOLVER_LLF = 1, SOLVER_HLL = 2, SOLVER_HLLC = 3, SOLVER_TWOSHOCK = 10;

#ifdef TURB
static constexpr int TURB_GS = 64;
#endif

typedef __half thalf;   // tp / TPRE kind for the shared tile

// Dynamic shared-memory footprint: the f16 tile (5 prim + refined) + 6 face tiles
// (5 prim each). At ns3 = 26376 B (fits the 48KB static cap); at ns4 = 53568 B
// (exceeds it) -> we use DYNAMIC shared (opt-in to 100KB) so ns4's lower halo is
// reachable. The block_reduce svalues(32) stays separate static shared.
static constexpr int SMEM_BYTES =
    (int)((5*TILE + 10*IX_SZ + 10*IY_SZ + 10*IZ_SZ)*sizeof(thalf) + TILE*sizeof(unsigned char));

// ---- f-read isolation experiment (GRAV builds only) ------------------------
// The ONLY code difference between GRAV=1 (fast) and GRAV=0 (slow) is the 6
// divergent f reads/cell (3 in the dt-fold, 3 in the velocity predictor) vs the
// constant_gravity broadcast. To find which reads drive the +44%, build GRAV=1
// (f allocated + read) and selectively swap a site back to the broadcast:
//   -DFEXP_DT_OFF    dt-fold uses constant_gravity (predictor still reads f)
//   -DFEXP_PRED_OFF  predictor uses constant_gravity (dt-fold still reads f)
//   both             == GRAV=0 instruction pattern, but with f still allocated
#if defined(GRAV) && !defined(FEXP_DT_OFF)
#define DT_USE_F 1
#else
#define DT_USE_F 0
#endif
#if defined(GRAV) && !defined(FEXP_PRED_OFF)
#define PRED_USE_F 1
#else
#define PRED_USE_F 0
#endif

// ---- ABI flat indexing into the Fortran device arrays (0-based, col-major) -
//   uold(cell,var,oct) : cell 1..8, var 1..NVAR, oct 1-based
//   f(cell,dim,oct)    : cell 1..8, dim 1..3
//   nbor(ind,subgrid)  : ind 1..SUBGRIDSIZE, subgrid 1-based
#define UOLD(c,v,o)  uold[((size_t)((o)-1)*NVAR + ((v)-1))*8 + ((c)-1)]
#define UNEW(c,v,o)  unew[((size_t)((o)-1)*NVAR + ((v)-1))*8 + ((c)-1)]
#define FARR(c,d,o)  f[((size_t)((o)-1)*NDIM + ((d)-1))*8 + ((c)-1)]
#define NBORA(i,sg)  nbor[((size_t)((sg)-1))*SUBGRIDSIZE + ((i)-1)]

// mirror of amr/oct_commons.f90 (NDIM=3, nhilbert=1); sizeof==64 (S0-verified)
struct Oct {
    long long hkey;
    int       ckey[3];
    int       refined[8];   // logical(4); read != 0
    int       lev;
    int       superoct;
};

// ---- small per-cell value structs (fp32 math) ------------------------------
struct Prim { float density, velocity_x, velocity_y, velocity_z, pressure; };
struct Cons { float density, momentum_x, momentum_y, momentum_z, energy; };

// ---- shared-tile views (pointers into __shared__ scratch) ------------------
struct Tile {                       // subgrid_6x6x6cell_primitive (TD^3)
    thalf *d, *vx, *vy, *vz, *p;
    unsigned char *ref;
    __device__ __forceinline__ int idx(int i,int j,int k) const { return i + TD*(j + TD*k); }
};
struct Face {                       // directional interface/flux tile
    thalf *d, *vx, *vy, *vz, *p;
    int nx, ny;
    __device__ __forceinline__ int idx(int i,int j,int k) const { return i + nx*(j + ny*k); }
};

// =================== per-cell physics (faithful to .cuf) ====================
__device__ __forceinline__ float magnitude_squared(float x,float y,float z){
    return x*x + (y*y + z*z);
}
__device__ __forceinline__ float compute_pressure(const Cons& c,float gamma,float smallr,float smallc2){
    float smallp = smallc2/gamma;
    float msq = magnitude_squared(c.momentum_x,c.momentum_y,c.momentum_z);
    float dens = fmaxf(c.density, smallr);
    float eint = c.energy - 0.5f*msq/dens;
    return fmaxf((gamma-1.0f)*eint, dens*smallp);
}
__device__ __forceinline__ float compute_energy(const Prim& p,float gamma){
    float vsq = magnitude_squared(p.velocity_x,p.velocity_y,p.velocity_z);
    return (p.pressure/(gamma-1.0f)) + 0.5f*p.density*vsq;
}
__device__ __forceinline__ float sound_speed(const Prim& p,float gamma){
    return sqrtf(gamma*p.pressure/p.density);
}
__device__ __forceinline__ Prim conserved_2_primitive(const Cons& c,float gamma,float smallr,float smallc2){
    Prim p;
    p.density    = fmaxf(c.density, smallr);
    p.velocity_x = c.momentum_x/p.density;
    p.velocity_y = c.momentum_y/p.density;
    p.velocity_z = c.momentum_z/p.density;
    p.pressure   = compute_pressure(c,gamma,smallr,smallc2);
    return p;
}
__device__ __forceinline__ Cons primitive_2_conserved(const Prim& p,float gamma){
    Cons c;
    c.density    = p.density;
    c.momentum_x = p.velocity_x*p.density;
    c.momentum_y = p.velocity_y*p.density;
    c.momentum_z = p.velocity_z*p.density;
    c.energy     = compute_energy(p,gamma);
    return c;
}
// slope_type factor: 0=1st order, 1=minmod, 2=moncen
__device__ __forceinline__ float slope_moncen(float left,float middle,float right,int slope){
    float sl = middle-left, sr = right-middle, sc = 0.5f*(sl+sr);
    float factor = (float)slope;
    if (sl*sr <= 0.0f) return 0.0f;
    else if (sl > 0.0f){ float mm = fminf(sl,sr); return fminf(factor*mm, sc); }
    else              { float mm = fmaxf(sl,sr); return fmaxf(factor*mm, sc); }
}

// =================== Riemann solvers ========================================
__device__ __forceinline__ float hll_flux(float sl,float sr,float fl,float fr,float cl,float cr){
    return (sr*fl - sl*fr + sr*sl*(cr-cl)) / (sr-sl);
}
__device__ Cons hll_fluxes(Prim L,Prim R,float gamma,float smallr,float smallc2,bool llf){
    float smallp = smallc2/gamma;
    L.density  = fmaxf(L.density, smallr);  R.density  = fmaxf(R.density, smallr);
    L.pressure = fmaxf(L.pressure, smallp*L.density);
    R.pressure = fmaxf(R.pressure, smallp*R.density);
    float cl = sound_speed(L,gamma), cr = sound_speed(R,gamma);
    float sl, sr;
    if (llf){
        float smax = fmaxf(fabsf(L.velocity_x)+cl, fabsf(R.velocity_x)+cr);
        sl = -smax; sr = smax;
    } else {
        sl = fminf(fminf(L.velocity_x,R.velocity_x) - fmaxf(cl,cr), 0.0f);
        sr = fmaxf(fmaxf(L.velocity_x,R.velocity_x) + fmaxf(cl,cr), 0.0f);
    }
    Cons lc = primitive_2_conserved(L,gamma), rc = primitive_2_conserved(R,gamma);
    Cons lf, rf, flux;
    lf.density    = lc.momentum_x;
    lf.momentum_x = L.pressure + L.velocity_x*lc.momentum_x;
    lf.momentum_y = L.velocity_x*lc.momentum_y;
    lf.momentum_z = L.velocity_x*lc.momentum_z;
    lf.energy     = L.velocity_x*(L.pressure + lc.energy);
    rf.density    = rc.momentum_x;
    rf.momentum_x = R.pressure + R.velocity_x*rc.momentum_x;
    rf.momentum_y = R.velocity_x*rc.momentum_y;
    rf.momentum_z = R.velocity_x*rc.momentum_z;
    rf.energy     = R.velocity_x*(R.pressure + rc.energy);
    flux.density    = hll_flux(sl,sr,lf.density,   rf.density,   lc.density,   rc.density);
    flux.momentum_x = hll_flux(sl,sr,lf.momentum_x,rf.momentum_x,lc.momentum_x,rc.momentum_x);
    flux.momentum_y = hll_flux(sl,sr,lf.momentum_y,rf.momentum_y,lc.momentum_y,rc.momentum_y);
    flux.momentum_z = hll_flux(sl,sr,lf.momentum_z,rf.momentum_z,lc.momentum_z,rc.momentum_z);
    flux.energy     = hll_flux(sl,sr,lf.energy,    rf.energy,    lc.energy,    rc.energy);
    return flux;
}
__device__ Cons hllc_fluxes(Prim L,Prim R,float gamma,float smallr,float smallc2){
    float smallp = smallc2/gamma;
    L.density  = fmaxf(L.density, smallr);  R.density  = fmaxf(R.density, smallr);
    L.pressure = fmaxf(L.pressure, smallp*L.density);
    R.pressure = fmaxf(R.pressure, smallp*R.density);
    float rl=L.density, rr=R.density, ul=L.velocity_x, ur=R.velocity_x;
    float vl=L.velocity_y, vr=R.velocity_y, wl=L.velocity_z, wr=R.velocity_z;
    float pl=L.pressure, pr=R.pressure;
    float entho = 1.0f/(gamma-1.0f);
    float el=pl*entho, er=pr*entho;
    float ekinl=0.5f*rl*magnitude_squared(ul,vl,wl), ekinr=0.5f*rr*magnitude_squared(ur,vr,wr);
    float etotl=el+ekinl, etotr=er+ekinr;
    float cl=sound_speed(L,gamma), cr=sound_speed(R,gamma);
    float sl=fminf(ul,ur)-fmaxf(cl,cr), sr=fmaxf(ul,ur)+fmaxf(cl,cr);
    float rcl=rl*(ul-sl), rcr=rr*(sr-ur);
    float ustar=(rcr*ur+rcl*ul+(pl-pr))/(rcr+rcl);
    float pstar=(rcr*pl+rcl*pr+rcl*rcr*(ul-ur))/(rcr+rcl);
    float rstarl=rl*(sl-ul)/(sl-ustar), rstarr=rr*(sr-ur)/(sr-ustar);
    float etotstarl=((sl-ul)*etotl-pl*ul+pstar*ustar)/(sl-ustar);
    float etotstarr=((sr-ur)*etotr-pr*ur+pstar*ustar)/(sr-ustar);
    float ro,uo,vo,wo,po,etoto;
    if (sl>0.0f)          { ro=rl;     uo=ul;    vo=vl; wo=wl; po=pl;    etoto=etotl; }
    else if (ustar>0.0f)  { ro=rstarl; uo=ustar; vo=vl; wo=wl; po=pstar; etoto=etotstarl; }
    else if (sr>0.0f)     { ro=rstarr; uo=ustar; vo=vr; wo=wr; po=pstar; etoto=etotstarr; }
    else                  { ro=rr;     uo=ur;    vo=vr; wo=wr; po=pr;    etoto=etotr; }
    Cons flux;
    flux.density    = ro*uo;
    flux.momentum_x = ro*uo*uo + po;
    flux.momentum_y = ro*uo*vo;
    flux.momentum_z = ro*uo*wo;
    flux.energy     = (etoto+po)*uo;
    return flux;
}
__device__ Cons twoshock_fluxes(Prim L,Prim R,float gamma,float smallr,float smallc2){
    const float tiny = 1e-20f;
    L.density=fmaxf(L.density,smallr); R.density=fmaxf(R.density,smallr);
    L.pressure=fmaxf(L.pressure,tiny); R.pressure=fmaxf(R.pressure,tiny);
    float pratio = fmaxf(L.pressure,R.pressure)/fmaxf(fminf(L.pressure,R.pressure),tiny);
    if (pratio > 2.0f) return hllc_fluxes(L,R,gamma,smallr,smallc2);
    float qa=(gamma+1.0f)/(2.0f*gamma), gp1=gamma+1.0f;
    float cl=sqrtf(gamma*L.pressure*L.density), cr=sqrtf(gamma*R.pressure*R.density);
    float ps=fmaxf((cr*L.pressure+cl*R.pressure+cr*cl*(L.velocity_x-R.velocity_x))/(cr+cl),tiny);
    float old_ps=ps, ubl=0.0f, ubr=0.0f, dpdul=0.0f, dpdur=0.0f, delta;
    bool converged=false;
    for (int n=2;n<=8;n++){
        if (!converged){
            float zl=cl*sqrtf(1.0f+qa*(ps/L.pressure-1.0f));
            float zr=cr*sqrtf(1.0f+qa*(ps/R.pressure-1.0f));
            ubl=L.velocity_x-(ps-L.pressure)/zl;
            ubr=R.velocity_x+(ps-R.pressure)/zr;
            dpdul=-4.0f*zl*zl*zl/L.density/(4.0f*zl*zl/L.density-gp1*(ps-L.pressure));
            dpdur= 4.0f*zr*zr*zr/R.density/(4.0f*zr*zr/R.density-gp1*(ps-R.pressure));
            ps=fmaxf(ps+(ubr-ubl)*dpdur*dpdul/(dpdur-dpdul),tiny);
            delta=ps-old_ps; old_ps=ps;
            converged = fabsf(delta/ps) < 1e-7f;
        }
    }
    float pbar=ps;
    float ubar=ubl+(ubr-ubl)*dpdur/(dpdur-dpdul);
    float sn = (-ubar>=0.0f) ? 1.0f : -1.0f;
    Prim q0 = (sn<0.0f) ? L : R;
    float c0=sqrtf(fmaxf(gamma*q0.pressure/q0.density,tiny));
    float z0=c0*q0.density*sqrtf(fmaxf(1.0f+qa*(pbar/q0.pressure-1.0f),tiny));
    float dbar=1.0f/(1.0f/q0.density-(pbar-q0.pressure)/fmaxf(z0*z0,tiny));
    float cbar=sqrtf(fmaxf(gamma*pbar/dbar,tiny));
    float l0, lbar;
    if (pbar<q0.pressure){ l0=q0.velocity_x*sn+c0; lbar=sn*ubar+cbar; }
    else                 { l0=q0.velocity_x*sn+z0/q0.density; lbar=l0; }
    float width=fmaxf(l0-lbar,tiny);
    float frac=fminf(fmaxf((0.0f-lbar)/width,0.0f),1.0f);
    float pbv=q0.pressure*frac+pbar*(1.0f-frac);
    float dbv=q0.density*frac+dbar*(1.0f-frac);
    float ubv=q0.velocity_x*frac+ubar*(1.0f-frac);
    if (lbar>=0.0f){ pbv=pbar; dbv=dbar; ubv=ubar; }
    if (l0<0.0f){ pbv=q0.pressure; dbv=q0.density; ubv=q0.velocity_x; }
    Prim up = (ubv>0.0f) ? L : R;
    float etot=pbv/(gamma-1.0f)+0.5f*dbv*magnitude_squared(ubv,up.velocity_y,up.velocity_z);
    Cons flux;
    flux.density    = dbv*ubv;
    flux.momentum_x = flux.density*ubv + pbv;
    flux.momentum_y = flux.density*up.velocity_y;
    flux.momentum_z = flux.density*up.velocity_z;
    flux.energy     = (etot+pbv)*ubv;
    return flux;
}
__device__ __forceinline__ Cons riemann_fluxes(Prim L,Prim R,float gamma,float smallr,float smallc2,int riemann){
    if      (riemann==SOLVER_LLF)      return hll_fluxes(L,R,gamma,smallr,smallc2,true);
    else if (riemann==SOLVER_HLL)      return hll_fluxes(L,R,gamma,smallr,smallc2,false);
    else if (riemann==SOLVER_HLLC)     return hllc_fluxes(L,R,gamma,smallr,smallc2);
    else if (riemann==SOLVER_TWOSHOCK) return twoshock_fluxes(L,R,gamma,smallr,smallc2);
    else                               return hll_fluxes(L,R,gamma,smallr,smallc2,true);
}

// =================== Local PPM (slope_type=10) helpers ======================
__device__ __forceinline__ float lppm_minmod3(float a,float b,float c){
    if (a*b<=0.0f || a*c<=0.0f) return 0.0f;
    return copysignf(fminf(fabsf(a),fminf(fabsf(b),fabsf(c))), a);
}
__device__ __forceinline__ float lppm_mc(float l,float m,float r){
    return lppm_minmod3(2.0f*(m-l), 2.0f*(r-m), 0.5f*(r-l));
}
__device__ __forceinline__ void lppm_monotonize(float ql,float q0,float qr,float& lo,float& hi){
    float dq=qr-ql;
    float diff=(q0-0.5f*(ql+qr))*dq;
    if ((qr-q0)*(q0-ql) <= 0.0f){ lo=q0; hi=q0; }
    else if (diff > dq*dq/6.0f){ lo=3.0f*q0-2.0f*qr; hi=qr; }
    else if (diff < -dq*dq/6.0f){ lo=ql; hi=3.0f*q0-2.0f*ql; }
    else { lo=ql; hi=qr; }
}
__device__ __forceinline__ void lppm_edges(float qm,float q0,float qp,float& ql,float& qr){
    float slope=0.25f*(qp-qm);
    float curve=(qm-2.0f*q0+qp)/12.0f;
    lppm_monotonize(q0-slope+curve, q0, q0+slope+curve, ql, qr);
}
__device__ __forceinline__ float lppm_avg(float ql,float qa,float qr,float lo,float hi){
    float b=-4.0f*ql+6.0f*qa-2.0f*qr;
    float c=3.0f*(ql-2.0f*qa+qr);
    return ql + 0.5f*b*(lo+hi) + c*(lo*lo+lo*hi+hi*hi)/3.0f;
}
__device__ __forceinline__ bool strong_pressure_jump(float a,float b,float c){
    const float tiny=1e-20f;
    float hi=fmaxf(a,fmaxf(b,c));
    float lo=fmaxf(fminf(a,fminf(b,c)),tiny);
    return hi/lo > 2.0f;
}
__device__ __forceinline__ Prim lppm_half_slope(const Prim& l,const Prim& m,const Prim& r){
    Prim s;
    s.density   =0.5f*lppm_mc(l.density,   m.density,   r.density);
    s.velocity_x=0.5f*lppm_mc(l.velocity_x,m.velocity_x,r.velocity_x);
    s.velocity_y=0.5f*lppm_mc(l.velocity_y,m.velocity_y,r.velocity_y);
    s.velocity_z=0.5f*lppm_mc(l.velocity_z,m.velocity_z,r.velocity_z);
    s.pressure  =0.5f*lppm_mc(l.pressure,  m.pressure,  r.pressure);
    return s;
}
__device__ __forceinline__ Prim transverse_source(const Prim& m,const Prim& s,float gamma,int dir){
    Prim src;
    float vdir,svel;
    if (dir==0){ vdir=m.velocity_x; svel=s.velocity_x; }
    else if (dir==1){ vdir=m.velocity_y; svel=s.velocity_y; }
    else { vdir=m.velocity_z; svel=s.velocity_z; }
    src.density   =-vdir*s.density - svel*m.density;
    src.velocity_x=-vdir*s.velocity_x;
    src.velocity_y=-vdir*s.velocity_y;
    src.velocity_z=-vdir*s.velocity_z;
    if (dir==0) src.velocity_x -= s.pressure/m.density;
    if (dir==1) src.velocity_y -= s.pressure/m.density;
    if (dir==2) src.velocity_z -= s.pressure/m.density;
    src.pressure  =-vdir*s.pressure - svel*gamma*m.pressure;
    return src;
}
__device__ __forceinline__ void add_transverse_source(Prim& q,const Prim& src,float dtdx,const Prim& m){
    q.density   +=dtdx*src.density;
    q.velocity_x+=dtdx*src.velocity_x;
    q.velocity_y+=dtdx*src.velocity_y;
    q.velocity_z+=dtdx*src.velocity_z;
    q.pressure  +=dtdx*src.pressure;
    if (q.density<=0.0f || q.pressure<=0.0f) q=m;
}
__device__ __forceinline__ Prim prim_add(const Prim& a,const Prim& b){
    Prim c;
    c.density=a.density+b.density;
    c.velocity_x=a.velocity_x+b.velocity_x;
    c.velocity_y=a.velocity_y+b.velocity_y;
    c.velocity_z=a.velocity_z+b.velocity_z;
    c.pressure=a.pressure+b.pressure;
    return c;
}
__device__ __forceinline__ Prim sg_load(const Tile& sg,int i,int j,int k){
    int q=sg.idx(i,j,k);
    Prim p;
    p.density   =__half2float(sg.d[q]);
    p.velocity_x=__half2float(sg.vx[q]);
    p.velocity_y=__half2float(sg.vy[q]);
    p.velocity_z=__half2float(sg.vz[q]);
    p.pressure  =__half2float(sg.p[q]);
    return p;
}
__device__ void local_ppm_trace(const Prim& l,const Prim& m,const Prim& r,float gamma,float dtdx,Prim& qplus,Prim& qminus){
    const float smallr=1e-10f, smallp=1e-30f;
    float dl,dr,ul,ur,vl,vr,wl,wr,pl,pr,cs,aa,bb;
    lppm_edges(l.density,   m.density,   r.density,   dl,dr);
    lppm_edges(l.velocity_x,m.velocity_x,r.velocity_x,ul,ur);
    lppm_edges(l.velocity_y,m.velocity_y,r.velocity_y,vl,vr);
    lppm_edges(l.velocity_z,m.velocity_z,r.velocity_z,wl,wr);
    lppm_edges(l.pressure,  m.pressure,  r.pressure,  pl,pr);
    cs=sqrtf(gamma*fmaxf(m.pressure,smallp)/fmaxf(m.density,smallr));
    lppm_monotonize(dl,m.density,   dr,aa,bb); dl=aa; dr=bb;
    lppm_monotonize(ul,m.velocity_x,ur,aa,bb); ul=aa; ur=bb;
    lppm_monotonize(vl,m.velocity_y,vr,aa,bb); vl=aa; vr=bb;
    lppm_monotonize(wl,m.velocity_z,wr,aa,bb); wl=aa; wr=bb;
    lppm_monotonize(pl,m.pressure,  pr,aa,bb); pl=aa; pr=bb;
    if (dl<=0.0f || dr<=0.0f){ dl=m.density; dr=m.density; }
    if (pl<=0.0f || pr<=0.0f){ pl=m.pressure; pr=m.pressure; }
    dl=fmaxf(dl,smallr); dr=fmaxf(dr,smallr);
    pl=fmaxf(pl,smallp); pr=fmaxf(pr,smallp);
    for (int side=0; side<2; side++){
        bool right = (side==0);
        float rg,ug,vg,wg,pg;
        if (right){ rg=dr; ug=ur; vg=vr; wg=wr; pg=pr; }
        else      { rg=dl; ug=ul; vg=vl; wg=wl; pg=pl; }
        float od=0.0f,ou=0.0f,ov=0.0f,ow=0.0f,op=0.0f;
        float lambda1=m.velocity_x-cs, lambda2=m.velocity_x, lambda3=m.velocity_x+cs;
        for (int wave=1; wave<=3; wave++){
            float lam = (wave==1)?lambda1:((wave==2)?lambda2:lambda3);
            if (right){ if (!(lam>0.0f)) continue; }
            else      { if (!(lam<0.0f)) continue; }
            float sigma=fminf(fabsf(lam)*dtdx,1.0f);
            float a,b;
            if (right){ a=1.0f-sigma; b=1.0f; } else { a=0.0f; b=sigma; }
            float du =lppm_avg(ul,m.velocity_x,ur,a,b)-ug;
            float dpr=lppm_avg(pl,m.pressure,  pr,a,b)-pg;
            if (wave==1 || wave==3){
                float amp;
                if (wave==1){ amp=-m.density*du/(2.0f*cs)+dpr/(2.0f*cs*cs); ou += (-cs/m.density)*amp; }
                else        { amp= m.density*du/(2.0f*cs)+dpr/(2.0f*cs*cs); ou += ( cs/m.density)*amp; }
                od += amp;
                op += cs*cs*amp;
            } else {
                od += (lppm_avg(dl,m.density,   dr,a,b)-rg-dpr/(cs*cs));
                ov += (lppm_avg(vl,m.velocity_y,vr,a,b)-vg);
                ow += (lppm_avg(wl,m.velocity_z,wr,a,b)-wg);
            }
        }
        Prim q;
        q.density=rg+od; q.velocity_x=ug+ou; q.velocity_y=vg+ov; q.velocity_z=wg+ow; q.pressure=pg+op;
        if (q.density<=0.0f || q.pressure<=0.0f) q=m;
        if (right) qplus=q; else qminus=q;
    }
}

// =================== block-min reduction + double atomicMin =================
__device__ __forceinline__ double atomicMinDouble(double* addr,double val){
    unsigned long long* a=(unsigned long long*)addr;
    unsigned long long old=*a, assumed;
    do {
        assumed=old;
        double cur=__longlong_as_double(assumed);
        if (cur<=val) break;
        old=atomicCAS(a, assumed, __double_as_longlong(val));
    } while (assumed!=old);
    return __longlong_as_double(old);
}
__device__ double blockReduceMin(double v){
    for (int off=warpSize/2; off>0; off>>=1)
        v=fmin(v, __shfl_down_sync(0xffffffffu, v, off));
    __shared__ double s[32];
    int lane=threadIdx.x & (warpSize-1);
    int wid =threadIdx.x / warpSize;
    if (lane==0) s[wid]=v;
    __syncthreads();
    int nwarps=blockDim.x/warpSize;
    v=(threadIdx.x < nwarps) ? s[lane] : HUGE_VAL;
    if (wid==0)
        for (int off=warpSize/2; off>0; off>>=1)
            v=fmin(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

// index_1Dto3D (0-based)
__device__ __forceinline__ void idx1Dto3D(int index,int nx,int ny,int& xid,int& yid,int& zid){
    zid=index/(nx*ny);
    yid=(index - zid*nx*ny)/nx;
    xid=index - zid*nx*ny - yid*nx;
}
__device__ __forceinline__ void idx1Dto2D(int index,int nx,int& xid,int& yid){
    yid=index/nx; xid=index-yid*nx;
}

// ============================================================================
// THE KERNEL
// ============================================================================
#ifndef CUMINBLK
#define CUMINBLK 3      // min blocks/SM: 3 matches the Fortran kernel's tuned occupancy
#endif
__global__ void __launch_bounds__(256, CUMINBLK) hydro_integrator_kernel_cuda(
    Oct* __restrict__ grid, float* __restrict__ uold, float* __restrict__ unew,
    const float* __restrict__ f, const int* __restrict__ father, const int* __restrict__ nbor,
    int head_idx, int num_subgrids, int ngridmax, int ilevel, int levelmin, int levelmax,
    float gamma, float smallr, float smallc2, float dt, float dx,
    int slope, int riemann, const double* __restrict__ constant_gravity, int base_write,
    float courant_factor, double* __restrict__ dt_out, int cfl_sqrt3
#ifdef TURB
    , const float* __restrict__ afield_now, const float* __restrict__ d_skip,
    float boxlen, float turb_min_rho, int do_turb
#endif
){
    // ---- shared scratch: DYNAMIC shared (all thalf arrays first for alignment,
    // refined uchar last). Lets the ns4 tile exceed the 48KB static cap.
    extern __shared__ char smem[];
    thalf* sp = (thalf*)smem;
    thalf* ls_d=sp; thalf* ls_vx=sp+TILE; thalf* ls_vy=sp+2*TILE; thalf* ls_vz=sp+3*TILE; thalf* ls_p=sp+4*TILE;
    thalf* fp = sp + 5*TILE;
    thalf *lix_d=fp,*lix_vx=fp+IX_SZ,*lix_vy=fp+2*IX_SZ,*lix_vz=fp+3*IX_SZ,*lix_p=fp+4*IX_SZ; fp+=5*IX_SZ;
    thalf *rix_d=fp,*rix_vx=fp+IX_SZ,*rix_vy=fp+2*IX_SZ,*rix_vz=fp+3*IX_SZ,*rix_p=fp+4*IX_SZ; fp+=5*IX_SZ;
    thalf *liy_d=fp,*liy_vx=fp+IY_SZ,*liy_vy=fp+2*IY_SZ,*liy_vz=fp+3*IY_SZ,*liy_p=fp+4*IY_SZ; fp+=5*IY_SZ;
    thalf *riy_d=fp,*riy_vx=fp+IY_SZ,*riy_vy=fp+2*IY_SZ,*riy_vz=fp+3*IY_SZ,*riy_p=fp+4*IY_SZ; fp+=5*IY_SZ;
    thalf *liz_d=fp,*liz_vx=fp+IZ_SZ,*liz_vy=fp+2*IZ_SZ,*liz_vz=fp+3*IZ_SZ,*liz_p=fp+4*IZ_SZ; fp+=5*IZ_SZ;
    thalf *riz_d=fp,*riz_vx=fp+IZ_SZ,*riz_vy=fp+2*IZ_SZ,*riz_vz=fp+3*IZ_SZ,*riz_p=fp+4*IZ_SZ; fp+=5*IZ_SZ;
    unsigned char* ls_ref=(unsigned char*)fp;

    Tile ls{ls_d,ls_vx,ls_vy,ls_vz,ls_p,ls_ref};
    Face lix{lix_d,lix_vx,lix_vy,lix_vz,lix_p,IX_NX,IX_NY};
    Face rix{rix_d,rix_vx,rix_vy,rix_vz,rix_p,IX_NX,IX_NY};
    Face liy{liy_d,liy_vx,liy_vy,liy_vz,liy_p,IY_NX,IY_NY};
    Face riy{riy_d,riy_vx,riy_vy,riy_vz,riy_p,IY_NX,IY_NY};
    Face liz{liz_d,liz_vx,liz_vy,liz_vz,liz_p,IZ_NX,IZ_NY};
    Face riz{riz_d,riz_vx,riz_vy,riz_vz,riz_p,IZ_NX,IZ_NY};

    float dtdx = dt/dx;
    int block_idx  = blockIdx.x;     // already 0-based
    int thread_idx = threadIdx.x;    // 0-based
    if (block_idx >= num_subgrids) return;
    int subgrid_idx = head_idx + block_idx;   // 1-based

    // ========================================================================
    // 1) subgrid_conserved_2_primitive : cooperative LOAD + c2p + dt-fold + grav
    // ========================================================================
    {
        const int work_size = 2*NS+4;            // 10
        const int owned_lo = 2, owned_hi = 2*NS+1;
        double dt_min = HUGE_VAL;

#ifdef WIDELOAD
        // -------------------------------------------------------------------
        // STAGE 2: float4 wide loads. The uold cell dim is contiguous (cell is
        // fastest), so cells 1..4 / 5..8 of an oct are 16-byte-aligned float4
        // groups (device base 256B-aligned; var stride = 8 floats = 2 float4).
        // One thread loads a whole half-oct (4 cells) as 5 float4 = 1.25 load
        // instr/cell vs 5 scalar LDG.32/cell -> cuts the L1/TEX request rate
        // (the throttle that walls GRAV=0). c2p/dt/predictor stay per-cell.
        // -------------------------------------------------------------------
        const int n_halfoct = SUBGRIDSIZE * 2;   // 125*2 = 250
        for (int w = thread_idx; w < n_halfoct; w += blockDim.x){
            int oct = w >> 1;                    // nbor index 0..124
            int half = w & 1;                    // 0 -> cells 1..4, 1 -> cells 5..8
            int src = NBORA(oct+1, subgrid_idx); // 1-based oct
            int c0 = half*4;                     // 0-based cell base (0 or 4)
            int isg,jsg,ksg; idx1Dto3D(oct, NSP2, NSP2, isg, jsg, ksg);

            const float4* b = reinterpret_cast<const float4*>(&uold[((size_t)(src-1)*NVAR)*8 + c0]);
            float4 vd=b[0], vmx=b[2], vmy=b[4], vmz=b[6], ve=b[8];  // var stride = 2 float4

            #pragma unroll
            for (int cc=0; cc<4; cc++){
                int cell_idx = c0 + cc + 1;      // 1-based
                int ci,cj,ck; idx1Dto3D(cell_idx-1, 2, 2, ci, cj, ck);
                int i=ci+2*isg, j=cj+2*jsg, k=ck+2*ksg;
                Cons conserved;
                conserved.density   =(&vd.x)[cc];
                conserved.momentum_x=(&vmx.x)[cc];
                conserved.momentum_y=(&vmy.x)[cc];
                conserved.momentum_z=(&vmz.x)[cc];
                conserved.energy    =(&ve.x)[cc];

                Prim primitive = conserved_2_primitive(conserved,gamma,smallr,smallc2);
                if (base_write){
                    if ( i>=owned_lo && i<=owned_hi && j>=owned_lo && j<=owned_hi &&
                         k>=owned_lo && k<=owned_hi && grid[src-1].refined[cell_idx-1]==0 ){
                        float cs=sqrtf(gamma*primitive.pressure/primitive.density);
                        float ctot;
                        if (cfl_sqrt3){
                            ctot=sqrtf(3.0f)*fmaxf(fabsf(primitive.velocity_x)+cs,
                                       fmaxf(fabsf(primitive.velocity_y)+cs, fabsf(primitive.velocity_z)+cs));
                        } else {
                            ctot=fabsf(primitive.velocity_x)+fabsf(primitive.velocity_y)+fabsf(primitive.velocity_z)+3.0f*cs;
                        }
                        float grav;
#if DT_USE_F
                        grav=fabsf(FARR(cell_idx,1,src))+fabsf(FARR(cell_idx,2,src))+fabsf(FARR(cell_idx,3,src));
#else
                        grav=fabsf((float)constant_gravity[0])+fabsf((float)constant_gravity[1])+fabsf((float)constant_gravity[2]);
#endif
                        grav=fmaxf(grav*dx/(ctot*ctot), 0.0001f);
                        float dtl=dx/ctot*(sqrtf(1.0f+2.0f*courant_factor*grav)-1.0f)/grav;
                        dt_min=fmin(dt_min,(double)dtl);
                    }
                }
#if PRED_USE_F
                primitive.velocity_x += FARR(cell_idx,1,src)*0.5f*dt;
                primitive.velocity_y += FARR(cell_idx,2,src)*0.5f*dt;
                primitive.velocity_z += FARR(cell_idx,3,src)*0.5f*dt;
#elif defined(FEXP_CG_FP64)
                primitive.velocity_x = (float)((double)primitive.velocity_x + constant_gravity[0]*0.5*(double)dt);
                primitive.velocity_y = (float)((double)primitive.velocity_y + constant_gravity[1]*0.5*(double)dt);
                primitive.velocity_z = (float)((double)primitive.velocity_z + constant_gravity[2]*0.5*(double)dt);
#else
                primitive.velocity_x += (float)constant_gravity[0]*0.5f*dt;
                primitive.velocity_y += (float)constant_gravity[1]*0.5f*dt;
                primitive.velocity_z += (float)constant_gravity[2]*0.5f*dt;
#endif
                int q=ls.idx(i,j,k);
                ls.d[q] =__float2half(primitive.density);
                ls.vx[q]=__float2half(primitive.velocity_x);
                ls.vy[q]=__float2half(primitive.velocity_y);
                ls.vz[q]=__float2half(primitive.velocity_z);
                ls.p[q] =__float2half(primitive.pressure);
                ls.ref[q]=(grid[src-1].refined[cell_idx-1]!=0) ? 1 : 0;
            }
        }
        if (base_write){
            dt_min=blockReduceMin(dt_min);
            if (thread_idx==0) atomicMinDouble(&dt_out[0], dt_min);
        }
        __syncthreads();
#else
        // software-pipelined prefetch (mirrors the Fortran load loop)
        int work_idx = thread_idx;
        bool have_next = (work_idx <= work_size*work_size*work_size - 1);
        Cons cons_n; int src_n=0, cell_n=0, in_=0, jn_=0, kn_=0;
        if (have_next){
            int isg,jsg,ksg;
            idx1Dto3D(work_idx/8, work_size/2, work_size/2, isg, jsg, ksg);
            src_n = NBORA(work_idx/8 + 1, subgrid_idx);
            cell_n = (work_idx & 7) + 1;
            int ci,cj,ck; idx1Dto3D(cell_n-1, 2, 2, ci, cj, ck);
            in_=ci+2*isg; jn_=cj+2*jsg; kn_=ck+2*ksg;
            cons_n.density   =UOLD(cell_n,1,src_n);
            cons_n.momentum_x=UOLD(cell_n,2,src_n);
            cons_n.momentum_y=UOLD(cell_n,3,src_n);
            cons_n.momentum_z=UOLD(cell_n,4,src_n);
            cons_n.energy    =UOLD(cell_n,5,src_n);
        }
        while (have_next){
            Cons conserved = cons_n;
            int source_idx = src_n, cell_idx = cell_n;
            int i=in_, j=jn_, k=kn_;

            work_idx += blockDim.x;
            have_next = (work_idx <= work_size*work_size*work_size - 1);
            if (have_next){
                int isg,jsg,ksg;
                idx1Dto3D(work_idx/8, work_size/2, work_size/2, isg, jsg, ksg);
                src_n = NBORA(work_idx/8 + 1, subgrid_idx);
                cell_n = (work_idx & 7) + 1;
                int ci,cj,ck; idx1Dto3D(cell_n-1, 2, 2, ci, cj, ck);
                in_=ci+2*isg; jn_=cj+2*jsg; kn_=ck+2*ksg;
                cons_n.density   =UOLD(cell_n,1,src_n);
                cons_n.momentum_x=UOLD(cell_n,2,src_n);
                cons_n.momentum_y=UOLD(cell_n,3,src_n);
                cons_n.momentum_z=UOLD(cell_n,4,src_n);
                cons_n.energy    =UOLD(cell_n,5,src_n);
            }

            Prim primitive = conserved_2_primitive(conserved,gamma,smallr,smallc2);

            // folded cmpdt (single-level): CFL dt for owned non-refined leaf cells
            if (base_write){
                if ( i>=owned_lo && i<=owned_hi && j>=owned_lo && j<=owned_hi &&
                     k>=owned_lo && k<=owned_hi && grid[source_idx-1].refined[cell_idx-1]==0 ){
                    float cs=sqrtf(gamma*primitive.pressure/primitive.density);
                    float ctot;
                    if (cfl_sqrt3){
                        ctot=sqrtf(3.0f)*fmaxf(fabsf(primitive.velocity_x)+cs,
                                   fmaxf(fabsf(primitive.velocity_y)+cs, fabsf(primitive.velocity_z)+cs));
                    } else {
                        ctot=fabsf(primitive.velocity_x)+fabsf(primitive.velocity_y)+fabsf(primitive.velocity_z)+3.0f*cs;
                    }
                    float grav;
#if DT_USE_F
                    grav=fabsf(FARR(cell_idx,1,source_idx))+fabsf(FARR(cell_idx,2,source_idx))+fabsf(FARR(cell_idx,3,source_idx));
#else
                    grav=fabsf((float)constant_gravity[0])+fabsf((float)constant_gravity[1])+fabsf((float)constant_gravity[2]);
#endif
                    grav=fmaxf(grav*dx/(ctot*ctot), 0.0001f);
                    float dtl=dx/ctot*(sqrtf(1.0f+2.0f*courant_factor*grav)-1.0f)/grav;
                    dt_min=fmin(dt_min,(double)dtl);
                }
            }

            // gravity predictor
#if PRED_USE_F
#ifdef FEXP_PRED_BCAST
            // broadcast f-read: all threads hit the SAME address (1 cache line).
            // f is all-zeros in a turb run, so the VALUE is identical to the
            // divergent read (0) -- only the access pattern (1 line vs many)
            // differs. Isolates MLP/divergence from raw load-instruction count.
            primitive.velocity_x += FARR(1,1,1)*0.5f*dt;
            primitive.velocity_y += FARR(1,2,1)*0.5f*dt;
            primitive.velocity_z += FARR(1,3,1)*0.5f*dt;
#else
            primitive.velocity_x += FARR(cell_idx,1,source_idx)*0.5f*dt;
            primitive.velocity_y += FARR(cell_idx,2,source_idx)*0.5f*dt;
            primitive.velocity_z += FARR(cell_idx,3,source_idx)*0.5f*dt;
#endif
#elif defined(FEXP_CG_FP64)
            // [diagnostic only] fp64 constant-gravity predictor: this single
            // fp64-promoted term (constant_gravity is real(kind=8)) cost the
            // ENTIRE GRAV=0<->GRAV=1 gap (~44%) on GA102's 1:64 fp64 path. Kept
            // behind a flag to reproduce; the default below is fp32.
            primitive.velocity_x = (float)((double)primitive.velocity_x + constant_gravity[0]*0.5*(double)dt);
            primitive.velocity_y = (float)((double)primitive.velocity_y + constant_gravity[1]*0.5*(double)dt);
            primitive.velocity_z = (float)((double)primitive.velocity_z + constant_gravity[2]*0.5*(double)dt);
#else
            // fp32 constant-gravity predictor (cast cg to float ONCE): the whole
            // sim is fp32, so doing this term in fp64 was pointless precision at
            // 1:64 throughput. This is the fix that closes the GRAV=0 gap.
            primitive.velocity_x += (float)constant_gravity[0]*0.5f*dt;
            primitive.velocity_y += (float)constant_gravity[1]*0.5f*dt;
            primitive.velocity_z += (float)constant_gravity[2]*0.5f*dt;
#endif
            int q=ls.idx(i,j,k);
            ls.d[q] =__float2half(primitive.density);
            ls.vx[q]=__float2half(primitive.velocity_x);
            ls.vy[q]=__float2half(primitive.velocity_y);
            ls.vz[q]=__float2half(primitive.velocity_z);
            ls.p[q] =__float2half(primitive.pressure);
            ls.ref[q]=(grid[source_idx-1].refined[cell_idx-1]!=0) ? 1 : 0;
        }
        if (base_write){
            dt_min=blockReduceMin(dt_min);
            if (thread_idx==0) atomicMinDouble(&dt_out[0], dt_min);
        }
        __syncthreads();
#endif // WIDELOAD
    }

#ifndef PERF_MEMFLOOR   // diagnostic: skip trace+riemann (+2 barriers, interface-tile traffic)
    // ========================================================================
    // 2) trace_3d : MUSCL-Hancock / Local-PPM reconstruction over the tile
    // ========================================================================
    {
        const int work_size = 2*NS+2;            // 8
        float smallp = smallr*smallc2;
        for (int work_idx=thread_idx; work_idx <= work_size*work_size*work_size-1; work_idx += blockDim.x){
            int i,j,k; idx1Dto3D(work_idx, work_size, work_size, i, j, k);
            i+=1; j+=1; k+=1;

            Prim cell = sg_load(ls,i,j,k);
            Prim m0 = cell;
            Prim slopes_x, slopes_y, slopes_z;

            bool use_ppm = (slope==10);
            if (use_ppm){
                if ( strong_pressure_jump(__half2float(ls.p[ls.idx(i-1,j,k)]),m0.pressure,__half2float(ls.p[ls.idx(i+1,j,k)])) ||
                     strong_pressure_jump(__half2float(ls.p[ls.idx(i,j-1,k)]),m0.pressure,__half2float(ls.p[ls.idx(i,j+1,k)])) ||
                     strong_pressure_jump(__half2float(ls.p[ls.idx(i,j,k-1)]),m0.pressure,__half2float(ls.p[ls.idx(i,j,k+1)])) )
                    use_ppm=false;
            }

            Prim srcx, srcy, srcz, na, nb;
            if (use_ppm){
                na=sg_load(ls,i-1,j,k); nb=sg_load(ls,i+1,j,k);
                srcx=transverse_source(m0, lppm_half_slope(na,m0,nb), gamma, 0);
                na=sg_load(ls,i,j-1,k); nb=sg_load(ls,i,j+1,k);
                srcy=transverse_source(m0, lppm_half_slope(na,m0,nb), gamma, 1);
                na=sg_load(ls,i,j,k-1); nb=sg_load(ls,i,j,k+1);
                srcz=transverse_source(m0, lppm_half_slope(na,m0,nb), gamma, 2);
            } else {
                int es = slope; if (slope==10) es=2;
                #define LSF(arr,a,b,c) __half2float(ls.arr[ls.idx(a,b,c)])
                slopes_x.density   =0.5f*slope_moncen(LSF(d, i-1,j,k),cell.density,   LSF(d, i+1,j,k),es);
                slopes_x.velocity_x=0.5f*slope_moncen(LSF(vx,i-1,j,k),cell.velocity_x,LSF(vx,i+1,j,k),es);
                slopes_x.velocity_y=0.5f*slope_moncen(LSF(vy,i-1,j,k),cell.velocity_y,LSF(vy,i+1,j,k),es);
                slopes_x.velocity_z=0.5f*slope_moncen(LSF(vz,i-1,j,k),cell.velocity_z,LSF(vz,i+1,j,k),es);
                slopes_x.pressure  =0.5f*slope_moncen(LSF(p, i-1,j,k),cell.pressure,  LSF(p, i+1,j,k),es);
                slopes_y.density   =0.5f*slope_moncen(LSF(d, i,j-1,k),cell.density,   LSF(d, i,j+1,k),es);
                slopes_y.velocity_x=0.5f*slope_moncen(LSF(vx,i,j-1,k),cell.velocity_x,LSF(vx,i,j+1,k),es);
                slopes_y.velocity_y=0.5f*slope_moncen(LSF(vy,i,j-1,k),cell.velocity_y,LSF(vy,i,j+1,k),es);
                slopes_y.velocity_z=0.5f*slope_moncen(LSF(vz,i,j-1,k),cell.velocity_z,LSF(vz,i,j+1,k),es);
                slopes_y.pressure  =0.5f*slope_moncen(LSF(p, i,j-1,k),cell.pressure,  LSF(p, i,j+1,k),es);
                slopes_z.density   =0.5f*slope_moncen(LSF(d, i,j,k-1),cell.density,   LSF(d, i,j,k+1),es);
                slopes_z.velocity_x=0.5f*slope_moncen(LSF(vx,i,j,k-1),cell.velocity_x,LSF(vx,i,j,k+1),es);
                slopes_z.velocity_y=0.5f*slope_moncen(LSF(vy,i,j,k-1),cell.velocity_y,LSF(vy,i,j,k+1),es);
                slopes_z.velocity_z=0.5f*slope_moncen(LSF(vz,i,j,k-1),cell.velocity_z,LSF(vz,i,j,k+1),es);
                slopes_z.pressure  =0.5f*slope_moncen(LSF(p, i,j,k-1),cell.pressure,  LSF(p, i,j,k+1),es);
                #undef LSF
                Prim st;
                st.density   =-cell.velocity_x*slopes_x.density   -cell.velocity_y*slopes_y.density   -cell.velocity_z*slopes_z.density   -(slopes_x.velocity_x+slopes_y.velocity_y+slopes_z.velocity_z)*cell.density;
                st.velocity_x=-cell.velocity_x*slopes_x.velocity_x-cell.velocity_y*slopes_y.velocity_x-cell.velocity_z*slopes_z.velocity_x-slopes_x.pressure/cell.density;
                st.velocity_y=-cell.velocity_x*slopes_x.velocity_y-cell.velocity_y*slopes_y.velocity_y-cell.velocity_z*slopes_z.velocity_y-slopes_y.pressure/cell.density;
                st.velocity_z=-cell.velocity_x*slopes_x.velocity_z-cell.velocity_y*slopes_y.velocity_z-cell.velocity_z*slopes_z.velocity_z-slopes_z.pressure/cell.density;
                st.pressure  =-cell.velocity_x*slopes_x.pressure  -cell.velocity_y*slopes_y.pressure  -cell.velocity_z*slopes_z.pressure  -(slopes_x.velocity_x+slopes_y.velocity_y+slopes_z.velocity_z)*gamma*cell.pressure;
                cell.density   +=dtdx*st.density;
                cell.velocity_x+=dtdx*st.velocity_x;
                cell.velocity_y+=dtdx*st.velocity_y;
                cell.velocity_z+=dtdx*st.velocity_z;
                cell.pressure  +=dtdx*st.pressure;
            }

            Prim fl, fr, qp, qn, rl, rm, rr, tsrc;

            // ----- X faces -----
            if (use_ppm){
                na=sg_load(ls,i-1,j,k); nb=sg_load(ls,i+1,j,k);
                local_ppm_trace(na,m0,nb,gamma,dtdx,fl,fr);
                tsrc=prim_add(srcy,srcz);
                add_transverse_source(fl,tsrc,dtdx,m0);
                add_transverse_source(fr,tsrc,dtdx,m0);
            } else {
                fr.density=cell.density-slopes_x.density; fr.velocity_x=cell.velocity_x-slopes_x.velocity_x; fr.velocity_y=cell.velocity_y-slopes_x.velocity_y; fr.velocity_z=cell.velocity_z-slopes_x.velocity_z; fr.pressure=cell.pressure-slopes_x.pressure;
                fl.density=cell.density+slopes_x.density; fl.velocity_x=cell.velocity_x+slopes_x.velocity_x; fl.velocity_y=cell.velocity_y+slopes_x.velocity_y; fl.velocity_z=cell.velocity_z+slopes_x.velocity_z; fl.pressure=cell.pressure+slopes_x.pressure;
            }
            if (i>1 && (j>1 && j<work_size) && (k>1 && k<work_size)){
                int q=rix.idx(i-2,j-2,k-2);
                rix.d[q]=__float2half(fr.density); rix.vx[q]=__float2half(fr.velocity_x); rix.vy[q]=__float2half(fr.velocity_y); rix.vz[q]=__float2half(fr.velocity_z); rix.p[q]=__float2half(fr.pressure);
                if (__half2float(rix.d[q])<smallr) rix.d[q]=ls.d[ls.idx(i,j,k)];
                if (__half2float(rix.p[q])<smallp) rix.p[q]=ls.p[ls.idx(i,j,k)];
            }
            if (i<work_size && (j>1 && j<work_size) && (k>1 && k<work_size)){
                int q=lix.idx(i-1,j-2,k-2);
                lix.d[q]=__float2half(fl.density); lix.vx[q]=__float2half(fl.velocity_x); lix.vy[q]=__float2half(fl.velocity_y); lix.vz[q]=__float2half(fl.velocity_z); lix.p[q]=__float2half(fl.pressure);
                if (__half2float(lix.d[q])<smallr) lix.d[q]=ls.d[ls.idx(i,j,k)];
                if (__half2float(lix.p[q])<smallp) lix.p[q]=ls.p[ls.idx(i,j,k)];
            }

            // ----- Y faces (rotate vx<-vy<-vz<-vx) -----
            if (use_ppm){
                na=sg_load(ls,i,j-1,k); nb=sg_load(ls,i,j+1,k);
                rl.density=na.density; rl.velocity_x=na.velocity_y; rl.velocity_y=na.velocity_z; rl.velocity_z=na.velocity_x; rl.pressure=na.pressure;
                rm.density=m0.density; rm.velocity_x=m0.velocity_y; rm.velocity_y=m0.velocity_z; rm.velocity_z=m0.velocity_x; rm.pressure=m0.pressure;
                rr.density=nb.density; rr.velocity_x=nb.velocity_y; rr.velocity_y=nb.velocity_z; rr.velocity_z=nb.velocity_x; rr.pressure=nb.pressure;
                local_ppm_trace(rl,rm,rr,gamma,dtdx,qp,qn);
                fl.density=qp.density; fl.velocity_x=qp.velocity_z; fl.velocity_y=qp.velocity_x; fl.velocity_z=qp.velocity_y; fl.pressure=qp.pressure;
                fr.density=qn.density; fr.velocity_x=qn.velocity_z; fr.velocity_y=qn.velocity_x; fr.velocity_z=qn.velocity_y; fr.pressure=qn.pressure;
                tsrc=prim_add(srcx,srcz);
                add_transverse_source(fl,tsrc,dtdx,m0);
                add_transverse_source(fr,tsrc,dtdx,m0);
            } else {
                fr.density=cell.density-slopes_y.density; fr.velocity_x=cell.velocity_x-slopes_y.velocity_x; fr.velocity_y=cell.velocity_y-slopes_y.velocity_y; fr.velocity_z=cell.velocity_z-slopes_y.velocity_z; fr.pressure=cell.pressure-slopes_y.pressure;
                fl.density=cell.density+slopes_y.density; fl.velocity_x=cell.velocity_x+slopes_y.velocity_x; fl.velocity_y=cell.velocity_y+slopes_y.velocity_y; fl.velocity_z=cell.velocity_z+slopes_y.velocity_z; fl.pressure=cell.pressure+slopes_y.pressure;
            }
            if ((i>1 && i<work_size) && j>1 && (k>1 && k<work_size)){
                int q=riy.idx(i-2,j-2,k-2);
                riy.d[q]=__float2half(fr.density); riy.vx[q]=__float2half(fr.velocity_x); riy.vy[q]=__float2half(fr.velocity_y); riy.vz[q]=__float2half(fr.velocity_z); riy.p[q]=__float2half(fr.pressure);
                if (__half2float(riy.d[q])<smallr) riy.d[q]=ls.d[ls.idx(i,j,k)];
                if (__half2float(riy.p[q])<smallp) riy.p[q]=ls.p[ls.idx(i,j,k)];
            }
            if ((i>1 && i<work_size) && j<work_size && (k>1 && k<work_size)){
                int q=liy.idx(i-2,j-1,k-2);
                liy.d[q]=__float2half(fl.density); liy.vx[q]=__float2half(fl.velocity_x); liy.vy[q]=__float2half(fl.velocity_y); liy.vz[q]=__float2half(fl.velocity_z); liy.p[q]=__float2half(fl.pressure);
                if (__half2float(liy.d[q])<smallr) liy.d[q]=ls.d[ls.idx(i,j,k)];
                if (__half2float(liy.p[q])<smallp) liy.p[q]=ls.p[ls.idx(i,j,k)];
            }

            // ----- Z faces (rotate vx<-vz<-vy<-vx) -----
            if (use_ppm){
                na=sg_load(ls,i,j,k-1); nb=sg_load(ls,i,j,k+1);
                rl.density=na.density; rl.velocity_x=na.velocity_z; rl.velocity_y=na.velocity_x; rl.velocity_z=na.velocity_y; rl.pressure=na.pressure;
                rm.density=m0.density; rm.velocity_x=m0.velocity_z; rm.velocity_y=m0.velocity_x; rm.velocity_z=m0.velocity_y; rm.pressure=m0.pressure;
                rr.density=nb.density; rr.velocity_x=nb.velocity_z; rr.velocity_y=nb.velocity_x; rr.velocity_z=nb.velocity_y; rr.pressure=nb.pressure;
                local_ppm_trace(rl,rm,rr,gamma,dtdx,qp,qn);
                fl.density=qp.density; fl.velocity_x=qp.velocity_y; fl.velocity_y=qp.velocity_z; fl.velocity_z=qp.velocity_x; fl.pressure=qp.pressure;
                fr.density=qn.density; fr.velocity_x=qn.velocity_y; fr.velocity_y=qn.velocity_z; fr.velocity_z=qn.velocity_x; fr.pressure=qn.pressure;
                tsrc=prim_add(srcx,srcy);
                add_transverse_source(fl,tsrc,dtdx,m0);
                add_transverse_source(fr,tsrc,dtdx,m0);
            } else {
                fr.density=cell.density-slopes_z.density; fr.velocity_x=cell.velocity_x-slopes_z.velocity_x; fr.velocity_y=cell.velocity_y-slopes_z.velocity_y; fr.velocity_z=cell.velocity_z-slopes_z.velocity_z; fr.pressure=cell.pressure-slopes_z.pressure;
                fl.density=cell.density+slopes_z.density; fl.velocity_x=cell.velocity_x+slopes_z.velocity_x; fl.velocity_y=cell.velocity_y+slopes_z.velocity_y; fl.velocity_z=cell.velocity_z+slopes_z.velocity_z; fl.pressure=cell.pressure+slopes_z.pressure;
            }
            if ((i>1 && i<work_size) && (j>1 && j<work_size) && k>1){
                int q=riz.idx(i-2,j-2,k-2);
                riz.d[q]=__float2half(fr.density); riz.vx[q]=__float2half(fr.velocity_x); riz.vy[q]=__float2half(fr.velocity_y); riz.vz[q]=__float2half(fr.velocity_z); riz.p[q]=__float2half(fr.pressure);
                if (__half2float(riz.d[q])<smallr) riz.d[q]=ls.d[ls.idx(i,j,k)];
                if (__half2float(riz.p[q])<smallp) riz.p[q]=ls.p[ls.idx(i,j,k)];
            }
            if ((i>1 && i<work_size) && (j>1 && j<work_size) && k<work_size){
                int q=liz.idx(i-2,j-2,k-1);
                liz.d[q]=__float2half(fl.density); liz.vx[q]=__float2half(fl.velocity_x); liz.vy[q]=__float2half(fl.velocity_y); liz.vz[q]=__float2half(fl.velocity_z); liz.p[q]=__float2half(fl.pressure);
                if (__half2float(liz.d[q])<smallr) liz.d[q]=ls.d[ls.idx(i,j,k)];
                if (__half2float(liz.p[q])<smallp) liz.p[q]=ls.p[ls.idx(i,j,k)];
            }
        }
        __syncthreads();
    }

    // ========================================================================
    // 3) riemann_driver : per-face HLLC/LLF; fluxes overwrite the left tiles
    // ========================================================================
    {
        const int ias = (2*NS+1)*(2*NS)*(2*NS);
        for (int work_idx=thread_idx; work_idx <= ias*3-1; work_idx += blockDim.x){
            int i,j,k;
            Prim L,R; Cons flux;
            if (work_idx < ias){
                idx1Dto3D(work_idx, 2*NS+1, 2*NS, i, j, k);
                int q=lix.idx(i,j,k);
                L.density=__half2float(lix.d[q]); L.velocity_x=__half2float(lix.vx[q]); L.velocity_y=__half2float(lix.vy[q]); L.velocity_z=__half2float(lix.vz[q]); L.pressure=__half2float(lix.p[q]);
                int qr=rix.idx(i,j,k);
                R.density=__half2float(rix.d[qr]); R.velocity_x=__half2float(rix.vx[qr]); R.velocity_y=__half2float(rix.vy[qr]); R.velocity_z=__half2float(rix.vz[qr]); R.pressure=__half2float(rix.p[qr]);
                flux=riemann_fluxes(L,R,gamma,smallr,smallc2,riemann);
                lix.d[q]=__float2half(flux.density); lix.vx[q]=__float2half(flux.momentum_x); lix.vy[q]=__float2half(flux.momentum_y); lix.vz[q]=__float2half(flux.momentum_z); lix.p[q]=__float2half(flux.energy);
            } else if (work_idx < 2*ias){
                idx1Dto3D(work_idx-ias, 2*NS, 2*NS+1, i, j, k);
                int q=liy.idx(i,j,k);
                L.density=__half2float(liy.d[q]); L.velocity_x=__half2float(liy.vy[q]); L.velocity_y=__half2float(liy.vz[q]); L.velocity_z=__half2float(liy.vx[q]); L.pressure=__half2float(liy.p[q]);
                int qr=riy.idx(i,j,k);
                R.density=__half2float(riy.d[qr]); R.velocity_x=__half2float(riy.vy[qr]); R.velocity_y=__half2float(riy.vz[qr]); R.velocity_z=__half2float(riy.vx[qr]); R.pressure=__half2float(riy.p[qr]);
                flux=riemann_fluxes(L,R,gamma,smallr,smallc2,riemann);
                liy.d[q]=__float2half(flux.density); liy.vx[q]=__float2half(flux.momentum_z); liy.vy[q]=__float2half(flux.momentum_x); liy.vz[q]=__float2half(flux.momentum_y); liy.p[q]=__float2half(flux.energy);
            } else {
                idx1Dto3D(work_idx-2*ias, 2*NS, 2*NS, i, j, k);
                int q=liz.idx(i,j,k);
                L.density=__half2float(liz.d[q]); L.velocity_x=__half2float(liz.vz[q]); L.velocity_y=__half2float(liz.vx[q]); L.velocity_z=__half2float(liz.vy[q]); L.pressure=__half2float(liz.p[q]);
                int qr=riz.idx(i,j,k);
                R.density=__half2float(riz.d[qr]); R.velocity_x=__half2float(riz.vz[qr]); R.velocity_y=__half2float(riz.vx[qr]); R.velocity_z=__half2float(riz.vy[qr]); R.pressure=__half2float(riz.p[qr]);
                flux=riemann_fluxes(L,R,gamma,smallr,smallc2,riemann);
                liz.d[q]=__float2half(flux.density); liz.vx[q]=__float2half(flux.momentum_y); liz.vy[q]=__float2half(flux.momentum_z); liz.vz[q]=__float2half(flux.momentum_x); liz.p[q]=__float2half(flux.energy);
            }
        }
        __syncthreads();
    }
#endif // PERF_MEMFLOOR

    // ========================================================================
    // 4) zero_fine_fluxes (only when finer level exists)
    // ========================================================================
    if (ilevel < levelmax){
        const int ias = (2*NS+1)*(2*NS)*(2*NS);
        for (int work_idx=thread_idx; work_idx <= ias*3-1; work_idx += blockDim.x){
            int i,j,k;
            if (work_idx < ias){
                idx1Dto3D(work_idx, 2*NS+1, 2*NS, i, j, k);
                if (ls.ref[ls.idx(i+1,j+2,k+2)] || ls.ref[ls.idx(i+2,j+2,k+2)]){
                    int q=lix.idx(i,j,k); lix.d[q]=0; lix.vx[q]=0; lix.vy[q]=0; lix.vz[q]=0; lix.p[q]=0;
                }
            } else if (work_idx < 2*ias){
                idx1Dto3D(work_idx-ias, 2*NS, 2*NS+1, i, j, k);
                if (ls.ref[ls.idx(i+2,j+1,k+2)] || ls.ref[ls.idx(i+2,j+2,k+2)]){
                    int q=liy.idx(i,j,k); liy.d[q]=0; liy.vx[q]=0; liy.vy[q]=0; liy.vz[q]=0; liy.p[q]=0;
                }
            } else {
                idx1Dto3D(work_idx-2*ias, 2*NS, 2*NS, i, j, k);
                if (ls.ref[ls.idx(i+2,j+2,k+1)] || ls.ref[ls.idx(i+2,j+2,k+2)]){
                    int q=liz.idx(i,j,k); liz.d[q]=0; liz.vx[q]=0; liz.vy[q]=0; liz.vz[q]=0; liz.p[q]=0;
                }
            }
        }
        __syncthreads();
    }

    // ========================================================================
    // 5) conservative_update : unew = base + dt/dx*divF (+ fused turb driving)
    // ========================================================================
    {
        const int work_size = 2*NS;              // 6
        for (int work_idx=thread_idx; work_idx <= work_size*work_size*work_size-1; work_idx += blockDim.x){
            int isg,jsg,ksg; idx1Dto3D(work_idx/8, work_size/2, work_size/2, isg, jsg, ksg);
            isg+=1; jsg+=1; ksg+=1;
            int ind_nbor = 1 + isg + NSP2*jsg + NSP2SQ*ksg;
            int oct_idx = NBORA(ind_nbor, subgrid_idx);
            int cell_idx = (work_idx & 7) + 1;
            int i,j,k; idx1Dto3D(cell_idx-1, 2, 2, i, j, k);
            i += 2*(isg-1); j += 2*(jsg-1); k += 2*(ksg-1);

            Cons up;
#ifdef PERF_MEMFLOOR
            up.density=0.f; up.momentum_x=0.f; up.momentum_y=0.f; up.momentum_z=0.f; up.energy=0.f;
#else
            up.density   =(__half2float(lix.d[lix.idx(i,j,k)]) -__half2float(lix.d[lix.idx(i+1,j,k)]) )*dtdx;
            up.momentum_x=(__half2float(lix.vx[lix.idx(i,j,k)])-__half2float(lix.vx[lix.idx(i+1,j,k)]))*dtdx;
            up.momentum_y=(__half2float(lix.vy[lix.idx(i,j,k)])-__half2float(lix.vy[lix.idx(i+1,j,k)]))*dtdx;
            up.momentum_z=(__half2float(lix.vz[lix.idx(i,j,k)])-__half2float(lix.vz[lix.idx(i+1,j,k)]))*dtdx;
            up.energy    =(__half2float(lix.p[lix.idx(i,j,k)]) -__half2float(lix.p[lix.idx(i+1,j,k)]) )*dtdx;

            up.density   +=(__half2float(liy.d[liy.idx(i,j,k)]) -__half2float(liy.d[liy.idx(i,j+1,k)]) )*dtdx;
            up.momentum_x+=(__half2float(liy.vx[liy.idx(i,j,k)])-__half2float(liy.vx[liy.idx(i,j+1,k)]))*dtdx;
            up.momentum_y+=(__half2float(liy.vy[liy.idx(i,j,k)])-__half2float(liy.vy[liy.idx(i,j+1,k)]))*dtdx;
            up.momentum_z+=(__half2float(liy.vz[liy.idx(i,j,k)])-__half2float(liy.vz[liy.idx(i,j+1,k)]))*dtdx;
            up.energy    +=(__half2float(liy.p[liy.idx(i,j,k)]) -__half2float(liy.p[liy.idx(i,j+1,k)]) )*dtdx;

            up.density   +=(__half2float(liz.d[liz.idx(i,j,k)]) -__half2float(liz.d[liz.idx(i,j,k+1)]) )*dtdx;
            up.momentum_x+=(__half2float(liz.vx[liz.idx(i,j,k)])-__half2float(liz.vx[liz.idx(i,j,k+1)]))*dtdx;
            up.momentum_y+=(__half2float(liz.vy[liz.idx(i,j,k)])-__half2float(liz.vy[liz.idx(i,j,k+1)]))*dtdx;
            up.momentum_z+=(__half2float(liz.vz[liz.idx(i,j,k)])-__half2float(liz.vz[liz.idx(i,j,k+1)]))*dtdx;
            up.energy    +=(__half2float(liz.p[liz.idx(i,j,k)]) -__half2float(liz.p[liz.idx(i,j,k+1)]) )*dtdx;
#endif

            float b1,b2,b3,b4,b5;
            if (base_write){
                b1=UOLD(cell_idx,1,oct_idx); b2=UOLD(cell_idx,2,oct_idx); b3=UOLD(cell_idx,3,oct_idx);
                b4=UOLD(cell_idx,4,oct_idx); b5=UOLD(cell_idx,5,oct_idx);
            } else {
                b1=UNEW(cell_idx,1,oct_idx); b2=UNEW(cell_idx,2,oct_idx); b3=UNEW(cell_idx,3,oct_idx);
                b4=UNEW(cell_idx,4,oct_idx); b5=UNEW(cell_idx,5,oct_idx);
            }
            b1+=up.density; b2+=up.momentum_x; b3+=up.momentum_y; b4+=up.momentum_z; b5+=up.energy;
#ifdef TURB
            if (do_turb && b1>=turb_min_rho){
                int tbmin[3], tbmax[3];
                float tw1[3], tw2[3];
                for (int td=1; td<=NDIM; td++){
                    float txc=(2*grid[oct_idx-1].ckey[td-1] + ((cell_idx-1)/(1<<(td-1)) & 1) + 0.5f)*dx - d_skip[td-1];
                    float trr=txc/boxlen*(float)TURB_GS;
                    tbmin[td-1]=(int)floorf(trr); tbmax[td-1]=(int)ceilf(trr);
                    if (tbmin[td-1]==TURB_GS) tbmin[td-1]=0;
                    if (tbmax[td-1]==TURB_GS) tbmax[td-1]=0;
                    tw1[td-1]=trr-floorf(trr);
                    tw2[td-1]=1.0f-tw1[td-1];
                }
                #define AF(t,a,b,c) afield_now[((t)-1) + 3*((a) + TURB_GS*((b) + TURB_GS*(c)))]
                float tff[3];
                for (int td=1; td<=3; td++){
                    tff[td-1]=
                        AF(td,tbmin[0],tbmin[1],tbmin[2])*tw2[0]*tw2[1]*tw2[2] +
                        AF(td,tbmax[0],tbmin[1],tbmin[2])*tw1[0]*tw2[1]*tw2[2] +
                        AF(td,tbmin[0],tbmax[1],tbmin[2])*tw2[0]*tw1[1]*tw2[2] +
                        AF(td,tbmax[0],tbmax[1],tbmin[2])*tw1[0]*tw1[1]*tw2[2] +
                        AF(td,tbmin[0],tbmin[1],tbmax[2])*tw2[0]*tw2[1]*tw1[2] +
                        AF(td,tbmax[0],tbmin[1],tbmax[2])*tw1[0]*tw2[1]*tw1[2] +
                        AF(td,tbmin[0],tbmax[1],tbmax[2])*tw2[0]*tw1[1]*tw1[2] +
                        AF(td,tbmax[0],tbmax[1],tbmax[2])*tw1[0]*tw1[1]*tw1[2];
                }
                #undef AF
                float trmax=fmaxf(b1,smallr);
                float tener=b5;
                tener=fmaxf(tener-0.5f*b2*b2/trmax, b1*smallc2);
                tener=fmaxf(tener-0.5f*b3*b3/trmax, b1*smallc2);
                tener=fmaxf(tener-0.5f*b4*b4/trmax, b1*smallc2);
                b2+=trmax*tff[0]*dt;
                b3+=trmax*tff[1]*dt;
                b4+=trmax*tff[2]*dt;
                tener=fmaxf(tener+0.5f*b2*b2/trmax, b1*smallc2);
                tener=fmaxf(tener+0.5f*b3*b3/trmax, b1*smallc2);
                tener=fmaxf(tener+0.5f*b4*b4/trmax, b1*smallc2);
                b5=tener;
            }
#endif
            UNEW(cell_idx,1,oct_idx)=b1; UNEW(cell_idx,2,oct_idx)=b2; UNEW(cell_idx,3,oct_idx)=b3;
            UNEW(cell_idx,4,oct_idx)=b4; UNEW(cell_idx,5,oct_idx)=b5;
        }
    }

    // ========================================================================
    // 6) coarse_cell_update (only when coarser level exists)
    // ========================================================================
    if (ilevel > levelmin){
        const float inv_twotondim = 1.0f/(float)TWOTONDIM;
        const int ias = NS*NS;                   // nsubgrid^(ndim-1)
        const float cfs = dtdx*inv_twotondim;
        int t = thread_idx;
        Cons flux; flux.density=flux.momentum_x=flux.momentum_y=flux.momentum_z=flux.energy=0.0f;
        int isg=0,jsg=0,ksg=0; int sel=-1;

        if (t < ias){                                   // left X
            int wj,wk; idx1Dto2D(t, NS, wj, wk); isg=0; jsg=wj+1; ksg=wk+1; sel=0;
            for (int j=2*jsg-2; j<=2*jsg-1; j++) for (int k=2*ksg-2; k<=2*ksg-1; k++){
                flux.density   -=__half2float(lix.d[lix.idx(0,j,k)])*cfs;
                flux.momentum_x-=__half2float(lix.vx[lix.idx(0,j,k)])*cfs;
                flux.momentum_y-=__half2float(lix.vy[lix.idx(0,j,k)])*cfs;
                flux.momentum_z-=__half2float(lix.vz[lix.idx(0,j,k)])*cfs;
                flux.energy    -=__half2float(lix.p[lix.idx(0,j,k)])*cfs;
            }
        } else if (t < 2*ias){                          // right X
            int wj,wk; idx1Dto2D(t-ias, NS, wj, wk); isg=NS+1; jsg=wj+1; ksg=wk+1; sel=0;
            for (int j=2*jsg-2; j<=2*jsg-1; j++) for (int k=2*ksg-2; k<=2*ksg-1; k++){
                flux.density   +=__half2float(lix.d[lix.idx(2*NS,j,k)])*cfs;
                flux.momentum_x+=__half2float(lix.vx[lix.idx(2*NS,j,k)])*cfs;
                flux.momentum_y+=__half2float(lix.vy[lix.idx(2*NS,j,k)])*cfs;
                flux.momentum_z+=__half2float(lix.vz[lix.idx(2*NS,j,k)])*cfs;
                flux.energy    +=__half2float(lix.p[lix.idx(2*NS,j,k)])*cfs;
            }
        } else if (t < 3*ias){                          // left Y
            int wi,wk; idx1Dto2D(t-2*ias, NS, wi, wk); isg=wi+1; jsg=0; ksg=wk+1; sel=0;
            for (int i=2*isg-2; i<=2*isg-1; i++) for (int k=2*ksg-2; k<=2*ksg-1; k++){
                flux.density   -=__half2float(liy.d[liy.idx(i,0,k)])*cfs;
                flux.momentum_x-=__half2float(liy.vx[liy.idx(i,0,k)])*cfs;
                flux.momentum_y-=__half2float(liy.vy[liy.idx(i,0,k)])*cfs;
                flux.momentum_z-=__half2float(liy.vz[liy.idx(i,0,k)])*cfs;
                flux.energy    -=__half2float(liy.p[liy.idx(i,0,k)])*cfs;
            }
        } else if (t < 4*ias){                          // right Y
            int wi,wk; idx1Dto2D(t-3*ias, NS, wi, wk); isg=wi+1; jsg=NS+1; ksg=wk+1; sel=0;
            for (int i=2*isg-2; i<=2*isg-1; i++) for (int k=2*ksg-2; k<=2*ksg-1; k++){
                flux.density   +=__half2float(liy.d[liy.idx(i,2*NS,k)])*cfs;
                flux.momentum_x+=__half2float(liy.vx[liy.idx(i,2*NS,k)])*cfs;
                flux.momentum_y+=__half2float(liy.vy[liy.idx(i,2*NS,k)])*cfs;
                flux.momentum_z+=__half2float(liy.vz[liy.idx(i,2*NS,k)])*cfs;
                flux.energy    +=__half2float(liy.p[liy.idx(i,2*NS,k)])*cfs;
            }
        } else if (t < 5*ias){                          // left Z
            int wi,wj; idx1Dto2D(t-4*ias, NS, wi, wj); isg=wi+1; jsg=wj+1; ksg=0; sel=0;
            for (int i=2*isg-2; i<=2*isg-1; i++) for (int j=2*jsg-2; j<=2*jsg-1; j++){
                flux.density   -=__half2float(liz.d[liz.idx(i,j,0)])*cfs;
                flux.momentum_x-=__half2float(liz.vx[liz.idx(i,j,0)])*cfs;
                flux.momentum_y-=__half2float(liz.vy[liz.idx(i,j,0)])*cfs;
                flux.momentum_z-=__half2float(liz.vz[liz.idx(i,j,0)])*cfs;
                flux.energy    -=__half2float(liz.p[liz.idx(i,j,0)])*cfs;
            }
        } else if (t < 6*ias){                          // right Z
            int wi,wj; idx1Dto2D(t-5*ias, NS, wi, wj); isg=wi+1; jsg=wj+1; ksg=NS+1; sel=0;
            for (int i=2*isg-2; i<=2*isg-1; i++) for (int j=2*jsg-2; j<=2*jsg-1; j++){
                flux.density   +=__half2float(liz.d[liz.idx(i,j,2*NS)])*cfs;
                flux.momentum_x+=__half2float(liz.vx[liz.idx(i,j,2*NS)])*cfs;
                flux.momentum_y+=__half2float(liz.vy[liz.idx(i,j,2*NS)])*cfs;
                flux.momentum_z+=__half2float(liz.vz[liz.idx(i,j,2*NS)])*cfs;
                flux.energy    +=__half2float(liz.p[liz.idx(i,j,2*NS)])*cfs;
            }
        }
        if (sel==0){
            int ind_nbor = 1 + isg + NSP2*jsg + NSP2SQ*ksg;
            int source_idx = NBORA(ind_nbor, subgrid_idx);
            if (source_idx > ngridmax){
                int father_idx = father[source_idx-1];
                int ci=grid[source_idx-1].ckey[0]-2*grid[father_idx-1].ckey[0];
                int cj=grid[source_idx-1].ckey[1]-2*grid[father_idx-1].ckey[1];
                int ck=grid[source_idx-1].ckey[2]-2*grid[father_idx-1].ckey[2];
                int cell_idx=1+ci+2*cj+4*ck;
                atomicAdd(&UNEW(cell_idx,1,father_idx), flux.density);
                atomicAdd(&UNEW(cell_idx,2,father_idx), flux.momentum_x);
                atomicAdd(&UNEW(cell_idx,3,father_idx), flux.momentum_y);
                atomicAdd(&UNEW(cell_idx,4,father_idx), flux.momentum_z);
                atomicAdd(&UNEW(cell_idx,5,father_idx), flux.energy);
            }
        }
    }
}

// ============================================================================
// extern "C" host launcher (called from gpu_runner.cuf via bind(c))
// ============================================================================
extern "C" void launch_hydro_integrator(
    void* grid, void* uold, void* unew, void* f, void* father, void* nbor,
    int head_idx, int num_subgrids, int ngridmax, int ilevel, int levelmin, int levelmax,
    float gamma, float smallr, float smallc2, float dt, float dx,
    int slope, int riemann, void* constant_gravity, int base_write,
    float courant_factor, void* dt_out, int cfl_sqrt3
#ifdef TURB
    , void* afield_now, void* d_skip, float boxlen, float turb_min_rho, int do_turb
#endif
){
    int threads = 256;
    int blocks = num_subgrids;
    if (blocks <= 0) return;
    // Opt into >48KB dynamic shared (needed at ns4); harmless at ns3. Set once.
    static bool attr_set = false;
    if (!attr_set){
        cudaFuncSetAttribute(hydro_integrator_kernel_cuda, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        attr_set = true;
    }
    hydro_integrator_kernel_cuda<<<blocks, threads, SMEM_BYTES>>>(
        (Oct*)grid, (float*)uold, (float*)unew, (const float*)f, (const int*)father, (const int*)nbor,
        head_idx, num_subgrids, ngridmax, ilevel, levelmin, levelmax,
        gamma, smallr, smallc2, dt, dx,
        slope, riemann, (const double*)constant_gravity, base_write,
        courant_factor, (double*)dt_out, cfl_sqrt3
#ifdef TURB
        , (const float*)afield_now, (const float*)d_skip, boxlen, turb_min_rho, do_turb
#endif
    );
}

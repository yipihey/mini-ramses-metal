// ============================================================================
// cuda_mhd_integrator.cu
//
// Stage-1 faithful CUDA-C translation of mhd_integrator_kernel (the default
// PLM Dedner GLM-MHD integrator, gpu/gpu_hydro.cuf:3215). Sister to the hydro
// port (cuda_hydro_integrator.cu); same CUKERNEL scaffolding (extern"C" +
// bind(c), nvcc -> .o, linked +-lstdc++).
//
// Pipeline (single-level turb, the production hot path): load+c2p+dt-fold+grav
// predictor -> trace_3d_mhd_plm (MUSCL-Hancock) -> riemann_driver_mhd (HLLD/
// HLL/LLF + GLM cleaning at speed ch) -> mhd_conservative_update (+fused turb
// + uniform psi damping glm_fac). No zero_fine/coarse reflux (AMR TODO in the
// Fortran kernel too). 9-var f16 shared tiles.
//
// Build flags mirror the GLM-MHD f16/ns3 build: NPRE=4 (dp=float, fp32 math),
// TPRE=2 (tp=__half tile), NSUB=3, NVAR=9, GLMMHD, optional GRAV/TURB.
// ============================================================================

#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>

#ifndef NSUB
#define NSUB 3
#endif
static constexpr int NS    = NSUB;
static constexpr int NDIM  = 3;
static constexpr int NVAR  = 9;                 // rho, mom_x/y/z, E, Bx, By, Bz, psi
static constexpr int TWOTONDIM = 8;
static constexpr int TD    = 2*NS + 4;          // tile per-dim (0..2*NS+3)
static constexpr int TILE  = TD*TD*TD;
static constexpr int NSP2  = NS + 2;
static constexpr int NSP2SQ = NSP2*NSP2;
static constexpr int SUBGRIDSIZE = NSP2*NSP2*NSP2;

static constexpr int IX_NX = 2*NS+1, IX_NY = 2*NS,   IX_NZ = 2*NS;
static constexpr int IY_NX = 2*NS,   IY_NY = 2*NS+1, IY_NZ = 2*NS;
static constexpr int IZ_NX = 2*NS,   IZ_NY = 2*NS,   IZ_NZ = 2*NS+1;
static constexpr int IX_SZ = IX_NX*IX_NY*IX_NZ;
static constexpr int IY_SZ = IY_NX*IY_NY*IY_NZ;
static constexpr int IZ_SZ = IZ_NX*IZ_NY*IZ_NZ;

static constexpr int SOLVER_LLF = 1, SOLVER_HLL = 2, SOLVER_HLLD = 4;

#ifdef TURB
static constexpr int TURB_GS = 64;
#endif

typedef __half thalf;

#define UOLD(c,v,o)  uold[((size_t)((o)-1)*NVAR + ((v)-1))*8 + ((c)-1)]
#define UNEW(c,v,o)  unew[((size_t)((o)-1)*NVAR + ((v)-1))*8 + ((c)-1)]
#define FARR(c,d,o)  f[((size_t)((o)-1)*NDIM + ((d)-1))*8 + ((c)-1)]
#define NBORA(i,sg)  nbor[((size_t)((sg)-1))*SUBGRIDSIZE + ((i)-1)]

struct Oct { long long hkey; int ckey[3]; int refined[8]; int lev; int superoct; };

// 9-var primitive / conserved (fp32 math)
struct Prim { float density, velocity_x, velocity_y, velocity_z, pressure, bx, by, bz, psi; };
struct Cons { float density, momentum_x, momentum_y, momentum_z, energy, bx, by, bz, psi; };

struct Tile {                       // subgrid_6x6x6cell_mhd
    thalf *d,*vx,*vy,*vz,*p,*bx,*by,*bz,*psi; unsigned char *ref;
    __device__ __forceinline__ int idx(int i,int j,int k) const { return i + TD*(j + TD*k); }
};
struct Face {
    thalf *d,*vx,*vy,*vz,*p,*bx,*by,*bz,*psi; int nx,ny;
    __device__ __forceinline__ int idx(int i,int j,int k) const { return i + nx*(j + ny*k); }
};

__device__ __forceinline__ float magsq(float x,float y,float z){ return x*x + (y*y + z*z); }

// =================== MHD per-cell physics (faithful to .cuf) =================
__device__ __forceinline__ float mhd_pressure(const Cons& c,float gamma,float smallr,float smallc2){
    float smallp=smallc2/gamma;
    float rho=fmaxf(c.density,smallr);
    float ekin=0.5f*magsq(c.momentum_x,c.momentum_y,c.momentum_z)/rho;
    float emag=0.5f*magsq(c.bx,c.by,c.bz);
    return fmaxf((gamma-1.0f)*(c.energy-ekin-emag), rho*smallp);
}
__device__ __forceinline__ Prim mhd_c2p(const Cons& c,float gamma,float smallr,float smallc2){
    float rho=fmaxf(c.density,smallr); Prim p;
    p.density=rho; p.velocity_x=c.momentum_x/rho; p.velocity_y=c.momentum_y/rho; p.velocity_z=c.momentum_z/rho;
    p.pressure=mhd_pressure(c,gamma,smallr,smallc2);
    p.bx=c.bx; p.by=c.by; p.bz=c.bz; p.psi=c.psi; return p;
}
__device__ __forceinline__ Cons mhd_p2c(const Prim& p,float gamma){
    float ekin=0.5f*p.density*magsq(p.velocity_x,p.velocity_y,p.velocity_z);
    float emag=0.5f*magsq(p.bx,p.by,p.bz); Cons c;
    c.density=p.density; c.momentum_x=p.density*p.velocity_x; c.momentum_y=p.density*p.velocity_y; c.momentum_z=p.density*p.velocity_z;
    c.energy=p.pressure/(gamma-1.0f)+ekin+emag; c.bx=p.bx; c.by=p.by; c.bz=p.bz; c.psi=p.psi; return c;
}
__device__ __forceinline__ float mhd_fast_n(const Prim& p,float gamma,float bn){
    float rho=p.density, c2=gamma*p.pressure/rho, b2=magsq(p.bx,p.by,p.bz)/rho, d2=0.5f*(b2+c2);
    return sqrtf(d2 + sqrtf(fmaxf(d2*d2 - c2*bn*bn/rho, 0.0f)));
}
__device__ __forceinline__ float mhd_fast_x(const Prim& p,float gamma){ return mhd_fast_n(p,gamma,p.bx); }
__device__ __forceinline__ Cons mhd_phys_flux(const Prim& p,float gamma){
    float vx=p.velocity_x, b2=magsq(p.bx,p.by,p.bz), ptot=p.pressure+0.5f*b2;
    float etot=p.pressure/(gamma-1.0f)+0.5f*p.density*magsq(p.velocity_x,p.velocity_y,p.velocity_z)+0.5f*b2;
    float vdotb=p.velocity_x*p.bx+p.velocity_y*p.by+p.velocity_z*p.bz; Cons f;
    f.density=p.density*vx;
    f.momentum_x=p.density*vx*vx+ptot-p.bx*p.bx;
    f.momentum_y=p.density*vx*p.velocity_y-p.bx*p.by;
    f.momentum_z=p.density*vx*p.velocity_z-p.bx*p.bz;
    f.energy=(etot+ptot)*vx-p.bx*vdotb;
    f.bx=0.0f; f.by=vx*p.by-p.velocity_y*p.bx; f.bz=vx*p.bz-p.velocity_z*p.bx; f.psi=0.0f;
    return f;
}
__device__ __forceinline__ Cons mhd_hll_combine(float sl,float sr,const Cons& fL,const Cons& fR,const Cons& uL,const Cons& uR){
    if (sl>=0.0f) return fL;
    if (sr<=0.0f) return fR;
    float inv=1.0f/(sr-sl); Cons f;
    f.density   =(sr*fL.density   -sl*fR.density   +sr*sl*(uR.density   -uL.density   ))*inv;
    f.momentum_x=(sr*fL.momentum_x-sl*fR.momentum_x+sr*sl*(uR.momentum_x-uL.momentum_x))*inv;
    f.momentum_y=(sr*fL.momentum_y-sl*fR.momentum_y+sr*sl*(uR.momentum_y-uL.momentum_y))*inv;
    f.momentum_z=(sr*fL.momentum_z-sl*fR.momentum_z+sr*sl*(uR.momentum_z-uL.momentum_z))*inv;
    f.energy    =(sr*fL.energy    -sl*fR.energy    +sr*sl*(uR.energy    -uL.energy    ))*inv;
    f.bx=(sr*fL.bx-sl*fR.bx+sr*sl*(uR.bx-uL.bx))*inv;
    f.by=(sr*fL.by-sl*fR.by+sr*sl*(uR.by-uL.by))*inv;
    f.bz=(sr*fL.bz-sl*fR.bz+sr*sl*(uR.bz-uL.bz))*inv;
    f.psi=(sr*fL.psi-sl*fR.psi+sr*sl*(uR.psi-uL.psi))*inv;
    return f;
}
__device__ __forceinline__ void glm_pair(float bnL,float bnR,float psiL,float psiR,float ch,
                                          float& bn_star,float& fbn,float& fpsi){
    bn_star=0.5f*(bnL+bnR)-0.5f*(psiR-psiL)/ch;
    float psi_star=0.5f*(psiL+psiR)-0.5f*ch*(bnR-bnL);
    fbn=psi_star; fpsi=ch*ch*bn_star;
}
__device__ Cons llf_mhd_fluxes(Prim L,Prim R,float gamma,float smallr,float smallc2,float ch){
    L.density=fmaxf(L.density,smallr); R.density=fmaxf(R.density,smallr);
    float bn_star,fbn,fpsi; glm_pair(L.bx,R.bx,L.psi,R.psi,ch,bn_star,fbn,fpsi);
    L.bx=bn_star; R.bx=bn_star;
    Cons fL=mhd_phys_flux(L,gamma), fR=mhd_phys_flux(R,gamma);
    Cons uL=mhd_p2c(L,gamma), uR=mhd_p2c(R,gamma);
    float smax=fmaxf(fabsf(L.velocity_x)+mhd_fast_x(L,gamma), fabsf(R.velocity_x)+mhd_fast_x(R,gamma));
    smax=fmaxf(smax,ch);
    Cons flux=mhd_hll_combine(-smax,smax,fL,fR,uL,uR);
    flux.bx=fbn; flux.psi=fpsi; return flux;
}
__device__ Cons hll_mhd_fluxes(Prim L,Prim R,float gamma,float smallr,float smallc2,float ch){
    L.density=fmaxf(L.density,smallr); R.density=fmaxf(R.density,smallr);
    float bn_star,fbn,fpsi; glm_pair(L.bx,R.bx,L.psi,R.psi,ch,bn_star,fbn,fpsi);
    L.bx=bn_star; R.bx=bn_star;
    Cons fL=mhd_phys_flux(L,gamma), fR=mhd_phys_flux(R,gamma);
    Cons uL=mhd_p2c(L,gamma), uR=mhd_p2c(R,gamma);
    float cfL=mhd_fast_x(L,gamma), cfR=mhd_fast_x(R,gamma);
    float sl=fminf(fminf(L.velocity_x,R.velocity_x)-fmaxf(cfL,cfR),0.0f);
    float sr=fmaxf(fmaxf(L.velocity_x,R.velocity_x)+fmaxf(cfL,cfR),0.0f);
    Cons flux=mhd_hll_combine(sl,sr,fL,fR,uL,uR);
    flux.bx=fbn; flux.psi=fpsi; return flux;
}
__device__ Cons hlld_mhd_fluxes(Prim L,Prim R,float gamma,float smallr,float smallc2,float ch){
    L.density=fmaxf(L.density,smallr); R.density=fmaxf(R.density,smallr);
    float entho=1.0f/(gamma-1.0f);
    float bn_star,fbn,fpsi; glm_pair(L.bx,R.bx,L.psi,R.psi,ch,bn_star,fbn,fpsi);
    float A=bn_star, sgnm=copysignf(1.0f,A); L.bx=A; R.bx=A;
    float rl=L.density,ul=L.velocity_x,vl=L.velocity_y,wl=L.velocity_z,pl=L.pressure,bl=L.by,cl=L.bz;
    float ekinl=0.5f*(ul*ul+vl*vl+wl*wl)*rl, emagl=0.5f*(A*A+bl*bl+cl*cl);
    float etotl=pl*entho+ekinl+emagl, ptotl=pl+emagl, vdotbl=ul*A+vl*bl+wl*cl;
    float rr=R.density,ur=R.velocity_x,vr=R.velocity_y,wr=R.velocity_z,pr=R.pressure,br=R.by,cr=R.bz;
    float ekinr=0.5f*(ur*ur+vr*vr+wr*wr)*rr, emagr=0.5f*(A*A+br*br+cr*cr);
    float etotr=pr*entho+ekinr+emagr, ptotr=pr+emagr, vdotbr=ur*A+vr*br+wr*cr;
    float cfastl=mhd_fast_x(L,gamma), cfastr=mhd_fast_x(R,gamma);
    float SL=fminf(ul,ur)-fmaxf(cfastl,cfastr), SR=fmaxf(ul,ur)+fmaxf(cfastl,cfastr);
    float rcl=rl*(ul-SL), rcr=rr*(SR-ur);
    float ustar=(rcr*ur+rcl*ul+(ptotl-ptotr))/(rcr+rcl);
    float ptotstar=(rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);
    float rstarl=rl*(SL-ul)/(SL-ustar), estar=rl*(SL-ul)*(SL-ustar)-A*A, el=rl*(SL-ul)*(SL-ul)-A*A;
    float vstarl,bstarl,wstarl,cstarl;
    if (fabsf(estar)<1e-4f*A*A){ vstarl=vl; bstarl=bl; wstarl=wl; cstarl=cl; }
    else { vstarl=vl-A*bl*(ustar-ul)/estar; bstarl=bl*el/estar; wstarl=wl-A*cl*(ustar-ul)/estar; cstarl=cl*el/estar; }
    float vdotbstarl=ustar*A+vstarl*bstarl+wstarl*cstarl;
    float etotstarl=((SL-ul)*etotl-ptotl*ul+ptotstar*ustar+A*(vdotbl-vdotbstarl))/(SL-ustar);
    float sqrrstarl=sqrtf(rstarl), calfvenl=fabsf(A)/sqrrstarl, SAL=ustar-calfvenl;
    float rstarr=rr*(SR-ur)/(SR-ustar); estar=rr*(SR-ur)*(SR-ustar)-A*A; float er=rr*(SR-ur)*(SR-ur)-A*A;
    float vstarr,bstarr,wstarr,cstarr;
    if (fabsf(estar)<1e-4f*A*A){ vstarr=vr; bstarr=br; wstarr=wr; cstarr=cr; }
    else { vstarr=vr-A*br*(ustar-ur)/estar; bstarr=br*er/estar; wstarr=wr-A*cr*(ustar-ur)/estar; cstarr=cr*er/estar; }
    float vdotbstarr=ustar*A+vstarr*bstarr+wstarr*cstarr;
    float etotstarr=((SR-ur)*etotr-ptotr*ur+ptotstar*ustar+A*(vdotbr-vdotbstarr))/(SR-ustar);
    float sqrrstarr=sqrtf(rstarr), calfvenr=fabsf(A)/sqrrstarr, SAR=ustar+calfvenr;
    float vstarstar=(sqrrstarl*vstarl+sqrrstarr*vstarr+sgnm*(bstarr-bstarl))/(sqrrstarl+sqrrstarr);
    float wstarstar=(sqrrstarl*wstarl+sqrrstarr*wstarr+sgnm*(cstarr-cstarl))/(sqrrstarl+sqrrstarr);
    float bstarstar=(sqrrstarl*bstarr+sqrrstarr*bstarl+sgnm*sqrrstarl*sqrrstarr*(vstarr-vstarl))/(sqrrstarl+sqrrstarr);
    float cstarstar=(sqrrstarl*cstarr+sqrrstarr*cstarl+sgnm*sqrrstarl*sqrrstarr*(wstarr-wstarl))/(sqrrstarl+sqrrstarr);
    float vdotbstarstar=ustar*A+vstarstar*bstarstar+wstarstar*cstarstar;
    float etotstarstarl=etotstarl-sgnm*sqrrstarl*(vdotbstarl-vdotbstarstar);
    float etotstarstarr=etotstarr+sgnm*sqrrstarr*(vdotbstarr-vdotbstarstar);
    float ro,uo,vo,wo,bo,co,ptoto,etoto,vdotbo;
    if (SL>0.0f)        { ro=rl;     uo=ul;    vo=vl;        wo=wl;        bo=bl;        co=cl;        ptoto=ptotl;    etoto=etotl;        vdotbo=vdotbl; }
    else if (SAL>0.0f)  { ro=rstarl; uo=ustar; vo=vstarl;    wo=wstarl;    bo=bstarl;    co=cstarl;    ptoto=ptotstar; etoto=etotstarl;     vdotbo=vdotbstarl; }
    else if (ustar>0.0f){ ro=rstarl; uo=ustar; vo=vstarstar; wo=wstarstar; bo=bstarstar; co=cstarstar; ptoto=ptotstar; etoto=etotstarstarl; vdotbo=vdotbstarstar; }
    else if (SAR>0.0f)  { ro=rstarr; uo=ustar; vo=vstarstar; wo=wstarstar; bo=bstarstar; co=cstarstar; ptoto=ptotstar; etoto=etotstarstarr; vdotbo=vdotbstarstar; }
    else if (SR>0.0f)   { ro=rstarr; uo=ustar; vo=vstarr;    wo=wstarr;    bo=bstarr;    co=cstarr;    ptoto=ptotstar; etoto=etotstarr;     vdotbo=vdotbstarr; }
    else                { ro=rr;     uo=ur;    vo=vr;        wo=wr;        bo=br;        co=cr;        ptoto=ptotr;    etoto=etotr;        vdotbo=vdotbr; }
    Cons flux;
    flux.density=ro*uo;
    flux.momentum_x=ro*uo*uo+ptoto-A*A;
    flux.momentum_y=ro*uo*vo-A*bo;
    flux.momentum_z=ro*uo*wo-A*co;
    flux.energy=(etoto+ptoto)*uo-A*vdotbo;
    flux.bx=fbn; flux.by=bo*uo-A*vo; flux.bz=co*uo-A*wo; flux.psi=fpsi;
    return flux;
}
__device__ __forceinline__ Cons mhd_riemann(Prim L,Prim R,float gamma,float smallr,float smallc2,float ch,
                                            int riemann,float sw_dmin,float sw_pmin){
    bool use_llf=false;
    float rmin=fminf(L.density,R.density), pmin=fminf(L.pressure,R.pressure);
    if (sw_dmin>0.0f && rmin<sw_dmin) use_llf=true;
    if (sw_pmin>0.0f && pmin<sw_pmin) use_llf=true;
    if (use_llf)                    return llf_mhd_fluxes(L,R,gamma,smallr,smallc2,ch);
    else if (riemann==SOLVER_HLL)   return hll_mhd_fluxes(L,R,gamma,smallr,smallc2,ch);
    else if (riemann==SOLVER_HLLD)  return hlld_mhd_fluxes(L,R,gamma,smallr,smallc2,ch);
    else                            return llf_mhd_fluxes(L,R,gamma,smallr,smallc2,ch);
}
// rotate primitive so axis dir(1/2/3) is normal
__device__ __forceinline__ Prim mhd_rot_to(Prim p,int dir){
    Prim r=p;
    if (dir==2){ r.velocity_x=p.velocity_y; r.velocity_y=p.velocity_z; r.velocity_z=p.velocity_x; r.bx=p.by; r.by=p.bz; r.bz=p.bx; }
    else if (dir==3){ r.velocity_x=p.velocity_z; r.velocity_y=p.velocity_x; r.velocity_z=p.velocity_y; r.bx=p.bz; r.by=p.bx; r.bz=p.by; }
    return r;
}
__device__ __forceinline__ Cons mhd_rot_flux_from(Cons f,int dir){
    Cons g=f;
    if (dir==2){ g.momentum_x=f.momentum_z; g.momentum_y=f.momentum_x; g.momentum_z=f.momentum_y; g.bx=f.bz; g.by=f.bx; g.bz=f.by; }
    else if (dir==3){ g.momentum_x=f.momentum_y; g.momentum_y=f.momentum_z; g.momentum_z=f.momentum_x; g.bx=f.by; g.by=f.bz; g.bz=f.bx; }
    return g;
}
// slope_moncen (shared with hydro)
__device__ __forceinline__ float slope_moncen(float left,float middle,float right,int slope){
    float sl=middle-left, sr=right-middle, sc=0.5f*(sl+sr), factor=(float)slope;
    if (sl*sr<=0.0f) return 0.0f;
    else if (sl>0.0f){ float mm=fminf(sl,sr); return fminf(factor*mm,sc); }
    else            { float mm=fmaxf(sl,sr); return fmaxf(factor*mm,sc); }
}
__device__ __forceinline__ Prim mhd_slope(const Prim& l,const Prim& m,const Prim& r,int slope){
    Prim s;
    s.density   =0.5f*slope_moncen(l.density,   m.density,   r.density,   slope);
    s.velocity_x=0.5f*slope_moncen(l.velocity_x,m.velocity_x,r.velocity_x,slope);
    s.velocity_y=0.5f*slope_moncen(l.velocity_y,m.velocity_y,r.velocity_y,slope);
    s.velocity_z=0.5f*slope_moncen(l.velocity_z,m.velocity_z,r.velocity_z,slope);
    s.pressure  =0.5f*slope_moncen(l.pressure,  m.pressure,  r.pressure,  slope);
    s.bx=0.5f*slope_moncen(l.bx,m.bx,r.bx,slope);
    s.by=0.5f*slope_moncen(l.by,m.by,r.by,slope);
    s.bz=0.5f*slope_moncen(l.bz,m.bz,r.bz,slope);
    s.psi=0.5f*slope_moncen(l.psi,m.psi,r.psi,slope);
    return s;
}
__device__ __forceinline__ Prim mhd_padd(const Prim& a,const Prim& b,float w){
    Prim c;
    c.density=a.density+w*b.density; c.velocity_x=a.velocity_x+w*b.velocity_x;
    c.velocity_y=a.velocity_y+w*b.velocity_y; c.velocity_z=a.velocity_z+w*b.velocity_z;
    c.pressure=a.pressure+w*b.pressure; c.bx=a.bx+w*b.bx; c.by=a.by+w*b.by; c.bz=a.bz+w*b.bz; c.psi=a.psi+w*b.psi;
    return c;
}
__device__ __forceinline__ Cons mhd_cadd(const Cons& a,const Cons& b){
    Cons c;
    c.density=a.density+b.density; c.momentum_x=a.momentum_x+b.momentum_x; c.momentum_y=a.momentum_y+b.momentum_y;
    c.momentum_z=a.momentum_z+b.momentum_z; c.energy=a.energy+b.energy; c.bx=a.bx+b.bx; c.by=a.by+b.by; c.bz=a.bz+b.bz; c.psi=a.psi+b.psi;
    return c;
}
// MUSCL-Hancock conserved increment from one direction (ideal MHD flux, no GLM)
__device__ __forceinline__ Cons mhd_hancock_dir(const Prim& m0,const Prim& s,int dir,float gamma,float hdtdx){
    Cons fp=mhd_rot_flux_from(mhd_phys_flux(mhd_rot_to(mhd_padd(m0,s, 1.0f),dir),gamma),dir);
    Cons fm=mhd_rot_flux_from(mhd_phys_flux(mhd_rot_to(mhd_padd(m0,s,-1.0f),dir),gamma),dir);
    Cons du;
    du.density=-hdtdx*(fp.density-fm.density);
    du.momentum_x=-hdtdx*(fp.momentum_x-fm.momentum_x);
    du.momentum_y=-hdtdx*(fp.momentum_y-fm.momentum_y);
    du.momentum_z=-hdtdx*(fp.momentum_z-fm.momentum_z);
    du.energy=-hdtdx*(fp.energy-fm.energy);
    du.bx=-hdtdx*(fp.bx-fm.bx); du.by=-hdtdx*(fp.by-fm.by); du.bz=-hdtdx*(fp.bz-fm.bz); du.psi=-hdtdx*(fp.psi-fm.psi);
    return du;
}
__device__ __forceinline__ Prim sg_load(const Tile& sg,int i,int j,int k){
    int q=sg.idx(i,j,k); Prim p;
    p.density=__half2float(sg.d[q]); p.velocity_x=__half2float(sg.vx[q]); p.velocity_y=__half2float(sg.vy[q]);
    p.velocity_z=__half2float(sg.vz[q]); p.pressure=__half2float(sg.p[q]);
    p.bx=__half2float(sg.bx[q]); p.by=__half2float(sg.by[q]); p.bz=__half2float(sg.bz[q]); p.psi=__half2float(sg.psi[q]);
    return p;
}
__device__ __forceinline__ void store_face(const Face& a,int i,int j,int k,const Prim& p,float smallr,float smallp,float lrho,float lp){
    int q=a.idx(i,j,k);
    a.d[q]=__float2half(p.density); a.vx[q]=__float2half(p.velocity_x); a.vy[q]=__float2half(p.velocity_y);
    a.vz[q]=__float2half(p.velocity_z); a.p[q]=__float2half(p.pressure);
    a.bx[q]=__float2half(p.bx); a.by[q]=__float2half(p.by); a.bz[q]=__float2half(p.bz); a.psi[q]=__float2half(p.psi);
    if (__half2float(a.d[q])<smallr) a.d[q]=__float2half(lrho);
    if (__half2float(a.p[q])<smallp) a.p[q]=__float2half(lp);
}
__device__ __forceinline__ Prim load_face(const Face& a,int i,int j,int k){
    int q=a.idx(i,j,k); Prim p;
    p.density=__half2float(a.d[q]); p.velocity_x=__half2float(a.vx[q]); p.velocity_y=__half2float(a.vy[q]);
    p.velocity_z=__half2float(a.vz[q]); p.pressure=__half2float(a.p[q]);
    p.bx=__half2float(a.bx[q]); p.by=__half2float(a.by[q]); p.bz=__half2float(a.bz[q]); p.psi=__half2float(a.psi[q]);
    return p;
}
__device__ __forceinline__ void store_flux(const Face& a,int i,int j,int k,const Cons& f){
    int q=a.idx(i,j,k);
    a.d[q]=__float2half(f.density); a.vx[q]=__float2half(f.momentum_x); a.vy[q]=__float2half(f.momentum_y);
    a.vz[q]=__float2half(f.momentum_z); a.p[q]=__float2half(f.energy);
    a.bx[q]=__float2half(f.bx); a.by[q]=__float2half(f.by); a.bz[q]=__float2half(f.bz); a.psi[q]=__float2half(f.psi);
}

__device__ __forceinline__ double atomicMinDouble(double* addr,double val){
    unsigned long long* a=(unsigned long long*)addr; unsigned long long old=*a,assumed;
    do { assumed=old; double cur=__longlong_as_double(assumed); if (cur<=val) break;
         old=atomicCAS(a,assumed,__double_as_longlong(val)); } while (assumed!=old);
    return __longlong_as_double(old);
}
__device__ double blockReduceMin(double v){
    for (int off=warpSize/2; off>0; off>>=1) v=fmin(v,__shfl_down_sync(0xffffffffu,v,off));
    __shared__ double s[32];
    int lane=threadIdx.x&(warpSize-1), wid=threadIdx.x/warpSize;
    if (lane==0) s[wid]=v; __syncthreads();
    int nwarps=blockDim.x/warpSize;
    v=(threadIdx.x<nwarps)?s[lane]:HUGE_VAL;
    if (wid==0) for (int off=warpSize/2; off>0; off>>=1) v=fmin(v,__shfl_down_sync(0xffffffffu,v,off));
    return v;
}
__device__ __forceinline__ void idx1Dto3D(int index,int nx,int ny,int& xid,int& yid,int& zid){
    zid=index/(nx*ny); yid=(index-zid*nx*ny)/nx; xid=index-zid*nx*ny-yid*nx;
}
__device__ __forceinline__ bool strong_pjump(float a,float b,float c){
    const float tiny=1e-20f; float hi=fmaxf(a,fmaxf(b,c)), lo=fmaxf(fminf(a,fminf(b,c)),tiny); return hi/lo>2.0f;
}

// ============================================================================
#ifndef CUMINBLK
#define CUMINBLK 2      // GLM-MHD PLM is shared-limited at ~2 blocks/SM
#endif
__global__ void __launch_bounds__(256, CUMINBLK) mhd_integrator_kernel_cuda(
    Oct* __restrict__ grid, float* __restrict__ uold, float* __restrict__ unew,
    const float* __restrict__ f, const int* __restrict__ father, const int* __restrict__ nbor,
    int head_idx, int num_subgrids, int ngridmax, int ilevel, int levelmin, int levelmax,
    float gamma, float smallr, float smallc2, float ch, float dt, float dx,
    int slope, int riemann, float sw_dmin, float sw_pmin,
    const double* __restrict__ constant_gravity, float glm_fac, int base_write,
    float courant_factor, double* __restrict__ dt_out, int cfl_sqrt3
#ifdef TURB
    , const float* __restrict__ afield_now, const float* __restrict__ d_skip,
    float boxlen, float turb_min_rho, int do_turb
#endif
){
    __shared__ thalf ls_d[TILE],ls_vx[TILE],ls_vy[TILE],ls_vz[TILE],ls_p[TILE],ls_bx[TILE],ls_by[TILE],ls_bz[TILE],ls_psi[TILE];
    __shared__ unsigned char ls_ref[TILE];
    __shared__ thalf lx_d[IX_SZ],lx_vx[IX_SZ],lx_vy[IX_SZ],lx_vz[IX_SZ],lx_p[IX_SZ],lx_bx[IX_SZ],lx_by[IX_SZ],lx_bz[IX_SZ],lx_psi[IX_SZ];
    __shared__ thalf rx_d[IX_SZ],rx_vx[IX_SZ],rx_vy[IX_SZ],rx_vz[IX_SZ],rx_p[IX_SZ],rx_bx[IX_SZ],rx_by[IX_SZ],rx_bz[IX_SZ],rx_psi[IX_SZ];
    __shared__ thalf ly_d[IY_SZ],ly_vx[IY_SZ],ly_vy[IY_SZ],ly_vz[IY_SZ],ly_p[IY_SZ],ly_bx[IY_SZ],ly_by[IY_SZ],ly_bz[IY_SZ],ly_psi[IY_SZ];
    __shared__ thalf ry_d[IY_SZ],ry_vx[IY_SZ],ry_vy[IY_SZ],ry_vz[IY_SZ],ry_p[IY_SZ],ry_bx[IY_SZ],ry_by[IY_SZ],ry_bz[IY_SZ],ry_psi[IY_SZ];
    __shared__ thalf lz_d[IZ_SZ],lz_vx[IZ_SZ],lz_vy[IZ_SZ],lz_vz[IZ_SZ],lz_p[IZ_SZ],lz_bx[IZ_SZ],lz_by[IZ_SZ],lz_bz[IZ_SZ],lz_psi[IZ_SZ];
    __shared__ thalf rz_d[IZ_SZ],rz_vx[IZ_SZ],rz_vy[IZ_SZ],rz_vz[IZ_SZ],rz_p[IZ_SZ],rz_bx[IZ_SZ],rz_by[IZ_SZ],rz_bz[IZ_SZ],rz_psi[IZ_SZ];

    Tile ls{ls_d,ls_vx,ls_vy,ls_vz,ls_p,ls_bx,ls_by,ls_bz,ls_psi,ls_ref};
    Face lx{lx_d,lx_vx,lx_vy,lx_vz,lx_p,lx_bx,lx_by,lx_bz,lx_psi,IX_NX,IX_NY};
    Face rx{rx_d,rx_vx,rx_vy,rx_vz,rx_p,rx_bx,rx_by,rx_bz,rx_psi,IX_NX,IX_NY};
    Face ly{ly_d,ly_vx,ly_vy,ly_vz,ly_p,ly_bx,ly_by,ly_bz,ly_psi,IY_NX,IY_NY};
    Face ry{ry_d,ry_vx,ry_vy,ry_vz,ry_p,ry_bx,ry_by,ry_bz,ry_psi,IY_NX,IY_NY};
    Face lz{lz_d,lz_vx,lz_vy,lz_vz,lz_p,lz_bx,lz_by,lz_bz,lz_psi,IZ_NX,IZ_NY};
    Face rz{rz_d,rz_vx,rz_vy,rz_vz,rz_p,rz_bx,rz_by,rz_bz,rz_psi,IZ_NX,IZ_NY};

    float dtdx=dt/dx;
    int block_idx=blockIdx.x, thread_idx=threadIdx.x;
    if (block_idx>=num_subgrids) return;
    int subgrid_idx=head_idx+block_idx;

    // 1) load + c2p + dt-fold + gravity predictor -----------------------------
    {
        const int work_size=2*NS+4;
        const int owned_lo=2, owned_hi=2*NS+1;
        double dt_min=HUGE_VAL;
        for (int work_idx=thread_idx; work_idx<=work_size*work_size*work_size-1; work_idx+=blockDim.x){
            int isg,jsg,ksg; idx1Dto3D(work_idx/8, work_size/2, work_size/2, isg, jsg, ksg);
            int src=NBORA(work_idx/8+1, subgrid_idx);
            int cell=(work_idx&7)+1;
            int ci,cj,ck; idx1Dto3D(cell-1,2,2,ci,cj,ck);
            int i=ci+2*isg, j=cj+2*jsg, k=ck+2*ksg;
            Cons c;
            c.density=UOLD(cell,1,src); c.momentum_x=UOLD(cell,2,src); c.momentum_y=UOLD(cell,3,src);
            c.momentum_z=UOLD(cell,4,src); c.energy=UOLD(cell,5,src);
            c.bx=UOLD(cell,6,src); c.by=UOLD(cell,7,src); c.bz=UOLD(cell,8,src); c.psi=UOLD(cell,9,src);
            Prim p=mhd_c2p(c,gamma,smallr,smallc2);
            if (base_write){
                if ( i>=owned_lo && i<=owned_hi && j>=owned_lo && j<=owned_hi && k>=owned_lo && k<=owned_hi
                     && grid[src-1].refined[cell-1]==0 ){
                    float ctot;
                    if (cfl_sqrt3){
                        ctot=sqrtf(3.0f)*fmaxf(fabsf(p.velocity_x)+mhd_fast_n(p,gamma,p.bx),
                                   fmaxf(fabsf(p.velocity_y)+mhd_fast_n(p,gamma,p.by), fabsf(p.velocity_z)+mhd_fast_n(p,gamma,p.bz)));
                    } else {
                        ctot=(fabsf(p.velocity_x)+mhd_fast_n(p,gamma,p.bx))+(fabsf(p.velocity_y)+mhd_fast_n(p,gamma,p.by))+(fabsf(p.velocity_z)+mhd_fast_n(p,gamma,p.bz));
                    }
                    float grav;
#ifdef GRAV
                    grav=fabsf(FARR(cell,1,src))+fabsf(FARR(cell,2,src))+fabsf(FARR(cell,3,src));
#else
                    grav=fabsf((float)constant_gravity[0])+fabsf((float)constant_gravity[1])+fabsf((float)constant_gravity[2]);
#endif
                    grav=fmaxf(grav*dx/(ctot*ctot), 0.0001f);
                    float dtl=dx/ctot*(sqrtf(1.0f+2.0f*courant_factor*grav)-1.0f)/grav;
                    dt_min=fmin(dt_min,(double)dtl);
                }
            }
#ifdef GRAV
            p.velocity_x += FARR(cell,1,src)*0.5f*dt; p.velocity_y += FARR(cell,2,src)*0.5f*dt; p.velocity_z += FARR(cell,3,src)*0.5f*dt;
#else
            p.velocity_x += (float)constant_gravity[0]*0.5f*dt; p.velocity_y += (float)constant_gravity[1]*0.5f*dt; p.velocity_z += (float)constant_gravity[2]*0.5f*dt;
#endif
            int q=ls.idx(i,j,k);
            ls.d[q]=__float2half(p.density); ls.vx[q]=__float2half(p.velocity_x); ls.vy[q]=__float2half(p.velocity_y);
            ls.vz[q]=__float2half(p.velocity_z); ls.p[q]=__float2half(p.pressure);
            ls.bx[q]=__float2half(p.bx); ls.by[q]=__float2half(p.by); ls.bz[q]=__float2half(p.bz); ls.psi[q]=__float2half(p.psi);
            ls.ref[q]=(grid[src-1].refined[cell-1]!=0)?1:0;
        }
        if (base_write){ dt_min=blockReduceMin(dt_min); if (thread_idx==0) atomicMinDouble(&dt_out[0],dt_min); }
        __syncthreads();
    }

    // 2) trace_3d_mhd_plm : MUSCL-Hancock reconstruction ----------------------
    {
        const int work_size=2*NS+2;
        float smallp=smallr*smallc2;
        for (int work_idx=thread_idx; work_idx<=work_size*work_size*work_size-1; work_idx+=blockDim.x){
            int i,j,k; idx1Dto3D(work_idx,work_size,work_size,i,j,k); i+=1; j+=1; k+=1;
            Prim m0=sg_load(ls,i,j,k);
            float lrho=m0.density, lp=m0.pressure;
            Prim sx=mhd_slope(sg_load(ls,i-1,j,k),m0,sg_load(ls,i+1,j,k),slope);
            Prim sy=mhd_slope(sg_load(ls,i,j-1,k),m0,sg_load(ls,i,j+1,k),slope);
            Prim sz=mhd_slope(sg_load(ls,i,j,k-1),m0,sg_load(ls,i,j,k+1),slope);
            Cons u0=mhd_p2c(m0,gamma);
            Cons du=mhd_hancock_dir(m0,sx,1,gamma,0.5f*dtdx);
            du=mhd_cadd(du,mhd_hancock_dir(m0,sy,2,gamma,0.5f*dtdx));
            du=mhd_cadd(du,mhd_hancock_dir(m0,sz,3,gamma,0.5f*dtdx));
            Prim mh=mhd_c2p(mhd_cadd(u0,du),gamma,smallr,smallc2);
            if (mh.density<=smallr || mh.pressure<=smallp) mh=m0;
            Prim fr,fl;
            if (i>1 && (j>1&&j<work_size) && (k>1&&k<work_size)){ fr=mhd_padd(mh,sx,-1.0f); store_face(rx,i-2,j-2,k-2,fr,smallr,smallp,lrho,lp); }
            if (i<work_size && (j>1&&j<work_size) && (k>1&&k<work_size)){ fl=mhd_padd(mh,sx,1.0f); store_face(lx,i-1,j-2,k-2,fl,smallr,smallp,lrho,lp); }
            if ((i>1&&i<work_size) && j>1 && (k>1&&k<work_size)){ fr=mhd_padd(mh,sy,-1.0f); store_face(ry,i-2,j-2,k-2,fr,smallr,smallp,lrho,lp); }
            if ((i>1&&i<work_size) && j<work_size && (k>1&&k<work_size)){ fl=mhd_padd(mh,sy,1.0f); store_face(ly,i-2,j-1,k-2,fl,smallr,smallp,lrho,lp); }
            if ((i>1&&i<work_size) && (j>1&&j<work_size) && k>1){ fr=mhd_padd(mh,sz,-1.0f); store_face(rz,i-2,j-2,k-2,fr,smallr,smallp,lrho,lp); }
            if ((i>1&&i<work_size) && (j>1&&j<work_size) && k<work_size){ fl=mhd_padd(mh,sz,1.0f); store_face(lz,i-2,j-2,k-1,fl,smallr,smallp,lrho,lp); }
        }
        __syncthreads();
    }

    // 3) riemann_driver_mhd : HLLD/HLL/LLF + rotation + GLM --------------------
    {
        const int n=(2*NS+1)*(2*NS)*(2*NS);
        for (int work_idx=thread_idx; work_idx<=n*3-1; work_idx+=blockDim.x){
            int i,j,k,dir; Prim L,R; Cons flux;
            if (work_idx<n){ dir=1; idx1Dto3D(work_idx,2*NS+1,2*NS,i,j,k); L=load_face(lx,i,j,k); R=load_face(rx,i,j,k);
                flux=mhd_rot_flux_from(mhd_riemann(mhd_rot_to(L,dir),mhd_rot_to(R,dir),gamma,smallr,smallc2,ch,riemann,sw_dmin,sw_pmin),dir);
                store_flux(lx,i,j,k,flux); }
            else if (work_idx<2*n){ dir=2; idx1Dto3D(work_idx-n,2*NS,2*NS+1,i,j,k); L=load_face(ly,i,j,k); R=load_face(ry,i,j,k);
                flux=mhd_rot_flux_from(mhd_riemann(mhd_rot_to(L,dir),mhd_rot_to(R,dir),gamma,smallr,smallc2,ch,riemann,sw_dmin,sw_pmin),dir);
                store_flux(ly,i,j,k,flux); }
            else { dir=3; idx1Dto3D(work_idx-2*n,2*NS,2*NS,i,j,k); L=load_face(lz,i,j,k); R=load_face(rz,i,j,k);
                flux=mhd_rot_flux_from(mhd_riemann(mhd_rot_to(L,dir),mhd_rot_to(R,dir),gamma,smallr,smallc2,ch,riemann,sw_dmin,sw_pmin),dir);
                store_flux(lz,i,j,k,flux); }
        }
        __syncthreads();
    }

    // 4) mhd_conservative_update (+ fused turb + psi damping) ------------------
    {
        const int work_size=2*NS;
        for (int work_idx=thread_idx; work_idx<=work_size*work_size*work_size-1; work_idx+=blockDim.x){
            int isg,jsg,ksg; idx1Dto3D(work_idx/8, work_size/2, work_size/2, isg, jsg, ksg);
            isg+=1; jsg+=1; ksg+=1;
            int ind_nbor=1+isg+NSP2*jsg+NSP2SQ*ksg;
            int oct_idx=NBORA(ind_nbor, subgrid_idx);
            int cell_idx=(work_idx&7)+1;
            int i,j,k; idx1Dto3D(cell_idx-1,2,2,i,j,k); i+=2*(isg-1); j+=2*(jsg-1); k+=2*(ksg-1);
            #define FDIV(fld) ((__half2float(lx.fld[lx.idx(i,j,k)])-__half2float(lx.fld[lx.idx(i+1,j,k)])) \
                             + (__half2float(ly.fld[ly.idx(i,j,k)])-__half2float(ly.fld[ly.idx(i,j+1,k)])) \
                             + (__half2float(lz.fld[lz.idx(i,j,k)])-__half2float(lz.fld[lz.idx(i,j,k+1)])))*dtdx
            float d1=FDIV(d), d2=FDIV(vx), d3=FDIV(vy), d4=FDIV(vz), d5=FDIV(p), d6=FDIV(bx), d7=FDIV(by), d8=FDIV(bz), d9=FDIV(psi);
            #undef FDIV
            if (base_write){
                d1+=UOLD(cell_idx,1,oct_idx); d2+=UOLD(cell_idx,2,oct_idx); d3+=UOLD(cell_idx,3,oct_idx);
                d4+=UOLD(cell_idx,4,oct_idx); d5+=UOLD(cell_idx,5,oct_idx); d6+=UOLD(cell_idx,6,oct_idx);
                d7+=UOLD(cell_idx,7,oct_idx); d8+=UOLD(cell_idx,8,oct_idx); d9+=UOLD(cell_idx,9,oct_idx);
            } else {
                d1+=UNEW(cell_idx,1,oct_idx); d2+=UNEW(cell_idx,2,oct_idx); d3+=UNEW(cell_idx,3,oct_idx);
                d4+=UNEW(cell_idx,4,oct_idx); d5+=UNEW(cell_idx,5,oct_idx); d6+=UNEW(cell_idx,6,oct_idx);
                d7+=UNEW(cell_idx,7,oct_idx); d8+=UNEW(cell_idx,8,oct_idx); d9+=UNEW(cell_idx,9,oct_idx);
            }
#ifdef TURB
            if (do_turb && d1>=turb_min_rho){
                int tbmin[3],tbmax[3]; float tw1[3],tw2[3];
                for (int td=1; td<=NDIM; td++){
                    float txc=(2*grid[oct_idx-1].ckey[td-1]+((cell_idx-1)/(1<<(td-1))&1)+0.5f)*dx-d_skip[td-1];
                    float trr=txc/boxlen*(float)TURB_GS;
                    tbmin[td-1]=(int)floorf(trr); tbmax[td-1]=(int)ceilf(trr);
                    if (tbmin[td-1]==TURB_GS) tbmin[td-1]=0;
                    if (tbmax[td-1]==TURB_GS) tbmax[td-1]=0;
                    tw1[td-1]=trr-floorf(trr); tw2[td-1]=1.0f-tw1[td-1];
                }
                #define AF(t,a,b,c) afield_now[((t)-1)+3*((a)+TURB_GS*((b)+TURB_GS*(c)))]
                float tff[3];
                for (int td=1; td<=3; td++){
                    tff[td-1]=
                        AF(td,tbmin[0],tbmin[1],tbmin[2])*tw2[0]*tw2[1]*tw2[2] + AF(td,tbmax[0],tbmin[1],tbmin[2])*tw1[0]*tw2[1]*tw2[2] +
                        AF(td,tbmin[0],tbmax[1],tbmin[2])*tw2[0]*tw1[1]*tw2[2] + AF(td,tbmax[0],tbmax[1],tbmin[2])*tw1[0]*tw1[1]*tw2[2] +
                        AF(td,tbmin[0],tbmin[1],tbmax[2])*tw2[0]*tw2[1]*tw1[2] + AF(td,tbmax[0],tbmin[1],tbmax[2])*tw1[0]*tw2[1]*tw1[2] +
                        AF(td,tbmin[0],tbmax[1],tbmax[2])*tw2[0]*tw1[1]*tw1[2] + AF(td,tbmax[0],tbmax[1],tbmax[2])*tw1[0]*tw1[1]*tw1[2];
                }
                #undef AF
                float trmax=fmaxf(d1,smallr), tener=d5;
                tener=fmaxf(tener-0.5f*d2*d2/trmax, d1*smallc2);
                tener=fmaxf(tener-0.5f*d3*d3/trmax, d1*smallc2);
                tener=fmaxf(tener-0.5f*d4*d4/trmax, d1*smallc2);
                d2+=trmax*tff[0]*dt; d3+=trmax*tff[1]*dt; d4+=trmax*tff[2]*dt;
                tener=fmaxf(tener+0.5f*d2*d2/trmax, d1*smallc2);
                tener=fmaxf(tener+0.5f*d3*d3/trmax, d1*smallc2);
                tener=fmaxf(tener+0.5f*d4*d4/trmax, d1*smallc2);
                d5=tener;
            }
#endif
            UNEW(cell_idx,1,oct_idx)=d1; UNEW(cell_idx,2,oct_idx)=d2; UNEW(cell_idx,3,oct_idx)=d3;
            UNEW(cell_idx,4,oct_idx)=d4; UNEW(cell_idx,5,oct_idx)=d5; UNEW(cell_idx,6,oct_idx)=d6;
            UNEW(cell_idx,7,oct_idx)=d7; UNEW(cell_idx,8,oct_idx)=d8; UNEW(cell_idx,9,oct_idx)=d9*glm_fac;
        }
    }
}

extern "C" void launch_mhd_integrator(
    void* grid, void* uold, void* unew, void* f, void* father, void* nbor,
    int head_idx, int num_subgrids, int ngridmax, int ilevel, int levelmin, int levelmax,
    float gamma, float smallr, float smallc2, float ch, float dt, float dx,
    int slope, int riemann, float sw_dmin, float sw_pmin,
    void* constant_gravity, float glm_fac, int base_write,
    float courant_factor, void* dt_out, int cfl_sqrt3
#ifdef TURB
    , void* afield_now, void* d_skip, float boxlen, float turb_min_rho, int do_turb
#endif
){
    int threads=256, blocks=num_subgrids;
    if (blocks<=0) return;
    mhd_integrator_kernel_cuda<<<blocks,threads>>>(
        (Oct*)grid,(float*)uold,(float*)unew,(const float*)f,(const int*)father,(const int*)nbor,
        head_idx,num_subgrids,ngridmax,ilevel,levelmin,levelmax,
        gamma,smallr,smallc2,ch,dt,dx,slope,riemann,sw_dmin,sw_pmin,
        (const double*)constant_gravity,glm_fac,base_write,courant_factor,(double*)dt_out,cfl_sqrt3
#ifdef TURB
        ,(const float*)afield_now,(const float*)d_skip,boxlen,turb_min_rho,do_turb
#endif
    );
}

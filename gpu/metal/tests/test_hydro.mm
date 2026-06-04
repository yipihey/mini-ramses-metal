// test_hydro.mm — unit test for the Metal hydro numerical core (hydro.h) and the
// trivial state kernels (set_unew/set_uold in hydro.metal).  Checks the GPU fp32
// results against a host double-precision replica of the SAME formulas plus two
// analytic invariants (EOS round-trip identity; identical-state HLLC == physical
// flux).  No RAMSES run.  See tests/run_tests.sh for the build line.
#import <Metal/Metal.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#ifndef NDIM
#define NDIM 3
#endif
#include "../../ramses_metal.h"   // TWOTONDIM/NHVAR/SUBGRIDSIZE/Oct/HydroParams/UH/IDX3
// mirrors HydroParams (ramses_metal.h): doubles narrowed to float, 8B-ordered.
struct HP { float gamma,dt,dx,smallr,smallc,courant,fp_scale;
            int slope,riemann,head,num,ngridmax,ilevel,levelmin,levelmax; };
static_assert(sizeof(HP)==sizeof(HydroParams), "HP must match HydroParams layout");

// ---- host double-precision replica of hydro.h (the parity reference) --------
struct P { double r,u,v,w,p; };
struct C { double d,mx,my,mz,e; };
static double mag2(double x,double y,double z){ return x*x+(y*y+z*z); }
static double cpress(C c,double g){ return (g-1)*(c.e-0.5*mag2(c.mx,c.my,c.mz)/c.d); }
static double cenergy(P q,double g){ return q.p/(g-1)+0.5*q.r*mag2(q.u,q.v,q.w); }
static double csound(P q,double g){ return sqrt(g*q.p/q.r); }
static C p2c(P q,double g){ return {q.r,q.u*q.r,q.v*q.r,q.w*q.r,cenergy(q,g)}; }
static double minmod(double l,double m,double r){ double a=m-l,b=r-m;
  if(a*b<=0)return 0; return a>0?fmin(a,b):fmax(a,b); }
static double moncen(double l,double m,double r,int s){ double a=m-l,b=r-m,c=0.5*(a+b),f=s;
  if(a*b<=0)return 0; return a>0?fmin(f*fmin(a,b),c):fmax(f*fmax(a,b),c); }
static C hllc(P L,P R,double g){
  double smallr=1e-10, smp=1e-20/g;
  L.r=fmax(L.r,smallr); R.r=fmax(R.r,smallr);
  L.p=fmax(L.p,smp*L.r); R.p=fmax(R.p,smp*R.r);
  double entho=1/(g-1), el=L.p*entho, er=R.p*entho;
  double ekl=0.5*L.r*mag2(L.u,L.v,L.w), ekr=0.5*R.r*mag2(R.u,R.v,R.w);
  double etl=el+ekl, etr=er+ekr;
  double cl=csound(L,g), cr=csound(R,g);
  double sl=fmin(L.u,R.u)-fmax(cl,cr), sr=fmax(L.u,R.u)+fmax(cl,cr);
  double rcl=L.r*(L.u-sl), rcr=R.r*(sr-R.u);
  double us=(rcr*R.u+rcl*L.u+(L.p-R.p))/(rcr+rcl);
  double ps=(rcr*L.p+rcl*R.p+rcl*rcr*(L.u-R.u))/(rcr+rcl);
  double rsl=L.r*(sl-L.u)/(sl-us), rsr=R.r*(sr-R.u)/(sr-us);
  double esl=((sl-L.u)*etl-L.p*L.u+ps*us)/(sl-us);
  double esr=((sr-R.u)*etr-R.p*R.u+ps*us)/(sr-us);
  double ro,uo,vo,wo,po,eo;
  if(sl>0){ro=L.r;uo=L.u;vo=L.v;wo=L.w;po=L.p;eo=etl;}
  else if(us>0){ro=rsl;uo=us;vo=L.v;wo=L.w;po=ps;eo=esl;}
  else if(sr>0){ro=rsr;uo=us;vo=R.v;wo=R.w;po=ps;eo=esr;}
  else {ro=R.r;uo=R.u;vo=R.v;wo=R.w;po=R.p;eo=etr;}
  return {ro*uo, ro*uo*uo+po, ro*uo*vo, ro*uo*wo, (eo+po)*uo};
}
static double hllf(double sl,double sr,double lf,double rf,double lc,double rc){
  return (sr*lf-sl*rf+sr*sl*(rc-lc))/(sr-sl); }
static C hll(P L,P R,double g,bool llf){
  double smallr=1e-10, smp=1e-20/g;
  L.r=fmax(L.r,smallr); R.r=fmax(R.r,smallr);
  L.p=fmax(L.p,smp*L.r); R.p=fmax(R.p,smp*R.r);
  double cl=csound(L,g), cr=csound(R,g), sl,sr;
  if(llf){ double sm=fmax(fabs(L.u)+cl,fabs(R.u)+cr); sl=-sm; sr=sm; }
  else { sl=fmin(fmin(L.u,R.u)-fmax(cl,cr),0.0); sr=fmax(fmax(L.u,R.u)+fmax(cl,cr),0.0); }
  C Lc=p2c(L,g), Rc=p2c(R,g);
  C lf,rf;
  lf={Lc.mx, L.p+L.u*Lc.mx, L.u*Lc.my, L.u*Lc.mz, L.u*(L.p+Lc.e)};
  rf={Rc.mx, R.p+R.u*Rc.mx, R.u*Rc.my, R.u*Rc.mz, R.u*(R.p+Rc.e)};
  return { hllf(sl,sr,lf.d,rf.d,Lc.d,Rc.d), hllf(sl,sr,lf.mx,rf.mx,Lc.mx,Rc.mx),
           hllf(sl,sr,lf.my,rf.my,Lc.my,Rc.my), hllf(sl,sr,lf.mz,rf.mz,Lc.mz,Rc.mz),
           hllf(sl,sr,lf.e,rf.e,Lc.e,Rc.e) };
}

enum { S_LLF=1, S_HLL=2, S_HLLC=3 };
static C rdispatch(P a, P b, double g, int riemann){
  if(riemann==S_HLL)  return hll(a,b,g,false);
  if(riemann==S_HLLC) return hllc(a,b,g);
  return hll(a,b,g,true);  // LLF
}
// host double replica of trace_cell_1d / godunov_oct_1d (the parity reference)
static void trace1d(P l,P m,P r,double g,double dtdx,int slope,P& qL,P& qR){
  double smallr=1e-10, smallp=1e-10*(1e-10*1e-10);
  P s={0.5*moncen(l.r,m.r,r.r,slope),0.5*moncen(l.u,m.u,r.u,slope),
       0.5*moncen(l.v,m.v,r.v,slope),0.5*moncen(l.w,m.w,r.w,slope),0.5*moncen(l.p,m.p,r.p,slope)};
  P src={ -m.u*s.r - s.u*m.r, -m.u*s.u - s.p/m.r, -m.u*s.v, -m.u*s.w, -m.u*s.p - s.u*g*m.p };
  P pp={m.r+dtdx*src.r,m.u+dtdx*src.u,m.v+dtdx*src.v,m.w+dtdx*src.w,m.p+dtdx*src.p};
  qL={pp.r+s.r,pp.u+s.u,pp.v+s.v,pp.w+s.w,pp.p+s.p};
  if(qL.r<smallr)qL.r=m.r; if(qL.p<smallp)qL.p=m.p;
  qR={pp.r-s.r,pp.u-s.u,pp.v-s.v,pp.w-s.w,pp.p-s.p};
  if(qR.r<smallr)qR.r=m.r; if(qR.p<smallp)qR.p=m.p;
}
static void god1d(P sg[6],double g,double dtdx,int slope,int riemann,C du[2]){
  P qL[4],qR[4];
  for(int c=1;c<=4;++c) trace1d(sg[c-1],sg[c],sg[c+1],g,dtdx,slope,qL[c-1],qR[c-1]);
  C fa=rdispatch(qL[0],qR[1],g,riemann), fb=rdispatch(qL[1],qR[2],g,riemann), fc=rdispatch(qL[2],qR[3],g,riemann);
  du[0]={(fa.d-fb.d)*dtdx,(fa.mx-fb.mx)*dtdx,(fa.my-fb.my)*dtdx,(fa.mz-fb.mz)*dtdx,(fa.e-fb.e)*dtdx};
  du[1]={(fb.d-fc.d)*dtdx,(fb.mx-fc.mx)*dtdx,(fb.my-fc.my)*dtdx,(fb.mz-fc.mz)*dtdx,(fb.e-fc.e)*dtdx};
}

// ---- host double replica of the 3D Godunov (trace_cell_3d + sweeps) ---------
struct Tr{ P qLx,qRx,qLy,qRy,qLz,qRz; };
static P padd(P a,P b,double s){ return {a.r+s*b.r,a.u+s*b.u,a.v+s*b.v,a.w+s*b.w,a.p+s*b.p}; }
static Tr trace3d(const P* sg,int i,int j,int k,double g,double dtdx,int slope){
  double smallr=1e-10, smallp=1e-10*(1e-10*1e-10);
  #define G(a,b,c) sg[(a)+6*(b)+36*(c)]
  P m=G(i,j,k);
  P sx={0.5*moncen(G(i-1,j,k).r,m.r,G(i+1,j,k).r,slope),0.5*moncen(G(i-1,j,k).u,m.u,G(i+1,j,k).u,slope),0.5*moncen(G(i-1,j,k).v,m.v,G(i+1,j,k).v,slope),0.5*moncen(G(i-1,j,k).w,m.w,G(i+1,j,k).w,slope),0.5*moncen(G(i-1,j,k).p,m.p,G(i+1,j,k).p,slope)};
  P sy={0.5*moncen(G(i,j-1,k).r,m.r,G(i,j+1,k).r,slope),0.5*moncen(G(i,j-1,k).u,m.u,G(i,j+1,k).u,slope),0.5*moncen(G(i,j-1,k).v,m.v,G(i,j+1,k).v,slope),0.5*moncen(G(i,j-1,k).w,m.w,G(i,j+1,k).w,slope),0.5*moncen(G(i,j-1,k).p,m.p,G(i,j+1,k).p,slope)};
  P sz={0.5*moncen(G(i,j,k-1).r,m.r,G(i,j,k+1).r,slope),0.5*moncen(G(i,j,k-1).u,m.u,G(i,j,k+1).u,slope),0.5*moncen(G(i,j,k-1).v,m.v,G(i,j,k+1).v,slope),0.5*moncen(G(i,j,k-1).w,m.w,G(i,j,k+1).w,slope),0.5*moncen(G(i,j,k-1).p,m.p,G(i,j,k+1).p,slope)};
  #undef G
  double divu=sx.u+sy.v+sz.w;
  P src={ -m.u*sx.r-m.v*sy.r-m.w*sz.r-divu*m.r,
          -m.u*sx.u-m.v*sy.u-m.w*sz.u-sx.p/m.r,
          -m.u*sx.v-m.v*sy.v-m.w*sz.v-sy.p/m.r,
          -m.u*sx.w-m.v*sy.w-m.w*sz.w-sz.p/m.r,
          -m.u*sx.p-m.v*sy.p-m.w*sz.p-divu*g*m.p };
  P p=padd(m,src,dtdx);
  Tr t={padd(p,sx,1),padd(p,sx,-1),padd(p,sy,1),padd(p,sy,-1),padd(p,sz,1),padd(p,sz,-1)};
  P* q[6]={&t.qLx,&t.qRx,&t.qLy,&t.qRy,&t.qLz,&t.qRz};
  for(int n=0;n<6;n++){ if(q[n]->r<smallr)q[n]->r=m.r; if(q[n]->p<smallp)q[n]->p=m.p; }
  return t;
}
static C cdf(C a,C b,double s){ return {(a.d-b.d)*s,(a.mx-b.mx)*s,(a.my-b.my)*s,(a.mz-b.mz)*s,(a.e-b.e)*s}; }
static C cad(C a,C b){ return {a.d+b.d,a.mx+b.mx,a.my+b.my,a.mz+b.mz,a.e+b.e}; }
static C fx_(P L,P R,double g,int riem){ return rdispatch(L,R,g,riem); }
static C fy_(P L,P R,double g,int riem){ P Lr={L.r,L.v,L.w,L.u,L.p},Rr={R.r,R.v,R.w,R.u,R.p}; C f=rdispatch(Lr,Rr,g,riem); return {f.d,f.mz,f.mx,f.my,f.e}; }
static C fz_(P L,P R,double g,int riem){ P Lr={L.r,L.w,L.u,L.v,L.p},Rr={R.r,R.w,R.u,R.v,R.p}; C f=rdispatch(Lr,Rr,g,riem); return {f.d,f.my,f.mz,f.mx,f.e}; }
static void god3d(const P* sg,double g,double dtdx,int slope,int riem,C du[8]){
  for(int ck=0;ck<2;ck++)for(int cj=0;cj<2;cj++)for(int ci=0;ci<2;ci++){
    int I=ci+2,J=cj+2,K=ck+2;
    C fxl=fx_(trace3d(sg,I-1,J,K,g,dtdx,slope).qLx,trace3d(sg,I,J,K,g,dtdx,slope).qRx,g,riem);
    C fxr=fx_(trace3d(sg,I,J,K,g,dtdx,slope).qLx,trace3d(sg,I+1,J,K,g,dtdx,slope).qRx,g,riem);
    C fyl=fy_(trace3d(sg,I,J-1,K,g,dtdx,slope).qLy,trace3d(sg,I,J,K,g,dtdx,slope).qRy,g,riem);
    C fyr=fy_(trace3d(sg,I,J,K,g,dtdx,slope).qLy,trace3d(sg,I,J+1,K,g,dtdx,slope).qRy,g,riem);
    C fzl=fz_(trace3d(sg,I,J,K-1,g,dtdx,slope).qLz,trace3d(sg,I,J,K,g,dtdx,slope).qRz,g,riem);
    C fzr=fz_(trace3d(sg,I,J,K,g,dtdx,slope).qLz,trace3d(sg,I,J,K+1,g,dtdx,slope).qRz,g,riem);
    du[ci+2*cj+4*ck]=cad(cad(cdf(fxl,fxr,dtdx),cdf(fyl,fyr,dtdx)),cdf(fzl,fzr,dtdx));
  }
}

// host double replica of the AMR Godunov (flux zeroing + boundary-flux sums).
static void god1d_amr(P sg[6], bool ref[6], double g,double dtdx,int slope,int riem, C du[2], C bnd[2]){
  P qL[4],qR[4]; for(int c=1;c<=4;++c) trace1d(sg[c-1],sg[c],sg[c+1],g,dtdx,slope,qL[c-1],qR[c-1]);
  C fx[3];
  for(int a=0;a<3;++a){ fx[a]=rdispatch(qL[a],qR[a+1],g,riem); if(ref[a+1]||ref[a+2]) fx[a]=C{0,0,0,0,0}; }
  du[0]=cdf(fx[0],fx[1],dtdx); du[1]=cdf(fx[1],fx[2],dtdx);
  bnd[0]=fx[0]; bnd[1]=fx[2];
}
static void god3d_amr(const P* sg, const bool* ref, double g,double dtdx,int slope,int riem, C du[8], C bnd[6]){
  #define RI(a,b,c) ((a)+6*(b)+36*(c))
  C fx[3][2][2],fy[2][3][2],fz[2][2][3];
  for(int ck=0;ck<2;ck++)for(int cj=0;cj<2;cj++){int J=cj+2,K=ck+2;
    for(int a=0;a<3;a++){ C f=fx_(trace3d(sg,a+1,J,K,g,dtdx,slope).qLx,trace3d(sg,a+2,J,K,g,dtdx,slope).qRx,g,riem);
      if(ref[RI(a+1,J,K)]||ref[RI(a+2,J,K)]) f=C{0,0,0,0,0}; fx[a][cj][ck]=f; }}
  for(int ck=0;ck<2;ck++)for(int ci=0;ci<2;ci++){int I=ci+2,K=ck+2;
    for(int b=0;b<3;b++){ C f=fy_(trace3d(sg,I,b+1,K,g,dtdx,slope).qLy,trace3d(sg,I,b+2,K,g,dtdx,slope).qRy,g,riem);
      if(ref[RI(I,b+1,K)]||ref[RI(I,b+2,K)]) f=C{0,0,0,0,0}; fy[ci][b][ck]=f; }}
  for(int cj=0;cj<2;cj++)for(int ci=0;ci<2;ci++){int I=ci+2,J=cj+2;
    for(int c=0;c<3;c++){ C f=fz_(trace3d(sg,I,J,c+1,g,dtdx,slope).qLz,trace3d(sg,I,J,c+2,g,dtdx,slope).qRz,g,riem);
      if(ref[RI(I,J,c+1)]||ref[RI(I,J,c+2)]) f=C{0,0,0,0,0}; fz[ci][cj][c]=f; }}
  for(int ck=0;ck<2;ck++)for(int cj=0;cj<2;cj++)for(int ci=0;ci<2;ci++)
    du[ci+2*cj+4*ck]=cad(cad(cdf(fx[ci][cj][ck],fx[ci+1][cj][ck],dtdx),cdf(fy[ci][cj][ck],fy[ci][cj+1][ck],dtdx)),cdf(fz[ci][cj][ck],fz[ci][cj][ck+1],dtdx));
  for(int i=0;i<6;i++) bnd[i]=C{0,0,0,0,0};
  for(int cj=0;cj<2;cj++)for(int ck=0;ck<2;ck++){bnd[0]=cad(bnd[0],fx[0][cj][ck]);bnd[1]=cad(bnd[1],fx[2][cj][ck]);}
  for(int ci=0;ci<2;ci++)for(int ck=0;ck<2;ck++){bnd[2]=cad(bnd[2],fy[ci][0][ck]);bnd[3]=cad(bnd[3],fy[ci][2][ck]);}
  for(int ci=0;ci<2;ci++)for(int cj=0;cj<2;cj++){bnd[4]=cad(bnd[4],fz[ci][cj][0]);bnd[5]=cad(bnd[5],fz[ci][cj][2]);}
  #undef RI
}

static int g_fail = 0;
static void chk(const char* name, double got, double ref, double rtol) {
    double err = fabs(got-ref), den = fmax(1.0, fabs(ref));
    bool ok = (err/den) < rtol;
    if(!ok){ printf("    [FAIL] %-22s got=%.7g ref=%.7g relerr=%.2e\n",name,got,ref,err/den); g_fail++; }
}

int main(int argc, char** argv) {
    const char* libpath = argc>1 ? argv[1] : "/tmp/test_hydro.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if(!dev){ fprintf(stderr,"no Metal device\n"); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){ fprintf(stderr,"lib: %s\n", err.localizedDescription.UTF8String); return 2; }

        auto pso = [&](const char* n){
            id<MTLFunction> fn=[lib newFunctionWithName:@(n)];
            if(!fn){ fprintf(stderr,"no fn %s\n",n); exit(2); }
            id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:fn error:&err];
            if(!p){ fprintf(stderr,"pso %s: %s\n",n,err.localizedDescription.UTF8String); exit(2); }
            return p;
        };

        // ---- 1) numerical-core test --------------------------------------
        double g = 1.4;
        P L = {1.0, 0.3, -0.2, 0.1, 1.0};      // a generic subsonic left state
        P R = {0.125, 0.0, 0.0, 0.0, 0.1};     // Sod-like right state
        float in[15] = { (float)g, (float)L.r,(float)L.u,(float)L.v,(float)L.w,(float)L.p,
                         (float)R.r,(float)R.u,(float)R.v,(float)R.w,(float)R.p,
                         2.0f, 1.0f, 2.0f, 4.0f };  // slopefactor=2(moncen), l/m/r=1,2,4
        id<MTLBuffer> bin =[dev newBufferWithBytes:in length:sizeof(in) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout=[dev newBufferWithLength:32*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> core = pso("hydro_core_test");
        { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:core]; [e setBuffer:bin offset:0 atIndex:0]; [e setBuffer:bout offset:0 atIndex:1];
          [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
        float* o = (float*)bout.contents;

        const double RT=2e-6, RF=1e-4;   // round-trip vs fp32 eps; flux vs fp32-of-double
        // round-trip identity
        chk("rt.rho",o[0],L.r,RT); chk("rt.u",o[1],L.u,RT); chk("rt.v",o[2],L.v,RT);
        chk("rt.w",o[3],L.w,RT);   chk("rt.P",o[4],L.p,RT);
        // HLLC / HLL / LLF vs double replica
        C fc=hllc(L,R,g), fh=hll(L,R,g,false), fl=hll(L,R,g,true);
        chk("hllc.d",o[5],fc.d,RF);  chk("hllc.mx",o[6],fc.mx,RF); chk("hllc.my",o[7],fc.my,RF);
        chk("hllc.mz",o[8],fc.mz,RF); chk("hllc.e",o[9],fc.e,RF);
        chk("hll.d",o[10],fh.d,RF);  chk("hll.mx",o[11],fh.mx,RF); chk("hll.e",o[14],fh.e,RF);
        chk("llf.d",o[15],fl.d,RF);  chk("llf.mx",o[16],fl.mx,RF); chk("llf.e",o[19],fl.e,RF);
        // slope limiters
        chk("minmod(1,2,4)",o[20],minmod(1,2,4),RF);
        chk("moncen(1,2,4)",o[21],moncen(1,2,4,2),RF);
        // identical-state HLLC == physical flux F(L)
        C pf = { L.r*L.u, L.r*L.u*L.u+L.p, L.r*L.u*L.v, L.r*L.u*L.w, (cenergy(L,g)+L.p)*L.u };
        chk("idL.d",o[22],pf.d,RF); chk("idL.mx",o[23],pf.mx,RF); chk("idL.my",o[24],pf.my,RF);
        chk("idL.mz",o[25],pf.mz,RF); chk("idL.e",o[26],pf.e,RF);

        // ---- 1b) 1D Godunov update (trace + riemann + flux differencing) --
        id<MTLComputePipelineState> g1d = pso("godunov_1d_test");
        auto run_god1d = [&](P sg[6], double gam, double dtdx, int slope, int riem, float duout[10]){
            float gi[34]; gi[0]=gam; gi[1]=dtdx; gi[2]=slope; gi[3]=riem;
            for(int c=0;c<6;c++){ int o=4+c*5; gi[o]=sg[c].r; gi[o+1]=sg[c].u; gi[o+2]=sg[c].v; gi[o+3]=sg[c].w; gi[o+4]=sg[c].p; }
            id<MTLBuffer> bi=[dev newBufferWithBytes:gi length:sizeof(gi) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bo=[dev newBufferWithLength:10*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:g1d]; [e setBuffer:bi offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            for(int i=0;i<10;i++) duout[i]=((float*)bo.contents)[i];
        };
        auto cmp_god = [&](const char* tag, P sg[6], double gam, double dtdx, int slope, int riem){
            float gpu[10]; run_god1d(sg, gam, dtdx, slope, riem, gpu);
            C du[2]; god1d(sg, gam, dtdx, slope, riem, du);
            double ref[10]={du[0].d,du[0].mx,du[0].my,du[0].mz,du[0].e, du[1].d,du[1].mx,du[1].my,du[1].mz,du[1].e};
            const char* nm[10]={"c2.d","c2.mx","c2.my","c2.mz","c2.e","c3.d","c3.mx","c3.my","c3.mz","c3.e"};
            for(int i=0;i<10;i++){ char b[40]; snprintf(b,sizeof b,"%s.%s",tag,nm[i]); chk(b,gpu[i],ref[i],RF); }
        };
        // Sod-like jump (HLLC, minmod) — vs host-double replica
        { P sg[6]={{1,0,0,0,1},{1,0,0,0,1},{1,0,0,0,1},{0.125,0,0,0,0.1},{0.125,0,0,0,0.1},{0.125,0,0,0,0.1}};
          cmp_god("sodHLLC", sg, 1.4, 0.1, 1, S_HLLC); }
        // smooth ramp with transverse velocity (moncen, HLL) — exercises slopes + advection
        { P sg[6]; for(int c=0;c<6;c++){ double x=c; sg[c]={1.0+0.1*x, 0.2, 0.3, -0.1, 1.0+0.05*x}; }
          cmp_god("rampHLL", sg, 1.4, 0.15, 2, S_HLL); }
        // uniform state -> zero update (LLF)
        { P sg[6]; for(int c=0;c<6;c++) sg[c]={1.3,0.4,-0.2,0.1,0.9};
          float gpu[10]; run_god1d(sg,1.4,0.2,2,S_LLF,gpu);
          for(int i=0;i<10;i++) if(fabs(gpu[i])>1e-6){ printf("    [FAIL] uniform du[%d]=%.3e != 0\n",i,gpu[i]); g_fail++; } }

        // ---- 1c) 3D Godunov update (directional sweeps + velocity rotation)
        id<MTLComputePipelineState> g3d = pso("godunov_3d_test");
        auto run_god3d = [&](P sg[216], double gam, double dtdx, int slope, int riem, float duout[40]){
            float gi[4+216*5]; gi[0]=gam; gi[1]=dtdx; gi[2]=slope; gi[3]=riem;
            for(int c=0;c<216;c++){ int o=4+c*5; gi[o]=sg[c].r; gi[o+1]=sg[c].u; gi[o+2]=sg[c].v; gi[o+3]=sg[c].w; gi[o+4]=sg[c].p; }
            id<MTLBuffer> bi=[dev newBufferWithBytes:gi length:sizeof(gi) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bo=[dev newBufferWithLength:40*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:g3d]; [e setBuffer:bi offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            for(int i=0;i<40;i++) duout[i]=((float*)bo.contents)[i];
        };
        // 3D field: x-Sod jump + small transverse velocities (exercises all 3 sweeps + rotation)
        { P sg[216];
          for(int k=0;k<6;k++)for(int j=0;j<6;j++)for(int i=0;i<6;i++){
              bool left = i<3;
              sg[i+6*j+36*k] = { left?1.0:0.2, 0.05*(j-2), -0.03*(k-2), 0.02*(i-2), left?1.0:0.3 }; }
          float gpu[40]; run_god3d(sg,1.4,0.1,1,S_HLLC,gpu);
          C du[8]; god3d(sg,1.4,0.1,1,S_HLLC,du);
          double ref[40]; for(int n=0;n<8;n++){ ref[n*5]=du[n].d;ref[n*5+1]=du[n].mx;ref[n*5+2]=du[n].my;ref[n*5+3]=du[n].mz;ref[n*5+4]=du[n].e; }
          const char* vn[5]={"d","mx","my","mz","e"};
          for(int n=0;n<8;n++)for(int vv=0;vv<5;vv++){ char b[40]; snprintf(b,sizeof b,"g3d.c%d.%s",n,vn[vv]); chk(b,gpu[n*5+vv],ref[n*5+vv],RF); } }
        // 3D uniform -> zero update
        { P sg[216]; for(int c=0;c<216;c++) sg[c]={1.1,0.3,-0.2,0.15,0.7};
          float gpu[40]; run_god3d(sg,1.4,0.2,2,S_HLLC,gpu);
          for(int i=0;i<40;i++) if(fabs(gpu[i])>1e-6){ printf("    [FAIL] 3D uniform du[%d]=%.3e != 0\n",i,gpu[i]); g_fail++; } }

        // ---- 1d) hydro_godunov integrator (AMR): nbor subgrid gather + gravity
        //   predictor + AMR godunov (zero_fine_fluxes) accumulate into unew, plus
        //   the coarse-fine reflux (fixed-point atomics + finalize).  A central
        //   oct surrounded by its 3^NDIM neighbour-cube of octs (+ one cache oct
        //   + its coarse father).  Three scenarios vs the host-double replica:
        //     A) uniform               -> du == god_amr(ref=false)
        //     B) a refined -x halo cell -> that boundary flux zeroed in du
        //     C) -x neighbour is a cache (coarser) oct -> reflux to its parent
        {
            id<MTLComputePipelineState> gint = pso("hydro_godunov");
            id<MTLComputePipelineState> gfin = pso("hydro_reflux_finalize");
            const int noct = SUBGRIDSIZE;            // 3^NDIM real octs, one per cube
            const int NT = noct + 2, CACHE = noct+1, FATHER = noct+2;
            int center_cube = 1;
#if NDIM>=2
            center_cube += 3;
#endif
#if NDIM>=3
            center_cube += 9;
#endif
            const int center = center_cube + 1;
            const int mxcube = (NDIM==1) ? 0 : 12;   // -x neighbour cube slot (0-based)

            double gam=1.4, dt=0.1, dx=0.5, dtdx=dt/dx, halfdt=0.5*dt; float fp=(float)(1<<20);
            int slope=2, riem=S_HLLC;
            auto genprim=[&](int o,int c)->P{ return { 1.0+0.05*o+0.01*c, 0.1*o-0.05*c,
                                                       0.02*o, -0.03*c, 0.8+0.03*o+0.02*c }; };
            auto genfg=[&](int o,int c,int d)->double{ return d==1?0.01*o:d==2?-0.02*c:0.005*(o+c); };
            auto loadp=[&](int o,int c)->P{ P qq=genprim(o,c);
                qq.u+=genfg(o,c,1)*halfdt; qq.v+=genfg(o,c,2)*halfdt; qq.w+=genfg(o,c,3)*halfdt; return qq; };

            size_t nU=(size_t)NT*TWOTONDIM*NHVAR, nF=(size_t)NT*3*TWOTONDIM, nN=(size_t)NT*SUBGRIDSIZE;
            id<MTLBuffer> bU =[dev newBufferWithLength:nU*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bNn=[dev newBufferWithLength:nU*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bF =[dev newBufferWithLength:nF*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bNB=[dev newBufferWithLength:nN*sizeof(int)   options:MTLResourceStorageModeShared];
            id<MTLBuffer> bG =[dev newBufferWithLength:(size_t)NT*sizeof(Oct) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bFA=[dev newBufferWithLength:(size_t)NT*sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bRL=[dev newBufferWithLength:nU*sizeof(uint) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bRH=[dev newBufferWithLength:nU*sizeof(uint) options:MTLResourceStorageModeShared];
            float* U=(float*)bU.contents; float* Nn=(float*)bNn.contents; float* F=(float*)bF.contents;
            int* NB=(int*)bNB.contents; Oct* G=(Oct*)bG.contents; int* FA=(int*)bFA.contents;
            uint* RL=(uint*)bRL.contents; uint* RH=(uint*)bRH.contents;
            auto UHc  =[&](int c,int v,int o){ return ((o-1)*NHVAR+(v-1))*TWOTONDIM+(c-1); };
            auto IDX3c=[&](int c,int d,int o){ return ((o-1)*3   +(d-1))*TWOTONDIM+(c-1); };

            memset(G,0,(size_t)NT*sizeof(Oct));
            for(int o=1;o<=NT;++o){ FA[o-1]=0; for(int c=1;c<=TWOTONDIM;++c){
                P qq=genprim(o,c); C cc=p2c(qq,gam);
                U[UHc(c,1,o)]=cc.d;U[UHc(c,2,o)]=cc.mx;U[UHc(c,3,o)]=cc.my;U[UHc(c,4,o)]=cc.mz;U[UHc(c,5,o)]=cc.e;
                F[IDX3c(c,1,o)]=genfg(o,c,1);F[IDX3c(c,2,o)]=genfg(o,c,2);F[IDX3c(c,3,o)]=genfg(o,c,3); } }
            FA[CACHE-1]=FATHER; G[CACHE-1].ckey[0]=1;   // father ckey 0 -> parent cell = 1+1 = 2
            const int pcell = 2;

            HP hp{}; hp.gamma=gam;hp.dt=dt;hp.dx=dx;hp.fp_scale=fp;hp.slope=slope;hp.riemann=riem;
            hp.ngridmax=noct; hp.ilevel=2; hp.levelmin=1; hp.levelmax=99;
            id<MTLBuffer> bp2=[dev newBufferWithLength:sizeof(HP) options:MTLResourceStorageModeShared];

            auto setNB=[&](bool mxCache){ for(int k=0;k<SUBGRIDSIZE;++k) NB[(center-1)*SUBGRIDSIZE+k]=k+1;
                if(mxCache) NB[(center-1)*SUBGRIDSIZE+mxcube]=CACHE; };
            auto dispatch=[&](){
                for(size_t i=0;i<nU;++i){ Nn[i]=0; RL[i]=0; RH[i]=0; }
                hp.head=center; hp.num=1; memcpy(bp2.contents,&hp,sizeof(HP));
                id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
                [e setComputePipelineState:gint];
                [e setBuffer:bU offset:0 atIndex:0];[e setBuffer:bNn offset:0 atIndex:1];[e setBuffer:bF offset:0 atIndex:2];
                [e setBuffer:bNB offset:0 atIndex:3];[e setBuffer:bG offset:0 atIndex:4];[e setBuffer:bFA offset:0 atIndex:5];
                [e setBuffer:bRL offset:0 atIndex:6];[e setBuffer:bRH offset:0 atIndex:7];[e setBuffer:bp2 offset:0 atIndex:8];
                [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [e endEncoding];[cb commit];[cb waitUntilCompleted]; };
            auto finalize=[&](int head){
                HP h2=hp; h2.head=head; h2.num=1;
                id<MTLBuffer> bp3=[dev newBufferWithBytes:&h2 length:sizeof(HP) options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
                [e setComputePipelineState:gfin];
                [e setBuffer:bNn offset:0 atIndex:0];[e setBuffer:bRL offset:0 atIndex:1];[e setBuffer:bRH offset:0 atIndex:2];[e setBuffer:bp3 offset:0 atIndex:3];
                [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [e endEncoding];[cb commit];[cb waitUntilCompleted]; };

            const char* vn[5]={"d","mx","my","mz","e"};
            auto compdu=[&](const char* tag, C* du){ for(int cc=0;cc<TWOTONDIM;++cc)for(int vv=0;vv<5;++vv){
                double ref=vv==0?du[cc].d:vv==1?du[cc].mx:vv==2?du[cc].my:vv==3?du[cc].mz:du[cc].e;
                char b[48]; snprintf(b,sizeof b,"%s.c%d.%s",tag,cc,vn[vv]); chk(b,Nn[UHc(cc+1,vv+1,center)],ref,RF); } };

#if NDIM==1
            auto build=[&](bool mxCache, P sg[6], bool ref[6]){ for(int sx=0;sx<6;++sx){
                int cube=sx/2; int o=(mxCache&&cube==mxcube)?CACHE:cube+1;
                sg[sx]=loadp(o,1+(sx&1)); ref[sx]=false; } };
#else
            auto build=[&](bool mxCache, P sg[216], bool ref[216]){
                for(int sz=0;sz<6;++sz)for(int sy=0;sy<6;++sy)for(int sx=0;sx<6;++sx){
                    int cube=(sx/2)+3*(sy/2)+9*(sz/2); int o=(mxCache&&cube==mxcube)?CACHE:cube+1;
                    sg[sx+6*sy+36*sz]=loadp(o,1+(sx&1)+2*(sy&1)+4*(sz&1)); ref[sx+6*sy+36*sz]=false; } };
#endif
            // A) uniform
            setNB(false); dispatch();
#if NDIM==1
            { P sg[6]; bool ref[6]; build(false,sg,ref); C du[2],bnd[2]; god1d_amr(sg,ref,gam,dtdx,slope,riem,du,bnd); compdu("gint",du); }
#else
            { P sg[216]; bool ref[216]; build(false,sg,ref); C du[8],bnd[6]; god3d_amr(sg,ref,gam,dtdx,slope,riem,du,bnd); compdu("gint",du); }
#endif
            // B) one refined -x halo cell -> its boundary flux zeroed
            {
                int src = mxcube+1;                 // real oct in the -x slot
#if NDIM==1
                int rc=2, ridx=1;                   // sx=1 -> cell 2, subgrid idx 1
#else
                int rc=2, ridx=1+6*2+36*2;          // (sx=1,sy=2,sz=2) -> cell 2, idx 85
#endif
                G[src-1].refined[rc-1]=1;
                setNB(false); dispatch();
#if NDIM==1
                { P sg[6]; bool ref[6]; build(false,sg,ref); ref[ridx]=true; C du[2],bnd[2]; god1d_amr(sg,ref,gam,dtdx,slope,riem,du,bnd); compdu("gintR",du); }
#else
                { P sg[216]; bool ref[216]; build(false,sg,ref); ref[ridx]=true; C du[8],bnd[6]; god3d_amr(sg,ref,gam,dtdx,slope,riem,du,bnd); compdu("gintR",du); }
#endif
                G[src-1].refined[rc-1]=0;
            }
            // C) -x neighbour is a cache (coarser) oct -> reflux onto its parent cell
            {
                setNB(true); dispatch(); finalize(FATHER);
                double w = dtdx/(double)TWOTONDIM;
#if NDIM==1
                P sg[6]; bool ref[6]; build(true,sg,ref); C du[2],bnd[2]; god1d_amr(sg,ref,gam,dtdx,slope,riem,du,bnd);
#else
                P sg[216]; bool ref[216]; build(true,sg,ref); C du[8],bnd[6]; god3d_amr(sg,ref,gam,dtdx,slope,riem,du,bnd);
#endif
                double ex[5]={-bnd[0].d*w,-bnd[0].mx*w,-bnd[0].my*w,-bnd[0].mz*w,-bnd[0].e*w};  // -x face, sign -1
                for(int vv=0;vv<5;vv++){ char b[48]; snprintf(b,sizeof b,"reflux.%s",vn[vv]);
                    chk(b, Nn[UHc(pcell,vv+1,FATHER)], ex[vv], 5e-4); }
            }
        }

        // ---- 2) set_unew / set_uold copy kernels -------------------------
        const int noct=3;
        size_t nbytes = (size_t)noct*TWOTONDIM*NHVAR*sizeof(float);
        id<MTLBuffer> bu=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        float* U=(float*)bu.contents; float* N=(float*)bn.contents;
        for(int i=0;i<noct*TWOTONDIM*NHVAR;i++){ U[i]=1.0f+0.001f*i; N[i]=-7.0f; }
        HP hp{};
        hp.head=1; hp.num=noct; hp.gamma=1.4f;
        id<MTLBuffer> bp=[dev newBufferWithBytes:&hp length:sizeof(hp) options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> su=pso("set_unew");
        { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:su]; [e setBuffer:bu offset:0 atIndex:0]; [e setBuffer:bn offset:0 atIndex:1];
          [e setBuffer:bp offset:0 atIndex:2];
          [e dispatchThreads:MTLSizeMake(noct,1,1) threadsPerThreadgroup:MTLSizeMake(noct,1,1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
        int copyfail=0;
        for(int i=0;i<noct*TWOTONDIM*NHVAR;i++) if(N[i]!=U[i]) copyfail++;
        if(copyfail){ printf("    [FAIL] set_unew copied %d/%d wrong\n",copyfail,noct*TWOTONDIM*NHVAR); g_fail++; }
        // now perturb unew and set_uold back
        for(int i=0;i<noct*TWOTONDIM*NHVAR;i++) N[i]=2.0f*i+5.0f;
        id<MTLComputePipelineState> so=pso("set_uold");
        { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:so]; [e setBuffer:bu offset:0 atIndex:0]; [e setBuffer:bn offset:0 atIndex:1];
          [e setBuffer:bp offset:0 atIndex:2];
          [e dispatchThreads:MTLSizeMake(noct,1,1) threadsPerThreadgroup:MTLSizeMake(noct,1,1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
        for(int i=0;i<noct*TWOTONDIM*NHVAR;i++) if(U[i]!=N[i]) copyfail++;
        if(copyfail){ printf("    [FAIL] set_uold mismatch\n"); g_fail++; }

        // ---- 3) source coupling (sync_hydro / grav_hydro) ----------------
        {
            const int no=4; size_t n=(size_t)no*TWOTONDIM*NHVAR, nf=(size_t)no*3*TWOTONDIM;
            id<MTLBuffer> bUo=[dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bUn=[dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bf =[dev newBufferWithLength:nf*sizeof(float) options:MTLResourceStorageModeShared];
            float* Uo=(float*)bUo.contents; float* Un=(float*)bUn.contents; float* Fg=(float*)bf.contents;
            double gam=1.4, dt=0.07;
            auto UHc=[&](int c,int v,int o){ return ((o-1)*NHVAR+(v-1))*TWOTONDIM+(c-1); };
            auto IDX3c=[&](int c,int d,int o){ return ((o-1)*3+(d-1))*TWOTONDIM+(c-1); };
            auto gp=[&](int o,int c)->P{ return {1.0+0.1*o+0.02*c, 0.2*o-0.1*c, 0.05*o, -0.04*c, 0.9+0.05*o+0.03*c}; };
            auto gf=[&](int o,int c,int d)->double{ return d==1?0.3*o:d==2?-0.2*c:0.1*(o-c); };
            for(int o=1;o<=no;++o)for(int c=1;c<=TWOTONDIM;++c){ P qq=gp(o,c); C cc=p2c(qq,gam);
                Uo[UHc(c,1,o)]=cc.d;Uo[UHc(c,2,o)]=cc.mx;Uo[UHc(c,3,o)]=cc.my;Uo[UHc(c,4,o)]=cc.mz;Uo[UHc(c,5,o)]=cc.e;
                // unew starts from a DIFFERENT (post-godunov-like) state so rho_new!=rho_old
                P qn={qq.r*1.1, qq.u*0.9, qq.v, qq.w, qq.p*1.05}; C cn=p2c(qn,gam);
                Un[UHc(c,1,o)]=cn.d;Un[UHc(c,2,o)]=cn.mx;Un[UHc(c,3,o)]=cn.my;Un[UHc(c,4,o)]=cn.mz;Un[UHc(c,5,o)]=cn.e;
                Fg[IDX3c(c,1,o)]=gf(o,c,1);Fg[IDX3c(c,2,o)]=gf(o,c,2);Fg[IDX3c(c,3,o)]=gf(o,c,3); }
            HP h3{}; h3.gamma=gam; h3.dt=dt; h3.head=1; h3.num=no;
            id<MTLBuffer> bp3=[dev newBufferWithBytes:&h3 length:sizeof(HP) options:MTLResourceStorageModeShared];
            auto run=[&](const char* k, id<MTLBuffer> b0, id<MTLBuffer> b1, id<MTLBuffer> b2, bool three){
                id<MTLComputePipelineState> ps=pso(k);
                id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
                [e setComputePipelineState:ps]; [e setBuffer:b0 offset:0 atIndex:0]; [e setBuffer:b1 offset:0 atIndex:1];
                [e setBuffer:b2 offset:0 atIndex:2]; if(three)[e setBuffer:bp3 offset:0 atIndex:3];
                [e dispatchThreads:MTLSizeMake(no*TWOTONDIM,1,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,1,1)];
                [e endEncoding];[cb commit];[cb waitUntilCompleted]; };
            // sync_hydro(uold,f,P): uold velocities += f*dt
            run("sync_hydro", bUo, bf, bp3, true);
            for(int o=1;o<=no;++o)for(int c=1;c<=TWOTONDIM;++c){ P qq=gp(o,c);
                P pe={qq.r, qq.u+gf(o,c,1)*dt, qq.v+gf(o,c,2)*dt, qq.w+gf(o,c,3)*dt, qq.p}; C ce=p2c(pe,gam);
                double r[5]={ce.d,ce.mx,ce.my,ce.mz,ce.e}; const char* vn[5]={"d","mx","my","mz","e"};
                for(int vv=0;vv<5;vv++){ char b[40]; snprintf(b,sizeof b,"sync.o%dc%d.%s",o,c,vn[vv]); chk(b,Uo[UHc(c,vv+1,o)],r[vv],RF); } }
        }

        // grav_hydro (uold,unew,f,P): unew velocities += f*dt*rho_old/rho_new
        {
            const int no=3; size_t n=(size_t)no*TWOTONDIM*NHVAR, nf=(size_t)no*3*TWOTONDIM;
            id<MTLBuffer> bUo=[dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bUn=[dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bf =[dev newBufferWithLength:nf*sizeof(float) options:MTLResourceStorageModeShared];
            float* Uo=(float*)bUo.contents; float* Un=(float*)bUn.contents; float* Fg=(float*)bf.contents;
            double gam=1.4, dt=0.07;
            auto UHc=[&](int c,int v,int o){ return ((o-1)*NHVAR+(v-1))*TWOTONDIM+(c-1); };
            auto IDX3c=[&](int c,int d,int o){ return ((o-1)*3+(d-1))*TWOTONDIM+(c-1); };
            auto gp=[&](int o,int c)->P{ return {1.0+0.1*o+0.02*c, 0.2*o-0.1*c, 0.05*o, -0.04*c, 0.9+0.05*o+0.03*c}; };
            auto gf=[&](int o,int c,int d)->double{ return d==1?0.3*o:d==2?-0.2*c:0.1*(o-c); };
            for(int o=1;o<=no;++o)for(int c=1;c<=TWOTONDIM;++c){ P qq=gp(o,c); C co=p2c(qq,gam);
                Uo[UHc(c,1,o)]=co.d;
                P qn={qq.r*1.1, qq.u*0.9, qq.v, qq.w, qq.p*1.05}; C cn=p2c(qn,gam);
                Un[UHc(c,1,o)]=cn.d;Un[UHc(c,2,o)]=cn.mx;Un[UHc(c,3,o)]=cn.my;Un[UHc(c,4,o)]=cn.mz;Un[UHc(c,5,o)]=cn.e;
                Fg[IDX3c(c,1,o)]=gf(o,c,1);Fg[IDX3c(c,2,o)]=gf(o,c,2);Fg[IDX3c(c,3,o)]=gf(o,c,3); }
            HP h3{}; h3.gamma=gam; h3.dt=dt; h3.head=1; h3.num=no;
            id<MTLBuffer> bp3=[dev newBufferWithBytes:&h3 length:sizeof(HP) options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> ps=pso("grav_hydro");
            { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
              [e setComputePipelineState:ps]; [e setBuffer:bUo offset:0 atIndex:0]; [e setBuffer:bUn offset:0 atIndex:1];
              [e setBuffer:bf offset:0 atIndex:2]; [e setBuffer:bp3 offset:0 atIndex:3];
              [e dispatchThreads:MTLSizeMake(no*TWOTONDIM,1,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,1,1)];
              [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
            const char* vn[5]={"d","mx","my","mz","e"};
            for(int o=1;o<=no;++o)for(int c=1;c<=TWOTONDIM;++c){ P qq=gp(o,c);
                P qn={qq.r*1.1, qq.u*0.9, qq.v, qq.w, qq.p*1.05}; C cn=p2c(qn,gam); P pn=qn;
                double ro=p2c(qq,gam).d, rn=cn.d, fac=dt*ro/rn;
                P pe={pn.r, pn.u+gf(o,c,1)*fac, pn.v+gf(o,c,2)*fac, pn.w+gf(o,c,3)*fac, pn.p}; C ce=p2c(pe,gam);
                double r[5]={ce.d,ce.mx,ce.my,ce.mz,ce.e};
                for(int vv=0;vv<5;vv++){ char b[40]; snprintf(b,sizeof b,"grav.o%dc%d.%s",o,c,vn[vv]); chk(b,Un[UHc(c,vv+1,o)],r[vv],RF); } }
        }

        // ---- 4) cmpdt reduction (dt min + mass/ekin/eint sums) -----------
        {
            const int no=4; size_t n=(size_t)no*TWOTONDIM*NHVAR, nf=(size_t)no*3*TWOTONDIM;
            id<MTLBuffer> bUo=[dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bf =[dev newBufferWithLength:nf*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bG =[dev newBufferWithLength:(size_t)no*sizeof(Oct) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bRed=[dev newBufferWithLength:HYDRO_RED_N*sizeof(uint) options:MTLResourceStorageModeShared];
            float* Uo=(float*)bUo.contents; float* Fg=(float*)bf.contents; Oct* G=(Oct*)bG.contents; uint* R=(uint*)bRed.contents;
            double gam=1.4, dx=0.25, cf=0.8; float fp=(float)(1<<20);
            auto UHc=[&](int c,int v,int o){ return ((o-1)*NHVAR+(v-1))*TWOTONDIM+(c-1); };
            auto IDX3c=[&](int c,int d,int o){ return ((o-1)*3+(d-1))*TWOTONDIM+(c-1); };
            auto gp=[&](int o,int c)->P{ return {0.5+0.2*o+0.05*c, 0.3*o, -0.1*c, 0.05, 0.6+0.1*o+0.04*c}; };
            auto gf=[&](int o,int c,int d)->double{ return d==1?0.2*o:d==2?0.05*c:0.0; };
            memset(G,0,(size_t)no*sizeof(Oct));
            G[1].refined[0]=1;   // oct 2 cell 1 refined -> skipped
            for(int o=1;o<=no;++o)for(int c=1;c<=TWOTONDIM;++c){ P qq=gp(o,c); C cc=p2c(qq,gam);
                Uo[UHc(c,1,o)]=cc.d;Uo[UHc(c,2,o)]=cc.mx;Uo[UHc(c,3,o)]=cc.my;Uo[UHc(c,4,o)]=cc.mz;Uo[UHc(c,5,o)]=cc.e;
                Fg[IDX3c(c,1,o)]=gf(o,c,1);Fg[IDX3c(c,2,o)]=gf(o,c,2);Fg[IDX3c(c,3,o)]=gf(o,c,3); }
            for(int i=0;i<HYDRO_RED_N;i++) R[i]=0; { float fm=__FLT_MAX__; memcpy(&R[0],&fm,4); }
            HP h4{}; h4.gamma=gam; h4.dx=dx; h4.courant=cf; h4.fp_scale=fp; h4.head=1; h4.num=no;
            id<MTLBuffer> bp4=[dev newBufferWithBytes:&h4 length:sizeof(HP) options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> ps=pso("hydro_cmpdt");
            { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
              [e setComputePipelineState:ps]; [e setBuffer:bG offset:0 atIndex:0]; [e setBuffer:bUo offset:0 atIndex:1];
              [e setBuffer:bf offset:0 atIndex:2]; [e setBuffer:bRed offset:0 atIndex:3]; [e setBuffer:bp4 offset:0 atIndex:4];
              [e dispatchThreads:MTLSizeMake(no*TWOTONDIM,1,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,1,1)];
              [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
            // host reference over leaf cells
            double dtmin=1e300, mass=0, ekin=0, eint=0, vol=dx*dx*dx;
            for(int o=1;o<=no;++o)for(int c=1;c<=TWOTONDIM;++c){ if(o==2&&c==1) continue;  // refined
                P qq=gp(o,c); C cc=p2c(qq,gam);
                mass+=qq.r*vol; ekin+=cc.e*vol; eint+=qq.p/(gam-1)*vol;
                double cs=sqrt(gam*qq.p/qq.r), ctot=fabs(qq.u)+fabs(qq.v)+fabs(qq.w)+3*cs;
                double gr=(fabs(gf(o,c,1))+fabs(gf(o,c,2))+fabs(gf(o,c,3)))*dx/(ctot*ctot); gr=fmax(gr,1e-4);
                double dl=dx/ctot*(sqrt(1+2*cf*gr)-1)/gr; dtmin=fmin(dtmin,dl); }
            float gdt; memcpy(&gdt,&R[0],4);
            auto f2f=[&](int lo,int hi)->double{ long qv=((long)((unsigned long)R[hi]<<32))|(unsigned long)R[lo]; return (double)qv/(double)fp; };
            chk("cmpdt.dt", gdt, dtmin, 3e-3);   // fp32 cancellation in (sqrt(1+2cf·g)-1)/g
            chk("cmpdt.mass", f2f(1,2), mass, 2e-4);
            chk("cmpdt.ekin", f2f(3,4), ekin, 2e-4);
            chk("cmpdt.eint", f2f(5,6), eint, 2e-4);
        }

        // ---- 5) hydro_flag (density/pressure-gradient refinement) --------
        {
            const int noct=SUBGRIDSIZE, NT=noct;
            int cc0=1;
#if NDIM>=2
            cc0+=3;
#endif
#if NDIM>=3
            cc0+=9;
#endif
            const int center=cc0+1;
            size_t n=(size_t)NT*TWOTONDIM*NHVAR, nN=(size_t)NT*SUBGRIDSIZE;
            id<MTLBuffer> bUo=[dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bG =[dev newBufferWithLength:(size_t)NT*sizeof(Oct) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bNB=[dev newBufferWithLength:nN*sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bFL=[dev newBufferWithLength:(size_t)NT*TWOTONDIM*sizeof(int) options:MTLResourceStorageModeShared];
            float* Uo=(float*)bUo.contents; Oct* G=(Oct*)bG.contents; int* NB=(int*)bNB.contents; int* FL=(int*)bFL.contents;
            double gam=1.4; memset(G,0,(size_t)NT*sizeof(Oct)); for(size_t i=0;i<nN;i++)NB[i]=0;
            for(size_t i=0;i<(size_t)NT*TWOTONDIM;i++) FL[i]=0;
            auto UHc=[&](int c,int v,int o){ return ((o-1)*NHVAR+(v-1))*TWOTONDIM+(c-1); };
            // uniform low density everywhere, EXCEPT a sharp spike in the central oct -> big gradient
            auto setp=[&](int o,int c,P qq){ C cc=p2c(qq,gam);
                Uo[UHc(c,1,o)]=cc.d;Uo[UHc(c,2,o)]=cc.mx;Uo[UHc(c,3,o)]=cc.my;Uo[UHc(c,4,o)]=cc.mz;Uo[UHc(c,5,o)]=cc.e; };
            for(int o=1;o<=NT;++o)for(int c=1;c<=TWOTONDIM;++c) setp(o,c, P{1.0,0,0,0,1.0});
            for(int c=1;c<=TWOTONDIM;++c) setp(center,c, P{5.0,0,0,0,5.0});   // spike
            for(int k=0;k<SUBGRIDSIZE;++k) NB[(center-1)*SUBGRIDSIZE+k]=k+1;
            HydroFlagParams fpp{}; fpp.gamma=gam; fpp.err_grad_d=0.1f; fpp.err_grad_p=-1.0f; fpp.floor_d=1e-10f; fpp.floor_p=1e-10f;
            fpp.head_idx=center; fpp.num_octs=1;
            id<MTLBuffer> bpf=[dev newBufferWithBytes:&fpp length:sizeof(fpp) options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> ps=pso("hydro_flag");
            { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
              [e setComputePipelineState:ps]; [e setBuffer:bFL offset:0 atIndex:0]; [e setBuffer:bG offset:0 atIndex:1];
              [e setBuffer:bNB offset:0 atIndex:2]; [e setBuffer:bUo offset:0 atIndex:3]; [e setBuffer:bpf offset:0 atIndex:4];
              [e dispatchThreads:MTLSizeMake(TWOTONDIM,1,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,1,1)];
              [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
            // every central cell borders the jump (its neighbour across the oct face is density 1 vs 5):
            // 2*max(|5-1|/(5+1))=1.33 > 0.1 -> all central cells flagged.
            int nf=0; for(int c=1;c<=TWOTONDIM;++c) if(FL[(center-1)*TWOTONDIM+(c-1)]==1) nf++;
            if(nf!=TWOTONDIM){ printf("    [FAIL] hydro_flag flagged %d/%d central cells\n",nf,TWOTONDIM); g_fail++; }
            // a far oct (not the central, no gradient) must stay unflagged
            if(FL[0]!=0){ printf("    [FAIL] hydro_flag spuriously flagged a uniform oct\n"); g_fail++; }
        }

        printf("%s\n", g_fail==0 ? "PASS" : "FAIL");
        return g_fail==0 ? 0 : 1;
    }
}

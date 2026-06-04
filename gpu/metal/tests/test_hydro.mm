// test_hydro.mm — unit test for the Metal hydro numerical core (hydro.h) and the
// trivial state kernels (set_unew/set_uold in hydro.metal).  Checks the GPU fp32
// results against a host double-precision replica of the SAME formulas plus two
// analytic invariants (EOS round-trip identity; identical-state HLLC == physical
// flux).  No RAMSES run.  See tests/run_tests.sh for the build line.
#import <Metal/Metal.h>
#include <cstdio>
#include <cmath>

#ifndef NDIM
#define NDIM 3
#endif
#define TWOTONDIM (1 << NDIM)
#define NHVAR 5

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

        // ---- 2) set_unew / set_uold copy kernels -------------------------
        const int noct=3;
        size_t nbytes = (size_t)noct*TWOTONDIM*NHVAR*sizeof(float);
        id<MTLBuffer> bu=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        float* U=(float*)bu.contents; float* N=(float*)bn.contents;
        for(int i=0;i<noct*TWOTONDIM*NHVAR;i++){ U[i]=1.0f+0.001f*i; N[i]=-7.0f; }
        struct HP { float gamma,dt,dx,smallr,smallc,courant; int slope,riemann,head,num,ngridmax,ilevel; } hp{};
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

        printf("%s\n", g_fail==0 ? "PASS" : "FAIL");
        return g_fail==0 ? 0 : 1;
    }
}

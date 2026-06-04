//============================================================================
// metal_bridge_hydrotest.mm — bridge-level integration check for the Metal hydro
// orchestration.  Drives the per-level wrappers (mtl_godunov_fine = set_unew ->
// hydro_godunov -> grav_hydro -> set_uold, plus mtl_hydro_cmpdt and
// mtl_hydro_flag) on a uniform PERIODIC 1D chain of octs, with f=0 (pure hydro).
// A correct orchestration leaves uold = uold0 + du where du is the host-double
// MUSCL-Hancock/HLLC update gathered from the periodic neighbours.  Also checks
// cmpdt (dt>0, mass>0 == sum) and that hydro_flag fires on a density spike.
// Proves the bridge buffers + kernel sequence COMPOSE correctly.  No Ramses run.
//============================================================================
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void mtl_godunov_fine(int,int,int,int,int,double,double,double,int,int,double,double);
  double mtl_hydro_cmpdt(int,int,double,double,double,double,double*,double*,double*);
  void mtl_hydro_flag(int,int,double,double,double,double,double);
  void* mtl_ptr_grid(); void* mtl_ptr_nbor(); void* mtl_ptr_uold();
  void* mtl_ptr_f();    void* mtl_ptr_flag1();
  void mtl_drain(); void mtl_finalize();
}

// ---- compact host double replica of the 1D MUSCL-Hancock / HLLC core ----------
struct P { double r,u,v,w,p; };  struct C { double d,mx,my,mz,e; };
static double mag2(double x,double y,double z){ return x*x+(y*y+z*z); }
static double cpress(C c,double g){ return (g-1)*(c.e-0.5*mag2(c.mx,c.my,c.mz)/c.d); }
static double cenergy(P q,double g){ return q.p/(g-1)+0.5*q.r*mag2(q.u,q.v,q.w); }
static double csound(P q,double g){ return sqrt(g*q.p/q.r); }
static C p2c(P q,double g){ return {q.r,q.u*q.r,q.v*q.r,q.w*q.r,cenergy(q,g)}; }
static P c2p(C c,double g){ return {c.d,c.mx/c.d,c.my/c.d,c.mz/c.d,cpress(c,g)}; }
static double moncen(double l,double m,double r,int s){ double a=m-l,b=r-m,cc=0.5*(a+b),f=s;
  if(a*b<=0)return 0; return a>0?fmin(f*fmin(a,b),cc):fmax(f*fmax(a,b),cc); }
static C hllc(P L,P R,double g){
  double smallr=1e-10, smp=1e-20/g;
  L.r=fmax(L.r,smallr); R.r=fmax(R.r,smallr); L.p=fmax(L.p,smp*L.r); R.p=fmax(R.p,smp*R.r);
  double entho=1/(g-1), el=L.p*entho, er=R.p*entho;
  double ekl=0.5*L.r*mag2(L.u,L.v,L.w), ekr=0.5*R.r*mag2(R.u,R.v,R.w), etl=el+ekl, etr=er+ekr;
  double cl=csound(L,g), cr=csound(R,g), sl=fmin(L.u,R.u)-fmax(cl,cr), sr=fmax(L.u,R.u)+fmax(cl,cr);
  double rcl=L.r*(L.u-sl), rcr=R.r*(sr-R.u);
  double us=(rcr*R.u+rcl*L.u+(L.p-R.p))/(rcr+rcl), ps=(rcr*L.p+rcl*R.p+rcl*rcr*(L.u-R.u))/(rcr+rcl);
  double rsl=L.r*(sl-L.u)/(sl-us), rsr=R.r*(sr-R.u)/(sr-us);
  double esl=((sl-L.u)*etl-L.p*L.u+ps*us)/(sl-us), esr=((sr-R.u)*etr-R.p*R.u+ps*us)/(sr-us);
  double ro,uo,vo,wo,po,eo;
  if(sl>0){ro=L.r;uo=L.u;vo=L.v;wo=L.w;po=L.p;eo=etl;}
  else if(us>0){ro=rsl;uo=us;vo=L.v;wo=L.w;po=ps;eo=esl;}
  else if(sr>0){ro=rsr;uo=us;vo=R.v;wo=R.w;po=ps;eo=esr;}
  else {ro=R.r;uo=R.u;vo=R.v;wo=R.w;po=R.p;eo=etr;}
  return {ro*uo, ro*uo*uo+po, ro*uo*vo, ro*uo*wo, (eo+po)*uo};
}
static void trace1d(P l,P m,P r,double g,double dtdx,int slope,P& qL,P& qR){
  double smallr=1e-10, smallp=1e-10*(1e-10*1e-10);
  P s={0.5*moncen(l.r,m.r,r.r,slope),0.5*moncen(l.u,m.u,r.u,slope),0.5*moncen(l.v,m.v,r.v,slope),
       0.5*moncen(l.w,m.w,r.w,slope),0.5*moncen(l.p,m.p,r.p,slope)};
  P src={ -m.u*s.r - s.u*m.r, -m.u*s.u - s.p/m.r, -m.u*s.v, -m.u*s.w, -m.u*s.p - s.u*g*m.p };
  P pp={m.r+dtdx*src.r,m.u+dtdx*src.u,m.v+dtdx*src.v,m.w+dtdx*src.w,m.p+dtdx*src.p};
  qL={pp.r+s.r,pp.u+s.u,pp.v+s.v,pp.w+s.w,pp.p+s.p}; if(qL.r<smallr)qL.r=m.r; if(qL.p<smallp)qL.p=m.p;
  qR={pp.r-s.r,pp.u-s.u,pp.v-s.v,pp.w-s.w,pp.p-s.p}; if(qR.r<smallr)qR.r=m.r; if(qR.p<smallp)qR.p=m.p;
}
static void god1d(P sg[6],double g,double dtdx,int slope,C du[2]){
  P qL[4],qR[4]; for(int c=1;c<=4;++c) trace1d(sg[c-1],sg[c],sg[c+1],g,dtdx,slope,qL[c-1],qR[c-1]);
  C fa=hllc(qL[0],qR[1],g), fb=hllc(qL[1],qR[2],g), fc=hllc(qL[2],qR[3],g);
  du[0]={(fa.d-fb.d)*dtdx,(fa.mx-fb.mx)*dtdx,(fa.my-fb.my)*dtdx,(fa.mz-fb.mz)*dtdx,(fa.e-fb.e)*dtdx};
  du[1]={(fb.d-fc.d)*dtdx,(fb.mx-fc.mx)*dtdx,(fb.my-fc.my)*dtdx,(fb.mz-fc.mz)*dtdx,(fb.e-fc.e)*dtdx};
}

static int g_fail=0;
static void chk(const char* n,double got,double ref,double rt){ double e=fabs(got-ref),d=fmax(1.0,fabs(ref));
  if(e/d>=rt){ printf("    [FAIL] %-16s got=%.7g ref=%.7g rel=%.2e\n",n,got,ref,e/d); g_fail++; } }

int main(int argc,char** argv){
  const char* lib=argc>1?argv[1]:"/tmp/ramses_kernels.metallib";
  if(mtl_init(lib)) return 2;
  const int nl=8, ncell=nl;                        // 8 octs in a periodic 1D chain
  mtl_alloc_buffers(ncell, 1, 2*ncell+3, 8);

  Oct* grid=(Oct*)mtl_ptr_grid(); int* nbor=(int*)mtl_ptr_nbor();
  float* uold=(float*)mtl_ptr_uold(); float* f=(float*)mtl_ptr_f();
  auto UHc=[&](int c,int v,int o){ return ((o-1)*NHVAR+(v-1))*TWOTONDIM+(c-1); };

  // periodic 1D connectivity (SUBGRIDSIZE==3): slot 0,1,2 = offset -1,0,+1
  for(int o=0;o<nl;++o){ grid[o]=Oct{}; grid[o].lev=5; grid[o].ckey[0]=o;
    for(int s=-1;s<=1;++s) nbor[o*SUBGRIDSIZE+(s+1)]=((o+s+nl)%nl)+1; }
  memset(f,0,(size_t)ncell*TWOTONDIM*NF*sizeof(float));

  const double gam=1.4, dt=0.06, dx=0.3, dtdx=dt/dx; const int slope=2, riem=SOLVER_HLLC;
  auto gp=[&](int o,int c)->P{ double x=2*o+(c-1); return {1.0+0.2*sin(0.3*x), 0.15*cos(0.2*x), 0,0, 0.8+0.1*sin(0.25*x+1)}; };
  for(int o=1;o<=nl;++o)for(int c=1;c<=TWOTONDIM;++c){ P q=gp(o,c); C cc=p2c(q,gam);
    uold[UHc(c,1,o)]=cc.d;uold[UHc(c,2,o)]=cc.mx;uold[UHc(c,3,o)]=cc.my;uold[UHc(c,4,o)]=cc.mz;uold[UHc(c,5,o)]=cc.e; }

  // host reference: uold_new = uold0 + du  (periodic gather, f=0 -> no predictor/grav)
  std::vector<double> ref((size_t)nl*TWOTONDIM*NHVAR);
  for(int o=1;o<=nl;++o){ int oL=((o-1-1+nl)%nl)+1, oR=((o-1+1)%nl)+1;
    P sg[6]={gp(oL,1),gp(oL,2),gp(o,1),gp(o,2),gp(oR,1),gp(oR,2)};
    C du[2]; god1d(sg,gam,dtdx,slope,du);
    for(int c=1;c<=2;++c){ C c0=p2c(gp(o,c),gam); C du_c=du[c-1];
      ref[UHc(c,1,o)]=c0.d+du_c.d; ref[UHc(c,2,o)]=c0.mx+du_c.mx; ref[UHc(c,3,o)]=c0.my+du_c.my;
      ref[UHc(c,4,o)]=c0.mz+du_c.mz; ref[UHc(c,5,o)]=c0.e+du_c.e; } }

  // single uniform level -> ilevel==levelmin==levelmax: no reflux, no fine-flux zeroing
  mtl_godunov_fine(5, 1, nl, 5, 5, gam, dt, dx, slope, riem, 0.8, (double)(1<<20));
  mtl_drain();
  const char* vn[5]={"d","mx","my","mz","e"};
  for(int o=1;o<=nl;++o)for(int c=1;c<=TWOTONDIM;++c)for(int v=1;v<=5;++v){
    char b[40]; snprintf(b,sizeof b,"god.o%dc%d.%s",o,c,vn[v-1]);
    chk(b, uold[UHc(c,v,o)], ref[UHc(c,v,o)], 2e-4); }

  // cmpdt: dt>0 and mass == sum over cells of rho*dx^3
  double mass=0,ekin=0,eint=0;
  double cdt=mtl_hydro_cmpdt(1, nl, gam, dx, 0.8, (double)(1<<20), &mass, &ekin, &eint);
  double mref=0, vol=dx*dx*dx;
  for(int o=1;o<=nl;++o)for(int c=1;c<=TWOTONDIM;++c) mref += c2p(C{uold[UHc(c,1,o)],uold[UHc(c,2,o)],uold[UHc(c,3,o)],uold[UHc(c,4,o)],uold[UHc(c,5,o)]},gam).r*vol;
  if(!(cdt>0 && cdt<1e29)){ printf("    [FAIL] cmpdt dt=%.4g not positive/finite\n",cdt); g_fail++; }
  chk("cmpdt.mass", mass, mref, 2e-4);

  // hydro_flag: spike one oct's density -> its cells flagged; a flat oct not.
  { P q=gp(4,1); q.r*=6.0; C cc=p2c(q,gam);
    uold[UHc(1,1,4)]=cc.d;uold[UHc(1,2,4)]=cc.mx;uold[UHc(1,3,4)]=cc.my;uold[UHc(1,4,4)]=cc.mz;uold[UHc(1,5,4)]=cc.e; }
  int* flag1=(int*)mtl_ptr_flag1(); memset(flag1,0,(size_t)ncell*TWOTONDIM*sizeof(int));
  mtl_hydro_flag(1, nl, gam, 0.1, -1.0, 1e-10, 1e-10); mtl_drain();
  if(flag1[(4-1)*TWOTONDIM+0]!=1){ printf("    [FAIL] hydro_flag missed the density spike\n"); g_fail++; }

  printf("%s\n", g_fail==0?"PASS":"FAIL");
  mtl_finalize();
  return g_fail==0?0:1;
}

//============================================================================
// metal_bridge_mgtest.mm — exercise the bridge's multigrid V-cycle solver via
// its extern "C" API.  Sets S = L_g(phi_true) on a uniform periodic base grid,
// runs mtl_mg_solve, and checks the recovered potential.
//============================================================================
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void* mtl_ptr_grid(); void* mtl_ptr_nbor(); void* mtl_ptr_phi(); void* mtl_ptr_f();
  void mtl_mg_solve(int,int,int,int);
  void mtl_finalize();
}

static const int hhh[6][8]={{2,1,4,3,6,5,8,7},{2,1,4,3,6,5,8,7},{3,4,1,2,7,8,5,6},
                            {3,4,1,2,7,8,5,6},{5,6,7,8,1,2,3,4},{5,6,7,8,1,2,3,4}};
static const int iii[6][8]={{-1,0,-1,0,-1,0,-1,0},{0,1,0,1,0,1,0,1},{-1,-1,0,0,-1,-1,0,0},
                            {0,0,1,1,0,0,1,1},{-1,-1,-1,-1,0,0,0,0},{0,0,0,0,1,1,1,1}};

static int run(int nl, int ncycle) {
    int noct=nl*nl*nl, ncl=2*nl, ncells=noct*TWOTONDIM;
    mtl_alloc_buffers(noct, 1, 1031, 8);
    Oct*  grid=(Oct*)mtl_ptr_grid();
    int*  nbor=(int*)mtl_ptr_nbor();
    float* phi=(float*)mtl_ptr_phi();
    float* f  =(float*)mtl_ptr_f();
    auto oi=[&](int x,int y,int z){return x+y*nl+z*nl*nl;};
    for (int z=0;z<nl;++z) for (int y=0;y<nl;++y) for (int x=0;x<nl;++x){
        int o=oi(x,y,z); grid[o]=Oct{}; grid[o].ckey[0]=x;grid[o].ckey[1]=y;grid[o].ckey[2]=z;
        for (int kk=-1;kk<=1;++kk) for (int jj=-1;jj<=1;++jj) for (int ii=-1;ii<=1;++ii){
            int ind=1+(1+ii)+3*(1+jj)+9*(1+kk);
            nbor[o*27+(ind-1)]=oi((x+ii+nl)%nl,(y+jj+nl)%nl,(z+kk+nl)%nl)+1;
        }
    }
    std::vector<double> pt(ncells);
    for (int o=1;o<=noct;++o){int oo=o-1,fx=oo%nl,fy=(oo/nl)%nl,fz=oo/(nl*nl);
        for (int c=1;c<=TWOTONDIM;++c){int ix=(c-1)&1,iy=((c-1)>>1)&1,iz=((c-1)>>2)&1;
            double ux=(2*fx+ix+0.5)/ncl,uy=(2*fy+iy+0.5)/ncl,uz=(2*fz+iz+0.5)/ncl;
            pt[IDX2(c,o)]=cos(2*M_PI*ux)+cos(2*M_PI*uy)+cos(2*M_PI*uz);}}
    memset(phi,0,ncells*sizeof(float));
    for (int i=0;i<3*ncells;++i) f[i]=0.f;
    for (int o=1;o<=noct;++o) for (int c=1;c<=TWOTONDIM;++c){
        double nb=0;
        for (int idim=1;idim<=3;++idim) for(int inb=1;inb<=2;++inb){
            int dir=2*(idim-1)+inb,off=iii[dir-1][c-1],in=0,jn=0,kn=0;
            if(idim==1)in=off; else if(idim==2)jn=off; else kn=off;
            int ind=1+(1+in)+3*(1+jn)+9*(1+kn),onb=nbor[(o-1)*27+(ind-1)],cnb=hhh[dir-1][c-1];
            nb+=pt[IDX2(cnb,onb)];
        }
        f[IDX3(c,2,o)]=(float)(nb-(double)TWONDIM*pt[IDX2(c,o)]); f[IDX3(c,3,o)]=1.f;
    }
    mtl_mg_solve(nl, ncycle, 2, 2);
    double mg=0,mt=0; for(int i=0;i<ncells;++i){mg+=phi[i];mt+=pt[i];} mg/=ncells; mt/=ncells;
    double en=0,ed=0; for(int i=0;i<ncells;++i){double d=(phi[i]-mg)-(pt[i]-mt);en+=d*d;ed+=(pt[i]-mt)*(pt[i]-mt);}
    double err=sqrt(en/ed);
    bool ok=err<3e-3;
    printf("bridge MG nl=%d (%d octs) cycles=%d : %s  phi_err=%.2e\n", nl, noct, ncycle, ok?"PASS":"FAIL", err);
    return ok?0:1;
}

int main(int argc, char** argv) {
    const char* lib = argc>1?argv[1]:"/tmp/ramses_kernels.metallib";
    if (mtl_init(lib)) return 1;
    int f=0;
    f+=run(8, 6);
    f+=run(16, 8);
    f+=run(32, 10);
    printf("%s\n", f?"FAILURES":"ALL PASS");
    mtl_finalize();
    return f;
}

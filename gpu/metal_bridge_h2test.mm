//============================================================================
// metal_bridge_h2test.mm — validate H2 base-level grouped Poisson on the REAL
// mesh path: build a full periodic base grid (levelmin) + connectivity (H1),
// set rho so the grouped source S = fourpi*2^L*rho equals L_g(phi_true) for a
// known periodic sinusoid, run mtl_poisson_base, and compare recovered phi.
//============================================================================
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void mtl_build_connectivity(int,int,int,const int*,const int*,int,int,int);
  void mtl_poisson_base(int,int,float,float,int,int,int,int);
  void* mtl_ptr_grid(); void* mtl_ptr_rho(); void* mtl_ptr_phi();
  void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void mtl_finalize();
}

static int run(int levelmin, int ncycle) {
    const int nlevelmax = levelmin+2;
    int nl = 1<<(levelmin-1);              // octs per dim
    int noct = nl*nl*nl, ncl = 2*nl;       // cells per dim
    int hash_size = 2*noct+3;
    mtl_alloc_buffers(noct, 1, hash_size, nlevelmax);

    int*  ckey_max=(int*)mtl_ptr_ckey_max(); long* key_off=(long*)mtl_ptr_key_off();
    for (int L=0;L<=nlevelmax+1;++L){ckey_max[L]=0;key_off[L]=0;}
    for (int L=1;L<=nlevelmax;++L) ckey_max[L]=1<<(L-1);
    key_off[1]=1; for(int L=2;L<=nlevelmax;++L){long n=ckey_max[L-1];key_off[L]=key_off[L-1]+n*n*n;}
    std::vector<int> box_min(3*(nlevelmax+1),0), box_max(3*(nlevelmax+1),0);
    for (int L=1;L<=nlevelmax;++L) for(int d=0;d<3;++d) box_max[(L-1)*3+d]=ckey_max[L];

    // full periodic base grid in canonical order (father fix reads ckey anyway).
    Oct* grid=(Oct*)mtl_ptr_grid();
    for (int z=0;z<nl;++z) for(int y=0;y<nl;++y) for(int x=0;x<nl;++x){
        int o=x+y*nl+z*nl*nl; grid[o]=Oct{}; grid[o].lev=levelmin;
        grid[o].ckey[0]=x; grid[o].ckey[1]=y; grid[o].ckey[2]=z;
    }
    mtl_build_connectivity(noct, levelmin, nlevelmax, box_min.data(), box_max.data(), 1,1,1);

    // phi_true (periodic cos sum on cell centres) and S_true = 7-pt Laplacian.
    auto gidx=[&](int x,int y,int z){return ((x+ncl)%ncl)+((y+ncl)%ncl)*ncl+((z+ncl)%ncl)*ncl*ncl;};
    std::vector<double> pt((size_t)ncl*ncl*ncl);
    for (int z=0;z<ncl;++z) for(int y=0;y<ncl;++y) for(int x=0;x<ncl;++x)
        pt[gidx(x,y,z)] = cos(2*M_PI*x/ncl)+cos(2*M_PI*y/ncl)+cos(2*M_PI*z/ncl);

    float* rho=(float*)mtl_ptr_rho();
    float twoL=(float)(1<<levelmin);                 // coef = fourpi(=1)*2^L
    for (int o=0;o<noct;++o){int ox=grid[o].ckey[0],oy=grid[o].ckey[1],oz=grid[o].ckey[2];
        for (int c=1;c<=TWOTONDIM;++c){int i=(c-1)&1,j=((c-1)>>1)&1,k=((c-1)>>2)&1;
            int gx=2*ox+i,gy=2*oy+j,gz=2*oz+k;
            double S = pt[gidx(gx+1,gy,gz)]+pt[gidx(gx-1,gy,gz)]
                     + pt[gidx(gx,gy+1,gz)]+pt[gidx(gx,gy-1,gz)]
                     + pt[gidx(gx,gy,gz+1)]+pt[gidx(gx,gy,gz-1)] - 6.0*pt[gidx(gx,gy,gz)];
            rho[IDX2(c,o+1)] = (float)(S/twoL);     // S = twoL*rho  (offset=0)
        }}

    memset(mtl_ptr_phi(),0,(size_t)noct*TWOTONDIM*sizeof(float));
    mtl_poisson_base(nl, levelmin, 1.0f, 0.0f, ncycle, 2, 2, 1);

    float* phi=(float*)mtl_ptr_phi();
    double mg=0,mt=0; int nc=noct*TWOTONDIM;
    for(int o=0;o<noct;++o)for(int c=1;c<=TWOTONDIM;++c){int ox=grid[o].ckey[0],oy=grid[o].ckey[1],oz=grid[o].ckey[2];
        int i=(c-1)&1,j=((c-1)>>1)&1,k=((c-1)>>2)&1; mg+=phi[IDX2(c,o+1)]; mt+=pt[gidx(2*ox+i,2*oy+j,2*oz+k)];}
    mg/=nc; mt/=nc;
    double en=0,ed=0;
    for(int o=0;o<noct;++o)for(int c=1;c<=TWOTONDIM;++c){int ox=grid[o].ckey[0],oy=grid[o].ckey[1],oz=grid[o].ckey[2];
        int i=(c-1)&1,j=((c-1)>>1)&1,k=((c-1)>>2)&1;
        double d=(phi[IDX2(c,o+1)]-mg)-(pt[gidx(2*ox+i,2*oy+j,2*oz+k)]-mt); en+=d*d; ed+=pow(pt[gidx(2*ox+i,2*oy+j,2*oz+k)]-mt,2);}
    double err=sqrt(en/ed); bool ok=err<3e-3;
    printf("H2 base poisson L%d (%d octs) cyc=%d : %s  phi_err=%.2e\n", levelmin, noct, ncycle, ok?"PASS":"FAIL", err);
    return ok?0:1;
}

int main(int argc, char** argv){
    const char* lib=argc>1?argv[1]:"/tmp/ramses_kernels.metallib";
    if (mtl_init(lib)) return 1;
    int f=0; f+=run(4,12); f+=run(5,14);
    printf("%s\n", f?"FAILURES":"ALL PASS");
    mtl_finalize(); return f;
}

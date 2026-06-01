//============================================================================
// metal_bridge_mgprodtest.mm — exercise the PRODUCTION multigrid (mtl_poisson_level)
// on a clean uniform periodic base grid with a known zero-mean cos source.
// Sets B.rho = L_g(phi_true); fourpi=1, offset=0, vol_loc=1, dx=1 -> reset_rhs
// makes S=rho and the solver should recover phi_true (up to the gauge mean).
// This isolates whether the production V-cycle / kernels converge on a clean
// problem (vs the stale mtl_mg_solve path in metal_bridge_mgtest.mm).
//============================================================================
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void* mtl_ptr_grid(); void* mtl_ptr_nbor(); void* mtl_ptr_father();
  void* mtl_ptr_phi();  void* mtl_ptr_rho();
  void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void mtl_poisson_level(int,int,int,int,float,float,float,int,int,int,int,
                         float,float,float,const int*,const int*,int,int,int,float);
  void mtl_finalize();
}

// same-level 3^3 neighbour-cube index helpers (match the device convention)
static int run(int L) {
    int nl = 1<<(L-1);                 // octs per dim at level L
    int noct = nl*nl*nl, ncells = noct*TWOTONDIM;
    mtl_alloc_buffers(noct, 1, 1031, L+1);
    Oct*  grid = (Oct*)mtl_ptr_grid();
    int*  nbor = (int*)mtl_ptr_nbor();
    int*  fath = (int*)mtl_ptr_father();
    float* phi = (float*)mtl_ptr_phi();
    float* rho = (float*)mtl_ptr_rho();
    int*  ckmax= (int*)mtl_ptr_ckey_max();
    long* koff = (long*)mtl_ptr_key_off();
    auto oi=[&](int x,int y,int z){return ((x%nl+nl)%nl)+((y%nl+nl)%nl)*nl+((z%nl+nl)%nl)*nl*nl;};

    // per-level key params: oct-count-per-dim = 2^(lev-1); key_off=0 (pmap is per-level)
    for (int lev=0; lev<=L+1; ++lev){ ckmax[lev] = (lev>=1)?(1<<(lev-1)):1; koff[lev]=0; }
    std::vector<int> bmn(3*(L+1),0), bmx(3*(L+1),0);
    for (int lev=1; lev<=L; ++lev) for(int d=0;d<3;++d){ bmn[(lev-1)*3+d]=0; bmx[(lev-1)*3+d]=1<<(lev-1); }

    for (int z=0;z<nl;++z) for (int y=0;y<nl;++y) for (int x=0;x<nl;++x){
        int o=oi(x,y,z); grid[o]=Oct{}; grid[o].lev=L;
        grid[o].ckey[0]=x; grid[o].ckey[1]=y; grid[o].ckey[2]=z;
        fath[o]=0;                                       // base: no AMR parent
        for (int kk=-1;kk<=1;++kk) for (int jj=-1;jj<=1;++jj) for (int ii=-1;ii<=1;++ii){
            int ind=1+(1+ii)+3*(1+jj)+9*(1+kk);
            nbor[o*27+(ind-1)] = oi(x+ii,y+jj,z+kk)+1;   // 1-based, periodic
        }
    }
    // phi_true = cos waves (zero mean), rho = L_g(phi_true) = sum_nbr phi - 6 phi
    std::vector<double> pt(ncells);
    for (int o=1;o<=noct;++o){int oo=o-1,fx=oo%nl,fy=(oo/nl)%nl,fz=oo/(nl*nl);
        for (int c=1;c<=TWOTONDIM;++c){int ix=(c-1)&1,iy=((c-1)>>1)&1,iz=((c-1)>>2)&1;
            double ux=(2*fx+ix+0.5)/(2*nl),uy=(2*fy+iy+0.5)/(2*nl),uz=(2*fz+iz+0.5)/(2*nl);
            pt[IDX2(c,o)]=cos(2*M_PI*ux)+cos(2*M_PI*uy)+cos(2*M_PI*uz);}}
    static const int hhh[6][8]={{2,1,4,3,6,5,8,7},{2,1,4,3,6,5,8,7},{3,4,1,2,7,8,5,6},
                                {3,4,1,2,7,8,5,6},{5,6,7,8,1,2,3,4},{5,6,7,8,1,2,3,4}};
    static const int iii[6][8]={{-1,0,-1,0,-1,0,-1,0},{0,1,0,1,0,1,0,1},{-1,-1,0,0,-1,-1,0,0},
                                {0,0,1,1,0,0,1,1},{-1,-1,-1,-1,0,0,0,0},{0,0,0,0,1,1,1,1}};
    memset(phi,0,ncells*sizeof(float));
    for (int o=1;o<=noct;++o) for (int c=1;c<=TWOTONDIM;++c){
        double nb=0;
        for (int idim=1;idim<=3;++idim) for(int inb=1;inb<=2;++inb){
            int dir=2*(idim-1)+inb,off=iii[dir-1][c-1],in=0,jn=0,kn=0;
            if(idim==1)in=off; else if(idim==2)jn=off; else kn=off;
            int ind=1+(1+in)+3*(1+jn)+9*(1+kn),onb=nbor[(o-1)*27+(ind-1)],cnb=hhh[dir-1][c-1];
            nb+=pt[IDX2(cnb,onb)];
        }
        rho[IDX2(c,o)] = (float)(nb - (double)TWONDIM*pt[IDX2(c,o)]);   // L_g(phi_true)
    }
    // Optionally inject a nonzero MEAN into the RHS (RAMSES_TEST_BIAS) to test
    // whether an inconsistent (non-zero-mean) periodic RHS breaks convergence.
    const char* bs = getenv("RAMSES_TEST_BIAS");
    if (bs) { float bias=atof(bs); for(int i=0;i<ncells;++i) rho[i]+=bias;
              printf("  [bias=%.3g added to RHS]\n", bias); }
    // PRODUCTION solve: fourpi=1, offset=0, vol_loc=1, dx=1, has_coarse=0 (periodic base)
    mtl_poisson_level(L, 1, noct, noct, 1.0f, 0.0f, 1.0f, 0,
                      2, 2, 50, 1e-6f, 1.0f, 1.0f, bmn.data(), bmx.data(), 1,1,1, 0.0f);
    double mg=0,mt=0; for(int i=0;i<ncells;++i){mg+=phi[i];mt+=pt[i];} mg/=ncells; mt/=ncells;
    double en=0,ed=0; for(int i=0;i<ncells;++i){double d=(phi[i]-mg)-(pt[i]-mt);en+=d*d;ed+=(pt[i]-mt)*(pt[i]-mt);}
    double err=sqrt(en/ed);
    bool ok=err<1e-2;
    printf("PROD MG L=%d (%d octs, nl=%d) : %s  phi_err=%.3e\n", L, noct, nl, ok?"PASS":"FAIL", err);
    return ok?0:1;
}

int main(int argc, char** argv) {
    const char* lib = argc>1?argv[1]:"/tmp/ramses_kernels.metallib";
    int L = argc>2?atoi(argv[2]):4;     // one level per process -> no base-cache reuse
    if (mtl_init(lib)) return 1;
    int f=run(L);
    mtl_finalize();
    return f;
}

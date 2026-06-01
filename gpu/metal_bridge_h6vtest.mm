//============================================================================
// metal_bridge_h6vtest.mm — #31 INTEGRATED MG-solve check.  Drives the faithful
// per-leaf bridge wrappers (mtl_mg_build / restrict_mask / gauss_seidel /
// cmp_residual / restrict_residual / reset_corr / interpolate_correct /
// residual_norm2) through the SAME V-cycle that m_metal_multigrid +
// metal_recursive_mg run (a faithful mirror of multigrid_fine_commons.f90), on an
// isolated 8^3 block of level-5 octs (edge neighbours missing => Dirichlet phi=0,
// non-singular operator).  mask=1, a varied source S in f(:,2), phi=0.  A correct
// V-cycle drives ||residual|| down by many orders -> proves the per-leaf operators
// COMPOSE into a converging multigrid solver.  No Ramses run.
//============================================================================
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void mtl_build_connectivity(int,int,int,const int*,const int*,int,int,int);
  void mtl_build_mg_amr(int,int,int,int,const int*,const int*,int,int,int);
  void mtl_mg_build(int,int,int,int,int,const int*,const int*,int,int,int,int,float,int,float);
  int  mtl_mg_restrict_mask(int,int);
  void mtl_mg_gauss_seidel(int,int,int,int);
  void mtl_mg_cmp_residual(int,int);
  void mtl_mg_restrict_residual(int,int);
  void mtl_mg_reset_corr(int,int);
  void mtl_mg_interpolate_correct(int,int);
  double mtl_mg_residual_norm2(int);
  void* mtl_ptr_grid(); void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void* mtl_ptr_phi(); void* mtl_ptr_f();
  void mtl_finalize();
}

// ||residual|| at the fine level: cmp_residual fills B.f(:,1); residual_norm2 sums r^2.
static double resnorm(int L) { mtl_mg_cmp_residual(L,L); return sqrt(mtl_mg_residual_norm2(L)); }

int main(int argc, char** argv){
    const char* lib=argc>1?argv[1]:"/tmp/ramses_kernels.metallib";
    if (mtl_init(lib)) return 1;
    const int nlevelmax=7, L=5, nb=8, bnd=1, ngs=2;
    int n_fine=nb*nb*nb, mg_cap=256;
    mtl_alloc_buffers(n_fine, 1, 2*n_fine+3, nlevelmax);

    int* ckey_max=(int*)mtl_ptr_ckey_max(); long* key_off=(long*)mtl_ptr_key_off();
    for(int l=0;l<=nlevelmax+1;++l){ckey_max[l]=0;key_off[l]=0;}
    for(int l=1;l<=nlevelmax;++l) ckey_max[l]=1<<(l-1);
    key_off[1]=1; for(int l=2;l<=nlevelmax;++l){long n=ckey_max[l-1];key_off[l]=key_off[l-1]+n*n*n;}
    std::vector<int> bmin(3*nlevelmax,0), bmax(3*nlevelmax,0);
    for(int l=1;l<=nlevelmax;++l) for(int d=0;d<3;++d) bmax[(l-1)*3+d]=ckey_max[l];

    Oct* grid=(Oct*)mtl_ptr_grid(); int o=0;
    for(int z=0;z<nb;++z)for(int y=0;y<nb;++y)for(int x=0;x<nb;++x){
        grid[o]=Oct{}; grid[o].lev=L; grid[o].ckey[0]=x;grid[o].ckey[1]=y;grid[o].ckey[2]=z; ++o; }

    // fine connectivity (B.nbor; edge neighbours missing -> 0 -> Dirichlet) + hierarchy
    mtl_build_connectivity(n_fine, L, nlevelmax, bmin.data(), bmax.data(), 1,1,1);

    // masks = interior everywhere; source S in f(:,2); phi = 0.
    float* f=(float*)mtl_ptr_f();
    memset(mtl_ptr_phi(),0,(size_t)n_fine*TWOTONDIM*sizeof(float));
    for(int oo=1;oo<=n_fine;++oo) for(int c=1;c<=TWOTONDIM;++c){
        f[IDX3(c,3,oo)]=1.0f;                                  // mask interior
        f[IDX3(c,2,oo)]=(float)(((oo*3+c)%7)-3);               // varied grouped source S
    }

    // Build the MG hierarchy (whole thing on the ifine==L call) + restrict mask.
    for (int ifine=L; ifine>=bnd+1; --ifine)
        mtl_mg_build(L, ifine, 1, n_fine, mg_cap, bmin.data(), bmax.data(), 1,1,1, /*is_base*/0, 1.0f, /*has_coarse*/0, 0.0f);
    int levelmin_mg = bnd;
    for (int ifine=L; ifine>=bnd+1; --ifine) if (mtl_mg_restrict_mask(L, ifine)==1) { levelmin_mg=ifine; break; }

    // Recursive V-cycle (mirrors metal_recursive_mg).
    std::function<void(int)> rec = [&](int ifine){
        if (ifine <= levelmin_mg) {                            // coarsest: solve "directly"
            for(int i=0;i<2*ngs;++i){ mtl_mg_gauss_seidel(L,ifine,0,1); mtl_mg_gauss_seidel(L,ifine,0,0); }
            return;
        }
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,ifine,0,1); mtl_mg_gauss_seidel(L,ifine,0,0); }   // pre-smooth
        mtl_mg_cmp_residual(L,ifine); mtl_mg_restrict_residual(L,ifine);
        mtl_mg_reset_corr(L,ifine-1); rec(ifine-1); mtl_mg_interpolate_correct(L,ifine);
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,ifine,0,1); mtl_mg_gauss_seidel(L,ifine,0,0); }   // post-smooth
    };

    double r0 = resnorm(L);
    for (int iter=0; iter<20; ++iter) {
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,L,0,1); mtl_mg_gauss_seidel(L,L,0,0); }            // pre-smooth fine
        mtl_mg_cmp_residual(L,L);
        if (L > levelmin_mg) {                                                                          // coarse-grid correction
            mtl_mg_restrict_residual(L,L); mtl_mg_reset_corr(L,L-1); rec(L-1); mtl_mg_interpolate_correct(L,L);
        }
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,L,0,1); mtl_mg_gauss_seidel(L,L,0,0); }            // post-smooth fine
        double rk = resnorm(L);
        if (rk < 1e-6*r0) break;
    }
    double r1 = resnorm(L);
    double drop = r1/r0;
    bool ok = drop < 1e-4;
    printf("#31 integrated V-cycle: %s  ||r0||=%.4e ||r||=%.4e  drop=%.2e (%.1f orders)  levelmin_mg=%d\n",
           ok?"PASS":"FAIL", r0, r1, drop, -log10(drop), levelmin_mg);
    mtl_finalize();
    return ok?0:1;
}

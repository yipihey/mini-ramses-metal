//============================================================================
// symmetry_test.mm — coarse-fine momentum-bug localizer (Phase A, field channel).
//
// Stage 1 (this file): a 1D uniform refined block (Dirichlet edges) with a
// left-right SYMMETRIC source. A correct linear solve MUST give a symmetric phi
// and an ANTISYMMETRIC force. We run the SAME per-leaf V-cycle that
// m_metal_multigrid/metal_recursive_mg run (faithful mirror, as in
// metal_bridge_h6vtest.mm), then gradient_phi, and assert symmetry of every
// intermediate field. The FIRST field that loses symmetry localizes the broken
// operator (#6 gauss_seidel / #7 cmp_residual / #8 restrict_residual /
// #9 interpolate_correct / #10 gradient_phi). No Ramses run, no reference data —
// the oracle is the mirror invariant itself.
//
// Build (run from gpu/metal): full NDIM=1 metallib + bridge .o, link this harness.
//   see run_tests.sh "symmetry (NDIM=1 field channel)" block.
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
  void mtl_mg_build(int,int,int,int,int,const int*,const int*,int,int,int,int,float,int,float);
  int  mtl_mg_restrict_mask(int,int);
  void mtl_mg_make_mask(int);
  void mtl_mg_gauss_seidel(int,int,int,int);
  void mtl_mg_cmp_residual(int,int);
  void mtl_mg_restrict_residual(int,int);
  void mtl_mg_reset_corr(int,int);
  void mtl_mg_interpolate_correct(int,int);
  double mtl_mg_residual_norm2(int);
  void mtl_gradient_phi(int,int,float,float);
  void* mtl_ptr_grid(); void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void* mtl_ptr_phi(); void* mtl_ptr_f();
  void mtl_finalize();
}

static int g_nb, g_base;          // block: nb octs, ckey = base..base+nb-1, level L
static bool g_ok = true;

// Global 1D cell index of (cell c in {1..2}, oct index o in 0..nb-1): g = 2*o + (c-1).
// Mirror about the block centre: g <-> (2*nb-1 - g).  Returns the (cell,oct) of the mirror.
static inline void mirror_of(int c, int o, int &cm, int &om) {
    int g  = 2*o + (c-1);
    int gm = 2*g_nb - 1 - g;
    om = gm / 2; cm = (gm % 2) + 1;
}

// Check a per-cell field F is left-right SYMMETRIC (sign=+1) or ANTISYMMETRIC (sign=-1).
static void check(const float* F, int sign, const char* name, double tol) {
    double maxa = 0.0, scale = 0.0;
    for (int o = 0; o < g_nb; ++o)
        for (int c = 1; c <= TWOTONDIM; ++c) {
            int cm, om; mirror_of(c, o, cm, om);
            float a = F[IDX2(c, o+1)];
            float b = F[IDX2(cm, om+1)];
            maxa  = fmax(maxa, fabs((double)a - sign*(double)b));
            scale = fmax(scale, fabs((double)a));
        }
    // absolute floor: a field whose magnitude is itself at the fp32 noise floor is
    // "symmetric by being ~zero" -> don't divide noise by noise.
    if (scale < 1e-5) { printf("  %-26s SKIP (|F|max=%.2e ~ 0)\n", name, scale); return; }
    double rel = maxa / scale;
    bool ok = rel < tol;
    if (!ok) g_ok = false;
    printf("  %-26s %s  |F|max=%.3e  max|F-%sFm|=%.3e  rel=%.3e (tol %.0e)\n",
           name, ok ? "OK " : "FAIL", scale, sign < 0 ? "-" : "+", maxa, rel, tol);
}

// the f-array column accessor as a per-cell IDX2-style view (stride NF*TWOTONDIM per oct)
static void check_fcol(const float* f, int col, int sign, const char* name, double tol) {
    double maxa = 0.0, scale = 1e-30;
    for (int o = 0; o < g_nb; ++o)
        for (int c = 1; c <= TWOTONDIM; ++c) {
            int cm, om; mirror_of(c, o, cm, om);
            float a = f[IDX3(c, col, o+1)];
            float b = f[IDX3(cm, col, om+1)];
            maxa  = fmax(maxa, fabs((double)a - sign*(double)b));
            scale = fmax(scale, fabs((double)a));
        }
    if (scale < 1e-5) { printf("  %-26s SKIP (|F|max=%.2e ~ 0)\n", name, scale); return; }
    double rel = maxa / scale; bool ok = rel < tol;
    if (!ok) g_ok = false;
    printf("  %-26s %s  |F|max=%.3e  max=%.3e  rel=%.3e (tol %.0e)\n",
           name, ok ? "OK " : "FAIL", scale, maxa, rel, tol);
}

int main(int argc, char** argv) {
    const char* lib = argc > 1 ? argv[1] : "/tmp/test_bridge1d.metallib";
    if (mtl_init(lib)) { printf("mtl_init FAIL\n"); return 1; }

    // Uniform refined block at level L, centred so both edges have MISSING
    // neighbours (Dirichlet) -> symmetric boundary. nb < 2^(L-1).
    const int nlevelmax = 7, L = 6, bnd = 1, ngs = 2;
    g_nb = 16; g_base = (1 << (L-1))/2 - g_nb/2;     // ckey 8..23 at L=6 (level width 32)
    int n_fine = g_nb, mg_cap = 256;
    mtl_alloc_buffers(n_fine, 1, 2*n_fine+3, nlevelmax);

    int* ckey_max = (int*)mtl_ptr_ckey_max(); long* key_off = (long*)mtl_ptr_key_off();
    for (int l=0;l<=nlevelmax+1;++l){ckey_max[l]=0;key_off[l]=0;}
    for (int l=1;l<=nlevelmax;++l) ckey_max[l]=1<<(l-1);
    key_off[1]=1; for(int l=2;l<=nlevelmax;++l){long n=ckey_max[l-1]; key_off[l]=key_off[l-1]+n;} // 1D stride
    std::vector<int> bmin(3*nlevelmax,0), bmax(3*nlevelmax,0);
    for (int l=1;l<=nlevelmax;++l) for(int d=0;d<3;++d) bmax[(l-1)*3+d]=ckey_max[l];

    Oct* grid=(Oct*)mtl_ptr_grid(); memset(grid,0,(size_t)n_fine*sizeof(Oct));
    for (int o=0;o<g_nb;++o){ grid[o].lev=L; grid[o].ckey[0]=g_base+o; }

    mtl_build_connectivity(n_fine, L, nlevelmax, bmin.data(), bmax.data(), 1,1,1);

    // mask = interior everywhere (f(:,3)=1); phi=0; SYMMETRIC source S in f(:,2).
    float* f=(float*)mtl_ptr_f();
    memset(mtl_ptr_phi(),0,(size_t)n_fine*TWOTONDIM*sizeof(float));
    for (int o=0;o<g_nb;++o) for(int c=1;c<=TWOTONDIM;++c){
        f[IDX3(c,3,o+1)]=1.0f;
        int g = 2*o + (c-1);
        double xc = (g + 0.5) - g_nb;             // distance of cell centre from block centre
        f[IDX3(c,2,o+1)]=(float)(1.0/(1.0+0.25*xc*xc) - 0.2);   // symmetric, zero-ish mean
    }
    printf("Stage 1: 1D uniform refined block L=%d nb=%d ckey %d..%d (Dirichlet edges)\n",
           L, g_nb, g_base, g_base+g_nb-1);
    check_fcol(f, 2, +1, "source f(:,2)", 1e-6);
    check_fcol(f, 3, +1, "mask f(:,3)",   1e-6);

    // Build hierarchy + restrict mask (mirror of metal_recursive_mg setup).
    for (int ifine=L; ifine>=bnd+1; --ifine)
        mtl_mg_build(L, ifine, 1, n_fine, mg_cap, bmin.data(), bmax.data(), 1,1,1, 0, 1.0f, 0, 0.0f);
    int levelmin_mg = bnd;
    for (int ifine=L; ifine>=bnd+1; --ifine)
        if (mtl_mg_restrict_mask(L, ifine)==1){ levelmin_mg=ifine; break; }

    std::function<void(int)> rec = [&](int ifine){
        if (ifine <= levelmin_mg){
            for(int i=0;i<2*ngs;++i){ mtl_mg_gauss_seidel(L,ifine,0,1); mtl_mg_gauss_seidel(L,ifine,0,0);} return; }
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,ifine,0,1); mtl_mg_gauss_seidel(L,ifine,0,0);}
        mtl_mg_cmp_residual(L,ifine); mtl_mg_restrict_residual(L,ifine);
        mtl_mg_reset_corr(L,ifine-1); rec(ifine-1); mtl_mg_interpolate_correct(L,ifine);
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,ifine,0,1); mtl_mg_gauss_seidel(L,ifine,0,0);}
    };

    const float* phi=(const float*)mtl_ptr_phi();
    double r0 = (mtl_mg_cmp_residual(L,L), sqrt(mtl_mg_residual_norm2(L)));
    for (int iter=0; iter<20; ++iter){
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,L,0,1); mtl_mg_gauss_seidel(L,L,0,0);}
        mtl_mg_cmp_residual(L,L);
        if (L>levelmin_mg){ mtl_mg_restrict_residual(L,L); mtl_mg_reset_corr(L,L-1); rec(L-1); mtl_mg_interpolate_correct(L,L);}
        for(int i=0;i<ngs;++i){ mtl_mg_gauss_seidel(L,L,0,1); mtl_mg_gauss_seidel(L,L,0,0);}
        double rk=sqrt(mtl_mg_residual_norm2(L));
        if (rk < 1e-6*r0) break;
    }
    printf("converged ||r||/||r0|| reached.\n");

    // The solved potential MUST be symmetric.
    check(phi, +1, "phi (solved)", 1e-4);
    mtl_mg_cmp_residual(L,L);
    check_fcol(f, 1, +1, "residual f(:,1)", 1e-4);

    // gradient_phi overwrites f(:,1..ndim) with the force; x-force MUST be antisymmetric.
    mtl_gradient_phi(1, n_fine, 1.0f/(1<<L), 0.0f);
    check_fcol(f, 1, -1, "force fx (antisym)", 1e-4);

    // net mesh force Sum_cells fx * S   (should be ~0 by Newton's 3rd law if symmetric)
    double net=0.0;
    for (int o=0;o<g_nb;++o) for(int c=1;c<=TWOTONDIM;++c) net += (double)f[IDX3(c,1,o+1)];
    printf("  net Sum(fx) = %.3e (exp ~0)\n", net);
    if (fabs(net) > 1e-6) g_ok = false;

    printf("\nSTAGE-1 (field channel, interior) %s\n", g_ok ? "PASS" : "FAIL");
    mtl_finalize();
    return g_ok ? 0 : 1;
}

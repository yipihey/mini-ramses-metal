//============================================================================
// metal_bridge_h6test.mm — validate the H6 AMR multigrid hierarchy builder
// (mtl_build_mg_amr).  A refined 8^3 block of level-5 octs (ckey 0..7) should
// coarsen 512 -> 64 (L4) -> 8 (L3) -> 1 (L2); check father_mg chain (parent
// ckey == child ckey/2), per-level counts, and nbor_mg self/688neighbour.
//============================================================================
#include <cstdio>
#include <cstring>
#include <vector>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void mtl_build_mg_amr(int,int,int,int,const int*,const int*,int,int,int);
  void* mtl_ptr_grid(); void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void* mtl_ptr_grid_mg(); void* mtl_ptr_father_mg(); void* mtl_ptr_nbor_mg();
  int   mtl_mg_head(int); int mtl_mg_noct(int); int mtl_mg_bottom();
  void mtl_finalize();
}

int main(int argc, char** argv) {
    const char* lib = argc>1 ? argv[1] : "/tmp/ramses_kernels.metallib";
    if (mtl_init(lib)) return 1;

    const int nlevelmax = 7, L = 5, nb = 8;       // refined block nb^3 at level L
    int n_fine = nb*nb*nb;                          // 512
    int mg_cap = 256;
    mtl_alloc_buffers(n_fine, 1, 2*n_fine+3, nlevelmax);

    int*  ckey_max=(int*)mtl_ptr_ckey_max(); long* key_off=(long*)mtl_ptr_key_off();
    for (int l=0;l<=nlevelmax+1;++l){ckey_max[l]=0;key_off[l]=0;}
    for (int l=1;l<=nlevelmax;++l) ckey_max[l]=1<<(l-1);
    key_off[1]=1; for(int l=2;l<=nlevelmax;++l){long n=ckey_max[l-1];key_off[l]=key_off[l-1]+n*n*n;}
    std::vector<int> box_min(3*nlevelmax,0), box_max(3*nlevelmax,0);
    for (int l=1;l<=nlevelmax;++l) for(int d=0;d<3;++d) box_max[(l-1)*3+d]=ckey_max[l];

    // fine octs: ckey 0..7 per dim at level L
    Oct* grid=(Oct*)mtl_ptr_grid();
    int o=0;
    for (int z=0;z<nb;++z) for(int y=0;y<nb;++y) for(int x=0;x<nb;++x){
        grid[o]=Oct{}; grid[o].lev=L; grid[o].ckey[0]=x;grid[o].ckey[1]=y;grid[o].ckey[2]=z; ++o;
    }

    mtl_build_mg_amr(L, 1, n_fine, mg_cap, box_min.data(), box_max.data(), 1,1,1);

    Oct* gmg=(Oct*)mtl_ptr_grid_mg();
    int* father=(int*)mtl_ptr_father_mg();
    int* nbor_mg=(int*)mtl_ptr_nbor_mg();

    int bad=0;
    // per-level counts
    int exp_noct[6]={0,0,1,8,64,0}; // L2..L4 (index by level)
    printf("  MG levels: L4 noct=%d (exp 64), L3 noct=%d (exp 8), L2 noct=%d (exp 1), bottom=%d\n",
           mtl_mg_noct(4), mtl_mg_noct(3), mtl_mg_noct(2), mtl_mg_bottom());
    if (mtl_mg_noct(4)!=64) ++bad;
    if (mtl_mg_noct(3)!=8)  ++bad;
    if (mtl_mg_noct(2)!=1)  ++bad;
    if (mtl_mg_bottom()!=2) ++bad;

    // father chain for fine octs (child index 0..n_fine-1)
    for (int c=0;c<n_fine;++c){
        int fa = father[c];                          // 1-based grid_mg
        if (fa<1){ if(bad<6)printf("  fine %d father %d invalid\n",c,fa); ++bad; continue; }
        if (gmg[fa-1].lev!=L-1) ++bad;
        for (int d=0;d<3;++d) if (gmg[fa-1].ckey[d]!=grid[c].ckey[d]/2) { ++bad; break; }
    }
    // father chain for L4 MG octs (child index n_fine + (idx-1) for idx in mg level 4)
    int h4=mtl_mg_head(4), n4=mtl_mg_noct(4);
    for (int q=0;q<n4;++q){
        int gidx = h4-1 + q;                         // 0-based grid_mg index of L4 oct
        int fa = father[n_fine + gidx];              // its parent (L3)
        if (fa<1){ ++bad; continue; }
        if (gmg[fa-1].lev!=L-2) ++bad;
        for (int d=0;d<3;++d) if (gmg[fa-1].ckey[d]!=gmg[gidx].ckey[d]/2){ ++bad; break; }
    }
    // nbor_mg self-reference (ind 14 = offset 0,0,0) for L4 octs
    int self_ok=1;
    for (int q=0;q<n4;++q){ int gidx=h4-1+q; if (nbor_mg[gidx*27 + 13] != gidx+1) self_ok=0; }
    if (!self_ok) ++bad;

    printf("H6 MG hierarchy: %s  (fine=%d, total_mg=%d, father_bad=%d, self_ok=%d)\n",
           bad? "FAIL":"PASS", n_fine, mtl_mg_head(2)-1+mtl_mg_noct(2), bad, self_ok);
    mtl_finalize();
    return bad?1:0;
}

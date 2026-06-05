//============================================================================
// metal_bridge_cacheorch_test.mm — #30 cache-oct HOST ORCHESTRATION check (no run).
// Drives the bridge mtl_make_cache (predicate -> scan -> compute_cache_swap_table
// -> make_cache_octs -> hash_insert, per neighbour direction) on a real 1D
// coarse-fine patch and verifies the materialised cache (ghost) octs.
//
// Patch (NDIM=1, periodic): coarse level 2 = octs ckey{0,1} (idx 1,2); coarse
// ckey0 is refined -> fine level 3 octs ckey{0,1} (idx 3,4).  The fine block's
// outward neighbours (ckey 3 on the left via wrap, ckey 2 on the right) are NOT
// refined -> MISSING -> must become cache octs with father = coarse ckey1 (idx 2).
// ngridmax=4 (real octs 1..4); cache region 5..ncell.
// Expect: 2 cache octs created; idx5 ckey3 father2; idx6 ckey2 father2; the fine
// octs' missing-nbr slots patched to point at them.
// Build: bridge NDIM=1 .o + NDIM=1 metallib.
//============================================================================
#include <cstdio>
#include <cstring>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void mtl_set_cache_region(int);
  void mtl_build_connectivity(int,int,int,const int*,const int*,int,int,int);
  int  mtl_make_cache(int,int,int,int,int,int,int,float,int);
  void* mtl_ptr_grid(); void* mtl_ptr_nbor(); void* mtl_ptr_father();
  void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void mtl_finalize();
}

int main(int argc, char** argv){
    const char* lib=argc>1?argv[1]:"/tmp/test_bridge1d.metallib";
    if (mtl_init(lib)) return 1;
    const int nlevelmax=7, ngridmax=4, ncell=12;
    mtl_alloc_buffers(ncell, 1, 2*ncell+3, nlevelmax);
    mtl_set_cache_region(ngridmax);              // octs 1..4 real; 5..12 cache

    int* ckey_max=(int*)mtl_ptr_ckey_max(); long* key_off=(long*)mtl_ptr_key_off();
    for(int l=0;l<=nlevelmax+1;++l){ckey_max[l]=0;key_off[l]=0;}
    for(int l=1;l<=nlevelmax;++l) ckey_max[l]=1<<(l-1);
    key_off[1]=1; for(int l=2;l<=nlevelmax;++l){long n=ckey_max[l-1];key_off[l]=key_off[l-1]+n;}  // 1D stride
    int bmin[3*7]={0}, bmax[3*7]={0};
    for(int l=1;l<=nlevelmax;++l) for(int d=0;d<3;++d) bmax[(l-1)*3+d]=ckey_max[l];

    Oct* grid=(Oct*)mtl_ptr_grid(); memset(grid,0,(size_t)ncell*sizeof(Oct));
    grid[0].lev=2; grid[0].ckey[0]=0; grid[0].refined[0]=1;   // coarse ckey0 (refined)
    grid[1].lev=2; grid[1].ckey[0]=1;                         // coarse ckey1 (not refined)
    grid[2].lev=3; grid[2].ckey[0]=0;                         // fine ckey0
    grid[3].lev=3; grid[3].ckey[0]=1;                         // fine ckey1

    // connectivity over the 4 real octs (hash + father + same-level nbor; the fine
    // block's outward nbrs hash-miss -> 0); stores box_min/box_max in B.
    mtl_build_connectivity(ngridmax, 2, nlevelmax, bmin, bmax, 1,1,1);

    int created = mtl_make_cache(3, 3, 2, nlevelmax, 1, 0, 0, 0.0f, 0); // fine level 3, octs idx 3..4 (full cache)

    Oct* g=(Oct*)mtl_ptr_grid(); int* father=(int*)mtl_ptr_father(); int* nbor=(int*)mtl_ptr_nbor();
    int c5_ck=g[4].ckey[0], c5_lv=g[4].lev, c5_fa=father[4];     // cache oct idx 5
    int c6_ck=g[5].ckey[0], c6_lv=g[5].lev, c6_fa=father[5];     // cache oct idx 6
    int nb3 = nbor[(3-1)*SUBGRIDSIZE + 0];   // fine oct3, input_ind=1 (left) -> cube index 0
    int nb4 = nbor[(4-1)*SUBGRIDSIZE + 2];   // fine oct4, input_ind=3 (right)-> cube index 2
    printf("created=%d (exp 2)\n", created);
    printf("cache idx5: lev=%d ckey=%d father=%d (exp 3,3,2)\n", c5_lv,c5_ck,c5_fa);
    printf("cache idx6: lev=%d ckey=%d father=%d (exp 3,2,2)\n", c6_lv,c6_ck,c6_fa);
    printf("patched nbor: oct3.left=%d (exp 5)  oct4.right=%d (exp 6)\n", nb3, nb4);
    bool ok = created==2
           && c5_lv==3 && c5_ck==3 && c5_fa==2
           && c6_lv==3 && c6_ck==2 && c6_fa==2
           && nb3==5 && nb4==6;
    printf("%s\n", ok?"PASS":"FAIL");
    mtl_finalize();
    return ok?0:1;
}

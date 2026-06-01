//============================================================================
// metal_bridge_splittest.mm — validate the GPU particle split (level bucket +
// stable partition + reorder of the resident arrays).  A level-3 base grid with
// one refined cell; particles placed in a refined cell (must descend) and an
// unrefined cell (must stay).  Check n_stay and that the descended particle is
// moved after the stay particle.
//============================================================================
#include <cstdio>
#include <cstring>
#include <vector>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void mtl_build_connectivity(int,int,int,const int*,const int*,int,int,int);
  int  mtl_gpu_split_part(int,int,int,int,int,long);
  void mtl_gpu_sort_part(int,int,int);
  void* mtl_ptr_grid(); void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void* mtl_ptr_ipos(); void* mtl_ptr_mp(); void* mtl_ptr_vp(); void* mtl_ptr_levelp();
  void mtl_finalize();
}

int main(int argc, char** argv) {
    const char* lib = argc>1 ? argv[1] : "/tmp/ramses_kernels.metallib";
    if (mtl_init(lib)) return 1;
    const int nlevelmax=5, L=3, nx0=4;            // level-3 base, 4 octs/dim
    int noct=nx0*nx0*nx0, npm=64;
    mtl_alloc_buffers(noct, npm, 2*noct+3, nlevelmax);

    int*  ckey_max=(int*)mtl_ptr_ckey_max(); long* key_off=(long*)mtl_ptr_key_off();
    for(int l=0;l<=nlevelmax+1;++l){ckey_max[l]=0;key_off[l]=0;}
    for(int l=1;l<=nlevelmax;++l) ckey_max[l]=1<<(l-1);
    key_off[1]=1; for(int l=2;l<=nlevelmax;++l){long n=ckey_max[l-1];key_off[l]=key_off[l-1]+n*n*n;}
    std::vector<int> bmin(3*nlevelmax,0), bmax(3*nlevelmax,0);
    for(int l=1;l<=nlevelmax;++l)for(int d=0;d<3;++d) bmax[(l-1)*3+d]=ckey_max[l];

    Oct* grid=(Oct*)mtl_ptr_grid();
    for(int z=0;z<nx0;++z)for(int y=0;y<nx0;++y)for(int x=0;x<nx0;++x){
        int o=x+y*nx0+z*nx0*nx0; grid[o]=Oct{}; grid[o].lev=L;
        grid[o].ckey[0]=x;grid[o].ckey[1]=y;grid[o].ckey[2]=z;
    }
    grid[0].refined[0]=1;                          // oct(0,0,0) cell 1 is refined
    mtl_build_connectivity(noct, L, nlevelmax, bmin.data(), bmax.data(), 1,1,1);

    // particle A in oct0 cell1 (refined -> descend); B in oct0 cell2 (stay).
    long* ipos=(long*)mtl_ptr_ipos();
    float* mp=(float*)mtl_ptr_mp(); float* vp=(float*)mtl_ptr_vp(); int* lvl=(int*)mtl_ptr_levelp();
    long c0=1L<<44;                                // center of level-3 cell 0
    // A (ip=1): cell (0,0,0)
    ipos[0*npm+0]=c0; ipos[1*npm+0]=c0; ipos[2*npm+0]=c0; mp[0]=11.f; lvl[0]=L;
    vp[0*npm+0]=1.f; vp[1*npm+0]=2.f; vp[2*npm+0]=3.f;
    // B (ip=2): cell (1,0,0) -> ipos_x center of cell 1 = 1.5*2^45
    ipos[0*npm+1]=(1L<<45)+(1L<<44); ipos[1*npm+1]=c0; ipos[2*npm+1]=c0; mp[1]=22.f; lvl[1]=L;
    vp[0*npm+1]=4.f; vp[1*npm+1]=5.f; vp[2*npm+1]=6.f;

    int n_stay = mtl_gpu_split_part(L, 1, 2, 2*noct+3, ckey_max[L], key_off[L]);

    // after split: stay (B, mp=22) at pos 1, descended (A, mp=11) at pos 2.
    bool ok = (n_stay==1) && (mp[0]==22.f) && (mp[1]==11.f)
              && (vp[0*npm+0]==4.f) && (vp[0*npm+1]==1.f);   // B's vx then A's vx
    printf("GPU split: %s  n_stay=%d (exp 1)  mp=[%.0f,%.0f] (exp 22,11)  vx=[%.0f,%.0f] (exp 4,1)\n",
           ok?"PASS":"FAIL", n_stay, mp[0], mp[1], vp[0], vp[npm]);
    mtl_finalize();
    return ok?0:1;
}

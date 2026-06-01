//============================================================================
// metal_bridge_conntest.mm — validate the H1 connectivity builder
// (mtl_build_connectivity) on a synthetic 2-level AMR mesh: a full level-L0
// base grid plus one refined base-oct (its 8 child cells -> 8 level-(L0+1)
// octs).  father/nbor produced by the builder are checked against an
// independent brute-force (level,ckey)->index map with the same periodic wrap.
//============================================================================
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <array>
#include "ramses_metal.h"

extern "C" {
  int  mtl_init(const char*);
  void mtl_alloc_buffers(int,int,int,int);
  void mtl_build_connectivity(int,int,int,const int*,const int*,int,int,int);
  void* mtl_ptr_grid(); void* mtl_ptr_nbor(); void* mtl_ptr_father();
  void* mtl_ptr_ckey_max(); void* mtl_ptr_key_off();
  void mtl_finalize();
}

int main(int argc, char** argv) {
    const char* lib = argc>1 ? argv[1] : "/tmp/ramses_kernels.metallib";
    if (mtl_init(lib)) return 1;

    const int nlevelmax = 5, levelmin = 3;
    const int ncell = 256, hash_size = 1031;
    mtl_alloc_buffers(ncell, 1, hash_size, nlevelmax);

    // per-level ckey_max (1-based-padded) and key_off (level-separating, >0).
    int*  ckey_max = (int*) mtl_ptr_ckey_max();
    long* key_off  = (long*)mtl_ptr_key_off();
    for (int L=0;L<=nlevelmax+1;++L) ckey_max[L]=0, key_off[L]=0;
    for (int L=1;L<=nlevelmax;++L) ckey_max[L] = 1<<(L-1);
    key_off[1]=1;
    for (int L=2;L<=nlevelmax;++L) {
        long nxp=ckey_max[L-1]; key_off[L]=key_off[L-1]+nxp*nxp*nxp;
    }
    // periodic box [0, ckey_max(L)) at each level.
    std::vector<int> box_min(3*(nlevelmax+1),0), box_max(3*(nlevelmax+1),0);
    for (int L=1;L<=nlevelmax;++L) for (int d=0;d<3;++d) box_max[(L-1)*3+d]=ckey_max[L];

    // Build the mesh into B.grid: full level-3 base (nx=4 -> 64 octs) then the
    // 8 level-4 children of base oct at ckey (1,1,1).
    Oct* grid = (Oct*)mtl_ptr_grid();
    int  nx0  = ckey_max[levelmin];           // 4
    int  noct = 0;
    auto put = [&](int L,int x,int y,int z){ Oct o{}; o.lev=L; o.ckey[0]=x;o.ckey[1]=y;o.ckey[2]=z;
                                             grid[noct]=o; ++noct; };
    for (int z=0;z<nx0;++z) for (int y=0;y<nx0;++y) for (int x=0;x<nx0;++x) put(levelmin,x,y,z);
    const int px=1,py=1,pz=1;                 // refined base oct
    for (int i=0;i<2;++i) for (int j=0;j<2;++j) for (int k=0;k<2;++k)
        put(levelmin+1, 2*px+i, 2*py+j, 2*pz+k);

    mtl_build_connectivity(noct, levelmin, nlevelmax, box_min.data(), box_max.data(), 1,1,1);

    // Brute-force reference map (level,ckey)->1-based index.
    std::map<std::array<int,4>,int> ref;
    for (int o=1;o<=noct;++o)
        ref[{grid[o-1].lev,grid[o-1].ckey[0],grid[o-1].ckey[1],grid[o-1].ckey[2]}] = o;

    const int* nbor   = (const int*)mtl_ptr_nbor();
    const int* father = (const int*)mtl_ptr_father();
    int nbad_f=0, nbad_n=0;

    for (int o=1;o<=noct;++o) {
        int L=grid[o-1].lev, cx=grid[o-1].ckey[0], cy=grid[o-1].ckey[1], cz=grid[o-1].ckey[2];
        int fexp = (L>levelmin) ? ref.count({L-1,cx/2,cy/2,cz/2}) ? ref[{L-1,cx/2,cy/2,cz/2}] : 0
                                : 0;
        // level-3 base octs have no parent level in this mesh -> 0
        if (father[o-1]!=fexp) { if(nbad_f<5) printf("  father oct %d L%d: got %d exp %d\n",o,L,father[o-1],fexp); ++nbad_f; }
        for (int kk=0;kk<3;++kk) for (int jj=0;jj<3;++jj) for (int ii=0;ii<3;++ii) {
            int ind=1+ii+3*jj+9*kk;
            int nck[3]={cx+ii-1,cy+jj-1,cz+kk-1};
            for (int d=0;d<3;++d){ int bmn=box_min[(L-1)*3+d],bmx=box_max[(L-1)*3+d];
                if(nck[d]<bmn)nck[d]=bmx-1; if(nck[d]>=bmx)nck[d]=bmn; }
            int nexp = ref.count({L,nck[0],nck[1],nck[2]}) ? ref[{L,nck[0],nck[1],nck[2]}] : 0;
            int ngot = nbor[(o-1)*27+(ind-1)];
            if (ngot!=nexp){ if(nbad_n<5) printf("  nbor oct %d ind %d: got %d exp %d\n",o,ind,ngot,nexp); ++nbad_n; }
        }
    }
    // sanity: a level-3 oct's self neighbour (ind 14, offset 0,0,0) is itself.
    int self_ok = (nbor[(1-1)*27 + (14-1)] == 1);

    bool ok = (nbad_f==0 && nbad_n==0 && self_ok);
    printf("H1 connectivity: %s  (octs=%d, father_bad=%d, nbor_bad=%d, self_ok=%d)\n",
           ok?"PASS":"FAIL", noct, nbad_f, nbad_n, self_ok);
    mtl_finalize();
    return ok?0:1;
}

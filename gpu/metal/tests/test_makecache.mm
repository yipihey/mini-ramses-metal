// Unit test for make_cache_octs (refine.metal), NDIM=1 — the faithful coarse-fine
// boundary materialisation.  A cache (ghost) oct is created for the MISSING same-
// level neighbour of a refined oct; it must be a REAL oct with the correct father
// (the coarse parent, found via the hash at lev-1) and straight-injected f/phi/
// phi_old from the parent cell.  This is the boundary that replaced the invented
// inline ghost; getting the father right is what makes the CIC (and thus the
// coarse-fine force) momentum-faithful.
//
// Setup (periodic): fine oct (idx1) lev=3 ckey=2, missing its +1 neighbour
// (input_ind=3 in the 3^1 cube).  Coarse parent (idx2) lev=2 ckey=1 lives in the
// hash.  Cache oct ckey = 2+1 = 3; parent ckey = 3/2 = 1 -> oct2; parent cell =
// 1+(3-2*1) = 2.  Expect: cache.lev=3, cache.ckey=3, father=2, fine.nbor[2]=cache,
// and f/phi/phi_old injected from (cell2, oct2).
// Build: refine.metal + hash.metal -DNDIM=1 -> one metallib.
#import <Metal/Metal.h>
#include "ramses_metal.h"   // Oct, CacheParams, ConnParams, IDX2/IDX3, SUBGRIDSIZE
#include <cstdio>
#include <cstring>
#include <cmath>

static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, const char* n) {
    NSError* e=nil; id<MTLFunction> f=[l newFunctionWithName:@(n)];
    if(!f){fprintf(stderr,"no kernel %s\n",n);exit(2);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&e];
    if(!p){fprintf(stderr,"pso %s: %s\n",n,e.localizedDescription.UTF8String);exit(2);}
    return p;
}

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_rc1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState> psoHash  = pso(dev,lib,"hash_insert");
        id<MTLComputePipelineState> psoCache = pso(dev,lib,"make_cache_octs");

        const int nlev=10, hsz=17, ngridmax=10, ifree_cache=0, input_ind=3, ilevel_fine=3;
        const int NB = ngridmax + 4;                 // grid/father/etc capacity (incl cache region)
        const long KEY_OFF2=100, KEY_OFF3=200;

        id<MTLBuffer> bgrid=[dev newBufferWithLength:NB*sizeof(Oct) options:MTLResourceStorageModeShared];
        Oct* g=(Oct*)bgrid.contents; memset(g,0,NB*sizeof(Oct));
        g[0].lev=ilevel_fine; g[0].ckey[0]=2;        // oct1: refined fine oct (the subgrid)
        g[1].lev=2;           g[1].ckey[0]=1;        // oct2: coarse parent

        // level-indexed ckey_max / key_off
        id<MTLBuffer> bckm=[dev newBufferWithLength:(nlev+1)*sizeof(int)  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bkof=[dev newBufferWithLength:(nlev+1)*sizeof(long) options:MTLResourceStorageModeShared];
        int* ckm=(int*)bckm.contents; long* kof=(long*)bkof.contents;
        memset(ckm,0,(nlev+1)*sizeof(int)); memset(kof,0,(nlev+1)*sizeof(long));
        ckm[2]=2; kof[2]=KEY_OFF2;  ckm[3]=4; kof[3]=KEY_OFF3;     // 2^(L-1) octs/dim

        // box bounds per level: index [(lev-1)*3 + d]; lev=3 -> [6..8]
        id<MTLBuffer> bbmin=[dev newBufferWithLength:3*(nlev+1)*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bbmax=[dev newBufferWithLength:3*(nlev+1)*sizeof(int) options:MTLResourceStorageModeShared];
        memset(bbmin.contents,0,3*(nlev+1)*sizeof(int)); memset(bbmax.contents,0,3*(nlev+1)*sizeof(int));
        ((int*)bbmax.contents)[(ilevel_fine-1)*3+0] = ckm[ilevel_fine];   // 4 octs/dim at lev3

        id<MTLBuffer> bhk=[dev newBufferWithLength:hsz*sizeof(long) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bhv=[dev newBufferWithLength:hsz*sizeof(int)  options:MTLResourceStorageModeShared];
        memset(bhk.contents,0,hsz*sizeof(long)); memset(bhv.contents,0,hsz*sizeof(int));

        auto fz=[&](int n){ id<MTLBuffer> b=[dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared]; memset(b.contents,0,n*sizeof(float)); return b; };
        id<MTLBuffer> bf =fz(TWOTONDIM*NF*NB), bphi=fz(TWOTONDIM*NB), bpho=fz(TWOTONDIM*NB);
        id<MTLBuffer> bflag=[dev newBufferWithLength:TWOTONDIM*NB*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bfath=[dev newBufferWithLength:NB*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bnbor=[dev newBufferWithLength:NB*SUBGRIDSIZE*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bswap=[dev newBufferWithLength:4*sizeof(int) options:MTLResourceStorageModeShared];
        memset(bflag.contents,0,TWOTONDIM*NB*sizeof(int)); memset(bfath.contents,0,NB*sizeof(int));
        memset(bnbor.contents,0,NB*SUBGRIDSIZE*sizeof(int));
        // Coarse parent (oct2) self/centre neighbour: make_cache_octs now CIC-refines the
        // cache phi (interpol_phi) using the parent's 3^NDIM neighbour cells.  The parent's
        // real neighbours are absent in this minimal patch, so the CIC falls back to the
        // CENTRE father cell -> the parent cell -> reproduces the injected phi (0.9).  In a
        // full run mtl_build_connectivity sets this; here we set just the centre.
        ((int*)bnbor.contents)[(2-1)*SUBGRIDSIZE + (THREETONDIM-1)/2] = 2;
        // parent cell (cell2, oct2) fields to be injected.  phi is now CIC-interpolated from
        // the parent's cells, so set BOTH parent cells to 0.9 -> uniform coarse phi -> the CIC
        // gives 0.9 in the cache oct (a uniform field is reproduced exactly by interpol_phi).
        ((float*)bf.contents)[IDX3(2,1,2)] = 0.7f;
        ((float*)bphi.contents)[IDX2(1,2)] = 0.9f;   // parent cell 1
        ((float*)bphi.contents)[IDX2(2,2)] = 0.9f;   // parent cell 2
        ((float*)bpho.contents)[IDX2(2,2)] = 0.5f;
        ((int*)bswap.contents)[0] = 1;               // swap_local[0] = fine oct 1

        ConnParams CP{}; CP.num_octs=2; CP.hash_size=hsz; CP.nlevelmax=nlev; CP.head_idx=1;
        CacheParams P{}; P.num_octs=1; P.ngridmax=ngridmax; P.ifree_cache=ifree_cache; P.input_ind=input_ind;
        P.hash_size=hsz; P.nlevelmax=nlev; P.per[0]=1;

        // 1) hash (parent must be findable)
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoHash];
          [e setBuffer:bgrid offset:0 atIndex:0]; [e setBuffer:bhk offset:0 atIndex:1]; [e setBuffer:bhv offset:0 atIndex:2];
          [e setBuffer:bckm offset:0 atIndex:3]; [e setBuffer:bkof offset:0 atIndex:4]; [e setBytes:&CP length:sizeof(CP) atIndex:5];
          [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)]; [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        // 2) make_cache_octs
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoCache];
          [e setBuffer:bgrid offset:0 atIndex:0]; [e setBuffer:bflag offset:0 atIndex:1]; [e setBuffer:bf offset:0 atIndex:2];
          [e setBuffer:bphi offset:0 atIndex:3]; [e setBuffer:bpho offset:0 atIndex:4]; [e setBuffer:bswap offset:0 atIndex:5];
          [e setBuffer:bfath offset:0 atIndex:6]; [e setBuffer:bnbor offset:0 atIndex:7]; [e setBuffer:bhk offset:0 atIndex:8];
          [e setBuffer:bhv offset:0 atIndex:9]; [e setBuffer:bckm offset:0 atIndex:10]; [e setBuffer:bkof offset:0 atIndex:11];
          [e setBuffer:bbmin offset:0 atIndex:12]; [e setBuffer:bbmax offset:0 atIndex:13]; [e setBytes:&P length:sizeof(P) atIndex:14];
          [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        int cache = ngridmax + ifree_cache + 0 + 1;  // 11 (1-based)
        Oct* gc=&((Oct*)bgrid.contents)[cache-1];
        int  father = ((int*)bfath.contents)[cache-1];
        int  nbslot = ((int*)bnbor.contents)[(1-1)*SUBGRIDSIZE + (input_ind-1)];
        float fx  = ((float*)bf.contents)[IDX3(1,1,cache)];
        float ph  = ((float*)bphi.contents)[IDX2(1,cache)];
        float pho = ((float*)bpho.contents)[IDX2(1,cache)];
        printf("cache oct #%d: lev=%d (exp 3) ckey=%d (exp 3) father=%d (exp 2) fine.nbor[+1]=%d (exp %d)\n",
               cache, gc->lev, gc->ckey[0], father, nbslot, cache);
        printf("injected: f_x=%.4f (exp 0.7) phi=%.4f (exp 0.9) phi_old=%.4f (exp 0.5)\n", fx, ph, pho);
        bool ok = gc->lev==3 && gc->ckey[0]==3 && father==2 && nbslot==cache
               && fabs(fx-0.7f)<1e-6 && fabs(ph-0.9f)<1e-6 && fabs(pho-0.5f)<1e-6;
        printf("%s\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}

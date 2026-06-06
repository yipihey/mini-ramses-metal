// Unit test for the DM refinement flagging (flag.metal: flag_poisson +
// flag_enforce_subgrid), NDIM=1.  Validates:
//   (1) flag_poisson flags exactly the cells whose nref >= m_refine (the DM
//       GRAV trigger: poisson_flag_kernel),
//   (2) flag_enforce_subgrid promotes a partially-flagged oct to fully flagged
//       (refinement is per-oct, nsubgrid==1) and leaves an unflagged oct alone.
//
// Setup: 2 octs (TWOTONDIM=2 cells each).  nref[oct1,cell1]=5 (> m_refine=3),
// everything else 0.  After flag_poisson: flag1=[1,0,0,0].  After enforce_subgrid:
// flag1=[1,1,0,0].
// Build: flag.metal -DNDIM=1 -> metallib.
#import <Metal/Metal.h>
#include "ramses_metal.h"     // FlagParams, IDX2
#include <cstdio>
#include <cstring>

static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, const char* n) {
    NSError* e=nil; id<MTLFunction> f=[l newFunctionWithName:@(n)];
    if(!f){fprintf(stderr,"no kernel %s\n",n);exit(2);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&e];
    if(!p){fprintf(stderr,"pso %s: %s\n",n,e.localizedDescription.UTF8String);exit(2);}
    return p;
}

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_flag1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState> psoP = pso(dev,lib,"flag_poisson");
        id<MTLComputePipelineState> psoS = pso(dev,lib,"flag_enforce_subgrid");

        const int nocts=2, nflat=TWOTONDIM*nocts;
        id<MTLBuffer> bflag=[dev newBufferWithLength:nflat*sizeof(int)   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bnref=[dev newBufferWithLength:nflat*sizeof(float) options:MTLResourceStorageModeShared];
        memset(bflag.contents,0,nflat*sizeof(int)); memset(bnref.contents,0,nflat*sizeof(float));
        ((float*)bnref.contents)[IDX2(1,1)] = 5.0f;        // oct1 cell1: above threshold

        FlagParams P{}; P.head_idx=1; P.num_octs=nocts; P.m_refine=3.0f;

        // flag_poisson: 2D grid (cell x oct)
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoP];
          [e setBuffer:bflag offset:0 atIndex:0]; [e setBuffer:bnref offset:0 atIndex:1]; [e setBytes:&P length:sizeof(P) atIndex:2];
          [e dispatchThreads:MTLSizeMake(TWOTONDIM,nocts,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,nocts,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        int* f=(int*)bflag.contents;
        bool ok_p = f[0]==1 && f[1]==0 && f[2]==0 && f[3]==0;
        printf("after flag_poisson:        flag1=[%d %d %d %d]  (exp [1 0 0 0])  %s\n", f[0],f[1],f[2],f[3], ok_p?"ok":"BAD");

        // flag_enforce_subgrid: 1D over octs
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoS];
          [e setBuffer:bflag offset:0 atIndex:0]; [e setBytes:&P length:sizeof(P) atIndex:1];
          [e dispatchThreads:MTLSizeMake(nocts,1,1) threadsPerThreadgroup:MTLSizeMake(nocts,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        bool ok_s = f[0]==1 && f[1]==1 && f[2]==0 && f[3]==0;
        printf("after flag_enforce_subgrid: flag1=[%d %d %d %d]  (exp [1 1 0 0])  %s\n", f[0],f[1],f[2],f[3], ok_s?"ok":"BAD");

        bool ok = ok_p && ok_s;
        printf("%s\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}

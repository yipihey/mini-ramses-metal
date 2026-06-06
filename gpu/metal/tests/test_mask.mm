// Unit test for the AMR mask build: reset_mask_kernel + restrict_mask + volume_to_mask.
// 1 coarse father oct (idx 1) + 8 fine child octs (idx 2..9, ckey parity = child cell).
// Fine all unmasked (mask=1) -> each coarse cell gets sum_8 (1+1)/2/8 = 1 -> volume_to_mask
// 2*1-1 = 1.  So a fully-refined coarse cell -> mask 1.  Validates the hierarchy vs CUDA.
// Build: recompile mg.metal -DNDIM=3 -> /tmp/test_mg.metallib (as in test_epot).
#import <Metal/Metal.h>
#include "ramses_metal.h"     // Oct, MgParams, IDX3, NF, TWOTONDIM (-DNDIM=3)
#include <cstdio>
#include <cstring>
#include <cmath>

static id<MTLComputePipelineState> mkpso(id<MTLDevice> dev, id<MTLLibrary> lib, const char* name) {
    NSError* e=nil; id<MTLFunction> fn=[lib newFunctionWithName:@(name)];
    if(!fn){fprintf(stderr,"no kernel %s\n",name);exit(2);}
    id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:fn error:&e];
    if(!p){fprintf(stderr,"pso %s: %s\n",name,e.localizedDescription.UTF8String);exit(2);}
    return p;
}

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_mg.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        auto pso_reset = mkpso(dev,lib,"reset_mask_kernel");
        auto pso_rest  = mkpso(dev,lib,"restrict_mask");
        auto pso_vol   = mkpso(dev,lib,"volume_to_mask");

        const int noct = 9;                       // 1 coarse (idx1) + 8 fine (idx2..9)
        id<MTLBuffer> bgrid=[dev newBufferWithLength:noct*sizeof(Oct) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bfa  =[dev newBufferWithLength:noct*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bf   =[dev newBufferWithLength:noct*NF*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        memset(bgrid.contents,0,noct*sizeof(Oct)); memset(bf.contents,0,noct*NF*TWOTONDIM*sizeof(float));
        Oct* grid=(Oct*)bgrid.contents; int* fa=(int*)bfa.contents; float* f=(float*)bf.contents;
        grid[0].lev=1;                            // coarse father, ckey (0,0,0)
        for (int g=0; g<8; ++g) {                 // fine octs 2..9
            int o=2+g; grid[o-1].lev=2;
            grid[o-1].ckey[0]= g&1; grid[o-1].ckey[1]=(g>>1)&1; grid[o-1].ckey[2]=(g>>2)&1; // parity -> cell g+1
            fa[o-1]=1;                            // father (mg_idx) -> coarse oct 1
        }

        auto run2d = [&](id<MTLComputePipelineState> p, void(^bind)(id<MTLComputeCommandEncoder>), int nx, int ny){
            id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:p]; bind(e);
            [e dispatchThreads:MTLSizeMake(nx,ny,1) threadsPerThreadgroup:MTLSizeMake(nx,ny,1)];
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
        };

        // 1) fine mask = 1   (head_idx=2, num=8)
        MgParams Pf{}; Pf.head_idx=2; Pf.num_octs=8; float one=1.0f;
        run2d(pso_reset, ^(id<MTLComputeCommandEncoder> e){
            [e setBuffer:bf offset:0 atIndex:0]; [e setBytes:&Pf length:sizeof(Pf) atIndex:1]; [e setBytes:&one length:4 atIndex:2];
        }, TWOTONDIM, 8);
        // 2) zero coarse mask (head_idx=1, num=1, mask_val=0)
        MgParams Pc{}; Pc.head_idx=1; Pc.num_octs=1; float zero=0.0f;
        run2d(pso_reset, ^(id<MTLComputeCommandEncoder> e){
            [e setBuffer:bf offset:0 atIndex:0]; [e setBytes:&Pc length:sizeof(Pc) atIndex:1]; [e setBytes:&zero length:4 atIndex:2];
        }, TWOTONDIM, 1);
        // 3) restrict_mask fine->coarse (head_idx=2, head_father=2, num=8)  f_mg=f (same buffer)
        MgParams Pr{}; Pr.head_idx=2; Pr.head_father=2; Pr.num_octs=8;
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:pso_rest];
          [e setBuffer:bgrid offset:0 atIndex:0]; [e setBuffer:bfa offset:0 atIndex:1];
          [e setBuffer:bf offset:0 atIndex:2]; [e setBuffer:bf offset:0 atIndex:3];
          [e setBytes:&Pr length:sizeof(Pr) atIndex:4];
          [e dispatchThreads:MTLSizeMake(8,1,1) threadsPerThreadgroup:MTLSizeMake(8,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        // 4) volume_to_mask coarse (head_idx=1, num=1)
        run2d(pso_vol, ^(id<MTLComputeCommandEncoder> e){
            [e setBuffer:bf offset:0 atIndex:0]; [e setBytes:&Pc length:sizeof(Pc) atIndex:1];
        }, TWOTONDIM, 1);

        float dmax = 0.0f;
        for (int c=1; c<=TWOTONDIM; ++c) dmax = fmaxf(dmax, fabsf(f[IDX3(c,3,1)] - 1.0f));
        printf("AMR mask build: coarse mask cells, max|mask-1| = %.3e (fully-refined -> 1)\n", dmax);
        bool ok = dmax < 1e-5f;
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

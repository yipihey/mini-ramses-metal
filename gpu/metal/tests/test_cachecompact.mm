// Unit test for the cache-oct compaction front-end (refine.metal:
// init_prefix_sum_nbor -> [inclusive scan] -> compute_cache_swap_table), NDIM=1.
// This is the stream-compaction that turns "which octs have a missing same-level
// neighbour in direction input_ind" into the dense list make_cache_octs consumes.
//
// Setup: 5 octs at a level (head_idx=1).  In direction input_ind=1, octs 2 and 4
// have nbor==0 (missing); 1,3,5 have a real neighbour.
//   predicate  prefix_sum = [0,1,0,1,0]
//   incl. scan            = [0,1,1,2,2]
//   compute_cache_swap_table -> swap_local = [2,4], count = 2.
// (The inclusive scan is done on the host here; block_scan/uniform_add are covered
// by the scan-kernel tests under #26.)
// Build: refine.metal -DNDIM=1 -> metallib.
#import <Metal/Metal.h>
#include "ramses_metal.h"     // ScanParams, SUBGRIDSIZE
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
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_refine1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState> psoPred = pso(dev,lib,"init_prefix_sum_nbor");
        id<MTLComputePipelineState> psoSwap = pso(dev,lib,"compute_cache_swap_table");

        const int nocts=5, input_ind=1;
        id<MTLBuffer> bnbor=[dev newBufferWithLength:nocts*SUBGRIDSIZE*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bpre =[dev newBufferWithLength:nocts*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bswap=[dev newBufferWithLength:nocts*sizeof(int) options:MTLResourceStorageModeShared];
        int* nbor=(int*)bnbor.contents; memset(nbor,0,nocts*SUBGRIDSIZE*sizeof(int));
        memset(bpre.contents,0,nocts*sizeof(int)); memset(bswap.contents,-1,nocts*sizeof(int));
        // direction input_ind=1 -> nbor[(oct-1)*SUBGRIDSIZE + 0].  Octs 1,3,5 have a
        // real neighbour (set 7); octs 2,4 stay 0 (missing).
        for (int oct=1; oct<=nocts; ++oct) if (oct!=2 && oct!=4) nbor[(oct-1)*SUBGRIDSIZE + (input_ind-1)] = 7;

        ScanParams P{}; P.n=nocts; P.head_idx=1;

        // 1) predicate
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoPred];
          [e setBuffer:bnbor offset:0 atIndex:0]; [e setBuffer:bpre offset:0 atIndex:1];
          [e setBytes:&P length:sizeof(P) atIndex:2]; [e setBytes:&input_ind length:sizeof(int) atIndex:3];
          [e dispatchThreads:MTLSizeMake(nocts,1,1) threadsPerThreadgroup:MTLSizeMake(nocts,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        int* pre=(int*)bpre.contents;
        printf("predicate  = [%d %d %d %d %d]  (exp [0 1 0 1 0])\n", pre[0],pre[1],pre[2],pre[3],pre[4]);
        bool ok_p = pre[0]==0&&pre[1]==1&&pre[2]==0&&pre[3]==1&&pre[4]==0;

        // 2) host inclusive scan in place
        for (int i=1;i<nocts;++i) pre[i]+=pre[i-1];
        int count=pre[nocts-1];
        printf("incl. scan = [%d %d %d %d %d]  (count=%d, exp 2)\n", pre[0],pre[1],pre[2],pre[3],pre[4], count);

        // 3) scatter
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoSwap];
          [e setBuffer:bswap offset:0 atIndex:0]; [e setBuffer:bpre offset:0 atIndex:1]; [e setBytes:&P length:sizeof(P) atIndex:2];
          [e dispatchThreads:MTLSizeMake(nocts,1,1) threadsPerThreadgroup:MTLSizeMake(nocts,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        int* sw=(int*)bswap.contents;
        printf("swap_local = [%d %d]  (exp [2 4])\n", sw[0], sw[1]);
        bool ok = ok_p && count==2 && sw[0]==2 && sw[1]==4;
        printf("%s\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}

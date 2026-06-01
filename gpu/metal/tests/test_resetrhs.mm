// Unit test for reset_rhs_kernel (mg.metal), validating the FAITHFUL materialized-
// boundary read (the fix for the inline-ghost invention).  8 interior octs (1..8,
// ngridmax=8) + 1 cache oct (idx 9 > ngridmax) with uniform phi = PC.  oct 1's -x
// neighbour is patched to the cache oct.  dx=1 (oneoverdx2=1), fourpi=1, offset=0,
// vol_loc=1 -> interior S = rho; oct1 cell1 boundary S = rho - 2*(0.5*PC + 0.5*phi_c).
#import <Metal/Metal.h>
#include "ramses_metal.h"     // IDX2, IDX3, NF, TWOTONDIM, SUBGRIDSIZE, MgParams (-DNDIM=1)
#include <cstdio>
#include <cstring>
#include <cmath>

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_mg1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLFunction> fn=[lib newFunctionWithName:@"reset_rhs_kernel"];
        if(!fn){fprintf(stderr,"no reset_rhs_kernel\n");return 2;}
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:fn error:&err];
        if(!pso){fprintf(stderr,"pso: %s\n",err.localizedDescription.UTF8String);return 2;}

        const int N=8, NT=N+1;                       // 8 interior + 1 cache (idx 9)
        id<MTLBuffer> bphi=[dev newBufferWithLength:NT*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> brho=[dev newBufferWithLength:NT*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bf  =[dev newBufferWithLength:NT*NF*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bnb =[dev newBufferWithLength:NT*SUBGRIDSIZE*sizeof(int)   options:MTLResourceStorageModeShared];
        memset(bphi.contents,0,NT*TWOTONDIM*sizeof(float)); memset(brho.contents,0,NT*TWOTONDIM*sizeof(float));
        memset(bf.contents,0,NT*NF*TWOTONDIM*sizeof(float));
        float* phi=(float*)bphi.contents; float* rho=(float*)brho.contents; float* f=(float*)bf.contents; int* nb=(int*)bnb.contents;
        const float PC=0.7f, phic=0.2f, R=0.5f;
        for (int o=1;o<=NT;++o) for (int c=1;c<=TWOTONDIM;++c){ rho[IDX2(c,o)]=R; f[IDX3(c,3,o)]=1.0f; phi[IDX2(c,o)]=0.0f; }
        for (int c=1;c<=TWOTONDIM;++c) phi[IDX2(c,9)]=PC;          // cache oct uniform phi
        for (int c=1;c<=TWOTONDIM;++c) phi[IDX2(c,1)]=phic;        // oct 1 phi
        // periodic interior nbor for octs 1..8, then patch oct1's -x to the cache oct 9
        for (int o=1;o<=N;++o){ nb[(o-1)*SUBGRIDSIZE+0]=((o-2+N)%N)+1; nb[(o-1)*SUBGRIDSIZE+1]=o; nb[(o-1)*SUBGRIDSIZE+2]=(o%N)+1; }
        nb[(1-1)*SUBGRIDSIZE+0] = 9;                               // oct1 -x neighbour = cache oct (idx 9 > ngridmax)

        MgParams P{}; P.head_idx=1; P.num_octs=N; P.ngridmax=N; P.dx=1.0f; P.fourpi=1.0f; P.offset=0.0f; P.vol_loc=1.0f;
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:pso];
          [e setBuffer:bphi offset:0 atIndex:0]; [e setBuffer:brho offset:0 atIndex:1];
          [e setBuffer:bf offset:0 atIndex:2]; [e setBuffer:bnb offset:0 atIndex:3];
          [e setBytes:&P length:sizeof(P) atIndex:4];
          [e dispatchThreads:MTLSizeMake(TWOTONDIM,N,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,N,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        // oct1 cell1 has the -x cache boundary; oct4 cell1 is pure interior.
        float S_bnd = f[IDX3(1,2,1)], S_int = f[IDX3(1,2,4)];
        float exp_int = R;                                        // fourpi*(rho/vol - offset) = R
        float exp_bnd = R - 2.0f*(0.5f*PC + 0.5f*phic);           // - 2*oneoverdx2*phi_b
        printf("interior S  = %.5f  (exp %.5f)\n", S_int, exp_int);
        printf("boundary S  = %.5f  (exp %.5f, uses cache-oct phi)\n", S_bnd, exp_bnd);
        bool ok = fabsf(S_int-exp_int)<1e-5f && fabsf(S_bnd-exp_bnd)<1e-5f;
        printf("%s\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}

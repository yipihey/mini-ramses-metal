// Unit test for gradient_phi (mg.metal) on a 1D periodic uniform grid.
// phi[g]=cos(2 pi g/NC).  The gg/hh stencil reduces (for a uniform grid) to the
// flat 4th-order central difference: f[g] = a*(phi[g-1]-phi[g+1]) - b*(phi[g-2]-phi[g+2]),
// a=1/2*4/3/dx, b=1/4*1/3/dx.  Compare gradient_phi output to that, computed on the host
// from the SAME float phi -> validates the materialised-neighbour read + gg/hh stencil.
// Build: mg.metal -DNDIM=1 -> /tmp/test_mg1d.metallib (as in test_residual).
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
        id<MTLFunction> fn=[lib newFunctionWithName:@"gradient_phi"];
        if(!fn){fprintf(stderr,"no gradient_phi\n");return 2;}
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:fn error:&err];
        if(!pso){fprintf(stderr,"pso: %s\n",err.localizedDescription.UTF8String);return 2;}

        const int N=8, NC=N*2;
        id<MTLBuffer> bphi=[dev newBufferWithLength:N*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bf  =[dev newBufferWithLength:N*NF*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bnb =[dev newBufferWithLength:N*SUBGRIDSIZE*sizeof(int)   options:MTLResourceStorageModeShared];
        memset(bf.contents,0,N*NF*TWOTONDIM*sizeof(float));
        float* phi=(float*)bphi.contents; float* f=(float*)bf.contents; int* nb=(int*)bnb.contents;
        auto G=[&](int o,int c){return (o-1)*2+(c-1);};
        float phg[64];
        for (int o=1;o<=N;++o) for (int c=1;c<=TWOTONDIM;++c){ float v=cosf(2.0f*M_PI*(float)G(o,c)/(float)NC); phi[IDX2(c,o)]=v; phg[G(o,c)]=v; }
        for (int o=1;o<=N;++o){ nb[(o-1)*SUBGRIDSIZE+0]=((o-2+N)%N)+1; nb[(o-1)*SUBGRIDSIZE+1]=o; nb[(o-1)*SUBGRIDSIZE+2]=(o%N)+1; }

        MgParams P{}; P.head_idx=1; P.num_octs=N; P.ngridmax=N; P.dx=1.0f;
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:pso];
          [e setBuffer:bphi offset:0 atIndex:0]; [e setBuffer:bf offset:0 atIndex:1]; [e setBuffer:bnb offset:0 atIndex:2];
          [e setBytes:&P length:sizeof(P) atIndex:3];
          [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(N,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        float a=(0.5f*4.0f/3.0f)/1.0f, b=(0.25f*1.0f/3.0f)/1.0f, dmax=0.0f;
        for (int o=1;o<=N;++o) for (int c=1;c<=TWOTONDIM;++c){
            int g=G(o,c);
            float fexp = a*(phg[(g-1+NC)%NC]-phg[(g+1)%NC]) - b*(phg[(g-2+NC)%NC]-phg[(g+2)%NC]);
            dmax = fmaxf(dmax, fabsf(f[IDX3(c,1,o)] - fexp));
        }
        printf("gradient_phi vs flat 4th-order: max|f - fexp| = %.3e\n", dmax);
        bool ok = dmax < 1e-5f;
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

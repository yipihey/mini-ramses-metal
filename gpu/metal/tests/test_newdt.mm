// Unit test for newdt_part_reduce (part.metal), NDIM=1.  Known velocities ->
// vmax = max|v|, ekin = sum 0.5*m*v^2.  Validates the NDIM-generic velocity loop +
// the vmax atomic-max and the fixed-point ekin accumulation.
// Build: part.metal -DNDIM=1 -> metallib; kernel newdt_part_reduce.
#import <Metal/Metal.h>
#include "ramses_metal.h"     // IDXP, FP_SHIFT_RHO, ScanParams (-DNDIM=1)
#include <cstdio>
#include <cstring>
#include <cmath>

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_part1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLFunction> fn=[lib newFunctionWithName:@"newdt_part_reduce"];
        if(!fn){fprintf(stderr,"no newdt_part_reduce\n");return 2;}
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:fn error:&err];
        if(!pso){fprintf(stderr,"pso: %s\n",err.localizedDescription.UTF8String);return 2;}

        const int np=5, npm=8;                       // 5 particles, npartmax=8
        float vels[np] = {0.1f, -0.3f, 0.2f, 0.5f, -0.4f};
        id<MTLBuffer> bvp=[dev newBufferWithLength:npm*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bmp=[dev newBufferWithLength:npm*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bred=[dev newBufferWithLength:3*sizeof(uint) options:MTLResourceStorageModeShared];
        memset(bvp.contents,0,npm*sizeof(float)); memset(bred.contents,0,3*sizeof(uint));
        float* vp=(float*)bvp.contents; float* mp=(float*)bmp.contents;
        for (int i=0;i<np;++i){ vp[IDXP(i+1,1,npm)] = vels[i]; mp[i]=1.0f; }

        ScanParams P{}; P.head_idx=1; P.n=np; P.npartmax=npm;
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:pso];
          [e setBuffer:bvp offset:0 atIndex:0]; [e setBuffer:bmp offset:0 atIndex:1]; [e setBuffer:bred offset:0 atIndex:2];
          [e setBytes:&P length:sizeof(P) atIndex:3];
          [e dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
          [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        uint* red=(uint*)bred.contents;
        float vmax; memcpy(&vmax, &red[0], 4);                       // atomic_max_f bit pattern
        long q = (long)(((unsigned long)red[2] << 32) | (unsigned long)red[1]);
        double ekin = (double)q / (double)(1L << FP_SHIFT_RHO);
        double vexp=0, eexp=0; for (int i=0;i<np;++i){ vexp=fmax(vexp,fabs(vels[i])); eexp+=0.5*vels[i]*vels[i]; }
        printf("vmax = %.5f (exp %.5f);  ekin = %.6f (exp %.6f)\n", vmax, vexp, ekin, eexp);
        bool ok = fabs((double)vmax-vexp)<1e-6 && fabs(ekin-eexp)<1e-5;
        printf("%s\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}

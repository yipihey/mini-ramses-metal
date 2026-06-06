// Unit test for residual_norm (reduce.metal).  f(:,1)=r with one large + many
// small entries: r=[1e4,1,1,...] -> r^2=[1e8,1,1,...], Sum r^2 = 1e8 + (total-1).
// fp32 loses the +1s (1e8 ulp 8); df64 keeps them.  f(:,3)=1 (all unmasked).
// Build:
//   xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -DNDIM=3 -I.. -I../.. -c ../reduce.metal -o /tmp/red.air
//   xcrun -sdk macosx metallib /tmp/red.air -o /tmp/test_red.metallib
//   clang++ -fobjc-arc -O2 -DNDIM=3 -I../.. test_resnorm.mm -framework Metal -framework Foundation -o /tmp/test_resnorm
//   /tmp/test_resnorm /tmp/test_red.metallib
#import <Metal/Metal.h>
#include "ramses_metal.h"     // IDX3, NF, TWOTONDIM, MgParams  (-DNDIM=3)
#include <cstdio>
#include <cstring>
#include <cmath>

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_red.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"residual_norm"];
        if (!fn) { fprintf(stderr, "no residual_norm\n"); return 2; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        const int noct = 4, total = noct * TWOTONDIM;
        id<MTLBuffer> bf    = [dev newBufferWithLength:noct*NF*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bpart = [dev newBufferWithLength:2*sizeof(float) options:MTLResourceStorageModeShared];
        memset(bf.contents, 0, noct*NF*TWOTONDIM*sizeof(float));
        float* f = (float*)bf.contents;
        int gcell = 0;
        for (int o = 1; o <= noct; ++o)
            for (int c = 1; c <= TWOTONDIM; ++c) {
                f[IDX3(c, 1, o)] = (gcell == 0) ? 1.0e4f : 1.0f;   // r:  one 1e4, rest 1
                f[IDX3(c, 3, o)] = 1.0f;                            // mask: all unmasked
                ++gcell;
            }

        MgParams P{}; P.head_idx = 1; P.num_octs = noct;
        ((float*)bpart.contents)[0] = 0.0f; ((float*)bpart.contents)[1] = 0.0f;

        id<MTLCommandBuffer> cb = [qu commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bf    offset:0 atIndex:0];
        [e setBuffer:bpart offset:0 atIndex:1];
        [e setBytes:&P length:sizeof(P) atIndex:2];
        [e dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(total,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

        double norm = (double)((float*)bpart.contents)[0] + (double)((float*)bpart.contents)[1];
        double ref  = 1.0e8 + (double)(total - 1);            // 1e8 + 31
        printf("residual_norm (df64) = %.1f   reference = %.1f   err=%.3e\n", norm, ref, fabs(norm - ref));
        bool ok = fabs(norm - ref) < 1.0;
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

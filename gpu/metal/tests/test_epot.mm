// Unit test for cmp_epot (mg.metal).  noct octs, all cells unrefined, force
// components f(:,d)=d  ->  |f|^2 = 1+4+9 = 14 per cell (NDIM=3).
// epot partial (df64) must equal num_octs*TWOTONDIM*14.  Also test the
// refined-cell skip: mark one oct's cells refined -> excluded.
// Build:
//   xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -DNDIM=3 -I.. -I../.. -c ../mg.metal -o /tmp/mg.air
//   xcrun -sdk macosx metallib /tmp/mg.air -o /tmp/test_mg.metallib
//   clang++ -fobjc-arc -O2 -DNDIM=3 -I../.. test_epot.mm -framework Metal -framework Foundation -o /tmp/test_epot
//   /tmp/test_epot /tmp/test_mg.metallib
#import <Metal/Metal.h>
#include "ramses_metal.h"     // Oct, MgParams, IDX3, NF, TWOTONDIM  (-I../.. , -DNDIM=3)
#include <cstdio>
#include <cstring>
#include <cmath>

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_mg.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"cmp_epot"];
        if (!fn) { fprintf(stderr, "no cmp_epot kernel\n"); return 2; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        const int noct  = 4;
        const int total = noct * TWOTONDIM;
        const int nrefined_oct = 1;                 // oct #4 marked refined -> excluded

        id<MTLBuffer> bgrid = [dev newBufferWithLength:noct*sizeof(Oct) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bf    = [dev newBufferWithLength:noct*NF*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bpart = [dev newBufferWithLength:2*sizeof(float) options:MTLResourceStorageModeShared];  // 1 df64
        memset(bgrid.contents, 0, noct*sizeof(Oct));
        memset(bf.contents,    0, noct*NF*TWOTONDIM*sizeof(float));
        Oct* grid = (Oct*)bgrid.contents;
        float* f  = (float*)bf.contents;
        for (int o = 1; o <= noct; ++o)
            for (int c = 1; c <= TWOTONDIM; ++c) {
                for (int d = 1; d <= NDIM; ++d) f[IDX3(c, d, o)] = (float)d;   // |f|^2 = 1+4+9 = 14
                if (o > noct - nrefined_oct) grid[o-1].refined[c-1] = 1;       // last oct refined
            }

        MgParams P{}; P.head_idx = 1; P.num_octs = noct;
        ((float*)bpart.contents)[0] = 0.0f; ((float*)bpart.contents)[1] = 0.0f;

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bgrid offset:0 atIndex:0];
        [e setBuffer:bf    offset:0 atIndex:1];
        [e setBuffer:bpart offset:0 atIndex:2];
        [e setBytes:&P length:sizeof(P) atIndex:3];
        [e dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(total,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

        double epot = (double)((float*)bpart.contents)[0] + (double)((float*)bpart.contents)[1];
        double ref  = (double)((noct - nrefined_oct) * TWOTONDIM) * 14.0;   // refined oct excluded
        printf("cmp_epot partial = %.6f   reference = %.6f   err=%.3e\n", epot, ref, fabs(epot - ref));
        bool ok = fabs(epot - ref) < 1e-3;
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

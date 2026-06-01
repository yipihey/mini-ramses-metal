// Unit test for cmp_residual (mg.metal) on a 1D periodic uniform grid.
// Set f(:,2) = discrete Laplacian of a known phi (flat periodic, dx=1):
//   S[g] = phi[g-1] + phi[g+1] - 2*phi[g].
// Then the kernel's residual r = -(sum_nb - 2*phi) + S must be ~0 everywhere,
// which validates the residual operator AND the MG_iii/MG_hhh/mg_nbor neighbour
// connectivity (cell c of oct o must resolve to global cells g-1, g+1).
// Build:
//   xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -DNDIM=1 -I.. -I../.. -c ../mg.metal -o /tmp/mg1d.air
//   xcrun -sdk macosx metallib /tmp/mg1d.air -o /tmp/test_mg1d.metallib
//   clang++ -fobjc-arc -O2 -DNDIM=1 -I../.. test_residual.mm -framework Metal -framework Foundation -o /tmp/test_residual
//   /tmp/test_residual /tmp/test_mg1d.metallib
#import <Metal/Metal.h>
#include "ramses_metal.h"     // IDX2, IDX3, NF, TWOTONDIM, SUBGRIDSIZE, MgParams (-DNDIM=1)
#include <cstdio>
#include <cstring>
#include <cmath>

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_mg1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"cmp_residual"];
        if (!fn) { fprintf(stderr, "no cmp_residual\n"); return 2; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        const int N = 8;                 // octs (1D), 2 cells each -> 16 cells
        const int NC = N * 2;            // global cells
        id<MTLBuffer> bphi = [dev newBufferWithLength:N*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bf   = [dev newBufferWithLength:N*NF*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bnb  = [dev newBufferWithLength:N*SUBGRIDSIZE*sizeof(int)   options:MTLResourceStorageModeShared];
        memset(bf.contents, 0, N*NF*TWOTONDIM*sizeof(float));
        float* phi = (float*)bphi.contents;
        float* f   = (float*)bf.contents;
        int*   nb  = (int*)bnb.contents;

        // phi[g] = cos(2 pi g / NC), g = (o-1)*2 + (cell-1)
        auto G = [&](int o, int c){ return (o-1)*2 + (c-1); };
        float phg[64];
        for (int o = 1; o <= N; ++o)
            for (int c = 1; c <= TWOTONDIM; ++c) {
                float v = cosf(2.0f*M_PI*(float)G(o,c)/(float)NC);
                phi[IDX2(c,o)] = v; phg[G(o,c)] = v;
            }
        // nbor (1D 3-cube): offset 0=left, 1=self, 2=right (periodic, 1-based octs)
        for (int o = 1; o <= N; ++o) {
            nb[(o-1)*SUBGRIDSIZE + 0] = ((o-2+N)%N) + 1;
            nb[(o-1)*SUBGRIDSIZE + 1] = o;
            nb[(o-1)*SUBGRIDSIZE + 2] = (o%N) + 1;
        }
        // f(:,3)=mask=1 ; f(:,2)=S = phi[g-1]+phi[g+1]-2 phi[g]  (flat periodic, dx=1)
        for (int o = 1; o <= N; ++o)
            for (int c = 1; c <= TWOTONDIM; ++c) {
                int g = G(o,c);
                f[IDX3(c,3,o)] = 1.0f;
                f[IDX3(c,2,o)] = phg[(g-1+NC)%NC] + phg[(g+1)%NC] - 2.0f*phg[g];
            }

        MgParams P{}; P.head_idx = 1; P.num_octs = N; P.ngridmax = N; P.dx = 1.0f;

        id<MTLCommandBuffer> cb = [qu commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bphi offset:0 atIndex:0];
        [e setBuffer:bf   offset:0 atIndex:1];
        [e setBuffer:bnb  offset:0 atIndex:2];
        [e setBytes:&P length:sizeof(P) atIndex:3];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM, N, 1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM, N, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

        float rmax = 0.0f;
        for (int o = 1; o <= N; ++o)
            for (int c = 1; c <= TWOTONDIM; ++c)
                rmax = fmaxf(rmax, fabsf(f[IDX3(c,1,o)]));
        printf("cmp_residual: max|r| = %.3e  (S = discrete Laplacian -> residual must be ~0)\n", rmax);
        bool ok = rmax < 1e-4f;
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

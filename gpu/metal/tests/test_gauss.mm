// Unit test for gauss_seidel (mg.metal) on a 1D periodic uniform grid.
// phi = cos(2 pi g / NC) is the EXACT solution of L phi = S with S = discrete
// Laplacian (dx=1).  One red + one black GS sweep must leave phi UNCHANGED
// (the exact solution is the GS fixed point): phi_new=(sum_nb - dx2*S)/twondim = phi.
// Validates the smoother formula + red/black indexing + connectivity vs CUDA.
// Build: same as test_residual (mg.metal -DNDIM=1 -> metallib), kernel gauss_seidel.
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
        id<MTLFunction> fn = [lib newFunctionWithName:@"gauss_seidel"];
        if (!fn) { fprintf(stderr, "no gauss_seidel\n"); return 2; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        const int N = 8, NC = N * 2;
        id<MTLBuffer> bphi = [dev newBufferWithLength:N*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bf   = [dev newBufferWithLength:N*NF*TWOTONDIM*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bnb  = [dev newBufferWithLength:N*SUBGRIDSIZE*sizeof(int)   options:MTLResourceStorageModeShared];
        memset(bf.contents, 0, N*NF*TWOTONDIM*sizeof(float));
        float* phi = (float*)bphi.contents; float* f = (float*)bf.contents; int* nb = (int*)bnb.contents;
        auto G = [&](int o, int c){ return (o-1)*2 + (c-1); };
        float phg[64], phi0[64];
        for (int o = 1; o <= N; ++o) for (int c = 1; c <= TWOTONDIM; ++c) {
            float v = cosf(2.0f*M_PI*(float)G(o,c)/(float)NC);
            phi[IDX2(c,o)] = v; phg[G(o,c)] = v; phi0[IDX2(c,o)] = v;
        }
        for (int o = 1; o <= N; ++o) {
            nb[(o-1)*SUBGRIDSIZE+0] = ((o-2+N)%N)+1; nb[(o-1)*SUBGRIDSIZE+1] = o; nb[(o-1)*SUBGRIDSIZE+2] = (o%N)+1;
        }
        for (int o = 1; o <= N; ++o) for (int c = 1; c <= TWOTONDIM; ++c) {
            int g = G(o,c); f[IDX3(c,3,o)] = 1.0f;
            f[IDX3(c,2,o)] = phg[(g-1+NC)%NC] + phg[(g+1)%NC] - 2.0f*phg[g];
        }

        MgParams P{}; P.head_idx = 1; P.num_octs = N; P.ngridmax = N; P.dx = 1.0f;
        int safe = 0;
        const int nrb = TWOTONDIM / 2;                  // red (or black) cells per oct (=1 in 1D)

        for (int step = 0; step < 2; ++step) {          // red then black
            P.redstep = (step == 0) ? 1 : 0;
            id<MTLCommandBuffer> cb = [qu commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:bphi offset:0 atIndex:0];
            [e setBuffer:bf   offset:0 atIndex:1];
            [e setBuffer:bnb  offset:0 atIndex:2];
            [e setBytes:&P length:sizeof(P) atIndex:3];
            [e setBytes:&safe length:sizeof(int) atIndex:4];
            [e dispatchThreads:MTLSizeMake(nrb, N, 1) threadsPerThreadgroup:MTLSizeMake(nrb, N, 1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        float dmax = 0.0f;
        for (int o = 1; o <= N; ++o) for (int c = 1; c <= TWOTONDIM; ++c)
            dmax = fmaxf(dmax, fabsf(phi[IDX2(c,o)] - phi0[IDX2(c,o)]));
        printf("gauss_seidel fixed-point: max|phi_after - phi_exact| = %.3e\n", dmax);
        bool ok = dmax < 1e-4f;
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

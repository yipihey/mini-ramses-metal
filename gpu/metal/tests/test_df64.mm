// Self-contained Metal unit-test harness for df64.h.  Loads a test metallib,
// runs test_df64_sum, checks df64 >> fp32 accuracy vs the fp64 reference.
// Build:
//   xcrun -sdk macosx metal   -I .. -c test_df64.metal -o /tmp/test_df64.air
//   xcrun -sdk macosx metallib /tmp/test_df64.air -o /tmp/test_df64.metallib
//   clang++ -fobjc-arc -O2 test_df64.mm -framework Metal -framework Foundation -o /tmp/test_df64
//   /tmp/test_df64 /tmp/test_df64.metallib
#import <Metal/Metal.h>
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_df64.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"test_df64_sum"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        int   n = 4000000;
        float a = 1.0f / 3.0f;     // not representable -> fp32 sum drifts
        id<MTLBuffer> bin  = [dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [dev newBufferWithLength:4*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bn   = [dev newBufferWithLength:sizeof(int)   options:MTLResourceStorageModeShared];
        ((float*)bin.contents)[0] = a;
        ((int*)bn.contents)[0]    = n;

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bin offset:0 atIndex:0];
        [e setBuffer:bout offset:0 atIndex:1];
        [e setBuffer:bn offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

        float dfsum = ((float*)bout.contents)[0];
        float f32   = ((float*)bout.contents)[1];
        float php   = ((float*)bout.contents)[2];
        float plo   = ((float*)bout.contents)[3];
        double ref  = (double)n * (double)a;
        double prodref = (double)a * (double)a;

        double df_err  = fabs((double)dfsum - ref);
        double f32_err = fabs((double)f32   - ref);
        double prod_err= fabs(((double)php + (double)plo) - prodref);

        printf("N=%d  a=1/3\n", n);
        printf("  reference   = %.6f\n", ref);
        printf("  df64 sum    = %.6f   err=%.3e\n", dfsum, df_err);
        printf("  fp32 sum    = %.6f   err=%.3e\n", f32, f32_err);
        printf("  two_prod err= %.3e (hi+lo vs a*a)\n", prod_err);

        // df64 must be far more accurate than fp32, and two_prod near-exact.
        bool ok = (df_err < 1.0) && (f32_err > 100.0) && (prod_err < 1e-9);
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

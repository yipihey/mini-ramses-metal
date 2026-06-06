// Harness for the df64 block reduce.  256 threads, one threadgroup.
//   in = [1e7, 1, 1, ..., 1]  -> ref = 1e7 + 255.
// Build (see test_df64.mm): xcrun metal -fno-fast-math -I.. -c test_reduce.metal ...
#import <Metal/Metal.h>
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_reduce.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"test_reduce_df64"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        const int N = 256;
        id<MTLBuffer> bin  = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [dev newBufferWithLength:3*sizeof(float) options:MTLResourceStorageModeShared];
        float* in = (float*)bin.contents;
        in[0] = 1.0e8f;                              // ulp(1e8)=8 -> fp32 loses the +1s
        for (int i = 1; i < N; ++i) in[i] = 1.0f;

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bin offset:0 atIndex:0];
        [e setBuffer:bout offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(N,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

        double ref   = 1.0e8 + (double)(N - 1);          // 100000255
        double dfsum = (double)((float*)bout.contents)[0] + (double)((float*)bout.contents)[1]; // hi+lo
        float  f32   = ((float*)bout.contents)[2];
        double df_err  = fabs(dfsum - ref);
        double f32_err = fabs((double)f32 - ref);
        printf("reference   = %.1f\n", ref);
        printf("df64 reduce = %.1f   err=%.3e\n", dfsum, df_err);
        printf("fp32 reduce = %.1f   err=%.3e\n", (double)f32, f32_err);
        // df64 must recover the true sum (err ~ 0) and be strictly better than fp32.
        bool ok = (df_err < 1.0) && (f32_err > df_err);
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}

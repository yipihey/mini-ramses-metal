// Unit test for the CIC force gather + leapfrog kick (part.metal:
// gather_cic_force via kick_drift_part), NDIM=1.  This is the *reaction* half of
// the PM action-reaction; the deposit (cic_part_warp) is the action half.
//
// Partition-of-unity / adjoint check: a UNIFORM force field f==C must gather to
// exactly C (the CIC corner weights sum to 1, same weights as the deposit).
// With vp=0, action=1 (level-transition half-kick), dteff=dtnew=2:
//     vp_new = ff * 0.5 * dteff = ff = C.
// So recovering vp==C proves the gather returns the correct CIC-weighted value
// and that gather/deposit share the same weights (=> momentum conserving).
//
// Same 2-oct/level-2 grid + hash as test_cicdeposit.  Particle in cell1, frac0.3.
// Build: part.metal + hash.metal -DNDIM=1 -> one metallib.
#import <Metal/Metal.h>
#include "ramses_metal.h"     // Oct, PartParams, ConnParams, IDX3/IDXP, NBITS_POS
#include <cstdio>
#include <cstring>
#include <cmath>

static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, const char* n) {
    NSError* e=nil; id<MTLFunction> f=[l newFunctionWithName:@(n)];
    if(!f){fprintf(stderr,"no kernel %s\n",n);exit(2);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&e];
    if(!p){fprintf(stderr,"pso %s: %s\n",n,e.localizedDescription.UTF8String);exit(2);}
    return p;
}

int main(int argc, char** argv) {
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_pk1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState> psoHash = pso(dev,lib,"hash_insert");
        id<MTLComputePipelineState> psoKick = pso(dev,lib,"kick_drift_part");

        const int nlev=10, hsz=17, npm=4, ilevel=2, ncell=2;
        const long KEY_OFF=100, CKEY_MAX=2;
        const float C=0.5f;                          // uniform force field

        id<MTLBuffer> bgrid=[dev newBufferWithLength:2*sizeof(Oct) options:MTLResourceStorageModeShared];
        Oct* g=(Oct*)bgrid.contents; memset(g,0,2*sizeof(Oct));
        g[0].ckey[0]=0; g[0].lev=ilevel; g[1].ckey[0]=1; g[1].lev=ilevel;

        id<MTLBuffer> bckm=[dev newBufferWithLength:(nlev+1)*sizeof(int)  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bkof=[dev newBufferWithLength:(nlev+1)*sizeof(long) options:MTLResourceStorageModeShared];
        int* ckm=(int*)bckm.contents; long* kof=(long*)bkof.contents;
        memset(ckm,0,(nlev+1)*sizeof(int)); memset(kof,0,(nlev+1)*sizeof(long));
        ckm[ilevel]=(int)CKEY_MAX; kof[ilevel]=KEY_OFF;

        id<MTLBuffer> bhk=[dev newBufferWithLength:hsz*sizeof(long) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bhv=[dev newBufferWithLength:hsz*sizeof(int)  options:MTLResourceStorageModeShared];
        memset(bhk.contents,0,hsz*sizeof(long)); memset(bhv.contents,0,hsz*sizeof(int));

        // force field f(twotondim, ndim, ncell) == C everywhere
        const int fflat=TWOTONDIM*NF*ncell;
        id<MTLBuffer> bf=[dev newBufferWithLength:fflat*sizeof(float) options:MTLResourceStorageModeShared];
        { float* f=(float*)bf.contents; for(int i=0;i<fflat;++i) f[i]=C; }

        const double frac=0.3;
        long ipos1 = (long)llround((1.0+frac) * ldexp(1.0,(double)(NBITS_POS-ilevel)));
        id<MTLBuffer> bipos=[dev newBufferWithLength:npm*sizeof(long) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bvp  =[dev newBufferWithLength:npm*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> blev =[dev newBufferWithLength:npm*sizeof(int)  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bper =[dev newBufferWithLength:3*sizeof(int) options:MTLResourceStorageModeShared];
        memset(bipos.contents,0,npm*sizeof(long)); memset(bvp.contents,0,npm*sizeof(float));
        ((long*)bipos.contents)[IDXP(1,1,npm)] = ipos1;
        ((int*)blev.contents)[0]=ilevel;             // levelp[ipart-1]
        { int* p=(int*)bper.contents; p[0]=1; p[1]=0; p[2]=0; }

        ConnParams CP{}; CP.num_octs=2; CP.hash_size=hsz; CP.nlevelmax=nlev; CP.head_idx=1; CP.per[0]=1;
        PartParams P{}; P.dtnew=2.0f; P.dtold=2.0f; P.hash_size=hsz; P.ilevel=ilevel;
        P.head_idx=1; P.num_parts=1; P.npartmax=npm; P.ngridmax=2; P.action_part=1;  // half-kick only

        // 1) build the hash
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoHash];
          [e setBuffer:bgrid offset:0 atIndex:0]; [e setBuffer:bhk offset:0 atIndex:1]; [e setBuffer:bhv offset:0 atIndex:2];
          [e setBuffer:bckm offset:0 atIndex:3]; [e setBuffer:bkof offset:0 atIndex:4]; [e setBytes:&CP length:sizeof(CP) atIndex:5];
          [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)]; [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        // 2) kick (gathers the force, applies half-kick)
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoKick];
          [e setBuffer:bipos offset:0 atIndex:0]; [e setBuffer:bvp offset:0 atIndex:1]; [e setBuffer:blev offset:0 atIndex:2];
          [e setBuffer:bf offset:0 atIndex:3]; [e setBuffer:bhk offset:0 atIndex:4]; [e setBuffer:bhv offset:0 atIndex:5];
          [e setBuffer:bckm offset:0 atIndex:6]; [e setBuffer:bkof offset:0 atIndex:7]; [e setBuffer:bper offset:0 atIndex:8];
          [e setBytes:&P length:sizeof(P) atIndex:9];
          [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        float vp = ((float*)bvp.contents)[IDXP(1,1,npm)];
        printf("gathered force (vp after half-kick, dt=2) = %.6f  (exp %.6f = uniform C)\n", vp, C);
        bool ok = fabs(vp - C) < 1e-5f;
        printf("%s\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}

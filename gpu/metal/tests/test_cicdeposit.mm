// Unit test for the CIC mass deposit (rho.metal: build_src_part + cic_part_warp),
// NDIM=1.  The deposit is the momentum-critical kernel (adjoint of gather_cic_force).
// Validates: (1) correct cell-centered CIC weights, (2) mass conservation
// (sum rho == mp), (3) the segmented warp-scan + fixed-point tail round-trip,
// (4) hash_insert + build_src_part feeding the deposit (the real pipeline).
//
// Setup: ilevel=2 -> 4 cells (0..3) = 2 octs (oct0=cells0,1; oct1=cells2,3),
// periodic.  One particle (mp=1) sits in cell 1 of oct0 at sub-cell frac=0.3.
// Cell-centered CIC: wx=[0.5-0.3, 1-|0.3-0.5|, max(0,0.3-0.5)] = [0.2, 0.8, 0].
//   k=1 ox=-1 -> tgt cell0 (oct0,cell1)  w=0.2
//   k=2 ox= 0 -> tgt cell1 (oct0,cell2)  w=0.8
//   k=3 ox=+1 -> tgt cell2 (oct1,cell1)  w=0   (skipped)
// => rho[IDX2(1,1)]=0.2, rho[IDX2(2,1)]=0.8, rest 0; total 1.0.
//
// Build: rho.metal + hash.metal -DNDIM=1 -> one metallib.
#import <Metal/Metal.h>
#include "ramses_metal.h"     // Oct, CicParams, BoxParams, ConnParams, IDX2/IDXP, NBITS_POS, FP_SHIFT_RHO
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
    const char* libpath = argc > 1 ? argv[1] : "/tmp/test_rho1d.metallib";
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState> psoHash = pso(dev,lib,"hash_insert");
        id<MTLComputePipelineState> psoSrc  = pso(dev,lib,"build_src_part");
        id<MTLComputePipelineState> psoCic  = pso(dev,lib,"cic_part_warp");

        const int nlev=10, hsz=17, npm=4, ilevel=2;
        const long KEY_OFF=100, CKEY_MAX=2;          // 2 octs/dim at level 2; nonzero key base

        // --- grid: 2 octs at level 2, ckey 0 and 1 ---
        id<MTLBuffer> bgrid=[dev newBufferWithLength:2*sizeof(Oct) options:MTLResourceStorageModeShared];
        Oct* g=(Oct*)bgrid.contents; memset(g,0,2*sizeof(Oct));
        g[0].ckey[0]=0; g[0].lev=ilevel;
        g[1].ckey[0]=1; g[1].lev=ilevel;

        // --- level-indexed ckey_max / key_off ---
        id<MTLBuffer> bckm=[dev newBufferWithLength:(nlev+1)*sizeof(int)  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bkof=[dev newBufferWithLength:(nlev+1)*sizeof(long) options:MTLResourceStorageModeShared];
        int* ckm=(int*)bckm.contents; long* kof=(long*)bkof.contents;
        memset(ckm,0,(nlev+1)*sizeof(int)); memset(kof,0,(nlev+1)*sizeof(long));
        ckm[ilevel]=(int)CKEY_MAX; kof[ilevel]=KEY_OFF;

        // --- hash table (host-zeroed; hash_insert fills it) ---
        id<MTLBuffer> bhk=[dev newBufferWithLength:hsz*sizeof(long) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bhv=[dev newBufferWithLength:hsz*sizeof(int)  options:MTLResourceStorageModeShared];
        memset(bhk.contents,0,hsz*sizeof(long)); memset(bhv.contents,0,hsz*sizeof(int));

        // --- particle: 1 particle, cell 1 (oct0), frac 0.3 ---
        const double frac=0.3;
        long ipos1 = (long)llround((1.0+frac) * ldexp(1.0,(double)(NBITS_POS-ilevel)));
        id<MTLBuffer> bipos=[dev newBufferWithLength:npm*sizeof(long) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bmp  =[dev newBufferWithLength:npm*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsort=[dev newBufferWithLength:npm*sizeof(int)  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bisp =[dev newBufferWithLength:(npm+1)*sizeof(int) options:MTLResourceStorageModeShared];
        memset(bipos.contents,0,npm*sizeof(long)); memset(bmp.contents,0,npm*sizeof(float));
        memset(bsort.contents,0,npm*sizeof(int));  memset(bisp.contents,0,(npm+1)*sizeof(int));
        ((long*)bipos.contents)[IDXP(1,1,npm)] = ipos1;
        ((float*)bmp.contents)[0] = 1.0f;            // mp[ipart-1], ipart=1
        ((int*)bsort.contents)[0] = 1;               // sortp[head_idx-1]=particle 1

        // --- rho accumulators: twotondim*ncell = 2*2 = 4 ---
        const int ncell=2, nflat=TWOTONDIM*ncell;
        auto mkz=[&](int n){ id<MTLBuffer> b=[dev newBufferWithLength:n*sizeof(uint) options:MTLResourceStorageModeShared]; memset(b.contents,0,n*sizeof(uint)); return b; };
        id<MTLBuffer> brl=mkz(nflat), brh=mkz(nflat), bnl=mkz(nflat), bnh=mkz(nflat);

        // --- params ---
        ConnParams CP{}; CP.num_octs=2; CP.hash_size=hsz; CP.nlevelmax=nlev; CP.head_idx=1; CP.per[0]=1;
        CicParams P{}; P.fp_scale_rho=ldexp(1.0f,FP_SHIFT_RHO); P.key_off=KEY_OFF; P.ckey_max=(int)CKEY_MAX;
        P.hash_size=hsz; P.ilevel=ilevel; P.head_idx=1; P.num_parts=1; P.npartmax=npm; P.m_refine=-1.0f; // skip nref
        BoxParams BX{}; BX.box_min[0]=0; BX.box_max[0]=(1<<ilevel); BX.periodic[0]=1;

        // 1) hash_insert
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoHash];
          [e setBuffer:bgrid offset:0 atIndex:0]; [e setBuffer:bhk offset:0 atIndex:1]; [e setBuffer:bhv offset:0 atIndex:2];
          [e setBuffer:bckm offset:0 atIndex:3]; [e setBuffer:bkof offset:0 atIndex:4]; [e setBytes:&CP length:sizeof(CP) atIndex:5];
          [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)]; [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        // 2) build_src_part
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoSrc];
          [e setBuffer:bipos offset:0 atIndex:0]; [e setBuffer:bsort offset:0 atIndex:1]; [e setBuffer:bisp offset:0 atIndex:2];
          [e setBuffer:bhk offset:0 atIndex:3]; [e setBuffer:bhv offset:0 atIndex:4]; [e setBytes:&P length:sizeof(P) atIndex:5];
          [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e endEncoding];[cb commit];[cb waitUntilCompleted]; }
        int combined=((int*)bisp.contents)[1];       // isp_swap[ipart=1]

        // 3) cic_part_warp  (1 threadgroup of 32 = 1 simdgroup; only lane 0 valid)
        { id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
          [e setComputePipelineState:psoCic];
          [e setBuffer:bsort offset:0 atIndex:0]; [e setBuffer:bisp offset:0 atIndex:1]; [e setBuffer:bgrid offset:0 atIndex:2];
          [e setBuffer:bhk offset:0 atIndex:3]; [e setBuffer:bhv offset:0 atIndex:4]; [e setBuffer:bipos offset:0 atIndex:5];
          [e setBuffer:bmp offset:0 atIndex:6]; [e setBuffer:brl offset:0 atIndex:7]; [e setBuffer:brh offset:0 atIndex:8];
          [e setBuffer:bnl offset:0 atIndex:9]; [e setBuffer:bnh offset:0 atIndex:10];
          [e setBytes:&P length:sizeof(P) atIndex:11]; [e setBytes:&BX length:sizeof(BX) atIndex:12];
          [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e endEncoding];[cb commit];[cb waitUntilCompleted]; }

        uint* rl=(uint*)brl.contents; uint* rh=(uint*)brh.contents;
        double rho[4], total=0;
        for (int i=0;i<nflat;++i){ long q=(long)(((unsigned long)rh[i]<<32)|(unsigned long)rl[i]); rho[i]=(double)q/ldexp(1.0,FP_SHIFT_RHO); total+=rho[i]; }
        double exp_[4]={0.2,0.8,0.0,0.0};
        printf("combined=%d (igrid=%d icell=%d)\n", combined, combined>>5, combined&31);
        printf("rho = [%.5f %.5f %.5f %.5f]  (exp [0.2 0.8 0 0])\n", rho[0],rho[1],rho[2],rho[3]);
        printf("total deposited = %.6f (exp 1.0 = mp)\n", total);
        bool ok = combined==((1<<5)|2) && fabs(total-1.0)<1e-5;
        for(int i=0;i<4;++i) ok = ok && fabs(rho[i]-exp_[i])<1e-5;
        printf("%s\n", ok?"PASS":"FAIL");
        return ok?0:1;
    }
}

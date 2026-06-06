// Adjoint (momentum-conservation) test of the CIC deposit vs the force gather at
// a COARSE-FINE BOUNDARY, NDIM=1.  This isolates the physical reason the GPU
// over-energizes at refined levels: the per-level net force Sum f*rho is nonzero
// (measured ~1e-7 at refined levels vs ~1e-12 at the base), which means the
// gather is not the exact transpose of the deposit at the boundary.
//
// Momentum theorem: total force = Sum_i m_i a(x_i) = Sum_i m_i Sum_c W_g(x_i,c) f_c.
// The self-force vanishes (Sum f*rho = 0) iff the gather weights W_g equal the
// deposit weights W_d for every particle and cell.  Equivalently, for any cell
// field v:   <v, D m>  ==  Sum_i m_i * gather_v(x_i)   <=>   W_g == W_d  (adjoint).
//
// We choose v = 1 on FINE (level-2) cells, 0 on COARSE (level-1) cells, and one
// particle of unit mass at the edge fine cell whose CIC stencil reaches a MISSING
// fine neighbour (the boundary):
//   LHS = <v, D m> = total mass the deposit places on FINE cells  (per-corner: keeps it)
//   RHS = gather_v(x) = the gathered FINE-indicator field          (all-or-nothing: 0,
//                       because a missing corner forces the whole force to COARSE)
// adjoint  <=> LHS == RHS.  defect = LHS - RHS.
//
// CONTROL: the same particle on a UNIFORM fine grid (the missing oct present) must
// be adjoint (defect ~ 0).  BOUNDARY: defect is O(1) -> the bug.
//
// Build: rho.metal + part.metal + hash.metal -DNDIM=1 -> one metallib.
#import <Metal/Metal.h>
#include "ramses_metal.h"
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

static id<MTLDevice> dev; static id<MTLCommandQueue> qu; static id<MTLLibrary> lib;
static id<MTLComputePipelineState> psoHash, psoSrc, psoCic, psoKick;

static id<MTLBuffer> buf(int nbytes){ id<MTLBuffer> b=[dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared]; memset(b.contents,0,nbytes); return b; }

// Run one case.  octsL2 = number of level-2 octs present (1 = boundary, 2 = uniform).
// Returns LHS (fine mass deposited) and RHS (gathered fine-indicator) via out[].
static void run_case(int octsL2, double frac, double out[2], const char* tag) {
    const int nlev=10, hsz=37, npm=4, ilevel=2;
    const int L1=ilevel-1;
    const long KOF1=100, KOF2=200;           // distinct key bases per level
    const int  CKM1=1,   CKM2=2;             // octs/dim: L1=2^0, L2=2^1
    // oct list: [0..octsL2-1] = L2 octs (ckey 0,1), then 1 coarse L1 oct (ckey 0).
    const int noct = octsL2 + 1;
    const int icoarse = octsL2;              // 0-based index of the coarse oct -> igrid=octsL2+1

    id<MTLBuffer> bgrid=buf(noct*sizeof(Oct)); Oct* g=(Oct*)bgrid.contents;
    for (int o=0;o<octsL2;++o){ g[o].ckey[0]=o; g[o].lev=ilevel; }
    g[icoarse].ckey[0]=0; g[icoarse].lev=L1;

    id<MTLBuffer> bckm=buf((nlev+1)*sizeof(int)); id<MTLBuffer> bkof=buf((nlev+1)*sizeof(long));
    int* ckm=(int*)bckm.contents; long* kof=(long*)bkof.contents;
    ckm[L1]=CKM1; kof[L1]=KOF1; ckm[ilevel]=CKM2; kof[ilevel]=KOF2;

    id<MTLBuffer> bhk=buf(hsz*sizeof(long)); id<MTLBuffer> bhv=buf(hsz*sizeof(int));

    // particle: unit mass, edge fine cell (cell index = octsL2*2 - 1 = last present fine cell), frac
    int edgecell = octsL2*2 - 1;             // boundary: cell1 (oct0); uniform: cell3 (oct1)
    long ipos1 = (long)llround(((double)edgecell + frac) * ldexp(1.0,(double)(NBITS_POS-ilevel)));
    id<MTLBuffer> bipos=buf(npm*sizeof(long)); id<MTLBuffer> bmp=buf(npm*sizeof(float));
    id<MTLBuffer> bsort=buf(npm*sizeof(int));  id<MTLBuffer> bisp=buf((npm+1)*sizeof(int));
    id<MTLBuffer> bvp=buf(npm*sizeof(float));  id<MTLBuffer> blev=buf(npm*sizeof(int));
    id<MTLBuffer> bper=buf(3*sizeof(int));
    ((long*)bipos.contents)[IDXP(1,1,npm)] = ipos1;
    ((float*)bmp.contents)[0]=1.0f; ((int*)bsort.contents)[0]=1; ((int*)blev.contents)[0]=ilevel;
    ((int*)bper.contents)[0]=1;

    const int nflat=TWOTONDIM*noct;
    id<MTLBuffer> brl=buf(nflat*sizeof(uint)),brh=buf(nflat*sizeof(uint)),bnl=buf(nflat*sizeof(uint)),bnh=buf(nflat*sizeof(uint));
    const int fflat=TWOTONDIM*NF*noct;
    id<MTLBuffer> bf=buf(fflat*sizeof(float)); float* f=(float*)bf.contents;
    // v = fine-indicator: f=1 on the L2 octs (igrid 1..octsL2), 0 on the coarse oct.
    for (int o=1;o<=octsL2;++o) for(int c=1;c<=TWOTONDIM;++c) f[IDX3(c,1,o)]=1.0f;

    ConnParams CP{}; CP.num_octs=noct; CP.hash_size=hsz; CP.nlevelmax=nlev; CP.head_idx=1; CP.per[0]=1;
    CicParams P{}; P.fp_scale_rho=ldexp(1.0f,FP_SHIFT_RHO); P.key_off=KOF2; P.ckey_max=CKM2;
    P.hash_size=hsz; P.ilevel=ilevel; P.head_idx=1; P.num_parts=1; P.npartmax=npm; P.m_refine=-1.0f;
    P.ngridmax=noct;   // real-oct bound; deposit skips dst_igrid>ngridmax (cache octs) and ==0 (missing)
    BoxParams BX{}; BX.box_min[0]=0; BX.box_max[0]=(1<<ilevel); BX.periodic[0]=1;
    PartParams PK{}; PK.dtnew=2.0f; PK.dtold=2.0f; PK.hash_size=hsz; PK.ilevel=ilevel;
    PK.head_idx=1; PK.num_parts=1; PK.npartmax=npm; PK.ngridmax=noct; PK.action_part=1;

    auto run=[&](id<MTLComputePipelineState> ps, void(^enc)(id<MTLComputeCommandEncoder>), int tgx, int tgt){
        id<MTLCommandBuffer> cb=[qu commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:ps]; enc(e);
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tgt,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; (void)tgx; };

    // hash_insert (all octs, both levels)
    run(psoHash,^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:bgrid offset:0 atIndex:0];[e setBuffer:bhk offset:0 atIndex:1];[e setBuffer:bhv offset:0 atIndex:2];
        [e setBuffer:bckm offset:0 atIndex:3];[e setBuffer:bkof offset:0 atIndex:4];[e setBytes:&CP length:sizeof(CP) atIndex:5];}, noct, noct);
    // build_src_part
    run(psoSrc,^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:bipos offset:0 atIndex:0];[e setBuffer:bsort offset:0 atIndex:1];[e setBuffer:bisp offset:0 atIndex:2];
        [e setBuffer:bhk offset:0 atIndex:3];[e setBuffer:bhv offset:0 atIndex:4];[e setBytes:&P length:sizeof(P) atIndex:5];},1,1);
    // cic_part_warp (deposit)
    run(psoCic,^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:bsort offset:0 atIndex:0];[e setBuffer:bisp offset:0 atIndex:1];[e setBuffer:bgrid offset:0 atIndex:2];
        [e setBuffer:bhk offset:0 atIndex:3];[e setBuffer:bhv offset:0 atIndex:4];[e setBuffer:bipos offset:0 atIndex:5];
        [e setBuffer:bmp offset:0 atIndex:6];[e setBuffer:brl offset:0 atIndex:7];[e setBuffer:brh offset:0 atIndex:8];
        [e setBuffer:bnl offset:0 atIndex:9];[e setBuffer:bnh offset:0 atIndex:10];
        [e setBytes:&P length:sizeof(P) atIndex:11];[e setBytes:&BX length:sizeof(BX) atIndex:12];},1,32);
    // kick (gather the fine-indicator field, half-kick -> vp = gathered value)
    run(psoKick,^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:bipos offset:0 atIndex:0];[e setBuffer:bvp offset:0 atIndex:1];[e setBuffer:blev offset:0 atIndex:2];
        [e setBuffer:bf offset:0 atIndex:3];[e setBuffer:bhk offset:0 atIndex:4];[e setBuffer:bhv offset:0 atIndex:5];
        [e setBuffer:bckm offset:0 atIndex:6];[e setBuffer:bkof offset:0 atIndex:7];[e setBuffer:bper offset:0 atIndex:8];
        [e setBytes:&PK length:sizeof(PK) atIndex:9];},1,1);

    // LHS = total mass deposited on FINE (L2) octs
    uint* rl=(uint*)brl.contents; uint* rh=(uint*)brh.contents;
    double fine_mass=0, all_mass=0;
    for (int o=1;o<=noct;++o) for(int c=1;c<=TWOTONDIM;++c){
        int idx=IDX2(c,o); long q=(long)(((unsigned long)rh[idx]<<32)|(unsigned long)rl[idx]);
        double m=(double)q/ldexp(1.0,FP_SHIFT_RHO); all_mass+=m; if(o<=octsL2) fine_mass+=m; }
    double rhs = ((float*)bvp.contents)[IDXP(1,1,npm)];     // gathered fine-indicator
    out[0]=fine_mass; out[1]=rhs;
    printf("  [%s] deposit: fine_mass=%.4f total=%.4f | gather(fine-indicator)=%.4f | defect=%.4f\n",
           tag, fine_mass, all_mass, rhs, fine_mass-rhs);
}

int main(int argc, char** argv) {
    const char* libpath = argc>1?argv[1]:"/tmp/test_adj1d.metallib";
    @autoreleasepool {
        dev=MTLCreateSystemDefaultDevice(); qu=[dev newCommandQueue];
        NSError* err=nil; lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if(!lib){fprintf(stderr,"lib: %s\n",err.localizedDescription.UTF8String);return 2;}
        psoHash=pso(dev,lib,"hash_insert"); psoSrc=pso(dev,lib,"build_src_part");
        psoCic=pso(dev,lib,"cic_part_warp"); psoKick=pso(dev,lib,"kick_drift_part");

        printf("Adjoint (momentum) test: <fine-indicator, deposit(m)> vs gather(fine-indicator)\n");
        printf("  adjoint  => LHS==RHS (defect 0).  net force ~ defect.\n");
        double ctrl[2], bnd[2];
        run_case(2, 0.7, ctrl, "UNIFORM ");   // both L2 octs present: no boundary
        run_case(1, 0.7, bnd,  "BOUNDARY");    // oct1 missing: edge particle straddles coarse-fine

        bool ctrl_adjoint = fabs(ctrl[0]-ctrl[1]) < 1e-4;
        bool bnd_nonadjoint = fabs(bnd[0]-bnd[1]) > 0.1 && bnd[0] > 0.1 && bnd[1] < 1e-3;
        printf("\nUNIFORM  adjoint (control passes): %s\n", ctrl_adjoint?"yes":"NO");
        printf("BOUNDARY non-adjoint (bug present): %s\n", bnd_nonadjoint?"yes":"no");
        if (ctrl_adjoint && bnd_nonadjoint) {
            printf("CONFIRMED: at a coarse-fine boundary the deposit keeps mass on the FINE\n"
                   "cells (per-corner) but the gather takes the force ENTIRELY from the COARSE\n"
                   "level (all-or-nothing) -> gather != deposit^T -> Sum f*rho != 0 -> the\n"
                   "refined-level momentum leak that drives the over-energization.\n");
            printf("PASS\n"); return 0;
        }
        printf("FAIL (mechanism not as hypothesised; inspect maps above)\n"); return 1;
    }
}

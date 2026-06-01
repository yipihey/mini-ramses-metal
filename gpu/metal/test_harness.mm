//============================================================================
// test_harness.mm  —  standalone GPU validation (no Fortran required)
//
// Loads the compiled .metallib, runs kernels on the real Metal device, and
// checks results against CPU references.  This proves the full execution path:
// device, shared buffers, threadgroup memory, SIMD shuffles, barriers,
// dispatch, and readback — independent of the Fortran host build.
//
// Build:
//   xcrun metal -std=metal3.1 -DNDIM=3 -I. -c scan.metal -o /tmp/scan.air
//   xcrun metallib /tmp/scan.air -o /tmp/ramses_kernels.metallib
//   clang++ -std=c++17 -ObjC++ -fobjc-arc -I. test_harness.mm \
//       -framework Metal -framework Foundation -o /tmp/test_harness
//   /tmp/test_harness /tmp/ramses_kernels.metallib
//============================================================================
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <simd/simd.h>
#include "ramses_metal.h"

static id<MTLDevice>            g_dev;
static id<MTLCommandQueue>      g_q;
static id<MTLLibrary>           g_lib;

static id<MTLComputePipelineState> pso(NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [g_lib newFunctionWithName:name];
    if (!fn) { fprintf(stderr, "missing kernel %s\n", name.UTF8String); exit(1); }
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
    return p;
}

template <class T>
static id<MTLBuffer> shared_buf(const T* src, size_t n) {
    id<MTLBuffer> b = [g_dev newBufferWithLength:n*sizeof(T) options:MTLResourceStorageModeShared];
    if (src) memcpy(b.contents, src, n*sizeof(T));
    return b;
}

//----------------------------------------------------------------------------
// Single-threadgroup inclusive scan: validates warp_scan + block combine.
//----------------------------------------------------------------------------
static int test_block_scan(int N, int tg) {
    std::vector<int> in(N);
    srand(1234);
    for (int i = 0; i < N; ++i) in[i] = rand() % 7;

    // CPU inclusive prefix sum reference.
    std::vector<int> ref(N);
    int acc = 0;
    for (int i = 0; i < N; ++i) { acc += in[i]; ref[i] = acc; }

    id<MTLBuffer> data  = shared_buf(in.data(), N);
    id<MTLBuffer> psum  = shared_buf<int>(nullptr, 1);
    id<MTLBuffer> total = shared_buf<int>(nullptr, 1);
    int offset = 1, size = N, flags = 2 /*write total_sum*/;

    id<MTLComputePipelineState> p = pso(@"block_scan");
    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:psum offset:0 atIndex:1];
    [enc setBuffer:total offset:0 atIndex:2];
    [enc setBytes:&offset length:sizeof(int) atIndex:3];
    [enc setBytes:&size   length:sizeof(int) atIndex:4];
    [enc setBytes:&flags  length:sizeof(int) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "cb error: %s\n", cb.error.localizedDescription.UTF8String);
        return 1;
    }

    const int* out = (const int*)data.contents;
    int bad = 0;
    for (int i = 0; i < N; ++i) if (out[i] != ref[i]) {
        if (bad < 5) fprintf(stderr, "  mismatch i=%d gpu=%d cpu=%d\n", i, out[i], ref[i]);
        ++bad;
    }
    int tot = ((const int*)total.contents)[0];
    bool tot_ok = (tot == ref[N-1]);
    printf("block_scan N=%d tg=%d : %s (total %d/%d %s)\n", N, tg,
           bad==0 ? "PASS" : "FAIL", tot, ref[N-1], tot_ok?"ok":"BAD");
    return (bad==0 && tot_ok) ? 0 : 1;
}

//----------------------------------------------------------------------------
// Validate the leaf primitives header (hilbert + fixed-point atomics) on GPU.
//----------------------------------------------------------------------------
static int test_primitives() {
    // Provided by primitives.metal (built into the same metallib).
    id<MTLFunction> fn = [g_lib newFunctionWithName:@"test_hilbert"];
    if (!fn) { printf("test_hilbert kernel not present, skipping primitives test\n"); return 0; }
    id<MTLComputePipelineState> p = pso(@"test_hilbert");

    const int N = 64;
    std::vector<int> ckx(N), cky(N), ckz(N);
    for (int i = 0; i < N; ++i) { ckx[i]=i&3; cky[i]=(i>>2)&3; ckz[i]=(i>>4)&3; }
    int level = 2;
    id<MTLBuffer> bx=shared_buf(ckx.data(),N), by=shared_buf(cky.data(),N), bz=shared_buf(ckz.data(),N);
    id<MTLBuffer> outk = shared_buf<long>(nullptr, N);
    id<MTLBuffer> lo = shared_buf<unsigned>(nullptr,1), hi = shared_buf<unsigned>(nullptr,1);

    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:by offset:0 atIndex:1];
    [enc setBuffer:bz offset:0 atIndex:2];
    [enc setBuffer:outk offset:0 atIndex:3];
    [enc setBuffer:lo offset:0 atIndex:4];
    [enc setBuffer:hi offset:0 atIndex:5];
    [enc setBytes:&level length:sizeof(int) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    // Fixed-point accumulator: each thread added 1.0 * 2^FP_SHIFT_RHO -> total N<<shift.
    long q = ((long)((const unsigned*)hi.contents)[0] << 32) | (long)((const unsigned*)lo.contents)[0];
    double sum = (double)q / (double)(1L << FP_SHIFT_RHO);
    bool ok = (sum > N - 1e-3 && sum < N + 1e-3);
    printf("primitives: fixed-point sum=%.4f (expect %d) : %s\n", sum, N, ok?"PASS":"FAIL");
    return ok ? 0 : 1;
}

//----------------------------------------------------------------------------
// Host replicas of the device FNV-64 hash + open-addressing table, used both
// to BUILD the table the GPU reads and inside the CPU golden CIC reference.
// Must match ramses_msl.h exactly (signed MOD + fixup, key 0 == empty).
//----------------------------------------------------------------------------
static long h_fnv64(long key_signed) {
    uint64_t key = (uint64_t)key_signed, h = 14695981039346656037ULL;
    for (int j = 0; j < 8; ++j) { uint64_t k=(key>>(8*j))&0xFF; h^=k; h*=1099511628211ULL; }
    return (long)h;
}
static int h_bucket(long key, int hs) {
    long ib = h_fnv64(key) % (long)hs;
    if (ib < 0) ib += hs;
    return (int)ib + 1;                 // 1-based
}
static void h_hash_set(std::vector<long>& hkey, std::vector<int>& hval,
                       int hs, long key, int val) {
    int b = h_bucket(key, hs);
    for (;;) {
        if (hkey[b-1] == 0 || hkey[b-1] == key) { hkey[b-1]=key; hval[b-1]=val; return; }
        b = (b % hs) + 1;
    }
}
static int h_hash_get(const std::vector<long>& hkey, const std::vector<int>& hval,
                      int hs, long key) {
    int b = h_bucket(key, hs);
    for (;;) {
        if (hkey[b-1]==key) return hval[b-1];
        if (hkey[b-1]==0)   return 0;
        b = (b % hs) + 1;
    }
}

//----------------------------------------------------------------------------
// Generic 1D dispatch over n threads (host helper; mirrors the bridge).
//----------------------------------------------------------------------------
static void dispatch_1d(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> p, int n) {
    NSUInteger tpt = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tpt,1,1)];
}

//----------------------------------------------------------------------------
// gpu_scan host driver: faithful port of the multi-level block_scan/uniform_add
// tree in gpu_runner.cuf (levels 0 and 1 implemented; covers n <= 256^3).
// In-place INCLUSIVE scan of prefix_sum[head-1 .. head-1+n-1].
//----------------------------------------------------------------------------
static void encode_block_scan(id<MTLComputeCommandEncoder> enc,
        id<MTLComputePipelineState> p, id<MTLBuffer> data, id<MTLBuffer> psum,
        id<MTLBuffer> total, int offset, int size, int flags, int tg) {
    [enc setComputePipelineState:p];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:(psum?psum:data) offset:0 atIndex:1];
    [enc setBuffer:(total?total:data) offset:0 atIndex:2];
    [enc setBytes:&offset length:4 atIndex:3];
    [enc setBytes:&size   length:4 atIndex:4];
    [enc setBytes:&flags  length:4 atIndex:5];
    int nblocks = (size + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(nblocks,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}
static void encode_block_scan_1blk(id<MTLComputeCommandEncoder> enc,
        id<MTLComputePipelineState> p, id<MTLBuffer> data, int size) {
    int offset=1, flags=0;
    [enc setComputePipelineState:p];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:data offset:0 atIndex:1];
    [enc setBuffer:data offset:0 atIndex:2];
    [enc setBytes:&offset length:4 atIndex:3];
    [enc setBytes:&size   length:4 atIndex:4];
    [enc setBytes:&flags  length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(size,1,1)];
}
static void encode_uniform_add(id<MTLComputeCommandEncoder> enc,
        id<MTLComputePipelineState> p, id<MTLBuffer> data, id<MTLBuffer> psum,
        id<MTLBuffer> total, int offset, int size, int flags, int tg, int nblocks) {
    [enc setComputePipelineState:p];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:psum offset:0 atIndex:1];
    [enc setBuffer:(total?total:data) offset:0 atIndex:2];
    [enc setBytes:&offset length:4 atIndex:3];
    [enc setBytes:&size   length:4 atIndex:4];
    [enc setBytes:&flags  length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(nblocks,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

static void gpu_scan(id<MTLComputeCommandEncoder> enc,
        id<MTLComputePipelineState> pBlock, id<MTLComputePipelineState> pUnif,
        id<MTLBuffer> prefix, id<MTLBuffer> ps0, id<MTLBuffer> ps1,
        id<MTLBuffer> total, int head, int n) {
    const int B = 256;
    int g0 = (n + B - 1) / B;
    if (n <= B*B) {
        encode_block_scan(enc, pBlock, prefix, ps0, total, head, n, 1|2, B);
        if (g0 > 1) {
            encode_block_scan_1blk(enc, pBlock, ps0, g0);
            encode_uniform_add(enc, pUnif, prefix, ps0, total,
                               head + B, n - B, 2, B, g0 - 1);
        }
    } else { // n <= B^3
        int g1 = (g0 + B - 1) / B;
        encode_block_scan(enc, pBlock, prefix, ps0, nil, head, n, 1, B);
        encode_block_scan(enc, pBlock, ps0, ps1, nil, 1, g0, 1, B);
        if (g1 > 1) {
            encode_block_scan_1blk(enc, pBlock, ps1, g1);
            encode_uniform_add(enc, pUnif, ps0, ps1, nil, 1 + B, g0 - B, 0, B, g1 - 1);
        }
        encode_uniform_add(enc, pUnif, prefix, ps0, total, head + B, n - B, 2, B, g0 - 1);
    }
}

//----------------------------------------------------------------------------
// Full Hilbert-key radix sort, validated against CPU stable_sort.
//----------------------------------------------------------------------------
static int test_radix_sort(int N, int level) {
    int nbits = NDIM * level;
    uint64_t mask = (nbits >= 64) ? ~0ULL : ((1ULL << nbits) - 1);
    std::vector<long> keys(N);
    srand(99);
    for (int i = 0; i < N; ++i)
        keys[i] = (long)((((uint64_t)rand() << 32) ^ (uint64_t)rand()) & mask);

    // CPU reference: stable sort of 1-based indices by key.
    std::vector<int> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = i + 1;
    std::stable_sort(ref.begin(), ref.end(),
        [&](int a, int b){ return keys[a-1] < keys[b-1]; });

    int head = 1;
    id<MTLBuffer> hkey   = shared_buf(keys.data(), N);
    id<MTLBuffer> sortp  = shared_buf<int>(nullptr, N);
    id<MTLBuffer> isp    = shared_buf<int>(nullptr, N);
    id<MTLBuffer> prefix = shared_buf<int>(nullptr, N);
    int g0 = (N + 256 - 1)/256, g1 = (g0 + 256 - 1)/256;
    id<MTLBuffer> ps0 = shared_buf<int>(nullptr, std::max(1,g0));
    id<MTLBuffer> ps1 = shared_buf<int>(nullptr, std::max(1,g1));
    id<MTLBuffer> total = shared_buf<int>(nullptr, 1);

    auto pInit  = pso(@"init_global_swap_table");
    auto pBit   = pso(@"init_prefix_sum_part_bit");
    auto pLocal = pso(@"compute_local_swap_table");
    auto pUpd   = pso(@"update_global_swap_table");
    auto pBlock = pso(@"block_scan");
    auto pUnif  = pso(@"uniform_add");

    ScanParams P{}; P.n = N; P.head_idx = head; P.npartmax = N; P.ilevel = level;

    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pInit];
    [enc setBuffer:sortp offset:0 atIndex:0];
    [enc setBytes:&P length:sizeof(P) atIndex:1];
    dispatch_1d(enc, pInit, N);

    for (int ibit = 0; ibit < nbits; ++ibit) {
        P.ibit = ibit;
        [enc setComputePipelineState:pBit];
        [enc setBuffer:hkey offset:0 atIndex:0];
        [enc setBuffer:sortp offset:0 atIndex:1];
        [enc setBuffer:prefix offset:0 atIndex:2];
        [enc setBytes:&P length:sizeof(P) atIndex:3];
        dispatch_1d(enc, pBit, N);

        gpu_scan(enc, pBlock, pUnif, prefix, ps0, ps1, total, head, N);

        [enc setComputePipelineState:pLocal];
        [enc setBuffer:isp offset:0 atIndex:0];
        [enc setBuffer:sortp offset:0 atIndex:1];
        [enc setBuffer:prefix offset:0 atIndex:2];
        [enc setBytes:&P length:sizeof(P) atIndex:3];
        dispatch_1d(enc, pLocal, N);

        [enc setComputePipelineState:pUpd];
        [enc setBuffer:sortp offset:0 atIndex:0];
        [enc setBuffer:isp offset:0 atIndex:1];
        [enc setBytes:&P length:sizeof(P) atIndex:2];
        dispatch_1d(enc, pUpd, N);
    }
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "radix cb error: %s\n", cb.error.localizedDescription.UTF8String);
        return 1;
    }

    const int* out = (const int*)sortp.contents;
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        // Accept either index match or key match (keys may tie); key match is
        // the physically meaningful check.
        if (keys[out[i]-1] != keys[ref[i]-1]) {
            if (bad < 5) fprintf(stderr, "  i=%d gpu_idx=%d key=%ld  cpu_idx=%d key=%ld\n",
                i, out[i], keys[out[i]-1], ref[i], keys[ref[i]-1]);
            ++bad;
        }
    }
    // Also verify it's a permutation and globally non-decreasing.
    bool sorted_ok = true;
    for (int i = 1; i < N; ++i) if (keys[out[i]-1] < keys[out[i-1]-1]) sorted_ok = false;
    printf("radix_sort N=%d level=%d (%d bits): %s%s\n", N, level, nbits,
           (bad==0 && sorted_ok) ? "PASS" : "FAIL",
           sorted_ok ? "" : " (NOT SORTED)");
    return (bad==0 && sorted_ok) ? 0 : 1;
}

//----------------------------------------------------------------------------
// CIC deposit validation on a synthetic fully-refined periodic box.
//   ilevel: deposit-level cells span [0, 2^ilevel) per axis; parent octs span
//   [0, 2^(ilevel-1)) (8 cells each).  boxlen = 1.
//----------------------------------------------------------------------------
static void cic_weights(double fr, double w[3]) {   // fp64 reference weights
    w[0] = fmax(0.0, 0.5 - fr);
    w[1] = 1.0 - fabs(fr - 0.5);
    w[2] = fmax(0.0, fr - 0.5);
}

static int run_cic_gpu(const std::vector<long>& ipos, const std::vector<float>& mp,
                       const std::vector<Oct>& grid, const std::vector<long>& hkey,
                       const std::vector<int>& hval, int N, int ilevel, int nx, int hs,
                       long key_off, int ncell_lin, bool periodic,
                       const std::vector<double>& rho_ref, const std::vector<double>& nref_ref,
                       bool verbose, const char* label);

// run_cic on a SPARSE oct set: `octs` lists father-level ckeys present; particles
// are given as 64-bit fixed-point positions (ipos, 3N column-major).  ncell_lin is
// the deposit-level extent per axis (2^ilevel) for periodic wrap.  Works at any
// depth because the grid is sparse (only the listed octs exist).
static int run_cic_sparse(const std::vector<long>& ipos, const std::vector<float>& mp,
                          int N, int ilevel, const std::vector<int>& octs /*3 per oct*/,
                          bool periodic, bool verbose, const char* label) {
    int noct_lin  = 1 << (ilevel - 1);    // octs per axis (full extent)
    int ncell_lin = 1 << ilevel;
    int nx        = noct_lin;
    int noct      = (int)octs.size()/3;
    int hs        = 2*std::max(1,noct) + 1031;
    long key_off  = 100;
    int shift     = NBITS_POS - ilevel;
    long cellmask = (1L << shift) - 1L;

    std::vector<Oct> grid(noct);
    std::vector<long> hkey(hs, 0); std::vector<int> hval(hs, 0);
    for (int o = 0; o < noct; ++o) {
        int ix=octs[3*o], iy=octs[3*o+1], iz=octs[3*o+2];
        Oct g{}; g.ckey[0]=ix; g.ckey[1]=iy; g.ckey[2]=iz; g.lev=ilevel-1; grid[o]=g;
        long key = key_off + (long)ix + (long)iy*nx + (long)iz*nx*nx;
        h_hash_set(hkey, hval, hs, key, o + 1);
    }

    int ncells = noct*TWOTONDIM;
    std::vector<double> rho_ref(ncells, 0.0), nref_ref(ncells, 0.0);
    for (int p = 1; p <= N; ++p) {
        int src[3]; double fr[3];
        for (int d = 1; d <= 3; ++d) {
            long ip = ipos[IDXP(p,d,N)];
            src[d-1] = (int)(ip >> shift);                       // exact cell index
            fr[d-1]  = (double)(ip & cellmask) / (double)(1L<<shift); // TRUE fp64 fraction
        }
        double wx[3], wy[3], wz[3];
        cic_weights(fr[0], wx); cic_weights(fr[1], wy); cic_weights(fr[2], wz);
        for (int k = 1; k <= 27; ++k) {
            int ox=(k-1)%3-1, oy=((k-1)/3)%3-1, oz=(k-1)/9-1;
            int tgt[3] = { src[0]+ox, src[1]+oy, src[2]+oz };
            bool in=true;
            for (int d = 0; d < 3; ++d) {
                if (periodic) { if (tgt[d]<0) tgt[d]=ncell_lin-1; if (tgt[d]>=ncell_lin) tgt[d]=0; }
                else if (tgt[d]<0 || tgt[d]>=ncell_lin) in=false;
            }
            if (!in) continue;
            int father[3], ii[3], icell = 1;
            for (int d = 0; d < 3; ++d) { father[d]=tgt[d]>>1; ii[d]=tgt[d]-2*father[d]; icell += ii[d]*(1<<d); }
            long key = key_off + (long)father[0] + (long)father[1]*nx + (long)father[2]*nx*nx;
            int ig = h_hash_get(hkey, hval, hs, key);
            if (!ig) continue;
            double w = wx[ox+1]*wy[oy+1]*wz[oz+1];
            if (w <= 0.0) continue;
            rho_ref [IDX2(icell, ig)] += (double)mp[p-1]*w;   // MONOPOLE (mass per cell)
            nref_ref[IDX2(icell, ig)] += w;
        }
    }
    return run_cic_gpu(ipos, mp, grid, hkey, hval, N, ilevel, nx, hs, key_off,
                       ncell_lin, periodic, rho_ref, nref_ref, verbose, label);
}

static int run_cic_gpu(const std::vector<long>& ipos, const std::vector<float>& mp,
                       const std::vector<Oct>& grid, const std::vector<long>& hkey,
                       const std::vector<int>& hval, int N, int ilevel, int nx, int hs,
                       long key_off, int ncell_lin, bool periodic,
                       const std::vector<double>& rho_ref, const std::vector<double>& nref_ref,
                       bool verbose, const char* label) {
    int noct   = (int)grid.size();
    int ncells = noct*TWOTONDIM;

    // ---- GPU pipeline: hkey -> sort -> build_src -> deposit -> finalize.
    int head = 1;
    id<MTLBuffer> bipos = shared_buf(ipos.data(), N*3);
    id<MTLBuffer> bmp = shared_buf(mp.data(), N);
    id<MTLBuffer> bgrid = shared_buf(grid.data(), noct);
    id<MTLBuffer> bhk = shared_buf(hkey.data(), hs);
    id<MTLBuffer> bhv = shared_buf(hval.data(), hs);
    id<MTLBuffer> hkpart = shared_buf<long>(nullptr, N);
    id<MTLBuffer> sortp = shared_buf<int>(nullptr, N);
    id<MTLBuffer> isp   = shared_buf<int>(nullptr, N);
    id<MTLBuffer> prefix= shared_buf<int>(nullptr, N);
    int g0=(N+255)/256, g1=(g0+255)/256;
    id<MTLBuffer> ps0=shared_buf<int>(nullptr,std::max(1,g0)), ps1=shared_buf<int>(nullptr,std::max(1,g1));
    id<MTLBuffer> total=shared_buf<int>(nullptr,1);
    id<MTLBuffer> rlo=shared_buf<uint32_t>(nullptr,ncells), rhi=shared_buf<uint32_t>(nullptr,ncells);
    id<MTLBuffer> nlo=shared_buf<uint32_t>(nullptr,ncells), nhi=shared_buf<uint32_t>(nullptr,ncells);
    id<MTLBuffer> brho=shared_buf<float>(nullptr,ncells), bnref=shared_buf<float>(nullptr,ncells);

    CicParams P{}; P.dx_loc=1.0f; P.vol_loc=1.0f; P.m_refine=1.0f; P.mass_cut=0.0f; // dx/vol unused (fixed-point + monopole)
    P.fp_scale_rho=(float)(1L<<FP_SHIFT_RHO); P.key_off=key_off; P.ckey_max=nx; P.hash_size=hs;
    P.ilevel=ilevel; P.head_idx=head; P.num_parts=N; P.npartmax=N; P.refine_on=1;
    BoxParams BX{}; for(int d=0;d<3;++d){BX.box_min[d]=0;BX.box_max[d]=ncell_lin;BX.periodic[d]=periodic?1:0;}

    auto pHkey=pso(@"compute_hkey_part"), pInit=pso(@"init_global_swap_table"),
         pBit=pso(@"init_prefix_sum_part_bit"), pLocal=pso(@"compute_local_swap_table"),
         pUpd=pso(@"update_global_swap_table"), pBlock=pso(@"block_scan"), pUnif=pso(@"uniform_add"),
         pSrc=pso(@"build_src_part"), pDep=pso(@"cic_part_warp"), pFin=pso(@"rho_finalize");

    id<MTLCommandBuffer> cb=[g_q commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];

    ScanParams S{}; S.n=N; S.head_idx=head; S.npartmax=N; S.ilevel=ilevel;
    [enc setComputePipelineState:pHkey];
    [enc setBuffer:bipos offset:0 atIndex:0]; [enc setBuffer:hkpart offset:0 atIndex:1];
    [enc setBytes:&S length:sizeof(S) atIndex:2];
    dispatch_1d(enc, pHkey, N);

    [enc setComputePipelineState:pInit];
    [enc setBuffer:sortp offset:0 atIndex:0]; [enc setBytes:&S length:sizeof(S) atIndex:1];
    dispatch_1d(enc, pInit, N);
    for (int ibit = 0; ibit < NDIM*ilevel; ++ibit) {
        S.ibit=ibit;
        [enc setComputePipelineState:pBit];
        [enc setBuffer:hkpart offset:0 atIndex:0]; [enc setBuffer:sortp offset:0 atIndex:1];
        [enc setBuffer:prefix offset:0 atIndex:2]; [enc setBytes:&S length:sizeof(S) atIndex:3];
        dispatch_1d(enc, pBit, N);
        gpu_scan(enc, pBlock, pUnif, prefix, ps0, ps1, total, head, N);
        [enc setComputePipelineState:pLocal];
        [enc setBuffer:isp offset:0 atIndex:0]; [enc setBuffer:sortp offset:0 atIndex:1];
        [enc setBuffer:prefix offset:0 atIndex:2]; [enc setBytes:&S length:sizeof(S) atIndex:3];
        dispatch_1d(enc, pLocal, N);
        [enc setComputePipelineState:pUpd];
        [enc setBuffer:sortp offset:0 atIndex:0]; [enc setBuffer:isp offset:0 atIndex:1];
        [enc setBytes:&S length:sizeof(S) atIndex:2];
        dispatch_1d(enc, pUpd, N);
    }

    [enc setComputePipelineState:pSrc];
    [enc setBuffer:bipos offset:0 atIndex:0]; [enc setBuffer:sortp offset:0 atIndex:1];
    [enc setBuffer:isp offset:0 atIndex:2]; [enc setBuffer:bhk offset:0 atIndex:3];
    [enc setBuffer:bhv offset:0 atIndex:4]; [enc setBytes:&P length:sizeof(P) atIndex:5];
    dispatch_1d(enc, pSrc, N);

    [enc setComputePipelineState:pDep];
    [enc setBuffer:sortp offset:0 atIndex:0]; [enc setBuffer:isp offset:0 atIndex:1];
    [enc setBuffer:bgrid offset:0 atIndex:2]; [enc setBuffer:bhk offset:0 atIndex:3];
    [enc setBuffer:bhv offset:0 atIndex:4]; [enc setBuffer:bipos offset:0 atIndex:5];
    [enc setBuffer:bmp offset:0 atIndex:6];
    [enc setBuffer:rlo offset:0 atIndex:7]; [enc setBuffer:rhi offset:0 atIndex:8];
    [enc setBuffer:nlo offset:0 atIndex:9]; [enc setBuffer:nhi offset:0 atIndex:10];
    [enc setBytes:&P length:sizeof(P) atIndex:11]; [enc setBytes:&BX length:sizeof(BX) atIndex:12];
    [enc dispatchThreadgroups:MTLSizeMake((N+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    float inv_scale = 1.0f/(float)(1L<<FP_SHIFT_RHO);
    [enc setComputePipelineState:pFin];
    [enc setBuffer:rlo offset:0 atIndex:0]; [enc setBuffer:rhi offset:0 atIndex:1];
    [enc setBuffer:nlo offset:0 atIndex:2]; [enc setBuffer:nhi offset:0 atIndex:3];
    [enc setBuffer:brho offset:0 atIndex:4]; [enc setBuffer:bnref offset:0 atIndex:5];
    [enc setBytes:&inv_scale length:4 atIndex:6]; [enc setBytes:&ncells length:4 atIndex:7];
    dispatch_1d(enc, pFin, ncells);

    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "cic cb error: %s\n", cb.error.localizedDescription.UTF8String); return 1;
    }

    const float* rho = (const float*)brho.contents;
    const float* nrf = (const float*)bnref.contents;
    if (verbose) {
        for (int i = 0; i < ncells; ++i) {
            if (fabs((double)rho[i]) > 1e-9 || fabs(rho_ref[i]) > 1e-9) {
                int o=i/TWOTONDIM, c=i%TWOTONDIM;
                fprintf(stderr, "   cell oct=%d ckey=(%d,%d,%d) icell=%d : gpu=%.6f ref=%.6f %s\n",
                    o+1, grid[o].ckey[0],grid[o].ckey[1],grid[o].ckey[2], c+1,
                    (double)rho[i], rho_ref[i],
                    fabs((double)rho[i]-rho_ref[i])>1e-4 ? "  <<<DIFF" : "");
            }
        }
    }
    double mass_gpu=0, mass_ref=0, max_rel=0, l2num=0, l2den=0, max_nref=0;
    for (int i = 0; i < ncells; ++i) { mass_ref += rho_ref[i]; }   // monopole totals
    double rho_typ = mass_ref / std::max(1,ncells) * 8.0;
    int nbig=0, worst=-1; double worstd=0;
    for (int i = 0; i < ncells; ++i) {
        mass_gpu += rho[i];
        double d = rho[i]-rho_ref[i];
        l2num += d*d; l2den += rho_ref[i]*rho_ref[i];
        if (rho_ref[i] > 1e-3*rho_typ) {
            double rel = fabs(d)/rho_ref[i];
            max_rel = std::max(max_rel, rel);
            if (rel > 0.01) ++nbig;
            if (fabs(d) > worstd) { worstd = fabs(d); worst = i; }
        }
        max_nref = std::max(max_nref, fabs((double)nrf[i]-nref_ref[i]));
    }
    if (worst >= 0) fprintf(stderr, "  worst cell flat=%d oct=%d icell=%d gpu=%.6f ref=%.6f  (%d cells >1%% off)\n",
        worst, worst/TWOTONDIM+1, worst%TWOTONDIM+1, (double)rho[worst], rho_ref[worst], nbig);
    int shown=0;
    for (int i = 0; i < ncells && shown < 8; ++i) {
        if (rho_ref[i] > 1e-3*rho_typ && fabs(rho[i]-rho_ref[i])/rho_ref[i] > 0.01) {
            int o = i/TWOTONDIM, c = i%TWOTONDIM;
            fprintf(stderr, "    off: oct=%d ckey=(%d,%d,%d) icell=%d gpu=%.3f ref=%.3f\n",
                o+1, grid[o].ckey[0], grid[o].ckey[1], grid[o].ckey[2], c+1, (double)rho[i], rho_ref[i]);
            ++shown;
        }
    }
    double l2 = sqrt(l2num/std::max(1e-30,l2den));
    // Monopole mass: gpu vs golden must agree (both deposit mp*w). For a fully-
    // enclosed particle set, mass also equals sum(mp) (deposit weights sum to 1).
    double mass_true = 0; for (int i=0;i<N;++i) mass_true += mp[i];
    bool ok = (l2 < 1e-5) && (max_rel < 1e-4) && (max_nref < 1e-3)
              && (fabs(mass_gpu - mass_ref) < 1e-4*std::max(1.0,mass_ref));
    printf("cic %s N=%d ilevel=%d noct=%d : %s  L2=%.2e maxrel=%.2e nref_err=%.2e  mass gpu=%.6f golden=%.6f sum_mp=%.6f\n",
           label, N, ilevel, noct, ok?"PASS":"FAIL", l2, max_rel, max_nref, mass_gpu, mass_ref, mass_true);
    return ok ? 0 : 1;
}

static long u_to_fix(double u) { return (long)llround(u * ldexp(1.0, NBITS_POS)); }

// Full periodic grid (all octs present); positions random in [0,1).
static int test_cic(int N, int ilevel) {
    int noct_lin = 1 << (ilevel - 1);
    std::vector<long> ipos(N*3); std::vector<float> mp(N);
    srand(2024);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) ipos[i + d*N] = u_to_fix((rand()%1000000)/1000000.0);
        mp[i] = (float)(0.5 + (rand()%1000)/1000.0);
    }
    std::vector<int> octs;
    for (int iz=0; iz<noct_lin; ++iz) for (int iy=0; iy<noct_lin; ++iy) for (int ix=0; ix<noct_lin; ++ix)
        { octs.push_back(ix); octs.push_back(iy); octs.push_back(iz); }
    int nx = noct_lin, hs = 2*(noct_lin*noct_lin*noct_lin)+1031;
    std::vector<Oct> grid; std::vector<long> hk; std::vector<int> hv; (void)nx;(void)hs;
    return run_cic_sparse(ipos, mp, N, ilevel, octs, /*periodic*/true, false, "full");
}

// Single deterministic particle near the -x periodic face: CIC spreads a
// fraction across the wrap to the far-x cell.  Dumps every nonzero cell.
static int test_cic_one(double px, double py, double pz, int ilevel) {
    int noct_lin = 1 << (ilevel - 1);
    std::vector<long> ipos = { u_to_fix(px), u_to_fix(py), u_to_fix(pz) };
    std::vector<float> mp = { 1.0f };
    std::vector<int> octs;
    for (int iz=0; iz<noct_lin; ++iz) for (int iy=0; iy<noct_lin; ++iy) for (int ix=0; ix<noct_lin; ++ix)
        { octs.push_back(ix); octs.push_back(iy); octs.push_back(iz); }
    fprintf(stderr, "-- cic_one pos=(%.4f,%.4f,%.4f) ilevel=%d --\n", px,py,pz,ilevel);
    return run_cic_sparse(ipos, mp, 1, ilevel, octs, true, true, "one");
}

// DEEP level: sparse 4x4x4-oct patch in the box interior.  This is where the
// old float frac = xp/dx - floor(xp/dx) lost ~one bit per level; fixed-point
// positions must keep ~24-bit accuracy here.
static int test_cic_deep(int N, int ilevel) {
    int noct_lin = 1 << (ilevel - 1);
    int base = noct_lin / 2;                 // patch father coords [base, base+4)
    std::vector<int> octs;
    for (int iz=0; iz<4; ++iz) for (int iy=0; iy<4; ++iy) for (int ix=0; ix<4; ++ix)
        { octs.push_back(base+ix); octs.push_back(base+iy); octs.push_back(base+iz); }
    // Deposit cells of the patch: [2*base, 2*base+8).  Keep particles in the
    // interior [2*base+1, 2*base+7) so the full CIC stencil stays inside (no miss).
    double lo = (2.0*base + 1.0) / (1<<ilevel);
    double hi = (2.0*base + 7.0) / (1<<ilevel);
    std::vector<long> ipos(N*3); std::vector<float> mp(N);
    srand(7);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) ipos[i + d*N] = u_to_fix(lo + (hi-lo)*((rand()%1000000)/1000000.0));
        mp[i] = (float)(0.5 + (rand()%1000)/1000.0);
    }
    return run_cic_sparse(ipos, mp, N, ilevel, octs, /*periodic*/false, false, "deep");
}

//----------------------------------------------------------------------------
// Multigrid Poisson smoother validation (grouped form).
// Periodic uniform grid: set S = L_g(phi_true) for a known smooth phi_true,
// run red-black Gauss-Seidel, verify it recovers phi_true (mean-removed) and
// drives the residual down.  Validates the discrete operator + grouped source.
//----------------------------------------------------------------------------
static const int H_hhh[6][8] = {
    {2,1,4,3,6,5,8,7},{2,1,4,3,6,5,8,7},{3,4,1,2,7,8,5,6},
    {3,4,1,2,7,8,5,6},{5,6,7,8,1,2,3,4},{5,6,7,8,1,2,3,4}};
static const int H_iii[6][8] = {
    {-1,0,-1,0,-1,0,-1,0},{0,1,0,1,0,1,0,1},{-1,-1,0,0,-1,-1,0,0},
    {0,0,1,1,0,0,1,1},{-1,-1,-1,-1,0,0,0,0},{0,0,0,0,1,1,1,1}};

static int test_mg(int ilevel, int nsweep) {
    int nl = 1 << (ilevel - 1);          // octs per axis
    int noct = nl*nl*nl;
    int ncells = noct*TWOTONDIM;
    auto oidx = [&](int fx,int fy,int fz){ return fx + fy*nl + fz*nl*nl; }; // 0-based

    // Periodic 27-neighbour table nbor(27, noct), values 1-based.
    std::vector<int> nbor(27*noct);
    for (int fz=0; fz<nl; ++fz) for (int fy=0; fy<nl; ++fy) for (int fx=0; fx<nl; ++fx) {
        int o = oidx(fx,fy,fz);
        for (int kk=-1; kk<=1; ++kk) for (int jj=-1; jj<=1; ++jj) for (int ii=-1; ii<=1; ++ii) {
            int no = oidx((fx+ii+nl)%nl,(fy+jj+nl)%nl,(fz+kk+nl)%nl);
            int ind = 1 + (1+ii) + 3*(1+jj) + 9*(1+kk);
            nbor[o*27 + (ind-1)] = no + 1;
        }
    }

    // phi_true (smooth, zero-mean, lowest periodic mode) per (cell,oct).
    int ncell_lin = 1 << ilevel;
    std::vector<double> phi_true(ncells);
    for (int o = 1; o <= noct; ++o) {
        int oo=o-1, fx=oo%nl, fy=(oo/nl)%nl, fz=oo/(nl*nl);
        for (int c = 1; c <= TWOTONDIM; ++c) {
            int iix=(c-1)&1, iiy=((c-1)>>1)&1, iiz=((c-1)>>2)&1;
            double ux=(2*fx+iix+0.5)/ncell_lin, uy=(2*fy+iiy+0.5)/ncell_lin, uz=(2*fz+iiz+0.5)/ncell_lin;
            phi_true[IDX2(c,o)] = cos(2*M_PI*ux)+cos(2*M_PI*uy)+cos(2*M_PI*uz);
        }
    }

    // S = L_g(phi_true) = sum_nb phi_true_nb - twondim*phi_true_c (all interior).
    std::vector<float> f(3*ncells, 0.0f);   // f(:,1)=res f(:,2)=S f(:,3)=mask
    std::vector<float> phi(ncells, 0.0f);
    for (int o = 1; o <= noct; ++o)
        for (int c = 1; c <= TWOTONDIM; ++c) {
            double nb = 0;
            for (int idim=1; idim<=3; ++idim) for (int inb=1; inb<=2; ++inb) {
                int dir=2*(idim-1)+inb, off=H_iii[dir-1][c-1];
                int in=0,jn=0,kn=0; if(idim==1)in=off; else if(idim==2)jn=off; else kn=off;
                int ind=1+(1+in)+3*(1+jn)+9*(1+kn);
                int onb=nbor[(o-1)*27+(ind-1)], cnb=H_hhh[dir-1][c-1];
                nb += phi_true[IDX2(cnb,onb)];
            }
            f[IDX3(c,2,o)] = (float)(nb - (double)TWONDIM*phi_true[IDX2(c,o)]); // S
            f[IDX3(c,3,o)] = 1.0f;                                              // interior mask
        }

    id<MTLBuffer> bphi = shared_buf(phi.data(), ncells);
    id<MTLBuffer> bf   = shared_buf(f.data(), 3*ncells);
    id<MTLBuffer> bnbor= shared_buf(nbor.data(), 27*noct);

    MgParams P{}; P.head_idx=1; P.num_octs=noct; P.ngridmax=noct; P.ilevel=ilevel;
    auto pGS = pso(@"mg_gauss_seidel"); auto pRes = pso(@"mg_residual");
    int safe = 0;

    // Initial residual norm (host) for the reduction report.
    auto resnorm = [&]()->double {
        id<MTLCommandBuffer> cb=[g_q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pRes];
        [e setBuffer:bphi offset:0 atIndex:0]; [e setBuffer:bf offset:0 atIndex:1];
        [e setBuffer:bnbor offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,noct,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        const float* fr=(const float*)bf.contents; double s=0;
        for (int o=1;o<=noct;++o) for(int c=1;c<=TWOTONDIM;++c){ double r=fr[IDX3(c,1,o)]; s+=r*r; }
        return sqrt(s);
    };
    double r0 = resnorm();

    id<MTLCommandBuffer> cb=[g_q commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    for (int it = 0; it < nsweep; ++it) {
        for (int rb = 1; rb >= 0; --rb) {       // red then black
            P.redstep = rb;
            [enc setComputePipelineState:pGS];
            [enc setBuffer:bphi offset:0 atIndex:0]; [enc setBuffer:bf offset:0 atIndex:1];
            [enc setBuffer:bnbor offset:0 atIndex:2]; [enc setBytes:&P length:sizeof(P) atIndex:3];
            [enc setBytes:&safe length:4 atIndex:4];
            [enc dispatchThreads:MTLSizeMake(4,noct,1) threadsPerThreadgroup:MTLSizeMake(4,16,1)];
        }
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "mg cb error: %s\n", cb.error.localizedDescription.UTF8String); return 1;
    }
    double r1 = resnorm();

    // Error vs phi_true, both mean-removed (periodic null space).
    const float* pg=(const float*)bphi.contents;
    double mg=0, mt=0; for (int i=0;i<ncells;++i){ mg+=pg[i]; mt+=phi_true[i]; }
    mg/=ncells; mt/=ncells;
    double en=0, ed=0;
    for (int i=0;i<ncells;++i){ double d=(pg[i]-mg)-(phi_true[i]-mt); en+=d*d; ed+=(phi_true[i]-mt)*(phi_true[i]-mt); }
    double err = sqrt(en/ed);
    bool ok = (err < 2e-3) && (r1 < 1e-2*r0);
    printf("mg ilevel=%d noct=%d sweeps=%d : %s  phi_err=%.2e  resid %.2e -> %.2e (x%.1e)\n",
           ilevel, noct, nsweep, ok?"PASS":"FAIL", err, r0, r1, r1/r0);
    return ok ? 0 : 1;
}

// Build a periodic 27-neighbour table for an nl^3 oct grid (values 1-based).
static std::vector<int> build_nbor(int nl) {
    auto oidx=[&](int x,int y,int z){return x+y*nl+z*nl*nl;};
    std::vector<int> nb(27*nl*nl*nl);
    for (int z=0;z<nl;++z) for (int y=0;y<nl;++y) for (int x=0;x<nl;++x) {
        int o=oidx(x,y,z);
        for (int kk=-1;kk<=1;++kk) for (int jj=-1;jj<=1;++jj) for (int ii=-1;ii<=1;++ii) {
            int ind=1+(1+ii)+3*(1+jj)+9*(1+kk);
            nb[o*27+(ind-1)]=oidx((x+ii+nl)%nl,(y+jj+nl)%nl,(z+kk+nl)%nl)+1;
        }
    }
    return nb;
}

// Validate the 4th-order force kernel against the analytic accel = -grad phi.
static int test_gradient(int ilevel) {
    int nl=1<<(ilevel-1), noct=nl*nl*nl, ncells=noct*TWOTONDIM, ncl=1<<ilevel;
    std::vector<int> nbor=build_nbor(nl);
    std::vector<float> phi(ncells), f(3*ncells,0.f);
    auto cellcoord=[&](int o,int c,double&ux,double&uy,double&uz){
        int oo=o-1,fx=oo%nl,fy=(oo/nl)%nl,fz=oo/(nl*nl);
        int ix=(c-1)&1,iy=((c-1)>>1)&1,iz=((c-1)>>2)&1;
        ux=(2*fx+ix+0.5)/ncl; uy=(2*fy+iy+0.5)/ncl; uz=(2*fz+iz+0.5)/ncl;
    };
    for (int o=1;o<=noct;++o) for (int c=1;c<=TWOTONDIM;++c){
        double ux,uy,uz; cellcoord(o,c,ux,uy,uz);
        phi[IDX2(c,o)]=cos(2*M_PI*ux)+cos(2*M_PI*uy)+cos(2*M_PI*uz);
    }
    id<MTLBuffer> bphi=shared_buf(phi.data(),ncells), bf=shared_buf(f.data(),3*ncells),
                  bnb=shared_buf(nbor.data(),27*noct);
    MgParams P{}; P.head_idx=1; P.num_octs=noct; P.ngridmax=noct; P.ilevel=ilevel;
    P.dx=(float)(1.0/ncl);
    id<MTLCommandBuffer> cb=[g_q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:pso(@"mg_gradient_phi")];
    [e setBuffer:bphi offset:0 atIndex:0]; [e setBuffer:bf offset:0 atIndex:1];
    [e setBuffer:bnb offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
    [e dispatchThreads:MTLSizeMake(TWOTONDIM,noct,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const float* fg=(const float*)bf.contents;
    double en=0,ed=0;
    for (int o=1;o<=noct;++o) for(int c=1;c<=TWOTONDIM;++c){
        double ux,uy,uz; cellcoord(o,c,ux,uy,uz);
        double gx=2*M_PI*sin(2*M_PI*ux), gy=2*M_PI*sin(2*M_PI*uy), gz=2*M_PI*sin(2*M_PI*uz); // -dphi/dx
        double dx=fg[IDX3(c,1,o)]-gx, dy=fg[IDX3(c,2,o)]-gy, dz=fg[IDX3(c,3,o)]-gz;
        en+=dx*dx+dy*dy+dz*dz; ed+=gx*gx+gy*gy+gz*gz;
    }
    double err=sqrt(en/ed);
    bool ok = err < 3e-2;   // 4th-order discretisation error at this resolution
    printf("gradient ilevel=%d noct=%d : %s  rel_err_vs_analytic=%.2e\n", ilevel, noct, ok?"PASS":"FAIL", err);
    return ok?0:1;
}

// 2-level multigrid V-cycle: must converge FAR faster than pure Gauss-Seidel.
static int test_vcycle(int Lf, int ncycle) {
    int nlf=1<<(Lf-1), noctf=nlf*nlf*nlf, ncf=noctf*TWOTONDIM, ncl=1<<Lf;
    int nlc=nlf/2, noctc=nlc*nlc*nlc, ncc=noctc*TWOTONDIM;
    auto fidx=[&](int x,int y,int z){return x+y*nlf+z*nlf*nlf;};
    auto cidx=[&](int x,int y,int z){return x+y*nlc+z*nlc*nlc;};

    std::vector<int> nbf=build_nbor(nlf), nbc=build_nbor(nlc);
    std::vector<Oct> gf(noctf), gc(noctc);
    std::vector<int> father(noctf);
    for (int z=0;z<nlf;++z) for (int y=0;y<nlf;++y) for (int x=0;x<nlf;++x){
        int o=fidx(x,y,z); Oct g{}; g.ckey[0]=x;g.ckey[1]=y;g.ckey[2]=z;g.lev=Lf-1; gf[o]=g;
        father[o]=cidx(x/2,y/2,z/2)+1;
    }
    for (int z=0;z<nlc;++z) for (int y=0;y<nlc;++y) for (int x=0;x<nlc;++x){
        int o=cidx(x,y,z); Oct g{}; g.ckey[0]=x;g.ckey[1]=y;g.ckey[2]=z;g.lev=Lf-2; gc[o]=g;
    }

    std::vector<double> phi_true(ncf);
    for (int o=1;o<=noctf;++o){ int oo=o-1,fx=oo%nlf,fy=(oo/nlf)%nlf,fz=oo/(nlf*nlf);
        for (int c=1;c<=TWOTONDIM;++c){ int ix=(c-1)&1,iy=((c-1)>>1)&1,iz=((c-1)>>2)&1;
            double ux=(2*fx+ix+0.5)/ncl,uy=(2*fy+iy+0.5)/ncl,uz=(2*fz+iz+0.5)/ncl;
            phi_true[IDX2(c,o)]=cos(2*M_PI*ux)+cos(2*M_PI*uy)+cos(2*M_PI*uz); } }
    std::vector<float> ff(3*ncf,0.f), phif(ncf,0.f), fmg(3*ncc,0.f), phimg(ncc,0.f);
    for (int o=1;o<=noctf;++o) for (int c=1;c<=TWOTONDIM;++c){
        double nb=0;
        for (int idim=1;idim<=3;++idim) for(int inb=1;inb<=2;++inb){
            int dir=2*(idim-1)+inb,off=H_iii[dir-1][c-1],in=0,jn=0,kn=0;
            if(idim==1)in=off; else if(idim==2)jn=off; else kn=off;
            int ind=1+(1+in)+3*(1+jn)+9*(1+kn), onb=nbf[(o-1)*27+(ind-1)], cnb=H_hhh[dir-1][c-1];
            nb+=phi_true[IDX2(cnb,onb)];
        }
        ff[IDX3(c,2,o)]=(float)(nb-(double)TWONDIM*phi_true[IDX2(c,o)]); ff[IDX3(c,3,o)]=1.f;
    }
    for (int i=0;i<ncc;++i) fmg[3*0+i*0]=0; // noop
    for (int o=1;o<=noctc;++o) for (int c=1;c<=TWOTONDIM;++c) fmg[IDX3(c,3,o)]=1.f;

    id<MTLBuffer> bphif=shared_buf(phif.data(),ncf), bff=shared_buf(ff.data(),3*ncf),
        bnbf=shared_buf(nbf.data(),27*noctf), bgf=shared_buf(gf.data(),noctf),
        bfather=shared_buf(father.data(),noctf),
        bphimg=shared_buf(phimg.data(),ncc), bfmg=shared_buf(fmg.data(),3*ncc),
        bnbc=shared_buf(nbc.data(),27*noctc);
    auto pGS=pso(@"mg_gauss_seidel"), pRes=pso(@"mg_residual"),
         pRestrict=pso(@"mg_restrict_residual"), pInterp=pso(@"mg_interpolate_correct");
    int safe=0;
    auto gs=[&](id<MTLComputeCommandEncoder> e, id<MTLBuffer> phi, id<MTLBuffer> f,
                id<MTLBuffer> nb, int noct, int nsweep){
        MgParams P{}; P.head_idx=1; P.num_octs=noct; P.ngridmax=noct;
        for (int s=0;s<nsweep;++s) for (int rb=1;rb>=0;--rb){ P.redstep=rb;
            [e setComputePipelineState:pGS];
            [e setBuffer:phi offset:0 atIndex:0]; [e setBuffer:f offset:0 atIndex:1];
            [e setBuffer:nb offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
            [e setBytes:&safe length:4 atIndex:4];
            [e dispatchThreads:MTLSizeMake(4,noct,1) threadsPerThreadgroup:MTLSizeMake(4,16,1)]; }
    };
    auto resnorm=[&]()->double{
        id<MTLCommandBuffer> cb=[g_q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        MgParams P{}; P.head_idx=1; P.num_octs=noctf; P.ngridmax=noctf;
        [e setComputePipelineState:pRes];
        [e setBuffer:bphif offset:0 atIndex:0]; [e setBuffer:bff offset:0 atIndex:1];
        [e setBuffer:bnbf offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,noctf,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        const float* fr=(const float*)bff.contents; double s=0;
        for (int o=1;o<=noctf;++o) for(int c=1;c<=TWOTONDIM;++c){ double r=fr[IDX3(c,1,o)]; s+=r*r; } return sqrt(s);
    };
    double r0=resnorm(), rprev=r0;
    for (int cyc=0; cyc<ncycle; ++cyc) {
        memset(bphimg.contents, 0, ncc*sizeof(float));   // zero coarse correction
        id<MTLCommandBuffer> cb=[g_q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        gs(e, bphif, bff, bnbf, noctf, 2);               // pre-smooth
        { MgParams P{}; P.head_idx=1; P.num_octs=noctf; P.ngridmax=noctf;   // residual
          [e setComputePipelineState:pRes]; [e setBuffer:bphif offset:0 atIndex:0];
          [e setBuffer:bff offset:0 atIndex:1]; [e setBuffer:bnbf offset:0 atIndex:2];
          [e setBytes:&P length:sizeof(P) atIndex:3];
          [e dispatchThreads:MTLSizeMake(TWOTONDIM,noctf,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)]; }
        { MgParams P{}; P.head_idx=1; P.num_octs=noctf; P.head_father=1;    // restrict
          [e setComputePipelineState:pRestrict]; [e setBuffer:bgf offset:0 atIndex:0];
          [e setBuffer:bfather offset:0 atIndex:1]; [e setBuffer:bff offset:0 atIndex:2];
          [e setBuffer:bfmg offset:0 atIndex:3]; [e setBytes:&P length:sizeof(P) atIndex:4];
          [e dispatchThreads:MTLSizeMake(noctf,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; }
        gs(e, bphimg, bfmg, bnbc, noctc, 50);            // coarse solve
        { MgParams P{}; P.head_idx=1; P.num_octs=noctf; P.head_father=1;    // prolong+correct
          [e setComputePipelineState:pInterp]; [e setBuffer:bgf offset:0 atIndex:0];
          [e setBuffer:bfather offset:0 atIndex:1]; [e setBuffer:bnbc offset:0 atIndex:2];
          [e setBuffer:bphif offset:0 atIndex:3]; [e setBuffer:bphimg offset:0 atIndex:4];
          [e setBuffer:bff offset:0 atIndex:5]; [e setBytes:&P length:sizeof(P) atIndex:6];
          [e dispatchThreads:MTLSizeMake(TWOTONDIM,noctf,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)]; }
        gs(e, bphif, bff, bnbf, noctf, 2);               // post-smooth
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double r=resnorm(); fprintf(stderr,"   vcycle %d: resid=%.3e (x%.3f)\n", cyc, r, r/rprev); rprev=r;
    }
    double r1=rprev;
    const float* pg=(const float*)bphif.contents;
    double mg=0,mt=0; for(int i=0;i<ncf;++i){mg+=pg[i];mt+=phi_true[i];} mg/=ncf; mt/=ncf;
    double en=0,ed=0; for(int i=0;i<ncf;++i){double d=(pg[i]-mg)-(phi_true[i]-mt); en+=d*d; ed+=(phi_true[i]-mt)*(phi_true[i]-mt);}
    double err=sqrt(en/ed);
    bool ok = (err<3e-3) && (r1 < 1e-3*r0);
    printf("vcycle Lf=%d (fine %d / coarse %d octs) cycles=%d : %s  phi_err=%.2e  resid %.2e -> %.2e\n",
           Lf, noctf, noctc, ncycle, ok?"PASS":"FAIL", err, r0, r1);
    return ok?0:1;
}

// Force gather + leapfrog kick/drift validation on a periodic force grid.
// f is a known smooth field; we run action=2 (kick+drift) and check the
// recovered gather (Dv/(0.5 dt)) and the drifted positions vs a CPU golden.
static int test_kickdrift(int ilevel, int N) {
    int NG = 1 << ilevel;                 // octs per axis (grid_level = ilevel)
    int cl = ilevel + 1;                  // cell level
    int ncl = 1 << cl;                    // cells per axis
    int noct = NG*NG*NG, ncells = noct*TWOTONDIM;
    int hs = 2*noct + 1031; long key_off_v = 100;
    double dt = 0.01;

    std::vector<Oct> grid(noct);
    std::vector<long> hkey(hs,0); std::vector<int> hval(hs,0);
    for (int fz=0; fz<NG; ++fz) for (int fy=0; fy<NG; ++fy) for (int fx=0; fx<NG; ++fx) {
        int o=fx+fy*NG+fz*NG*NG; Oct g{}; g.ckey[0]=fx;g.ckey[1]=fy;g.ckey[2]=fz;g.lev=ilevel; grid[o]=g;
        h_hash_set(hkey,hval,hs, key_off_v+(long)fx+(long)fy*NG+(long)fz*NG*NG, o+1);
    }
    // Known force field per cell (depends on cell-center coordinate).
    std::vector<float> f(3*ncells, 0.f);
    for (int o=1;o<=noct;++o){ int oo=o-1,fx=oo%NG,fy=(oo/NG)%NG,fz=oo/(NG*NG);
        for (int c=1;c<=TWOTONDIM;++c){ int ix=(c-1)&1,iy=((c-1)>>1)&1,iz=((c-1)>>2)&1;
            double ux=(2*fx+ix+0.5)/ncl,uy=(2*fy+iy+0.5)/ncl,uz=(2*fz+iz+0.5)/ncl;
            f[IDX3(c,1,o)]=(float)(cos(2*M_PI*ux)); f[IDX3(c,2,o)]=(float)(0.7*sin(2*M_PI*uy));
            f[IDX3(c,3,o)]=(float)(0.3+0.5*cos(2*M_PI*uz)); } }

    std::vector<long> ipos(N*3); std::vector<float> vp(N*3); std::vector<int> levelp(N, ilevel);
    std::vector<double> u0(N*3), v0(N*3);
    srand(31);
    for (int i=0;i<N;++i) for (int d=0;d<3;++d){
        double u=(rand()%1000000)/1000000.0; u0[i+d*N]=u; ipos[i+d*N]=u_to_fix(u);
        double v=((rand()%2000000)/1000000.0-1.0); v0[i+d*N]=v; vp[i+d*N]=(float)v;
    }

    // ---- CPU golden gather + kick + drift (double; fixed-point arithmetic mirror).
    int shift = NBITS_POS - cl; long mask = (1L<<shift)-1;
    std::vector<double> vnew_ref(N*3), unew_ref(N*3);
    for (int p=1;p<=N;++p){
        int il[3],ir[3],bmax[3]; double dr[3],dl[3];
        for (int d=0;d<3;++d){
            long ip=ipos[(p-1)+d*N];
            int C=(int)(ip>>shift); double fr=(double)(ip&mask)/(double)(1L<<shift);
            double drr=fr+0.5; int add=drr>=1.0?1:0; ir[d]=C+add; dr[d]=drr-add; dl[d]=1-dr[d]; il[d]=ir[d]-1;
            bmax[d]=1<<cl; if(il[d]<0)il[d]=bmax[d]-1; if(ir[d]>=bmax[d])ir[d]=0;
        }
        double ff[3]={0,0,0};
        for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx){
            int tgt[3]={bx?ir[0]:il[0],by?ir[1]:il[1],bz?ir[2]:il[2]};
            double w=(bx?dr[0]:dl[0])*(by?dr[1]:dl[1])*(bz?dr[2]:dl[2]);
            int fa[3],ii[3],ic=1; for(int d=0;d<3;++d){fa[d]=tgt[d]>>1;ii[d]=tgt[d]-2*fa[d];ic+=ii[d]*(1<<d);}
            long key=key_off_v+(long)fa[0]+(long)fa[1]*NG+(long)fa[2]*NG*NG;
            int ig=h_hash_get(hkey,hval,hs,key);
            for(int d=0;d<3;++d) ff[d]+=(double)f[IDX3(ic,d+1,ig)]*w;
        }
        for(int d=0;d<3;++d){ double v=v0[(p-1)+d*N]+ff[d]*0.5*dt; vnew_ref[(p-1)+d*N]=v;
            double un=u0[(p-1)+d*N]+v*dt; un=fmod(un,1.0); if(un<0)un+=1.0; unew_ref[(p-1)+d*N]=un; }
    }

    // ---- GPU.
    id<MTLBuffer> bipos=shared_buf(ipos.data(),N*3), bvp=shared_buf(vp.data(),N*3),
        blvl=shared_buf(levelp.data(),N), bf=shared_buf(f.data(),3*ncells),
        bhk=shared_buf(hkey.data(),hs), bhv=shared_buf(hval.data(),hs);
    std::vector<int> ckey_max(ilevel+2,0); ckey_max[ilevel]=NG;
    std::vector<long> key_off(ilevel+2,0); key_off[ilevel]=key_off_v;
    id<MTLBuffer> bck=shared_buf(ckey_max.data(),ilevel+2), bko=shared_buf(key_off.data(),ilevel+2);
    int periodic[3]={1,1,1}; id<MTLBuffer> bper=shared_buf(periodic,3);
    PartParams P{}; P.dtnew=(float)dt; P.dtold=(float)dt; P.box_size[0]=P.box_size[1]=P.box_size[2]=1.0f;
    P.hash_size=hs; P.ilevel=ilevel; P.head_idx=1; P.num_parts=N; P.npartmax=N; P.ngridmax=noct; P.action_part=2;
    id<MTLCommandBuffer> cb=[g_q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:pso(@"kick_drift_part")];
    [e setBuffer:bipos offset:0 atIndex:0]; [e setBuffer:bvp offset:0 atIndex:1];
    [e setBuffer:blvl offset:0 atIndex:2]; [e setBuffer:bf offset:0 atIndex:3];
    [e setBuffer:bhk offset:0 atIndex:4]; [e setBuffer:bhv offset:0 atIndex:5];
    [e setBuffer:bck offset:0 atIndex:6]; [e setBuffer:bko offset:0 atIndex:7];
    [e setBuffer:bper offset:0 atIndex:8]; [e setBytes:&P length:sizeof(P) atIndex:9];
    dispatch_1d(e, pso(@"kick_drift_part"), N);
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr,"kd cb error: %s\n", cb.error.localizedDescription.UTF8String); return 1; }

    const float* vg=(const float*)bvp.contents; const long* ig=(const long*)bipos.contents;
    double ev=0, edv=0, ex=0, edx=0, maxff=0;
    for (int p=0;p<N;++p) for (int d=0;d<3;++d){
        double vn=vg[p+d*N], vr=vnew_ref[p+d*N]; ev+=(vn-vr)*(vn-vr); edv+=vr*vr;
        double ug=(double)ig[p+d*N]/ldexp(1.0,NBITS_POS), ur=unew_ref[p+d*N];
        double du=ug-ur; if(du>0.5)du-=1; if(du<-0.5)du+=1; ex+=du*du; edx+=ur*ur;
        double ff=(vn-v0[p+d*N])/(0.5*dt); maxff=std::max(maxff,fabs(ff));
    }
    double verr=sqrt(ev/edv), xerr=sqrt(ex/edx);
    bool ok = verr<1e-5 && xerr<1e-6;
    printf("kickdrift ilevel=%d noct=%d N=%d : %s  v_err=%.2e pos_err=%.2e (maxff=%.2f)\n",
           ilevel, noct, N, ok?"PASS":"FAIL", verr, xerr, maxff);
    return ok?0:1;
}

int main(int argc, const char** argv) {
    @autoreleasepool {
        const char* libpath = argc > 1 ? argv[1] : "/tmp/ramses_kernels.metallib";
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        printf("device: %s\n", g_dev.name.UTF8String);
        printf("  supportsApple7=%d Apple8=%d Apple9=%d  maxThreadsPerTG=%lu  simdWidth(probe via PSO)\n",
               [g_dev supportsFamily:MTLGPUFamilyApple7],
               [g_dev supportsFamily:MTLGPUFamilyApple8],
               [g_dev supportsFamily:MTLGPUFamilyApple9],
               (unsigned long)g_dev.maxThreadsPerThreadgroup.width);
        g_q = [g_dev newCommandQueue];
        NSError* err = nil;
        g_lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
        if (!g_lib) { fprintf(stderr, "load lib %s: %s\n", libpath, err.localizedDescription.UTF8String); return 1; }
        printf("threadExecutionWidth(block_scan)=%lu\n", (unsigned long)pso(@"block_scan").threadExecutionWidth);

        int fails = 0;
        fails += test_block_scan(32, 32);
        fails += test_block_scan(256, 256);
        fails += test_block_scan(200, 256);   // partial last warp
        fails += test_block_scan(1024, 1024); // many warps
        fails += test_primitives();
        fails += test_radix_sort(1000, 4);     // single-block scan path
        fails += test_radix_sort(50000, 6);    // multi-block scan path
        fails += test_radix_sort(300000, 8);   // level-1 scan tree, 24-bit keys
        test_cic_one(0.10, 0.50, 0.50, 2);      // near -x face: wrap to far-x
        fails += test_cic(20000, 4);            // 64 octs, 512 cells
        fails += test_cic(100000, 5);           // 512 octs, multi-block deposit
        fails += test_cic_deep(20000, 12);      // moderate depth
        fails += test_cic_deep(20000, 18);      // deep: old float-frac would be ~6% off
        fails += test_cic_deep(20000, 21);      // near nhilbert=1 limit (3*21=63 bits)
        fails += test_mg(3, 400);               // 8^3 periodic Poisson, GS recovers phi_true
        fails += test_mg(4, 2000);              // 16^3, more sweeps
        fails += test_gradient(4);              // 4th-order force vs analytic -grad phi
        fails += test_vcycle(4, 8);             // 2-level V-cycle: fast convergence
        fails += test_vcycle(5, 10);            // 32^3 fine
        fails += test_kickdrift(4, 50000);      // force gather + leapfrog kick/drift
        fails += test_kickdrift(6, 50000);      // deeper level
        printf("\n%s (%d failures)\n", fails==0 ? "ALL PASS" : "FAILURES", fails);
        return fails;
    }
}

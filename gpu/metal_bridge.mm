//============================================================================
// metal_bridge.mm — Obj-C++ host bridge between the Fortran mini-ramses driver
// and the validated Metal compute kernels (gpu/metal/*.metal -> .metallib).
//
// Layering:  Fortran gpu_*.f90  ->  bind(C) shims here (extern "C")  ->  Metal.
//
// Holds persistent state: MTLDevice/queue, the loaded MTLLibrary, a pipeline
// cache, and a registry of MTLStorageModeShared buffers (the former CUDA
// `device, allocatable` arrays).  On Apple unified memory these buffers are
// CPU/GPU-coherent, so the Fortran host fills/reads them directly via pointers
// obtained from mtl_get_ptr_* + c_f_pointer (no explicit H<->D copies).
//
// This file is plain Metal-API Obj-C++ (no metal-cpp dependency).
//============================================================================
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "ramses_metal.h"
static double g_mg_hier=0.0, g_mg_vcyc=0.0;   // MG hierarchy-build vs V-cycle wall (s)
extern "C" void mtl_get_mg_times(double* h, double* v){ *h=g_mg_hier; *v=g_mg_vcyc; }
// Persistent per-level multigrid safe-mode flag (mirrors g%safe_mode in
// amr_commons.f90: initialised .false., only ever escalated to .true., never
// reset for the rest of the run).  Indexed by ilevel (<= nlevelmax).
static int g_mg_safe[256] = {0};

//---------------------------------------------------------------------------
namespace {
id<MTLDevice>           g_dev   = nil;
id<MTLCommandQueue>     g_queue = nil;
id<MTLLibrary>          g_lib   = nil;
std::unordered_map<std::string, id<MTLComputePipelineState>> g_pso;

struct Buffers {
    // Mesh (filled by host r_set_grid_device, read back by r_transfer_grid_host)
    id<MTLBuffer> grid, nbor, father, hash_key, hash_val, ckey_max, key_off, box_min, box_max;
    id<MTLBuffer> rho, nref, phi, f, phi_old;             // fields (float)
    id<MTLBuffer> flag1, flag2;                           // refinement map (int)
    id<MTLBuffer> rho_lo, rho_hi, nref_lo, nref_hi;       // fixed-point deposit accumulators
    // Particles (+ swap buffers for the on-GPU reorder)
    id<MTLBuffer> ipos, vp, mp, levelp, sortp, isp, hkey_part, idp;
    id<MTLBuffer> ipos2, vp2, mp2, levelp2, bucket, idp2;
    // Scan scratch
    id<MTLBuffer> prefix, ps0, ps1, total;
    // Refine: compaction scratch (grid + widest float field) + permutation + free counter
    id<MTLBuffer> grid2, fld2, swap, ifree_ctr, kill_ctr, redbuf, fbk;
    // Multigrid residual-norm partials (one float per threadgroup over the fine level)
    id<MTLBuffer> mgnorm, mgnorm_i;
    // Cache-oct (coarse-fine boundary ghost) compaction scratch: per-oct missing-nbor
    // predicate / inclusive scan, oct-indexed.
    id<MTLBuffer> cache_pre;
    int ncell, npartmax, hash_size, nlevelmax;
    int ngridmax;        // real-oct bound; cache (ghost) octs live at (ngridmax, ncell]
    int ifree_cache;     // next free cache slot offset (0-based, beyond ngridmax)
} B;

id<MTLComputePipelineState> pso(const char* name) {
    auto it = g_pso.find(name);
    if (it != g_pso.end()) return it->second;
    NSError* err = nil;
    id<MTLFunction> fn = [g_lib newFunctionWithName:@(name)];
    if (!fn) { fprintf(stderr, "[metal] missing kernel '%s'\n", name); return nil; }
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "[metal] pso '%s': %s\n", name, err.localizedDescription.UTF8String); return nil; }
    g_pso[name] = p;
    return p;
}

template <class T> id<MTLBuffer> newbuf(size_t n) {
    return [g_dev newBufferWithLength:n*sizeof(T) options:MTLResourceStorageModeShared];
}
void dispatch1d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int n) {
    NSUInteger t = MIN((NSUInteger)256, p.maxTotalThreadsPerThreadgroup);
    [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(t,1,1)];
}

//----------------------------------------------------------------------------
// Async command-buffer submission.  g_queue is a SERIAL queue, so committed
// command buffers execute in submission order -> data dependencies through the
// shared buffers are preserved WITHOUT a CPU-side waitUntilCompleted.  We only
// block (mtl_drain) when the HOST must touch a GPU buffer (read a result, or
// memset/memcpy a buffer that an in-flight cb might still be using).  This keeps
// the GPU queue full instead of paying a round-trip latency per dispatch.
static id<MTLCommandBuffer> g_lastcb = nil;
static inline void submit_async(id<MTLCommandBuffer> cb) { [cb commit]; g_lastcb = cb; }
// Block until all submitted GPU work has completed (serial queue -> waiting on
// the last command buffer drains everything before it).
extern "C" void mtl_drain(void) { if (g_lastcb) { [g_lastcb waitUntilCompleted]; g_lastcb = nil; } }

//---------------------------------------------------------------------------
// HOST-side hash (the GPU table is read-only via hash_get; all insertion is
// here on the CPU, off the per-step hot path).  These MUST reproduce the
// device fnv64/hash_bucket/hash_get in ramses_msl.h bit-for-bit so a key built
// here lands in the same bucket a kernel later probes.
//---------------------------------------------------------------------------
inline long h_fnv64(long key_signed) {
    unsigned long key = (unsigned long)key_signed;
    unsigned long h   = 14695981039346656037UL;          // fnv64_basis
    for (int j = 0; j < 8; ++j) {
        unsigned long k = (key >> (8 * j)) & 0xFFUL;
        h ^= k;
        h *= 1099511628211UL;                             // fnv64_prime
    }
    return (long)h;
}
inline int h_hash_bucket(long key, int hash_size) {
    long ib = h_fnv64(key) % (long)hash_size;             // C '%' == Fortran MOD (trunc)
    if (ib < 0) ib += hash_size;
    return (int)ib + 1;                                   // 1-based
}
// Linear-probe insert (serial: no atomics needed).  Sentinel 0 == empty.
inline void h_hash_set(long* hkey, int* hval, int hash_size, long key, int val) {
    int b = h_hash_bucket(key, hash_size);
    for (;;) {
        long cur = hkey[b - 1];
        if (cur == 0 || cur == key) { hkey[b - 1] = key; hval[b - 1] = val; return; }
        b = (b % hash_size) + 1;
    }
}
inline int h_hash_get(const long* hkey, const int* hval, int hash_size, long key) {
    int b = h_hash_bucket(key, hash_size);
    for (;;) {
        long cur = hkey[b - 1];
        if (cur == key) return hval[b - 1];
        if (cur == 0)   return 0;
        b = (b % hash_size) + 1;
    }
}
// Globally-unique integer key for an oct at (level, ckey), matching gpu_refine.cuf:
//   key = key_off(level) + ix + iy*nx + iz*nx*nx,  nx = ckey_max(level).
inline long h_oct_key(const int* ckey_max, const long* key_off, int level,
                      long ix, long iy, long iz) {
    long nx = (long)ckey_max[level];                      // 1-based-padded (slot L)
    return key_off[level] + ix + iy*nx + iz*nx*nx;
}
// NDIM-generic oct key from a Cartesian-key array (unused dims must be 0).
inline long h_oct_key_n(const int* ckey_max, const long* key_off, int level, const long* ck) {
    long nx = (long)ckey_max[level], key = key_off[level], st = 1;
    for (int d = 0; d < NDIM; ++d) { key += ck[d] * st; st *= nx; }
    return key;
}
// Read an Oct's Cartesian key into a 3-long array (unused dims = 0).
inline void h_read_ckey(const Oct& o, long ck[3]) {
    ck[0]=ck[1]=ck[2]=0;
    for (int d = 0; d < NDIM; ++d) ck[d] = o.ckey[d];
}

// --- device-wide inclusive scan (gpu_scan): block_scan -> spine -> uniform_add.
void enc_block_scan(id<MTLComputeCommandEncoder> e, id<MTLBuffer> data, id<MTLBuffer> psum,
                    id<MTLBuffer> total, int offset, int size, int flags, int tg, bool oneblk) {
    auto p = pso("block_scan");
    [e setComputePipelineState:p];
    [e setBuffer:data offset:0 atIndex:0];
    [e setBuffer:(psum?psum:data) offset:0 atIndex:1];
    [e setBuffer:(total?total:data) offset:0 atIndex:2];
    [e setBytes:&offset length:4 atIndex:3];
    [e setBytes:&size length:4 atIndex:4];
    [e setBytes:&flags length:4 atIndex:5];
    int nb = oneblk ? 1 : (size + tg - 1)/tg;
    int t  = oneblk ? size : tg;
    [e dispatchThreadgroups:MTLSizeMake(nb,1,1) threadsPerThreadgroup:MTLSizeMake(t,1,1)];
}
void enc_uniform_add(id<MTLComputeCommandEncoder> e, id<MTLBuffer> data, id<MTLBuffer> psum,
                     id<MTLBuffer> total, int offset, int size, int flags, int tg, int nblocks) {
    auto p = pso("uniform_add");
    [e setComputePipelineState:p];
    [e setBuffer:data offset:0 atIndex:0];
    [e setBuffer:psum offset:0 atIndex:1];
    [e setBuffer:(total?total:data) offset:0 atIndex:2];
    [e setBytes:&offset length:4 atIndex:3];
    [e setBytes:&size length:4 atIndex:4];
    [e setBytes:&flags length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(nblocks,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}
// --- Multigrid hierarchy (geometric coarsening of a uniform periodic base) ---
struct MgLevel { id<MTLBuffer> grid, nbor, phi, f, father; int nl, noct; };
std::vector<MgLevel> g_mg;
int g_mg_nl = -1;

static void fill_periodic_nbor(int* nbor, int nl) {
    auto oi=[&](int x,int y,int z){return x+y*nl+z*nl*nl;};
    for (int z=0;z<nl;++z) for (int y=0;y<nl;++y) for (int x=0;x<nl;++x){
        int o=oi(x,y,z);
        for (int kk=-1;kk<=1;++kk) for (int jj=-1;jj<=1;++jj) for (int ii=-1;ii<=1;++ii){
            int ind=1+(1+ii)+3*(1+jj)+9*(1+kk);
            nbor[o*27+(ind-1)]=oi((x+ii+nl)%nl,(y+jj+nl)%nl,(z+kk+nl)%nl)+1;
        }
    }
}
// Build the coarse hierarchy for a uniform periodic base of nl_fine^3 octs.
// Level 0 aliases the main buffers (B.grid/nbor/phi/f); levels >=1 are freshly
// allocated coarsenings down to nl==1.  father[L] maps level-L oct -> level-(L+1) oct.
static void build_mg_hierarchy(int nl_fine) {
    if (g_mg_nl == nl_fine) return;
    g_mg.clear();
    int nl = nl_fine, L = 0;
    while (true) {
        MgLevel lev; lev.nl = nl; lev.noct = nl*nl*nl;
        if (L == 0) { lev.grid=B.grid; lev.nbor=B.nbor; lev.phi=B.phi; lev.f=B.f; }
        else {
            lev.grid = newbuf<Oct>(lev.noct);
            lev.nbor = newbuf<int>((size_t)lev.noct*27);
            lev.phi  = newbuf<float>((size_t)lev.noct*TWOTONDIM);
            lev.f    = newbuf<float>((size_t)lev.noct*TWOTONDIM*NF);
            Oct* g=(Oct*)lev.grid.contents;
            for (int z=0;z<nl;++z) for (int y=0;y<nl;++y) for (int x=0;x<nl;++x){
                int o=x+y*nl+z*nl*nl; g[o]=Oct{}; g[o].ckey[0]=x;
#if NDIM>=2
                g[o].ckey[1]=y;
#endif
#if NDIM>=3
                g[o].ckey[2]=z;
#endif
            }
            fill_periodic_nbor((int*)lev.nbor.contents, nl);
            // coarse mask = interior everywhere
            float* ff=(float*)lev.f.contents;
            for (int o=1;o<=lev.noct;++o) for (int c=1;c<=TWOTONDIM;++c) ff[IDX3(c,3,o)]=1.0f;
        }
        lev.father = nil;
        g_mg.push_back(lev);
        if (nl == 1) break;
        nl /= 2; ++L;
    }
    // father maps for each level L -> L+1.  Level 0 is the REAL base mesh whose
    // octs are in Hilbert (not x+y*nl+z*nl^2) order, so read each oct's actual
    // ckey from B.grid; coarser levels (>=1) are canonical-order full grids.
    for (size_t L=0; L+1<g_mg.size(); ++L) {
        int nlf=g_mg[L].nl, nlc=g_mg[L+1].nl;
        g_mg[L].father = newbuf<int>(g_mg[L].noct);
        int* fa=(int*)g_mg[L].father.contents;
        if (L==0) {
            Oct* g=(Oct*)g_mg[0].grid.contents;
            for (int o=0;o<g_mg[0].noct;++o) {
                int cx=g[o].ckey[0], cy=0, cz=0;
#if NDIM>=2
                cy=g[o].ckey[1];
#endif
#if NDIM>=3
                cz=g[o].ckey[2];
#endif
                fa[o] = (cx/2)+(cy/2)*nlc+(cz/2)*nlc*nlc + 1;
            }
        } else {
            for (int z=0;z<nlf;++z) for (int y=0;y<nlf;++y) for (int x=0;x<nlf;++x)
                fa[x+y*nlf+z*nlf*nlf] = (x/2)+(y/2)*nlc+(z/2)*nlc*nlc + 1;
        }
    }
    g_mg_nl = nl_fine;
}

static void enc_gs(id<MTLComputeCommandEncoder> e, const MgLevel& lev, int nsweep, int safe) {
    MgParams P{}; P.head_idx=1; P.num_octs=lev.noct; P.ngridmax=lev.noct;
    for (int s=0;s<nsweep;++s) for (int rb=1;rb>=0;--rb){ P.redstep=rb;
        [e setComputePipelineState:pso("gauss_seidel")];
        [e setBuffer:lev.phi offset:0 atIndex:0]; [e setBuffer:lev.f offset:0 atIndex:1];
        [e setBuffer:lev.nbor offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
        [e setBytes:&safe length:4 atIndex:4];
        [e setBuffer:B.grid offset:0 atIndex:5]; [e setBuffer:B.father offset:0 atIndex:6];
        [e setBuffer:B.phi_old offset:0 atIndex:7];   // bound (use_ghost=0 -> unread)
        [e dispatchThreads:MTLSizeMake(4,lev.noct,1) threadsPerThreadgroup:MTLSizeMake(4,16,1)]; }
}
static void enc_residual(id<MTLComputeCommandEncoder> e, const MgLevel& lev) {
    MgParams P{}; P.head_idx=1; P.num_octs=lev.noct; P.ngridmax=lev.noct;
    [e setComputePipelineState:pso("cmp_residual")];
    [e setBuffer:lev.phi offset:0 atIndex:0]; [e setBuffer:lev.f offset:0 atIndex:1];
    [e setBuffer:lev.nbor offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
    [e setBuffer:B.grid offset:0 atIndex:4]; [e setBuffer:B.father offset:0 atIndex:5];
    [e setBuffer:B.phi_old offset:0 atIndex:6];   // bound (use_ghost=0 -> unread)
    [e dispatchThreads:MTLSizeMake(TWOTONDIM,lev.noct,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
}
static void enc_restrict(id<MTLComputeCommandEncoder> e, const MgLevel& fine, const MgLevel& coarse) {
    MgParams P{}; P.head_idx=1; P.num_octs=fine.noct; P.head_father=1;
    [e setComputePipelineState:pso("restrict_residual")];
    [e setBuffer:fine.grid offset:0 atIndex:0]; [e setBuffer:fine.father offset:0 atIndex:1];
    [e setBuffer:fine.f offset:0 atIndex:2]; [e setBuffer:coarse.f offset:0 atIndex:3];
    [e setBytes:&P length:sizeof(P) atIndex:4];
    dispatch1d(e, pso("restrict_residual"), fine.noct);
}
static void enc_interp(id<MTLComputeCommandEncoder> e, const MgLevel& fine, const MgLevel& coarse) {
    MgParams P{}; P.head_idx=1; P.num_octs=fine.noct; P.head_father=1;
    [e setComputePipelineState:pso("interpolate_correct")];
    [e setBuffer:fine.grid offset:0 atIndex:0]; [e setBuffer:fine.father offset:0 atIndex:1];
    [e setBuffer:coarse.nbor offset:0 atIndex:2]; [e setBuffer:fine.phi offset:0 atIndex:3];
    [e setBuffer:coarse.phi offset:0 atIndex:4]; [e setBuffer:fine.f offset:0 atIndex:5];
    [e setBytes:&P length:sizeof(P) atIndex:6];
    [e dispatchThreads:MTLSizeMake(TWOTONDIM,fine.noct,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
}

void gpu_scan(id<MTLComputeCommandEncoder> e, id<MTLBuffer> prefix, int head, int n) {
    const int Bk = 256; int g0 = (n + Bk - 1)/Bk;
    if (n <= Bk*Bk) {
        enc_block_scan(e, prefix, B.ps0, B.total, head, n, 1|2, Bk, false);
        if (g0 > 1) { enc_block_scan(e, B.ps0, nil, nil, 1, g0, 0, g0, true);
            enc_uniform_add(e, prefix, B.ps0, B.total, head+Bk, n-Bk, 2, Bk, g0-1); }
    } else {
        int g1 = (g0 + Bk - 1)/Bk;
        enc_block_scan(e, prefix, B.ps0, nil, head, n, 1, Bk, false);
        enc_block_scan(e, B.ps0, B.ps1, nil, 1, g0, 1, Bk, false);
        if (g1 > 1) { enc_block_scan(e, B.ps1, nil, nil, 1, g1, 0, g1, true);
            enc_uniform_add(e, B.ps0, B.ps1, nil, 1+Bk, g0-Bk, 0, Bk, g1-1); }
        enc_uniform_add(e, prefix, B.ps0, B.total, head+Bk, n-Bk, 2, Bk, g0-1);
    }
}
} // namespace

//---------------------------------------------------------------------------
extern "C" {

// Initialise device, queue, and load the kernel library.  Returns 0 on success.
int mtl_init(const char* metallib_path) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { fprintf(stderr, "[metal] no device\n"); return 1; }
        if (![g_dev supportsFamily:MTLGPUFamilyApple7]) {
            fprintf(stderr, "[metal] requires Apple7+ (atomics)\n"); return 2; }
        g_queue = [g_dev newCommandQueue];
        NSError* err = nil;
        g_lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path)] error:&err];
        if (!g_lib) { fprintf(stderr, "[metal] load '%s': %s\n", metallib_path,
                              err.localizedDescription.UTF8String); return 3; }
#if NDIM == 3
        static_assert(sizeof(Oct)==64, "Oct layout must match Fortran type oct (64 B)");
#endif
        // For NDIM<3 the Oct struct auto-sizes from NDIM/TWOTONDIM/NHILBERT; the
        // Fortran `type oct` uses the same params, so layouts still match.
        // NDIM GUARD: the loaded metallib MUST share this binary's NDIM.  Otherwise
        // kernels write each oct with the wrong TWOTONDIM stride while the host reads
        // another -> every (8/TWOTONDIM)-th oct nonzero, ~1000x-wrong force, SILENT
        // garbage (e.g. a 1D binary loading the default ../../bin NDIM=3 lib).  Probe
        // the lib's compile-time NDIM and abort loudly on mismatch.
        if (id<MTLComputePipelineState> pp = pso("probe_ndim")) {
            id<MTLBuffer> probe = newbuf<int>(2);
            ((int*)probe.contents)[0] = -1; ((int*)probe.contents)[1] = -1;
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pp];
            [e setBuffer:probe offset:0 atIndex:0];
            [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            int lib_ndim = ((int*)probe.contents)[0];
            if (lib_ndim != NDIM) {
                fprintf(stderr, "[metal] FATAL: metallib '%s' compiled NDIM=%d but this "
                        "binary is NDIM=%d (TWOTONDIM %d vs %d). Set RAMSES_METALLIB to the "
                        "matching NDIM=%d lib.\n", metallib_path, lib_ndim, NDIM,
                        (1<<lib_ndim), TWOTONDIM, NDIM);
                return 4;
            }
        } else {
            fprintf(stderr, "[metal] WARNING: metallib '%s' lacks probe_ndim kernel; "
                    "cannot verify NDIM match (rebuild the metallib).\n", metallib_path);
        }
        fprintf(stderr, "[metal] init on %s\n", g_dev.name.UTF8String);
        return 0;
    }
}

// Allocate all Shared device buffers.  Sizes from the host (ngridmax, npartmax, ...).
void mtl_alloc_buffers(int ncell, int npartmax, int hash_size, int nlevelmax) {
    B.ncell=ncell; B.npartmax=npartmax; B.hash_size=hash_size; B.nlevelmax=nlevelmax;
    size_t nc = (size_t)ncell, np = (size_t)npartmax;
    B.grid     = newbuf<Oct>(nc);
    B.nbor     = newbuf<int>(nc*SUBGRIDSIZE);
    B.father   = newbuf<int>(nc);
    B.hash_key = newbuf<long>((size_t)hash_size);
    B.hash_val = newbuf<int>((size_t)hash_size);
    B.ckey_max = newbuf<int>((size_t)nlevelmax+2);
    B.key_off  = newbuf<long>((size_t)nlevelmax+2);
    B.box_min  = newbuf<int>((size_t)3*nlevelmax+3);
    B.box_max  = newbuf<int>((size_t)3*nlevelmax+3);
    B.rho      = newbuf<float>(nc*TWOTONDIM);
    B.nref     = newbuf<float>(nc*TWOTONDIM);
    B.flag1    = newbuf<int>(nc*TWOTONDIM);
    B.flag2    = newbuf<int>(nc*TWOTONDIM);
    memset(B.nref.contents, 0, nc*TWOTONDIM*sizeof(float));  // zero before first flag (pre-deposit)
    memset(B.rho.contents,  0, nc*TWOTONDIM*sizeof(float));
    B.phi      = newbuf<float>(nc*TWOTONDIM);
    B.phi_old  = newbuf<float>(nc*TWOTONDIM);
    memset(B.phi_old.contents, 0, nc*TWOTONDIM*sizeof(float)); // valid before first save (tfrac=0 anyway)
    B.f        = newbuf<float>(nc*TWOTONDIM*NF);
    B.rho_lo   = newbuf<unsigned>(nc*TWOTONDIM); B.rho_hi  = newbuf<unsigned>(nc*TWOTONDIM);
    B.nref_lo  = newbuf<unsigned>(nc*TWOTONDIM); B.nref_hi = newbuf<unsigned>(nc*TWOTONDIM);
    B.ipos     = newbuf<long>(np*NDIM);
    B.vp       = newbuf<float>(np*NDIM);
    B.mp       = newbuf<float>(np);
    B.levelp   = newbuf<int>(np);
    B.idp      = newbuf<int>(np);          // particle unique label (diagnostic; carried through reorder)
    B.idp2     = newbuf<int>(np);
    B.sortp    = newbuf<int>(np);
    B.isp      = newbuf<int>(np);
    B.hkey_part= newbuf<long>(np);
    B.ipos2    = newbuf<long>(np*NDIM);
    B.vp2      = newbuf<float>(np*NDIM);
    B.mp2      = newbuf<float>(np);
    B.levelp2  = newbuf<int>(np);
    B.bucket   = newbuf<int>(np);
    B.prefix   = newbuf<int>(np);
    int g0=((int)np+255)/256, g1=(g0+255)/256;
    B.ps0 = newbuf<int>(g0>0?g0:1); B.ps1 = newbuf<int>(g1>0?g1:1); B.total = newbuf<int>(1);
    // Refine compaction scratch: grid copy + a float field wide enough for f (twotondim*ndim).
    B.grid2     = newbuf<Oct>(nc);
    B.fld2      = newbuf<float>(nc*TWOTONDIM*NF);
    B.swap      = newbuf<int>(nc);
    B.ifree_ctr = newbuf<int>(1);
    B.kill_ctr  = newbuf<int>(1);
    B.redbuf    = newbuf<unsigned>(3);   // newdt: [vmax bits, ekin_lo, ekin_hi]
    B.fbk       = newbuf<unsigned>(1);    // DIAG: make_initial_phi centre-cell-fallback hits
    // residual_norm writes one df64 (hi,lo = 2 floats) per threadgroup -> size x2.
    B.mgnorm    = newbuf<float>(2*((size_t)((nc + 255) / 256) + 1));  // MG residual-norm df64 partials
    B.mgnorm_i  = newbuf<float>(2*((size_t)((nc + 255) / 256) + 1));  // iter-1 initial-norm df64 partials
    B.cache_pre = newbuf<int>(nc);                                // cache-oct missing-nbor predicate/scan
    // Real-oct bound: cache (ghost) octs occupy (ngridmax, ncell].  When the caller
    // sizes ncell == ngridmax (no cache region) ngridmax == ncell and every nbor is
    // "real" -> the cache-oct boundary path is inert (identical to the pre-cache port).
    B.ngridmax   = ncell;
    B.ifree_cache = 0;
}

// Declare the real-oct bound: octs 1..ngridmax are real; (ngridmax, ncell] is the
// cache (coarse-fine ghost) region that mtl_make_cache fills.  Call after
// mtl_alloc_buffers with ngridmax < ncell to ACTIVATE the cache-oct boundary path
// (the kernels treat a neighbour > ngridmax as a materialised ghost).  ngridmax ==
// ncell (the default) keeps the path inert.
void mtl_set_cache_region(int ngridmax) {
    B.ngridmax = (ngridmax > 0 && ngridmax <= B.ncell) ? ngridmax : B.ncell;
    B.ifree_cache = 0;
}

// Per-level particle CFL reduction on the GPU (replaces CPU newdt_part): returns
// vmax = max|v| and ekin = sum 0.5 m v^2 over [head, head+num).  No host vp needed.
void mtl_newdt_part(int ilevel, int head, int num, int npartmax,
                    double* vmax_out, double* ekin_out) {
    unsigned* r = (unsigned*)B.redbuf.contents;
    r[0]=0u; r[1]=0u; r[2]=0u;
    if (num > 0) {
        @autoreleasepool {
            ScanParams S{}; S.n=num; S.head_idx=head; S.npartmax=npartmax; S.ilevel=ilevel;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("newdt_part_reduce")];
            [e setBuffer:B.vp offset:0 atIndex:0]; [e setBuffer:B.mp offset:0 atIndex:1];
            [e setBuffer:B.redbuf offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
            dispatch1d(e, pso("newdt_part_reduce"), num);
            [e endEncoding]; submit_async(cb);
        }
    }
    mtl_drain();                       // host reads the reduction result below
    float vm; memcpy(&vm, &r[0], sizeof(float));
    long q = (long)(((unsigned long)r[2] << 32) | (unsigned long)r[1]);
    *vmax_out = (double)vm;
    *ekin_out = (double)q / (double)(1L << FP_SHIFT_RHO);
}

// Expose buffer contents pointers so the Fortran host can c_f_pointer them.
void* mtl_ptr_grid()     { return B.grid.contents; }
void* mtl_ptr_nbor()     { return B.nbor.contents; }
void* mtl_ptr_father()   { return B.father.contents; }
void* mtl_ptr_flag1()    { return B.flag1.contents; }

// AMR refinement flagging on the GPU (mirrors m_flag_fine): reset flag1, propagate
// from level (ilevel+1) children, smooth (3 passes, thresholds 1/2/2), apply the
// nref refine threshold, smooth nexpand more times, optionally enforce rules.
// head/num = level ilevel octs; head1/num1 = level (ilevel+1) octs.  Leaves the
// refinement map in B.flag1 (host then copies it for the CPU refine).
void mtl_flag(int head, int num, int head1, int num1, int ngridmax,
              float m_refine, int nexpand, int do_rules) {
    if (num <= 0) return;
    const int n_nbor[3] = {1, 2, 2};
    // Per-pass L-R mirror-symmetry dump (RAMSES_FLAG_DUMP): after each flag pass, flush and
    // report total flagged + the count of cells whose flag1 differs from their mirror cell
    // (1D: mirror ckey = nx-1-c, mirror cell = 3-ind).  The FIRST pass with asym>0 localizes
    // where left-right symmetry breaks.  Diagnostic only; off unless the env is set.
    const bool dbg = (getenv("RAMSES_FLAG_DUMP") != nullptr);
    std::unordered_map<int,int> idx_by_ck;
    int nx_lev = 0;
    if (dbg) {
        const Oct* g = (const Oct*)B.grid.contents;
        const int* ckm = (const int*)B.ckey_max.contents;
        nx_lev = ckm[ g[head-1].lev ];
        for (int o=0;o<num;++o) idx_by_ck[ g[head-1+o].ckey[0] ] = head+o;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        FlagParams P{}; P.head_idx=head; P.num_octs=num; P.ngridmax=ngridmax;
        auto dump=[&](const char* lbl){
            if (!dbg) return;
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            const int* f = (const int*)B.flag1.contents;
            const Oct* g = (const Oct*)B.grid.contents;
            int total=0, asym=0, firstck=-1;
            for (int o=0;o<num;++o){
                int oct=head+o, c0=g[oct-1].ckey[0];
                auto it=idx_by_ck.find(nx_lev-1-c0);
                int moct=(it==idx_by_ck.end())?0:it->second;
                for (int ind=1; ind<=TWOTONDIM; ++ind){
                    int v=f[(oct-1)*TWOTONDIM+(ind-1)]; total+=v;
                    if (moct){ int mv=f[(moct-1)*TWOTONDIM+(2-ind)];
                               if (v!=mv){ asym++; if(firstck<0||c0<firstck) firstck=c0; } }
                }
            }
            fprintf(stderr,"[FLAGDUMP] %-14s total=%d asym=%d firstck=%d\n",lbl,total,asym,firstck);
            cb=[g_queue commandBuffer]; e=[cb computeCommandEncoder];
        };
        // reset flag1 (level ilevel)
        [e setComputePipelineState:pso("flag_reset")];
        [e setBuffer:B.flag1 offset:0 atIndex:0]; [e setBytes:&P length:sizeof(P) atIndex:1];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,num,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        dump("reset");
        // init_flag over level (ilevel+1): flag parent cells with refined/flagged children
        if (num1 > 0) {
            FlagParams P1{}; P1.head_idx=head1; P1.num_octs=num1; P1.ngridmax=ngridmax;
            [e setComputePipelineState:pso("flag_init")];
            [e setBuffer:B.flag1 offset:0 atIndex:0]; [e setBuffer:B.grid offset:0 atIndex:1];
            [e setBuffer:B.father offset:0 atIndex:2]; [e setBytes:&P1 length:sizeof(P1) atIndex:3];
            dispatch1d(e, pso("flag_init"), num1);
        }
        dump("init");
        // enforce_subgrid: DEFAULT SKIP (sgmode=0).  The CUDA enforce_subgrid_kernel
        // flags the WHOLE oct if any cell is flagged -- needed ONLY for nsubgrid>1
        // oct-grouping (CPU m_flag_fine calls r_ensure_subgrid #ifdef _CUDA && nsubgrid>1,
        // LAST).  The non-CUDA CPU reference and the Metal data model refine PER-CELL, so
        // running it (early OR late) over-flags whole octs -> deeper-refinement mesh
        // divergence (a=0.25 1-step: 27/24 vs CPU 26/21; full collapse ~10x the round-off
        // floor).  Skipping it makes GPU flag BIT-EXACT to CPU (verified a=0.25 dv=0.0).
        // RAMSES_FLAG_SUBGRID: 0=skip (default,correct), 1=early(legacy), 2=late.
        int sgmode = 0; { const char* s=getenv("RAMSES_FLAG_SUBGRID"); if(s) sgmode=atoi(s); }
        if (sgmode == 1) {
        [e setComputePipelineState:pso("flag_enforce_subgrid")];
        [e setBuffer:B.flag1 offset:0 atIndex:0]; [e setBytes:&P length:sizeof(P) atIndex:1];
        dispatch1d(e, pso("flag_enforce_subgrid"), num);
        }
        dump("subgrid");
        auto smooth=[&](){
            for (int s=0;s<3;++s) {
                FlagParams Q=P; Q.num_nbors=n_nbor[s];
                [e setComputePipelineState:pso("flag_count_nbor")];
                [e setBuffer:B.flag2 offset:0 atIndex:0]; [e setBuffer:B.flag1 offset:0 atIndex:1];
                [e setBuffer:B.nbor offset:0 atIndex:2]; [e setBytes:&Q length:sizeof(Q) atIndex:3];
                [e dispatchThreads:MTLSizeMake(TWOTONDIM,num,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
                [e setComputePipelineState:pso("flag_propagate")];
                [e setBuffer:B.flag1 offset:0 atIndex:0]; [e setBuffer:B.flag2 offset:0 atIndex:1];
                [e setBytes:&Q length:sizeof(Q) atIndex:2];
                [e dispatchThreads:MTLSizeMake(TWOTONDIM,num,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
                if (dbg) { char lb[24]; snprintf(lb,sizeof(lb),"smooth2.%d",s); dump(lb); }
            }
        };
        smooth();                                                 // step 2
        // poisson_flag: nref >= m_refine  (step 3)
        { FlagParams Q=P; Q.m_refine=m_refine;
          [e setComputePipelineState:pso("flag_poisson")];
          [e setBuffer:B.flag1 offset:0 atIndex:0]; [e setBuffer:B.nref offset:0 atIndex:1];
          [e setBytes:&Q length:sizeof(Q) atIndex:2];
          [e dispatchThreads:MTLSizeMake(TWOTONDIM,num,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)]; }
        dump("poisson");
        for (int ex=0; ex<nexpand; ++ex) { smooth(); if(dbg){char lb[24];snprintf(lb,sizeof(lb),"expand.%d",ex);dump(lb);} }   // step 4
        if (sgmode == 2) {   // late enforce_subgrid (CPU r_ensure_subgrid order: after expand)
            [e setComputePipelineState:pso("flag_enforce_subgrid")];
            [e setBuffer:B.flag1 offset:0 atIndex:0]; [e setBytes:&P length:sizeof(P) atIndex:1];
            dispatch1d(e, pso("flag_enforce_subgrid"), num);
            dump("subgrid_late");
        }
        if (do_rules) {
            [e setComputePipelineState:pso("flag_enforce_rules")];
            [e setBuffer:B.flag1 offset:0 atIndex:0]; [e setBuffer:B.nbor offset:0 atIndex:1];
            [e setBytes:&P length:sizeof(P) atIndex:2];
            dispatch1d(e, pso("flag_enforce_rules"), num);
            dump("rules");
        }
        [e endEncoding]; submit_async(cb);
    }
}
void* mtl_ptr_hash_key() { return B.hash_key.contents; }
void* mtl_ptr_hash_val() { return B.hash_val.contents; }
void* mtl_ptr_ckey_max() { return B.ckey_max.contents; }
void* mtl_ptr_key_off()  { return B.key_off.contents; }
void* mtl_ptr_rho()      { return B.rho.contents; }
void* mtl_ptr_nref()     { return B.nref.contents; }
void* mtl_ptr_phi()      { return B.phi.contents; }
void* mtl_ptr_phi_old()  { return B.phi_old.contents; }
void* mtl_ptr_f()        { return B.f.contents; }
void* mtl_ptr_ipos()     { return B.ipos.contents; }
void* mtl_ptr_vp()       { return B.vp.contents; }
void* mtl_ptr_mp()       { return B.mp.contents; }
void* mtl_ptr_levelp()   { return B.levelp.contents; }
void* mtl_ptr_idp()      { return B.idp.contents; }
void* mtl_ptr_sortp()    { return B.sortp.contents; }
void* mtl_ptr_hkey_part(){ return B.hkey_part.contents; }

// Raw byte copy of the host oct array (type(oct), 64 B each) into B.grid at
// 1-based slot `dst_head`.  Host and Oct layouts are identical (see header), so
// this is a plain memcpy (the unified-memory analogue of r_set_grid_device).
void mtl_copy_grid_in(const void* host_grid, int dst_head, int n) {
    memcpy((Oct*)B.grid.contents + (dst_head-1), host_grid, (size_t)n*sizeof(Oct));
}
// Copy the GPU-owned mesh back to the host (after GPU refine mutates B.grid).
void mtl_copy_grid_out(void* host_grid, int src_head, int n) {
    memcpy(host_grid, (Oct*)B.grid.contents + (src_head-1), (size_t)n*sizeof(Oct));
}

//---------------------------------------------------------------------------
// H6: AMR multigrid hierarchy (mirrors gpu_build_mg / make_father_octs /
// update_nbor_array_mg).  Given the fine AMR level's octs in B.grid[1..nF]
// (level L, ckey set), build the coarser MG sub-levels grid_mg by grouping
// octs under their parent (ckey/2), down to a 1-oct base.  Produces, in the
// CUDA layout:
//   grid_mg[]   : MG octs for levels L-1..bottom (contiguous per level)
//   father_mg[] : child -> parent MG oct.  Child index space = the fine octs
//                 (0..nF-1) followed by the grid_mg octs (offset nF), exactly
//                 the CUDA head_father = noct(ilevel)+m_mg%head(ifine).
//   nbor_mg[]   : 27 same-level neighbours per MG oct (0 if missing)
//   mg_head[L'] : first grid_mg index (1-based) of MG level L' ; mg_noct[L']
// phi_mg / f_mg are sized to grid_mg and zeroed (filled by the V-cycle).
//---------------------------------------------------------------------------
namespace {
struct MgAmr {
    id<MTLBuffer> grid_mg, father_mg, nbor_mg, phi_mg, f_mg;
    int ilevel, n_fine, n_mg, fine_head;      // fine octs, total MG octs, fine head in B.grid
    int mg_head[64], mg_noct[64];             // per MG level (1-based grid_mg index)
    int bottom;                               // coarsest MG level reached
    int cap = 0, fcap = 0;                    // current grid_mg / father_mg capacity (reused)
    // Base-level hierarchy cache (the levelmin grid is FIXED -> build once, reuse).
    id<MTLBuffer> grid_mg_b, father_mg_b, nbor_mg_b;
    int mg_head_b[64], mg_noct_b[64], n_mg_b, n_fine_b, bottom_b;
    bool base_cached = false;
} MG;
}

void* mtl_ptr_grid_mg()   { return MG.grid_mg.contents; }
void* mtl_ptr_father_mg() { return MG.father_mg.contents; }
void* mtl_ptr_nbor_mg()   { return MG.nbor_mg.contents; }
void* mtl_ptr_phi_mg()    { return MG.phi_mg.contents; }
void* mtl_ptr_f_mg()      { return MG.f_mg.contents; }
int   mtl_mg_head(int lev){ return (lev>=0 && lev<64) ? MG.mg_head[lev] : 0; }
int   mtl_mg_noct(int lev){ return (lev>=0 && lev<64) ? MG.mg_noct[lev] : 0; }
int   mtl_mg_bottom()     { return MG.bottom; }

// Build the MG hierarchy from the fine octs at B.grid[head_idx..head_idx+n_fine-1]
// (level ilevel).  mg_cap is the max number of grid_mg octs to allocate.
void mtl_build_mg_amr(int ilevel, int head_idx, int n_fine, int mg_cap,
                      const int* box_min, const int* box_max,
                      int per0, int per1, int per2, int is_base) {
    // Base level: the levelmin grid never changes, so its MG hierarchy is fixed.
    // Restore it from the cache (cheap memcpy) instead of re-grouping octs.
    // (RAMSES_MG_NOCACHE forces a full rebuild so MG_DUMP shows the base.)
    if (is_base && MG.base_cached && !getenv("RAMSES_MG_NOCACHE")) {
        MG.ilevel=ilevel; MG.n_fine=n_fine; MG.fine_head=head_idx;
        MG.n_mg=MG.n_mg_b; MG.bottom=MG.bottom_b;
        for (int i=0;i<64;++i){ MG.mg_head[i]=MG.mg_head_b[i]; MG.mg_noct[i]=MG.mg_noct_b[i]; }
        memcpy(MG.grid_mg.contents,   MG.grid_mg_b.contents,   (size_t)MG.n_mg_b*sizeof(Oct));
        memcpy(MG.nbor_mg.contents,   MG.nbor_mg_b.contents,   (size_t)MG.n_mg_b*SUBGRIDSIZE*sizeof(int));
        memcpy(MG.father_mg.contents, MG.father_mg_b.contents, (size_t)(MG.n_fine_b+MG.n_mg_b)*sizeof(int));
        memset(MG.phi_mg.contents, 0, (size_t)MG.n_mg_b*TWOTONDIM*sizeof(float));
        memset(MG.f_mg.contents,   0, (size_t)MG.n_mg_b*TWOTONDIM*NF*sizeof(float));
        return;
    }
    int* ckey_max = (int*) B.ckey_max.contents;
    long* key_off = (long*)B.key_off.contents;
    int per[3] = { per0, per1, per2 };
    MG.ilevel = ilevel; MG.n_fine = n_fine; MG.fine_head = head_idx;
    for (int i=0;i<64;++i){ MG.mg_head[i]=0; MG.mg_noct[i]=0; }

    // Persistent buffers: (re)allocate only when a level needs more capacity
    // than we've ever allocated (the base level is the largest, so after the
    // first solve these are reused for the rest of the run -- no per-call churn).
    int fcap_need = n_fine + mg_cap;
    if (!MG.grid_mg || mg_cap > MG.cap) {
        MG.cap = mg_cap;
        MG.grid_mg = newbuf<Oct>(MG.cap);
        MG.nbor_mg = newbuf<int>((size_t)MG.cap*SUBGRIDSIZE);
        MG.phi_mg  = newbuf<float>((size_t)MG.cap*TWOTONDIM);
        MG.f_mg    = newbuf<float>((size_t)MG.cap*TWOTONDIM*NF);
    }
    if (!MG.father_mg || fcap_need > MG.fcap) {
        MG.fcap = fcap_need;
        MG.father_mg = newbuf<int>((size_t)MG.fcap);
    }
    Oct*  grid     = (Oct*) B.grid.contents;
    Oct*  grid_mg  = (Oct*) MG.grid_mg.contents;
    int*  father   = (int*) MG.father_mg.contents;
    int*  nbor_mg  = (int*) MG.nbor_mg.contents;

    // Per MG level, map (ckey) -> 1-based grid_mg index, built by grouping the
    // children of the level above under their parent.
    int n_mg = 0;                          // running grid_mg oct count
    int lev_child = ilevel;                // level of the child octs being grouped
    int child_count = n_fine;
    // accessor for a child oct's ckey at lev_child: fine octs from B.grid,
    // MG octs from grid_mg at mg_head[lev_child].
    auto child_ckey = [&](int ci, int d)->long {
        if (lev_child == ilevel) return grid[(head_idx-1) + ci].ckey[d];   // fine octs at head_idx
        return grid_mg[MG.mg_head[lev_child]-1 + ci].ckey[d];
    };
    auto child_father_slot = [&](int ci)->int& {
        if (lev_child == ilevel) return father[ci];                 // 0..n_fine-1
        return father[n_fine + (MG.mg_head[lev_child]-1 + ci)];     // MG offset
    };

    // Persistent parent map: reused (cleared, not reallocated) across MG levels
    // and steps so the hierarchy build doesn't pay heap-alloc + rehash each call
    // (this host grouping is the dominant non-GPU cost; ~68s of the full run).
    static std::unordered_map<long,int> g_pmap;
    int lev = ilevel;
    while (child_count > 1 && lev > 1) {          // stop at a single-oct coarsest level
        int clev = lev - 1;                           // coarse MG level to create
        std::unordered_map<long,int>& pmap = g_pmap;  // parent key -> 1-based grid_mg idx
        pmap.clear(); pmap.reserve((size_t)child_count);
        int base = n_mg;                              // first grid_mg idx (0-based) for clev
        MG.mg_head[clev] = base + 1;                  // 1-based
        for (int ci = 0; ci < child_count; ++ci) {
            long ck[3] = {0,0,0}, pck[3] = {0,0,0};
            for (int d=0; d<NDIM; ++d) { ck[d] = child_ckey(ci,d); pck[d] = ck[d] / 2; }
            long pkey = h_oct_key_n(ckey_max, key_off, clev, pck);
            auto it = pmap.find(pkey);
            int pidx;
            if (it == pmap.end()) {
                pidx = ++n_mg;                        // new 1-based grid_mg oct
                Oct o{}; o.lev = clev;
                for (int d=0; d<NDIM; ++d) o.ckey[d] = (int)pck[d];
                grid_mg[pidx-1] = o;
                pmap[pkey] = pidx;
            } else pidx = it->second;
            child_father_slot(ci) = pidx;
        }
        MG.mg_noct[clev] = n_mg - base;
        // nbor_mg for the new MG octs at clev (3^NDIM same-level neighbours via pmap),
        // indexed cubeq = sum_d (off_d+1)*3^d to match the device mg_nbor/ccc scheme.
        for (int q = base; q < n_mg; ++q) {
            long ck[3] = {0,0,0};
            for (int d=0; d<NDIM; ++d) ck[d] = grid_mg[q].ckey[d];
            for (int cubeq = 0; cubeq < THREETONDIM; ++cubeq) {
                long nck[3] = {0,0,0}; int rem = cubeq;
                for (int d=0; d<NDIM; ++d) {
                    int o_d = (rem % 3) - 1; rem /= 3;
                    long c = ck[d] + o_d;
                    if (per[d]) {
                        int bmn=box_min[(clev-1)*3+d], bmx=box_max[(clev-1)*3+d];
                        if (c <  bmn) c = bmx-1;
                        if (c >= bmx) c = bmn;
                    }
                    nck[d] = c;
                }
                long nkey = h_oct_key_n(ckey_max, key_off, clev, nck);
                auto it = pmap.find(nkey);
                nbor_mg[q*SUBGRIDSIZE + cubeq] = (it==pmap.end()) ? 0 : it->second;
            }
        }
        MG.bottom = clev;
        lev = clev; lev_child = clev; child_count = MG.mg_noct[clev];
    }
    MG.n_mg = n_mg;
    memset(MG.phi_mg.contents, 0, (size_t)n_mg*TWOTONDIM*sizeof(float));
    memset(MG.f_mg.contents,   0, (size_t)n_mg*TWOTONDIM*NF*sizeof(float));

    // DIAG (RAMSES_MG_NBORDUMP): per coarse MG level, dump each oct's ckey + nbor_mg
    // (1D: [left,self,right] MG-oct indices) to verify the coarse-grid connectivity.
    // self must == the oct; left/right must be the ckey-/+1 periodic neighbours.
    if (getenv("RAMSES_MG_NBORDUMP") && is_base) {
        for (int clev = ilevel-1; clev >= MG.bottom && clev >= ilevel-3; --clev) {
            int h = MG.mg_head[clev], no = MG.mg_noct[clev];
            fprintf(stderr, "[MGNBOR] clev=%d noct=%d\n", clev, no);
            for (int o = h; o < h+no && o < h+8; ++o) {
                int L = nbor_mg[(o-1)*SUBGRIDSIZE + 0];
                int S = nbor_mg[(o-1)*SUBGRIDSIZE + 1];
                int R = nbor_mg[(o-1)*SUBGRIDSIZE + 2];
                int ckL = (L>=1)?grid_mg[L-1].ckey[0]:-1;
                int ckR = (R>=1)?grid_mg[R-1].ckey[0]:-1;
                fprintf(stderr, "  oct=%d ckey=%d | L=%d(ck%d) S=%d R=%d(ck%d) | selfOK=%d\n",
                        o, grid_mg[o-1].ckey[0], L, ckL, S, R, ckR, (S==o));
            }
        }
    }

    // Cache the base-level hierarchy (fixed grid) for reuse on later steps.
    if (is_base) {
        if (!MG.grid_mg_b) {
            MG.grid_mg_b   = newbuf<Oct>(n_mg>0?n_mg:1);
            MG.nbor_mg_b   = newbuf<int>((size_t)(n_mg>0?n_mg:1)*SUBGRIDSIZE);
            MG.father_mg_b = newbuf<int>((size_t)(n_fine+n_mg));
        }
        MG.n_mg_b=n_mg; MG.n_fine_b=n_fine; MG.bottom_b=MG.bottom;
        for (int i=0;i<64;++i){ MG.mg_head_b[i]=MG.mg_head[i]; MG.mg_noct_b[i]=MG.mg_noct[i]; }
        memcpy(MG.grid_mg_b.contents,   grid_mg,  (size_t)n_mg*sizeof(Oct));
        memcpy(MG.nbor_mg_b.contents,   nbor_mg,  (size_t)n_mg*SUBGRIDSIZE*sizeof(int));
        memcpy(MG.father_mg_b.contents, father,   (size_t)(n_fine+n_mg)*sizeof(int));
        MG.base_cached = true;
    }
}

//---------------------------------------------------------------------------
// H1: connectivity bridge.  Given B.grid (the host oct array, already copied
// in) and B.ckey_max/B.key_off (per-level, 1-based-padded), (re)build on the
// CPU the read-only flat structures the Metal gravity kernels consume:
//   - B.hash_key / B.hash_val : (level,ckey) -> oct index   (linear-probe)
//   - B.father                : oct -> parent oct at L-1     (0 if none)
//   - B.nbor (27 per oct)      : same-level 3x3x3 neighbours  (0 if missing)
// No atomicCAS / 64-bit atomics anywhere (that was the gpu_refine blocker);
// this is a serial O(num_octs*27) host pass, cheap vs the gravity solve.
// box_ckey_min/max are the host m%box_ckey_min/max (1:3,1:nlevelmax) flattened
// column-major: element (idim,ilevel) at box[(ilevel-1)*3 + (idim-1)].
//---------------------------------------------------------------------------
void mtl_build_connectivity(int num_octs, int levelmin, int nlevelmax,
                            const int* box_min, const int* box_max,
                            int per0, int per1, int per2) {
    (void)levelmin;
    Oct*  grid     = (Oct*) B.grid.contents;
    long* hkey     = (long*)B.hash_key.contents;
    int*  hval     = (int*) B.hash_val.contents;
    int*  ckey_max = (int*) B.ckey_max.contents;
    long* key_off  = (long*)B.key_off.contents;
    int*  nbor     = (int*) B.nbor.contents;
    int*  father   = (int*) B.father.contents;
    int   hs       = B.hash_size;
    int   per[3]   = { per0, per1, per2 };

    // (1) rebuild the read-only hash over all active octs (lev>0).
    memset(hkey, 0, (size_t)hs*sizeof(long));
    memset(hval, 0, (size_t)hs*sizeof(int));
    for (int o = 1; o <= num_octs; ++o) {
        int L = grid[o-1].lev;
        if (L < 1 || L > nlevelmax) continue;             // skip free/invalid slots
        long ck[3]; h_read_ckey(grid[o-1], ck);
        long key = h_oct_key_n(ckey_max, key_off, L, ck);
        h_hash_set(hkey, hval, hs, key, o);
    }

    // (2) father + 27 same-level neighbours per oct, ON THE GPU (#4).  The hash
    // is now immutable, so these are pure hash_get reads -> two kernels over the
    // octs replace the host loop entirely.  Copy the per-level box bounds into
    // device buffers first.
    int nlm3 = 3*B.nlevelmax;
    memcpy(B.box_min.contents, box_min, (size_t)nlm3*sizeof(int));
    memcpy(B.box_max.contents, box_max, (size_t)nlm3*sizeof(int));
    ConnParams P{}; P.num_octs=num_octs; P.hash_size=hs; P.nlevelmax=nlevelmax; P.head_idx=1;
    P.per[0]=per[0]; P.per[1]=per[1]; P.per[2]=per[2];
    @autoreleasepool {
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("conn_build_father")];
        [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.father offset:0 atIndex:1];
        [e setBuffer:B.hash_key offset:0 atIndex:2]; [e setBuffer:B.hash_val offset:0 atIndex:3];
        [e setBuffer:B.ckey_max offset:0 atIndex:4]; [e setBuffer:B.key_off offset:0 atIndex:5];
        [e setBytes:&P length:sizeof(P) atIndex:6];
        dispatch1d(e, pso("conn_build_father"), num_octs);
        [e setComputePipelineState:pso("conn_build_nbor")];
        [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.nbor offset:0 atIndex:1];
        [e setBuffer:B.hash_key offset:0 atIndex:2]; [e setBuffer:B.hash_val offset:0 atIndex:3];
        [e setBuffer:B.ckey_max offset:0 atIndex:4]; [e setBuffer:B.key_off offset:0 atIndex:5];
        [e setBuffer:B.box_min offset:0 atIndex:6]; [e setBuffer:B.box_max offset:0 atIndex:7];
        [e setBytes:&P length:sizeof(P) atIndex:8];
        dispatch1d(e, pso("conn_build_nbor"), num_octs);
        [e endEncoding]; submit_async(cb);
    }
}

// Full hash rebuild on the GPU: host-memset the table to the 0 sentinel, copy
// the (static) box bounds for nbor, then dispatch a parallel atomicCAS insert of
// every active oct.  Correct under per-level compaction (all oct indices shift
// each refine, so only a FULL rebuild is valid) yet fast (no serial host loop).
void mtl_conn_rebuild_hash(int num_octs, int nlevelmax, const int* box_min, const int* box_max) {
    mtl_drain();   // host memset/memcpy of hash+box must not race in-flight GPU
    int hs = B.hash_size;
    memset(B.hash_key.contents, 0, (size_t)hs*sizeof(long));
    memset(B.hash_val.contents, 0, (size_t)hs*sizeof(int));
    int nlm3 = 3*B.nlevelmax;
    memcpy(B.box_min.contents, box_min, (size_t)nlm3*sizeof(int));
    memcpy(B.box_max.contents, box_max, (size_t)nlm3*sizeof(int));
    @autoreleasepool {
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        ConnParams P{}; P.num_octs=num_octs; P.hash_size=hs; P.nlevelmax=nlevelmax; P.head_idx=1;
        [e setComputePipelineState:pso("hash_insert")];
        [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.hash_key offset:0 atIndex:1];
        [e setBuffer:B.hash_val offset:0 atIndex:2]; [e setBuffer:B.ckey_max offset:0 atIndex:3];
        [e setBuffer:B.key_off offset:0 atIndex:4]; [e setBytes:&P length:sizeof(P) atIndex:5];
        dispatch1d(e, pso("hash_insert"), num_octs);
        [e endEncoding]; submit_async(cb);
    }
#ifdef MTL_VERIFY_HASH
    {   // DEBUG: confirm every active oct is findable in the GPU-built table.
        long* hkey=(long*)B.hash_key.contents; int* hval=(int*)B.hash_val.contents;
        int* ckm=(int*)B.ckey_max.contents; long* koff=(long*)B.key_off.contents;
        Oct* grid=(Oct*)B.grid.contents; int bad=0;
        for (int o=1;o<=num_octs;++o){ int L=grid[o-1].lev; if(L<1||L>nlevelmax) continue;
            long vck[3]; h_read_ckey(grid[o-1], vck); long key=h_oct_key_n(ckm,koff,L,vck);
            if (h_hash_get(hkey,hval,B.hash_size,key)!=o) ++bad; }
        if (bad) fprintf(stderr,"[MTL_VERIFY_HASH] %d/%d octs NOT found in GPU hash\n",bad,num_octs);
    }
#endif
}

// Rebuild nbor for octs [nbor_head, nbor_head+nbor_num) and father for octs
// [father_head, father_head+father_num) on the GPU (per-level, cheap).  The hash
// + box buffers must already be current.
void mtl_conn_build_range(int nbor_head, int nbor_num, int father_head, int father_num,
                          int nlevelmax) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        if (father_num > 0) {
            ConnParams P{}; P.num_octs=father_num; P.hash_size=B.hash_size; P.nlevelmax=nlevelmax; P.head_idx=father_head;
            [e setComputePipelineState:pso("conn_build_father")];
            [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.father offset:0 atIndex:1];
            [e setBuffer:B.hash_key offset:0 atIndex:2]; [e setBuffer:B.hash_val offset:0 atIndex:3];
            [e setBuffer:B.ckey_max offset:0 atIndex:4]; [e setBuffer:B.key_off offset:0 atIndex:5];
            [e setBytes:&P length:sizeof(P) atIndex:6];
            dispatch1d(e, pso("conn_build_father"), father_num);
        }
        if (nbor_num > 0) {
            ConnParams P{}; P.num_octs=nbor_num; P.hash_size=B.hash_size; P.nlevelmax=nlevelmax; P.head_idx=nbor_head;
            P.per[0]=1; P.per[1]=1; P.per[2]=1;     // periodic (dmo); box bounds in B.box_*
            [e setComputePipelineState:pso("conn_build_nbor")];
            [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.nbor offset:0 atIndex:1];
            [e setBuffer:B.hash_key offset:0 atIndex:2]; [e setBuffer:B.hash_val offset:0 atIndex:3];
            [e setBuffer:B.ckey_max offset:0 atIndex:4]; [e setBuffer:B.key_off offset:0 atIndex:5];
            [e setBuffer:B.box_min offset:0 atIndex:6]; [e setBuffer:B.box_max offset:0 atIndex:7];
            [e setBytes:&P length:sizeof(P) atIndex:8];
            dispatch1d(e, pso("conn_build_nbor"), nbor_num);
        }
        [e endEncoding]; submit_async(cb);
    }
}

// ---------------------------------------------------------------------------
// Cache-oct (coarse-fine boundary ghost) materialisation for the octs in
// [head_idx, head_idx+num_octs) at level ilevel.  Mirrors gpu_runner.cuf 644-692:
// for each 3^NDIM neighbour direction, find octs whose same-level neighbour is
// MISSING (init_prefix_sum_nbor predicate), inclusive-scan, compute_cache_swap_table
// (compact), make_cache_octs (create a ghost oct with the correct father via hash +
// straight-injected f/phi/phi_old), insert each into the hash, and advance the cache
// free pointer.  Requires a CURRENT nbor + hash (call mtl_conn_build_range first) and
// a FRESH hash (cache entries are rebuilt each step).  Returns #cache octs created.
//
// INERT until a cache region is allocated (ncell > ngridmax): without it every nbor
// is "real" so there is nothing to materialise and the early return keeps the
// pre-cache behaviour bit-identical.
int mtl_make_cache(int ilevel, int head_idx, int num_octs, int nlevelmax,
                   int per0, int per1, int per2, float tfrac) {
    if (num_octs <= 0 || B.ncell <= B.ngridmax) return 0;   // no cache region -> inert
    const int CENTER = THREETONDIM/2 + 1;                   // self direction (off=0)
    B.ifree_cache = 0;                                      // rebuild this level's cache from scratch
    int created = 0;
    for (int input_ind = 1; input_ind <= THREETONDIM; ++input_ind) {
        if (input_ind == CENTER) continue;
        // (1) predicate -> B.cache_pre[oct-1]; (2) inclusive scan over [head,head+num)
        @autoreleasepool {
            ScanParams S{}; S.n=num_octs; S.head_idx=head_idx;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("init_prefix_sum_nbor")];
            [e setBuffer:B.nbor offset:0 atIndex:0]; [e setBuffer:B.cache_pre offset:0 atIndex:1];
            [e setBytes:&S length:sizeof(S) atIndex:2]; [e setBytes:&input_ind length:sizeof(int) atIndex:3];
            dispatch1d(e, pso("init_prefix_sum_nbor"), num_octs);
            gpu_scan(e, B.cache_pre, head_idx, num_octs);
            [e endEncoding]; submit_async(cb);
        }
        mtl_drain();
        int count = ((const int*)B.cache_pre.contents)[head_idx-1 + num_octs-1];  // scan total
        if (count <= 0) continue;
        if (B.ngridmax + B.ifree_cache + count > B.ncell) {     // cache region exhausted
            fprintf(stderr, "mtl_make_cache: cache region overflow (need %d, cap %d)\n",
                    B.ngridmax + B.ifree_cache + count, B.ncell);
            break;
        }
        // (3) compact -> B.swap; (4) materialise; (5) hash-insert the new cache octs
        @autoreleasepool {
            ScanParams S{}; S.n=num_octs; S.head_idx=head_idx;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("compute_cache_swap_table")];
            [e setBuffer:B.swap offset:0 atIndex:0]; [e setBuffer:B.cache_pre offset:0 atIndex:1];
            [e setBytes:&S length:sizeof(S) atIndex:2];
            dispatch1d(e, pso("compute_cache_swap_table"), num_octs);
            CacheParams P{}; P.num_octs=count; P.ngridmax=B.ngridmax; P.ifree_cache=B.ifree_cache;
            P.input_ind=input_ind; P.hash_size=B.hash_size; P.nlevelmax=nlevelmax;
            P.per[0]=per0; P.per[1]=per1; P.per[2]=per2; P.tfrac=tfrac;
            [e setComputePipelineState:pso("make_cache_octs")];
            [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.flag1 offset:0 atIndex:1];
            [e setBuffer:B.f offset:0 atIndex:2]; [e setBuffer:B.phi offset:0 atIndex:3];
            [e setBuffer:B.phi_old offset:0 atIndex:4]; [e setBuffer:B.swap offset:0 atIndex:5];
            [e setBuffer:B.father offset:0 atIndex:6]; [e setBuffer:B.nbor offset:0 atIndex:7];
            [e setBuffer:B.hash_key offset:0 atIndex:8]; [e setBuffer:B.hash_val offset:0 atIndex:9];
            [e setBuffer:B.ckey_max offset:0 atIndex:10]; [e setBuffer:B.key_off offset:0 atIndex:11];
            [e setBuffer:B.box_min offset:0 atIndex:12]; [e setBuffer:B.box_max offset:0 atIndex:13];
            [e setBytes:&P length:sizeof(P) atIndex:14];
            dispatch1d(e, pso("make_cache_octs"), count);
            ConnParams CP{}; CP.num_octs=count; CP.head_idx=B.ngridmax+B.ifree_cache+1;
            CP.hash_size=B.hash_size; CP.nlevelmax=nlevelmax;
            [e setComputePipelineState:pso("hash_insert")];
            [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.hash_key offset:0 atIndex:1];
            [e setBuffer:B.hash_val offset:0 atIndex:2]; [e setBuffer:B.ckey_max offset:0 atIndex:3];
            [e setBuffer:B.key_off offset:0 atIndex:4]; [e setBytes:&CP length:sizeof(CP) atIndex:5];
            dispatch1d(e, pso("hash_insert"), count);
            [e endEncoding]; submit_async(cb);
        }
        mtl_drain();
        B.ifree_cache += count; created += count;
    }
    // DIAG (RAMSES_CACHE_DUMP): dump each cache oct's GPU ckey + phi to check the
    // coarse-fine BC left-right symmetry (a symmetric pancake must give symmetric cache phi;
    // asymmetry here = the tilt source).  GPU grid ckey is correct (Fortran m%grid reads 0).
    {
        const char* cd = getenv("RAMSES_CACHE_DUMP");
        if (cd && cd[0] && ilevel <= 12 && created > 0) {
            Oct* g = (Oct*)B.grid.contents; float* ph = (float*)B.phi.contents;
            fprintf(stderr, "[CACHEDUMP] L%d ncache=%d\n", ilevel, created);
            for (int i = 0; i < created; ++i) {
                int c = B.ngridmax + i;   // 0-based cache oct index
                fprintf(stderr, "  ckey=%d phi=[%.6e %.6e]\n",
                        g[c].ckey[0], ph[c*TWOTONDIM+0], ph[c*TWOTONDIM+1]);
            }
        }
    }
    return created;
}

// ---------------------------------------------------------------------------
// GPU AMR refine (data_on_device): create child octs for flagged cells, mark
// derefined octs dead, compact the finer region (drop holes + group by level),
// rebuild hash + father + nbor.  head/noct point at the host m%head(levelmin)/
// m%noct(levelmin) arrays (indexed [L-levelmin]); updated in place.  *noct_used
// and *ifree are updated.  B.grid must be synced + flag1 resident on entry.
void mtl_refine(int ilevel, int levelmin, int nlevelmax, int* head, int* noct,
                int* noct_used, int* ifree, const int* box_min, const int* box_max,
                int* ncreate_out, int* nkill_out) {
    Oct* grid = (Oct*)B.grid.contents;
    int* ctr  = (int*)B.ifree_ctr.contents;
    int* kctr = (int*)B.kill_ctr.contents;
    *kctr = 0;
    auto HD = [&](int L){ return head[L-levelmin]; };
    auto NOC = [&](int L){ return noct[L-levelmin]; };
    auto TL = [&](int L){ return head[L-levelmin]+noct[L-levelmin]-1; };

    // DIAG: count flag1==1 in B.flag1 at refine entry (what refine_create will see).
    if (getenv("RAMSES_REFINE_DUMP")) {
        mtl_drain();
        int* fl=(int*)B.flag1.contents;
        long nf=0;
        for (int L=ilevel; L<=nlevelmax-1; ++L)
            for (int o=HD(L); o<HD(L)+NOC(L); ++o)
                for (int c=1;c<=TWOTONDIM;++c) if (fl[(o-1)*TWOTONDIM+(c-1)]==1) ++nf;
        fprintf(stderr,"[REFINE-IN] ilevel=%d B.flag1==1 count=%ld (over levels %d..%d)\n",
                ilevel, nf, ilevel, nlevelmax-1);
    }

    // ---- STEP 1: create child octs, top-down over ALL levels (the GPU refine is
    // called once at levelmin and does the whole hierarchy) ----
    *ctr = *noct_used + 1;
    for (int L=ilevel; L<=nlevelmax-1; ++L) {
        if (NOC(L) <= 0) continue;
        @autoreleasepool {
            RefineParams P{}; P.head_idx=HD(L); P.num_octs=NOC(L); P.nlevelmax=nlevelmax; P.hash_size=B.hash_size;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("refine_create")];
            [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.flag1 offset:0 atIndex:1];
            [e setBuffer:B.f offset:0 atIndex:2]; [e setBuffer:B.phi offset:0 atIndex:3];
            [e setBuffer:B.phi_old offset:0 atIndex:4];     // make_new_oct injects phi_old too
            [e setBuffer:B.ifree_ctr offset:0 atIndex:5]; [e setBytes:&P length:sizeof(P) atIndex:6];
            MTLSize tg = MTLSizeMake(TWOTONDIM, 64, 1);
            MTLSize grd = MTLSizeMake(TWOTONDIM, (NOC(L)+63)/64*64, 1);
            [e dispatchThreads:grd threadsPerThreadgroup:tg];
            [e endEncoding]; submit_async(cb);
        }
    }
    mtl_drain();                                          // host reads the create counter
    int num = *ctr - 1;                                   // highest used slot after create
    int ncreate = num - *noct_used;

    // Rebuild hash so derefine's parent lookup sees the new octs (only if any).
    if (ncreate > 0) mtl_conn_rebuild_hash(num, nlevelmax, box_min, box_max);

    // ---- STEP 2: derefine, bottom-up over all finer levels (mark lev=0) ----
    for (int L=nlevelmax; L>=ilevel+1; --L) {
        if (NOC(L) <= 0) continue;
        @autoreleasepool {
            RefineParams P{}; P.head_idx=HD(L); P.num_octs=NOC(L); P.nlevelmax=nlevelmax; P.hash_size=B.hash_size;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("refine_derefine")];
            [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.flag1 offset:0 atIndex:1];
            [e setBuffer:B.hash_key offset:0 atIndex:2]; [e setBuffer:B.hash_val offset:0 atIndex:3];
            [e setBuffer:B.ckey_max offset:0 atIndex:4]; [e setBuffer:B.key_off offset:0 atIndex:5];
            [e setBytes:&P length:sizeof(P) atIndex:6]; [e setBuffer:B.kill_ctr offset:0 atIndex:7];
            dispatch1d(e, pso("refine_derefine"), NOC(L));
            [e endEncoding]; submit_async(cb);
        }
    }
    mtl_drain();                                          // host reads kill counter + grid.lev below
    int nkill = *kctr;
    if (ncreate_out) *ncreate_out = ncreate;
    if (nkill_out)   *nkill_out   = nkill;

    // Nothing changed (redundant recursive refine call) -> mesh + layout unchanged.
    if (ncreate == 0 && nkill == 0) return;

    // ---- STEP 3: counting sort by level over the finer region [base, num] ----
    int base = TL(ilevel)+1;
    int cnt[80]={0}, nh[80]={0}, idx[80]={0};
    for (int o=base; o<=num; ++o) { int L=grid[o-1].lev; if (L>=1&&L<=nlevelmax) cnt[L]++; }
    nh[ilevel+1]=base;
    for (int L=ilevel+2; L<=nlevelmax; ++L) nh[L]=nh[L-1]+cnt[L-1];
    int* swap=(int*)B.swap.contents;
    for (int L=ilevel+1; L<=nlevelmax; ++L) idx[L]=nh[L];
    for (int o=base; o<=num; ++o) { int L=grid[o-1].lev; if (L>=1&&L<=nlevelmax){ swap[idx[L]-1]=o; idx[L]++; } }
    int nsurv=0; for (int L=ilevel+1; L<=nlevelmax; ++L) nsurv+=cnt[L];

    // ---- STEP 3b: Hilbert-key sort WITHIN each level (PARITY with CUDA gpu_refine
    // gpu_runner.cuf:559 + CPU refine_utils.f90 Step 4) ------------------------------
    // The level counting sort above leaves octs in atomic-creation order WITHIN each
    // level (refine_create's atomicadd is non-deterministic).  CPU and CUDA both then
    // LSD-radix-sort each level by Hilbert key so the layout is deterministic and
    // matches the CPU's.  Omitting it left Metal's intra-level order != CPU -> the
    // order-dependent fp32 gravity (Gauss-Seidel sweep / deposit reduction) diverged
    // from CPU at every oct-creation event (first L13 at a~0.093).  NHILBERT==1, so a
    // numeric sort on hkey[0] reproduces the radix (bit-order) result exactly; keys
    // are unique per oct (unique ckey).  Sort the swap entries (old oct indices) in
    // each level's [nh[L], nh[L]+cnt[L]) range by the oct's Hilbert key.
    int hsort_moved = 0;
    bool hsort_on = (getenv("RAMSES_NO_HSORT") == nullptr);
    for (int L=ilevel+1; L<=nlevelmax && hsort_on; ++L) {
        if (cnt[L] > 1) {
            int* lo = swap + (nh[L]-1);
            std::vector<int> before(lo, lo+cnt[L]);
            std::stable_sort(lo, lo+cnt[L], [grid](int a, int b){
                return (unsigned long long)grid[a-1].hkey[0]
                     < (unsigned long long)grid[b-1].hkey[0];
            });
            for (int k=0;k<cnt[L];++k) if (before[k]!=lo[k]) ++hsort_moved;
        }
    }
    if (getenv("RAMSES_REFINE_DUMP") && hsort_moved)
        fprintf(stderr,"[HSORT] ilevel=%d moved %d swap entries into Hilbert order\n", ilevel, hsort_moved);
    if (getenv("RAMSES_REFINE_DUMP")) {
        // count flagged-unrefined cells over the create levels (what SHOULD be created)
        int* fl=(int*)B.flag1.contents; Oct* g=(Oct*)grid;
        long nflag=0;
        for (int L=ilevel; L<=nlevelmax-1; ++L)
            for (int o=HD(L); o<HD(L)+NOC(L); ++o)
                for (int c=1;c<=TWOTONDIM;++c)
                    if (fl[(o-1)*TWOTONDIM+(c-1)]==1 && g[o-1].refined[c-1]==0) ++nflag;
        fprintf(stderr,"[REFINE] ilevel=%d flagged_unrefined=%ld ncreate=%d nkill=%d nsurv=%d  per-lev:",
                ilevel, nflag, ncreate, nkill, nsurv);
        for (int L=ilevel+1; L<=nlevelmax; ++L) if (cnt[L]) fprintf(stderr," L%d=%d", L, cnt[L]);
        fprintf(stderr,"\n");
    }

    // ---- gather grid + f + phi into scratch, blit the [base,base+nsurv) range back ----
    if (nsurv > 0) {
        @autoreleasepool {
            int bn[2] = {base, nsurv};
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("refine_gather_oct")];
            [e setBuffer:B.grid2 offset:0 atIndex:0]; [e setBuffer:B.grid offset:0 atIndex:1];
            [e setBuffer:B.swap offset:0 atIndex:2]; [e setBytes:&bn length:sizeof(bn) atIndex:3];
            dispatch1d(e, pso("refine_gather_oct"), nsurv);
            int wf = TWOTONDIM*NF;
            [e setComputePipelineState:pso("refine_gather_field")];
            [e setBuffer:B.fld2 offset:0 atIndex:0]; [e setBuffer:B.f offset:0 atIndex:1];
            [e setBuffer:B.swap offset:0 atIndex:2]; [e setBytes:&bn length:sizeof(bn) atIndex:3];
            [e setBytes:&wf length:sizeof(int) atIndex:4];
            [e dispatchThreads:MTLSizeMake(wf,nsurv,1) threadsPerThreadgroup:MTLSizeMake(wf,1,1)];
            [e endEncoding]; submit_async(cb);
        }
        // blit grid2 + fld2 -> grid + f for [base, base+nsurv)
        mtl_drain();   // host reads the gather output (grid2/fld2)
        memcpy((Oct*)B.grid.contents + (base-1), (Oct*)B.grid2.contents + (base-1), (size_t)nsurv*sizeof(Oct));
        memcpy((float*)B.f.contents + (size_t)(base-1)*TWOTONDIM*NF,
               (float*)B.fld2.contents + (size_t)(base-1)*TWOTONDIM*NF,
               (size_t)nsurv*TWOTONDIM*NF*sizeof(float));
        // phi (W=twotondim) via the same scratch
        @autoreleasepool {
            int bn[2] = {base, nsurv}; int wp = TWOTONDIM;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("refine_gather_field")];
            [e setBuffer:B.fld2 offset:0 atIndex:0]; [e setBuffer:B.phi offset:0 atIndex:1];
            [e setBuffer:B.swap offset:0 atIndex:2]; [e setBytes:&bn length:sizeof(bn) atIndex:3];
            [e setBytes:&wp length:sizeof(int) atIndex:4];
            [e dispatchThreads:MTLSizeMake(wp,nsurv,1) threadsPerThreadgroup:MTLSizeMake(wp,1,1)];
            [e endEncoding]; submit_async(cb);
        }
        mtl_drain();   // host reads the phi gather output (fld2)
        memcpy((float*)B.phi.contents + (size_t)(base-1)*TWOTONDIM,
               (float*)B.fld2.contents + (size_t)(base-1)*TWOTONDIM, (size_t)nsurv*TWOTONDIM*sizeof(float));
        // phi_old (W=twotondim) — surviving octs that change slot during compaction
        // MUST carry their phi_old too, else the next subcycle time-extrapolates a stale
        // boundary phi (interpol_phi tfrac>0) -> wrong coarse-fine boundary force.
        @autoreleasepool {
            int bn[2] = {base, nsurv}; int wp = TWOTONDIM;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("refine_gather_field")];
            [e setBuffer:B.fld2 offset:0 atIndex:0]; [e setBuffer:B.phi_old offset:0 atIndex:1];
            [e setBuffer:B.swap offset:0 atIndex:2]; [e setBytes:&bn length:sizeof(bn) atIndex:3];
            [e setBytes:&wp length:sizeof(int) atIndex:4];
            [e dispatchThreads:MTLSizeMake(wp,nsurv,1) threadsPerThreadgroup:MTLSizeMake(wp,1,1)];
            [e endEncoding]; submit_async(cb);
        }
        mtl_drain();
        memcpy((float*)B.phi_old.contents + (size_t)(base-1)*TWOTONDIM,
               (float*)B.fld2.contents + (size_t)(base-1)*TWOTONDIM, (size_t)nsurv*TWOTONDIM*sizeof(float));
        // flag1 (int) — gather with the INT kernel (NOT the float one): int 1 is a
        // denormal float and the GPU's FTZ would flush it to 0, clearing flags during
        // compaction -> next-step derefine mass-kills the refined region.
        @autoreleasepool {
            int bn[2] = {base, nsurv}; int wp = TWOTONDIM;
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso("refine_gather_int")];
            [e setBuffer:B.fld2 offset:0 atIndex:0]; [e setBuffer:B.flag1 offset:0 atIndex:1];
            [e setBuffer:B.swap offset:0 atIndex:2]; [e setBytes:&bn length:sizeof(bn) atIndex:3];
            [e setBytes:&wp length:sizeof(int) atIndex:4];
            [e dispatchThreads:MTLSizeMake(wp,nsurv,1) threadsPerThreadgroup:MTLSizeMake(wp,1,1)];
            [e endEncoding]; submit_async(cb);
        }
        mtl_drain();   // host reads the flag1 gather output (fld2)
        memcpy((int*)B.flag1.contents + (size_t)(base-1)*TWOTONDIM,
               (float*)B.fld2.contents + (size_t)(base-1)*TWOTONDIM, (size_t)nsurv*TWOTONDIM*sizeof(int));
    }

    // ---- update level layout + free pointers ----
    for (int L=ilevel+1; L<=nlevelmax; ++L) { head[L-levelmin]=nh[L]; noct[L-levelmin]=cnt[L]; }
    *noct_used = base - 1 + nsurv;
    *ifree     = *noct_used + 1;
    // NOTE: no final hash/nbor rebuild here — the compacted-mesh connectivity is
    // rebuilt lazily by the next consumer (a recursive mtl_refine rebuilds its
    // own hash after create; the gravity's metal_sync_mesh rebuilds hash+nbor).
    // The caller sets g_synced_ifree=-1 so that rebuild is forced.
}

// Hilbert-key LSD radix sort of particles [head_idx, head_idx+num_parts-1].
void mtl_sort_part(int ilevel, int head_idx, int num_parts, int npartmax) {
    if (num_parts <= 0) return;
    @autoreleasepool {
        ScanParams S{}; S.n=num_parts; S.head_idx=head_idx; S.npartmax=npartmax; S.ilevel=ilevel;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        // hilbert keys
        [e setComputePipelineState:pso("compute_hkey_part")];
        [e setBuffer:B.ipos offset:0 atIndex:0]; [e setBuffer:B.hkey_part offset:0 atIndex:1];
        [e setBytes:&S length:sizeof(S) atIndex:2];
        dispatch1d(e, pso("compute_hkey_part"), num_parts);
        // identity permutation
        [e setComputePipelineState:pso("init_global_swap_table")];
        [e setBuffer:B.sortp offset:0 atIndex:0]; [e setBytes:&S length:sizeof(S) atIndex:1];
        dispatch1d(e, pso("init_global_swap_table"), num_parts);
        for (int ibit=0; ibit<NDIM*ilevel; ++ibit) {
            S.ibit=ibit;
            [e setComputePipelineState:pso("init_prefix_sum_part_bit")];
            [e setBuffer:B.hkey_part offset:0 atIndex:0]; [e setBuffer:B.sortp offset:0 atIndex:1];
            [e setBuffer:B.prefix offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
            dispatch1d(e, pso("init_prefix_sum_part_bit"), num_parts);
            gpu_scan(e, B.prefix, head_idx, num_parts);
            [e setComputePipelineState:pso("compute_local_swap_table")];
            [e setBuffer:B.isp offset:0 atIndex:0]; [e setBuffer:B.sortp offset:0 atIndex:1];
            [e setBuffer:B.prefix offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
            dispatch1d(e, pso("compute_local_swap_table"), num_parts);
            [e setComputePipelineState:pso("update_global_swap_table")];
            [e setBuffer:B.sortp offset:0 atIndex:0]; [e setBuffer:B.isp offset:0 atIndex:1];
            [e setBytes:&S length:sizeof(S) atIndex:2];
            dispatch1d(e, pso("update_global_swap_table"), num_parts);
        }
        [e endEncoding]; submit_async(cb);
    }
}

// Reorder the resident particle arrays (ipos/vp/mp/levelp) over [head,head+num)
// by the permutation perm (1-based old index): gather each column into a swap,
// then blit the reordered range back.  Only the [head,head+num) range is touched
// (other levels' particles untouched).
static void reorder_particles(int head, int num, id<MTLBuffer> perm) {
    if (num <= 0) return;
    int npm = B.npartmax;
    ScanParams S{}; S.n=num; S.head_idx=head; S.npartmax=npm;
    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    for (int dim=1; dim<=NDIM; ++dim) {
        [e setComputePipelineState:pso("part_gather_long")];
        [e setBuffer:B.ipos2 offset:0 atIndex:0]; [e setBuffer:B.ipos offset:0 atIndex:1];
        [e setBuffer:perm offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
        [e setBytes:&dim length:4 atIndex:4];
        dispatch1d(e, pso("part_gather_long"), num);
        [e setComputePipelineState:pso("part_gather_float")];
        [e setBuffer:B.vp2 offset:0 atIndex:0]; [e setBuffer:B.vp offset:0 atIndex:1];
        [e setBuffer:perm offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
        [e setBytes:&dim length:4 atIndex:4];
        dispatch1d(e, pso("part_gather_float"), num);
    }
    int one=1;
    [e setComputePipelineState:pso("part_gather_float")];
    [e setBuffer:B.mp2 offset:0 atIndex:0]; [e setBuffer:B.mp offset:0 atIndex:1];
    [e setBuffer:perm offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
    [e setBytes:&one length:4 atIndex:4];
    dispatch1d(e, pso("part_gather_float"), num);
    [e setComputePipelineState:pso("part_gather_int")];
    [e setBuffer:B.levelp2 offset:0 atIndex:0]; [e setBuffer:B.levelp offset:0 atIndex:1];
    [e setBuffer:perm offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
    dispatch1d(e, pso("part_gather_int"), num);
    [e setComputePipelineState:pso("part_gather_int")];     // carry idp (label) with the reorder
    [e setBuffer:B.idp2 offset:0 atIndex:0]; [e setBuffer:B.idp offset:0 atIndex:1];
    [e setBuffer:perm offset:0 atIndex:2]; [e setBytes:&S length:sizeof(S) atIndex:3];
    dispatch1d(e, pso("part_gather_int"), num);
    [e endEncoding];
    // copy the reordered range back (per column)
    id<MTLBlitCommandEncoder> bl=[cb blitCommandEncoder];
    for (int dim=0; dim<NDIM; ++dim) {
        NSUInteger o8=((NSUInteger)dim*npm + (head-1))*sizeof(long), n8=(NSUInteger)num*sizeof(long);
        [bl copyFromBuffer:B.ipos2 sourceOffset:o8 toBuffer:B.ipos destinationOffset:o8 size:n8];
        NSUInteger o4=((NSUInteger)dim*npm + (head-1))*sizeof(float), n4=(NSUInteger)num*sizeof(float);
        [bl copyFromBuffer:B.vp2 sourceOffset:o4 toBuffer:B.vp destinationOffset:o4 size:n4];
    }
    NSUInteger om=((NSUInteger)(head-1))*sizeof(float), nm=(NSUInteger)num*sizeof(float);
    [bl copyFromBuffer:B.mp2 sourceOffset:om toBuffer:B.mp destinationOffset:om size:nm];
    NSUInteger oi=((NSUInteger)(head-1))*sizeof(int), ni=(NSUInteger)num*sizeof(int);
    [bl copyFromBuffer:B.levelp2 sourceOffset:oi toBuffer:B.levelp destinationOffset:oi size:ni];
    [bl copyFromBuffer:B.idp2 sourceOffset:oi toBuffer:B.idp destinationOffset:oi size:ni];
    [bl endEncoding];
    submit_async(cb);
}

// GPU Hilbert sort of [head,head+num) at ilevel (reorders the resident arrays).
void mtl_gpu_sort_part(int ilevel, int head_idx, int num_parts) {
    if (num_parts <= 0) return;
    @autoreleasepool {
        mtl_sort_part(ilevel, head_idx, num_parts, B.npartmax);   // builds sortp (permutation)
        reorder_particles(head_idx, num_parts, B.sortp);
    }
}

// GPU split: bucket [head,head+num) into stay(0)/descend(1) at ilevel, stable-
// partition (0s first), reorder the arrays.  Returns the number that stay (=>
// host sets headp(ilevel)/tailp and the descended block goes to ilevel+1).
int mtl_gpu_split_part(int ilevel, int head_idx, int num_parts,
                       int hash_size, int ckey_max, long key_off) {
    if (num_parts <= 0) return 0;
    @autoreleasepool {
        ScanParams S{}; S.n=num_parts; S.head_idx=head_idx; S.npartmax=B.npartmax;
        S.ilevel=ilevel; S.hash_size=hash_size; S.ckey_max=ckey_max; S.key_off=key_off;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("bucket_part")];
        [e setBuffer:B.ipos offset:0 atIndex:0]; [e setBuffer:B.bucket offset:0 atIndex:1];
        [e setBuffer:B.grid offset:0 atIndex:2]; [e setBuffer:B.hash_key offset:0 atIndex:3];
        [e setBuffer:B.hash_val offset:0 atIndex:4]; [e setBuffer:B.ckey_max offset:0 atIndex:5];
        [e setBuffer:B.key_off offset:0 atIndex:6]; [e setBytes:&S length:sizeof(S) atIndex:7];
        dispatch1d(e, pso("bucket_part"), num_parts);
        // prefix = inclusive scan of bucket (over the range); also identity sortp first
        [e setComputePipelineState:pso("init_global_swap_table")];
        [e setBuffer:B.sortp offset:0 atIndex:0]; [e setBytes:&S length:sizeof(S) atIndex:1];
        dispatch1d(e, pso("init_global_swap_table"), num_parts);
        [e endEncoding];
        // copy bucket range -> prefix, scan it
        { id<MTLBlitCommandEncoder> bl=[cb blitCommandEncoder];
          [bl copyFromBuffer:B.bucket sourceOffset:(NSUInteger)(head_idx-1)*sizeof(int)
                    toBuffer:B.prefix destinationOffset:(NSUInteger)(head_idx-1)*sizeof(int)
                        size:(NSUInteger)num_parts*sizeof(int)]; [bl endEncoding]; }
        { id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder];
          gpu_scan(e2, B.prefix, head_idx, num_parts);
          // stable split: build swap table (0s first, then 1s) into isp
          [e2 setComputePipelineState:pso("compute_local_swap_table")];
          [e2 setBuffer:B.isp offset:0 atIndex:0]; [e2 setBuffer:B.sortp offset:0 atIndex:1];
          [e2 setBuffer:B.prefix offset:0 atIndex:2]; [e2 setBytes:&S length:sizeof(S) atIndex:3];
          dispatch1d(e2, pso("compute_local_swap_table"), num_parts);
          [e2 endEncoding]; }
        submit_async(cb);
        mtl_drain();   // host reads the scan total below
        // total descended = prefix[last] (inclusive scan of bucket over the range)
        int* pre = (int*)B.prefix.contents;
        int n_desc = pre[head_idx-1 + num_parts-1];
        // reorder by the split permutation (isp), then n_stay = num - n_desc
        reorder_particles(head_idx, num_parts, B.isp);
        return num_parts - n_desc;
    }
}
// multi-level rho can be built RESIDENT on the GPU (each level deposits into its
// own octs, all accumulating into the two-word fixed-point words; no per-level
// re-zero).  rho is the MONOPOLE (mass/cell) -> the solver applies 4piG*2^L.
// Zero the deposit accumulators for the CELL range [cell_base, cell_base+ncells).
// rho_fine deposits only levels [ilevel..nlevelmax] (oct range [head(ilevel),
// noct_used]); under subcycling it is called again at a finer ilevel.  Zeroing
// (and finalizing) ONLY that level's cell range leaves the COARSER levels' rho/
// nref intact — exactly as the CPU rho_fine does.  Wiping the whole buffer (the
// old behaviour) zeroed level-7 nref when a subcycle deposited at level 10,
// making the end-of-step level-7 flag see nref==0 -> mass derefine oscillation.
void mtl_cic_zero(int cell_base, int ncells) {
    if (ncells <= 0) return;
    mtl_drain();   // host memset of the accumulators must not race in-flight GPU
    size_t off = (size_t)cell_base*sizeof(unsigned);
    size_t len = (size_t)ncells*sizeof(unsigned);
    memset((char*)B.rho_lo.contents  + off, 0, len);
    memset((char*)B.rho_hi.contents  + off, 0, len);
    memset((char*)B.nref_lo.contents + off, 0, len);
    memset((char*)B.nref_hi.contents + off, 0, len);
}
void mtl_cic_finalize(int cell_base, int ncells) {
    if (ncells <= 0) return;
    @autoreleasepool {
        float inv = 1.0f/(float)(1L<<FP_SHIFT_RHO);
        size_t uoff = (size_t)cell_base*sizeof(unsigned);
        size_t foff = (size_t)cell_base*sizeof(float);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("rho_finalize")];
        [e setBuffer:B.rho_lo offset:uoff atIndex:0]; [e setBuffer:B.rho_hi offset:uoff atIndex:1];
        [e setBuffer:B.nref_lo offset:uoff atIndex:2]; [e setBuffer:B.nref_hi offset:uoff atIndex:3];
        [e setBuffer:B.rho offset:foff atIndex:4]; [e setBuffer:B.nref offset:foff atIndex:5];
        [e setBytes:&inv length:4 atIndex:6]; [e setBytes:&ncells length:4 atIndex:7];
        dispatch1d(e, pso("rho_finalize"), ncells);
        [e endEncoding]; submit_async(cb);
    }
}
// Deposit one level's particles [head_idx, head_idx+num_parts) into the fixed-
// point accumulators (no zero, no finalize).  Particles are already sorted in
// the host array, so sortp is set to identity over the range first.
void mtl_cic_deposit(int ilevel, int head_idx, int num_parts, int npartmax,
                     int hash_size, int ckey_max, long key_off,
                     float m_refine, float mass_cut, int refine_on,
                     int periodic0, int periodic1, int periodic2) {
    if (num_parts <= 0) return;
    @autoreleasepool {
        CicParams P{}; P.dx_loc=1.0f; P.vol_loc=1.0f; P.m_refine=m_refine; P.mass_cut=mass_cut;
        P.fp_scale_rho=(float)(1L<<FP_SHIFT_RHO); P.key_off=key_off; P.ckey_max=ckey_max;
        P.hash_size=hash_size; P.ilevel=ilevel; P.head_idx=head_idx; P.num_parts=num_parts;
        P.npartmax=npartmax; P.refine_on=refine_on;
        P.ngridmax=B.ngridmax;   // skip depositing onto cache (ghost) octs (>ngridmax), matching CUDA deposit_rho
        BoxParams BX{}; for(int d=0;d<3;++d){ BX.box_min[d]=0; BX.box_max[d]=1<<ilevel; }
        BX.periodic[0]=periodic0; BX.periodic[1]=periodic1; BX.periodic[2]=periodic2;
        ScanParams S{}; S.n=num_parts; S.head_idx=head_idx; S.npartmax=npartmax; S.ilevel=ilevel;

        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        // sortp = identity over the range (host array already Hilbert-sorted)
        [e setComputePipelineState:pso("init_global_swap_table")];
        [e setBuffer:B.sortp offset:0 atIndex:0]; [e setBytes:&S length:sizeof(S) atIndex:1];
        dispatch1d(e, pso("init_global_swap_table"), num_parts);
        [e setComputePipelineState:pso("build_src_part")];
        [e setBuffer:B.ipos offset:0 atIndex:0]; [e setBuffer:B.sortp offset:0 atIndex:1];
        [e setBuffer:B.isp offset:0 atIndex:2]; [e setBuffer:B.hash_key offset:0 atIndex:3];
        [e setBuffer:B.hash_val offset:0 atIndex:4]; [e setBytes:&P length:sizeof(P) atIndex:5];
        dispatch1d(e, pso("build_src_part"), num_parts);
        [e setComputePipelineState:pso("cic_part_warp")];
        [e setBuffer:B.sortp offset:0 atIndex:0]; [e setBuffer:B.isp offset:0 atIndex:1];
        [e setBuffer:B.grid offset:0 atIndex:2]; [e setBuffer:B.hash_key offset:0 atIndex:3];
        [e setBuffer:B.hash_val offset:0 atIndex:4]; [e setBuffer:B.ipos offset:0 atIndex:5];
        [e setBuffer:B.mp offset:0 atIndex:6];
        [e setBuffer:B.rho_lo offset:0 atIndex:7]; [e setBuffer:B.rho_hi offset:0 atIndex:8];
        [e setBuffer:B.nref_lo offset:0 atIndex:9]; [e setBuffer:B.nref_hi offset:0 atIndex:10];
        [e setBytes:&P length:sizeof(P) atIndex:11]; [e setBytes:&BX length:sizeof(BX) atIndex:12];
        [e dispatchThreadgroups:MTLSizeMake((num_parts+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; submit_async(cb);
    }
}
// Single-level convenience (standalone tests): zero + deposit + finalize.
void mtl_cic_part(int ilevel, int head_idx, int num_parts, int npartmax,
                  int hash_size, int ckey_max, long key_off,
                  float m_refine, float mass_cut, int refine_on,
                  int periodic0, int periodic1, int periodic2) {
    if (num_parts <= 0) return;
    int allcells = B.ncell * TWOTONDIM;
    mtl_cic_zero(0, allcells);
    mtl_cic_deposit(ilevel, head_idx, num_parts, npartmax, hash_size, ckey_max, key_off,
                    m_refine, mass_cut, refine_on, periodic0, periodic1, periodic2);
    mtl_cic_finalize(0, allcells);
}

// One bundle of red-black Gauss-Seidel sweeps on the single (fine) level.
void mtl_gauss_seidel(int head_idx, int num_octs, int nsweep, int safe) {
    @autoreleasepool {
        MgParams P{}; P.head_idx=head_idx; P.num_octs=num_octs; P.ngridmax=B.ngridmax;   // real-oct bound: nbr>ngridmax => cache (ghost) oct
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        for (int s=0;s<nsweep;++s) for (int rb=1;rb>=0;--rb) {
            P.redstep=rb;
            [e setComputePipelineState:pso("gauss_seidel")];
            [e setBuffer:B.phi offset:0 atIndex:0]; [e setBuffer:B.f offset:0 atIndex:1];
            [e setBuffer:B.nbor offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
            [e setBytes:&safe length:4 atIndex:4];
            [e setBuffer:B.grid offset:0 atIndex:5]; [e setBuffer:B.father offset:0 atIndex:6];
            [e setBuffer:B.phi_old offset:0 atIndex:7];
            [e dispatchThreads:MTLSizeMake(4,num_octs,1) threadsPerThreadgroup:MTLSizeMake(4,16,1)];
        }
        [e endEncoding]; submit_async(cb);
    }
}

// Force f = -grad phi (4th order) on the fine level.
// Snapshot phi -> phi_old for one level's octs (gpu_save_phi_old).  Must run BEFORE
// the level's MG solve overwrites B.phi (make_initial_phi resets it), so the
// finer level's subcycle can time-extrapolate the coarse boundary phi.
void mtl_save_phi_old(int head_idx, int num_octs) {
    if (num_octs <= 0) return;
    @autoreleasepool {
        size_t off = (size_t)(head_idx-1)*TWOTONDIM*sizeof(float);
        size_t len = (size_t)num_octs*TWOTONDIM*sizeof(float);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLBlitCommandEncoder> b=[cb blitCommandEncoder];
        [b copyFromBuffer:B.phi sourceOffset:off toBuffer:B.phi_old destinationOffset:off size:len];
        [b endEncoding]; submit_async(cb);
    }
}

void mtl_gradient_phi(int head_idx, int num_octs, float dx, float tfrac) {
    @autoreleasepool {
        // gradient_phi: with cache octs ON the boundary neighbour is a materialised cache
        // oct (read directly); with cache OFF the neighbour index is 0 and gradient_phi
        // reconstructs it inline via mg_interpol_ghost -> needs grid/hash/ckey_max/key_off/
        // box/phi_old.  These are bound unconditionally (unused on the cache path).
        MgParams P{}; P.head_idx=head_idx; P.num_octs=num_octs; P.ngridmax=B.ngridmax;
        P.dx=dx; P.tfrac=tfrac; P.hash_size=B.hash_size;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("gradient_phi")];
        [e setBuffer:B.phi offset:0 atIndex:0];    [e setBuffer:B.f offset:0 atIndex:1];
        [e setBuffer:B.nbor offset:0 atIndex:2];   [e setBytes:&P length:sizeof(P) atIndex:3];
        [e setBuffer:B.grid offset:0 atIndex:4];   [e setBuffer:B.hash_key offset:0 atIndex:5];
        [e setBuffer:B.hash_val offset:0 atIndex:6];[e setBuffer:B.ckey_max offset:0 atIndex:7];
        [e setBuffer:B.key_off offset:0 atIndex:8];[e setBuffer:B.box_min offset:0 atIndex:9];
        [e setBuffer:B.box_max offset:0 atIndex:10];[e setBuffer:B.phi_old offset:0 atIndex:11];
        [e dispatchThreads:MTLSizeMake(num_octs,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e endEncoding]; submit_async(cb);
    }
}

// Leapfrog kick/drift: gather force (CIC) and update vp/ipos.
void mtl_kick_drift_part(int ilevel, int head_idx, int num_parts, int npartmax,
                         int hash_size, int action_part, float dtnew, float dtold,
                         float box0, float box1, float box2,
                         int periodic0, int periodic1, int periodic2) {
    if (num_parts <= 0) return;
    @autoreleasepool {
        PartParams P{}; P.dtnew=dtnew; P.dtold=dtold;
        P.box_size[0]=box0; P.box_size[1]=box1; P.box_size[2]=box2;
        P.hash_size=hash_size; P.ilevel=ilevel; P.head_idx=head_idx; P.num_parts=num_parts;
        // Gather rejects cache (ghost) octs (igrid>ngridmax -> coarse fallback), matching CUDA
        // gather_cic_force_part (their f is stale: gradient_phi does not update cache octs).
        // [Tested: reading cache octs did NOT fix the <vx> drift -> the drift is the boundary
        //  gradient asymmetry from piecewise-constant cache phi, not the gather adjoint.]
        P.npartmax=npartmax; P.ngridmax=B.ngridmax; P.action_part=action_part;
        int per[3]={periodic0,periodic1,periodic2};
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("kick_drift_part")];
        [e setBuffer:B.ipos offset:0 atIndex:0]; [e setBuffer:B.vp offset:0 atIndex:1];
        [e setBuffer:B.levelp offset:0 atIndex:2]; [e setBuffer:B.f offset:0 atIndex:3];
        [e setBuffer:B.hash_key offset:0 atIndex:4]; [e setBuffer:B.hash_val offset:0 atIndex:5];
        [e setBuffer:B.ckey_max offset:0 atIndex:6]; [e setBuffer:B.key_off offset:0 atIndex:7];
        [e setBytes:per length:sizeof(per) atIndex:8]; [e setBytes:&P length:sizeof(P) atIndex:9];
        dispatch1d(e, pso("kick_drift_part"), num_parts);
        [e endEncoding]; submit_async(cb);
    }
}


// Dispatch the grouped residual on the fine level into B.f(:,1) (for diagnostics).
void mtl_cmp_residual_fine(int n_fine) {
    @autoreleasepool {
        MgParams P{}; P.head_idx=1; P.num_octs=n_fine; P.ngridmax=n_fine;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("cmp_residual")];
        [e setBuffer:B.phi offset:0 atIndex:0]; [e setBuffer:B.f offset:0 atIndex:1];
        [e setBuffer:B.nbor offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
        [e setBuffer:B.grid offset:0 atIndex:4]; [e setBuffer:B.father offset:0 atIndex:5];
        [e setBuffer:B.phi_old offset:0 atIndex:6];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,n_fine,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; submit_async(cb);
    }
}

// ============================================================================
// CUDA-STYLE per-leaf multigrid ops, driven by the shared Fortran multigrid()
// driver (poisson/multigrid_fine_commons.f90).  Each mirrors a gpu_* routine in
// gpu_runner.cuf: the Fortran loop/convergence/levelmin_mg/safe-mode logic is
// reused verbatim; only these inner ops run on the GPU.  They reuse the SAME
// kernels and the persistent MG hierarchy that mtl_poisson_level builds, but the
// V-cycle structure now comes from Fortran (recursive_multigrid) instead of the
// hand-coded loop -- so convergence behaviour is identical to the CPU by
// construction.  Selectors fine-level=ilevel uses the MAIN B.* arrays; coarser
// MG levels use the MG.* arrays at mg_head[lv].
// ============================================================================
static struct { int ilevel, has_coarse; float dx_fine, tfrac; } g_mgc;
static inline id<MTLBuffer> MGPHI(int lv){ return lv==g_mgc.ilevel?B.phi :MG.phi_mg; }
static inline id<MTLBuffer> MGF  (int lv){ return lv==g_mgc.ilevel?B.f   :MG.f_mg;   }
static inline id<MTLBuffer> MGNB (int lv){ return lv==g_mgc.ilevel?B.nbor:MG.nbor_mg;}
static inline id<MTLBuffer> MGGR (int lv){ return lv==g_mgc.ilevel?B.grid:MG.grid_mg;}
static inline int MGHEAD(int lv){ return lv==g_mgc.ilevel?MG.fine_head :MG.mg_head[lv]; }
static inline int MGNUM (int lv){ return lv==g_mgc.ilevel?MG.n_fine    :MG.mg_noct[lv]; }
static inline int MGNGM (int lv){ return lv==g_mgc.ilevel?B.ngridmax   :MG.n_mg; }
static inline int MGHFA (int lv){ return lv==g_mgc.ilevel?1            :(MG.n_fine+MG.mg_head[lv]); }
static inline float MGDX(int lv){ return g_mgc.dx_fine * (float)(1 << (g_mgc.ilevel - lv)); }

// Build the whole MG hierarchy (geometry + father + nbor) once, on the build_mg
// call for the finest level; the coarser-level build_mg calls are then no-ops
// (the hierarchy already holds every level).  Masks are NOT set here -- the
// Fortran make_mask + restrict_mask leaves do that, so levelmin_mg is driven by
// the Fortran allmasked loop exactly as on the CPU.
void mtl_mg_build(int ilevel, int ifine, int head_idx, int n_fine, int mg_cap,
                  const int* box_min, const int* box_max, int per0,int per1,int per2,
                  int is_base, float dx_fine, int has_coarse, float tfrac) {
    if (ifine != ilevel) return;                       // whole hierarchy built on the first call
    g_mgc.ilevel=ilevel; g_mgc.has_coarse=has_coarse; g_mgc.dx_fine=dx_fine; g_mgc.tfrac=tfrac;
    mtl_build_mg_amr(ilevel, head_idx, n_fine, mg_cap, box_min, box_max, per0,per1,per2, is_base);
}

// Fine-level mask = +1 everywhere (gpu_make_mask / CPU make_mask set f(:,3)=1.0 on
// every active fine cell; the coarse-fine boundary is carried by the materialised
// cache octs, NOT by the fine mask).  Faithful reset_mask_kernel(mask_val=1); the
// host then drives the coarse mask restriction via mtl_mg_restrict_mask per level.
void mtl_mg_make_mask(int ilevel) {
    @autoreleasepool {
        MgParams P{}; P.head_idx=MG.fine_head; P.num_octs=MG.n_fine; P.ngridmax=B.ngridmax;   // real-oct bound: nbr>ngridmax => cache (ghost) oct
        float mask_val = 1.0f;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("reset_mask_kernel")];
        [e setBuffer:B.f offset:0 atIndex:0]; [e setBytes:&P length:sizeof(P) atIndex:1];
        [e setBytes:&mask_val length:sizeof(float) atIndex:2];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,MG.n_fine,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; submit_async(cb);
    }
}

// Initial guess at the fine level: coarse-fine boundary phi (interpol_phi, time-
// extrapolated by tfrac).  Snapshots phi->phi_old first so a finer subcycle can
// extrapolate this level later (matches r_save_phi_old + make_initial_phi).
void mtl_mg_make_initial_phi(int ilevel, float dx_fine, float tfrac, int has_coarse) {
    g_mgc.ilevel=ilevel; g_mgc.has_coarse=has_coarse; g_mgc.dx_fine=dx_fine; g_mgc.tfrac=tfrac;
    mtl_save_phi_old(MG.fine_head, MG.n_fine);
    if (!has_coarse) return;
    @autoreleasepool {
        MgParams P{}; P.head_idx=MG.fine_head; P.num_octs=MG.n_fine; P.ngridmax=B.ngridmax;   // real-oct bound: nbr>ngridmax => cache (ghost) oct
        P.head_father=MG.fine_head; P.ilevel=ilevel; P.dx=dx_fine; P.tfrac=tfrac; P.use_ghost=1;
        static bool s_fbk_diag = getenv("RAMSES_FALLBACK_DIAG") != nullptr;
        if (s_fbk_diag) { mtl_drain(); *(unsigned*)B.fbk.contents = 0; }
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("make_initial_phi")];
        [e setBuffer:B.grid offset:0 atIndex:0]; [e setBuffer:B.father offset:0 atIndex:1];
        [e setBuffer:B.nbor offset:0 atIndex:2]; [e setBuffer:B.phi offset:0 atIndex:3];
        [e setBytes:&P length:sizeof(P) atIndex:4]; [e setBuffer:B.phi_old offset:0 atIndex:5];
        [e setBuffer:B.fbk offset:0 atIndex:6];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,MG.n_fine,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; submit_async(cb);
        if (s_fbk_diag) { mtl_drain(); unsigned h=*(unsigned*)B.fbk.contents;
            long tot=(long)TWOTONDIM*MG.n_fine;
            fprintf(stderr,"[FALLBACK] ilevel=%d n_fine=%d corners=%ld hits=%u (%.4f%%)\n",
                    ilevel, MG.n_fine, tot, h, tot>0?100.0*h/tot:0.0); }
    }
}

// BC-modified RHS at the fine level: S = fourpi*(rho/vol_loc - offset), minus the
// Dirichlet boundary term where a same-level neighbour is absent (reset_rhs_kernel).
void mtl_mg_make_rhs(int ilevel, float fourpi, float offset, float vol_loc,
                     float dx_fine, int has_coarse, float tfrac) {
    g_mgc.ilevel=ilevel; g_mgc.has_coarse=has_coarse; g_mgc.dx_fine=dx_fine; g_mgc.tfrac=tfrac;
    @autoreleasepool {
        MgParams P{}; P.head_idx=MG.fine_head; P.num_octs=MG.n_fine; P.ngridmax=B.ngridmax;   // real-oct bound: nbr>ngridmax => cache (ghost) oct
        P.head_father=MG.fine_head; P.ilevel=ilevel; P.fourpi=fourpi; P.offset=offset;
        P.vol_loc=vol_loc; P.dx=dx_fine; P.tfrac=tfrac; P.use_ghost=has_coarse;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        P.hash_size=B.hash_size;
        [e setComputePipelineState:pso("reset_rhs_kernel")];
        [e setBuffer:B.phi offset:0 atIndex:0]; [e setBuffer:B.rho offset:0 atIndex:1];
        [e setBuffer:B.f offset:0 atIndex:2]; [e setBuffer:B.nbor offset:0 atIndex:3];
        [e setBytes:&P length:sizeof(P) atIndex:4];
        [e setBuffer:B.grid offset:0 atIndex:5];     [e setBuffer:B.hash_key offset:0 atIndex:6];
        [e setBuffer:B.hash_val offset:0 atIndex:7]; [e setBuffer:B.ckey_max offset:0 atIndex:8];
        [e setBuffer:B.key_off offset:0 atIndex:9];  [e setBuffer:B.box_min offset:0 atIndex:10];
        [e setBuffer:B.box_max offset:0 atIndex:11]; [e setBuffer:B.phi_old offset:0 atIndex:12];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,MG.n_fine,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; submit_async(cb);
    }
}

// Restrict the mask from level ifine into ifine-1 (continuous distance field),
// then convert volume->mask; return 1 if the coarser level is fully masked
// (allmasked -> the Fortran sets levelmin_mg=ifine and stops coarsening).
int mtl_mg_restrict_mask(int ilevel, int ifine) {
    int clev = ifine - 1, chd = MG.mg_head[clev], cnum = MG.mg_noct[clev];
    if (cnum <= 0) return 1;
    float* fmg = (float*)MG.f_mg.contents;
    mtl_drain();
    for (int q=chd; q<chd+cnum; ++q) for (int c=1;c<=TWOTONDIM;++c) fmg[IDX3(c,3,q)] = 0.0f;
    @autoreleasepool {
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        MgParams Pr{}; Pr.head_idx=MGHEAD(ifine); Pr.num_octs=MGNUM(ifine); Pr.head_father=MGHFA(ifine);
        [e setComputePipelineState:pso("restrict_mask")];
        [e setBuffer:MGGR(ifine) offset:0 atIndex:0]; [e setBuffer:MG.father_mg offset:0 atIndex:1];
        [e setBuffer:MGF(ifine) offset:0 atIndex:2]; [e setBuffer:MG.f_mg offset:0 atIndex:3];
        [e setBytes:&Pr length:sizeof(Pr) atIndex:4];
        dispatch1d(e, pso("restrict_mask"), MGNUM(ifine));
        MgParams Pv{}; Pv.head_idx=chd; Pv.num_octs=cnum;
        [e setComputePipelineState:pso("volume_to_mask")];
        [e setBuffer:MG.f_mg offset:0 atIndex:0]; [e setBytes:&Pv length:sizeof(Pv) atIndex:1];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,cnum,1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; submit_async(cb);
    }
    mtl_drain();
    double mmax = -1.0;
    for (int q=chd; q<chd+cnum; ++q) for (int c=1;c<=TWOTONDIM;++c) mmax = fmax(mmax,(double)fmg[IDX3(c,3,q)]);
    return (mmax <= 0.0) ? 1 : 0;
}

// One red or black Gauss-Seidel sweep at level ifine.
void mtl_mg_gauss_seidel(int ilevel, int ifine, int safe, int redstep) {
    if (MGNUM(ifine) <= 0) return;
    @autoreleasepool {
        const int HALF = TWOTONDIM/2;
        MgParams P{}; P.head_idx=MGHEAD(ifine); P.num_octs=MGNUM(ifine); P.ngridmax=MGNGM(ifine);
        P.use_ghost = (ifine==ilevel)?g_mgc.has_coarse:0; P.tfrac=(ifine==ilevel)?g_mgc.tfrac:0.0f;
        P.dx = MGDX(ifine); P.redstep = redstep;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("gauss_seidel")];
        [e setBuffer:MGPHI(ifine) offset:0 atIndex:0]; [e setBuffer:MGF(ifine) offset:0 atIndex:1];
        [e setBuffer:MGNB(ifine) offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
        [e setBytes:&safe length:4 atIndex:4];
        [e setBuffer:B.grid offset:0 atIndex:5]; [e setBuffer:B.father offset:0 atIndex:6];
        [e setBuffer:B.phi_old offset:0 atIndex:7];
        [e dispatchThreads:MTLSizeMake(HALF,MGNUM(ifine),1) threadsPerThreadgroup:MTLSizeMake(HALF,16,1)];
        [e endEncoding]; submit_async(cb);
    }
}

// Batched smoothing (Stage 3): encode nsweep red+black Gauss-Seidel sweeps into ONE
// command buffer / encoder instead of 2*nsweep separate command buffers.  Numerically
// IDENTICAL to calling mtl_mg_gauss_seidel 2*nsweep times: same kernel, same red-then-
// black order; the default compute encoder is serial with automatic hazard tracking, so
// each black sweep correctly sees the phi the preceding red sweep wrote.  Removes the
// per-sweep [g_queue commandBuffer]/commit overhead from the hot smoothing path.
void mtl_mg_smooth(int ilevel, int ifine, int safe, int nsweep) {
    if (MGNUM(ifine) <= 0 || nsweep <= 0) return;
    @autoreleasepool {
        const int HALF = TWOTONDIM/2;
        MgParams P{}; P.head_idx=MGHEAD(ifine); P.num_octs=MGNUM(ifine); P.ngridmax=MGNGM(ifine);
        P.use_ghost = (ifine==ilevel)?g_mgc.has_coarse:0; P.tfrac=(ifine==ilevel)?g_mgc.tfrac:0.0f;
        P.dx = MGDX(ifine);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("gauss_seidel")];
        [e setBuffer:MGPHI(ifine) offset:0 atIndex:0]; [e setBuffer:MGF(ifine) offset:0 atIndex:1];
        [e setBuffer:MGNB(ifine) offset:0 atIndex:2];
        [e setBytes:&safe length:4 atIndex:4];
        [e setBuffer:B.grid offset:0 atIndex:5]; [e setBuffer:B.father offset:0 atIndex:6];
        [e setBuffer:B.phi_old offset:0 atIndex:7];
        for (int s=0; s<nsweep; ++s) {
            for (int red=1; red>=0; --red) {
                P.redstep = red;
                [e setBytes:&P length:sizeof(P) atIndex:3];
                [e dispatchThreads:MTLSizeMake(HALF,MGNUM(ifine),1)
                    threadsPerThreadgroup:MTLSizeMake(HALF,16,1)];
            }
        }
        [e endEncoding]; submit_async(cb);
    }
}

// Residual at level ifine -> f(:,1).
void mtl_mg_cmp_residual(int ilevel, int ifine) {
    if (MGNUM(ifine) <= 0) return;
    @autoreleasepool {
        MgParams P{}; P.head_idx=MGHEAD(ifine); P.num_octs=MGNUM(ifine); P.ngridmax=MGNGM(ifine);
        P.use_ghost = (ifine==ilevel)?g_mgc.has_coarse:0; P.tfrac=(ifine==ilevel)?g_mgc.tfrac:0.0f;
        P.dx = MGDX(ifine);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("cmp_residual")];
        [e setBuffer:MGPHI(ifine) offset:0 atIndex:0]; [e setBuffer:MGF(ifine) offset:0 atIndex:1];
        [e setBuffer:MGNB(ifine) offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
        [e setBuffer:B.grid offset:0 atIndex:4]; [e setBuffer:B.father offset:0 atIndex:5];
        [e setBuffer:B.phi_old offset:0 atIndex:6];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,MGNUM(ifine),1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; submit_async(cb);
    }
}

// Restrict residual ifine -> ifine-1 RHS (mean).
void mtl_mg_restrict_residual(int ilevel, int ifine) {
    if (MGNUM(ifine) <= 0) return;
    @autoreleasepool {
        MgParams P{}; P.head_idx=MGHEAD(ifine); P.num_octs=MGNUM(ifine); P.head_father=MGHFA(ifine);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("restrict_residual")];
        [e setBuffer:MGGR(ifine) offset:0 atIndex:0]; [e setBuffer:MG.father_mg offset:0 atIndex:1];
        [e setBuffer:MGF(ifine) offset:0 atIndex:2]; [e setBuffer:MG.f_mg offset:0 atIndex:3];
        [e setBytes:&P length:sizeof(P) atIndex:4];
        dispatch1d(e, pso("restrict_residual"), MGNUM(ifine));
        [e endEncoding]; submit_async(cb);
    }
}

// Zero the correction (phi) at MG level ifine before solving it.
void mtl_mg_reset_corr(int ilevel, int ifine) {
    int n = MGNUM(ifine); if (n <= 0) return;
    @autoreleasepool {
        size_t off = (size_t)(MGHEAD(ifine)-1)*TWOTONDIM*sizeof(float);
        size_t len = (size_t)n*TWOTONDIM*sizeof(float);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLBlitCommandEncoder> bl=[cb blitCommandEncoder];
        [bl fillBuffer:MGPHI(ifine) range:NSMakeRange(off,len) value:0];
        [bl endEncoding]; submit_async(cb);
    }
}

// Interpolate coarse correction (ifine-1) and add into level ifine (CIC).
void mtl_mg_interpolate_correct(int ilevel, int ifine) {
    if (MGNUM(ifine) <= 0) return;
    @autoreleasepool {
        MgParams P{}; P.head_idx=MGHEAD(ifine); P.num_octs=MGNUM(ifine); P.head_father=MGHFA(ifine);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("interpolate_correct")];
        [e setBuffer:MGGR(ifine) offset:0 atIndex:0]; [e setBuffer:MG.father_mg offset:0 atIndex:1];
        [e setBuffer:MG.nbor_mg offset:0 atIndex:2]; [e setBuffer:MGPHI(ifine) offset:0 atIndex:3];
        [e setBuffer:MG.phi_mg offset:0 atIndex:4]; [e setBuffer:MGF(ifine) offset:0 atIndex:5];
        [e setBytes:&P length:sizeof(P) atIndex:6];
        [e dispatchThreads:MTLSizeMake(TWOTONDIM,MGNUM(ifine),1) threadsPerThreadgroup:MTLSizeMake(TWOTONDIM,8,1)];
        [e endEncoding]; submit_async(cb);
    }
}

// Zero-mean gauge pin for the periodic base (ilevel==levelmin): subtract the mean
// of phi so the constant null-space mode doesn't drift -> keeps phi O(physical) so
// deep-level 2nd-difference operators don't suffer fp32 cancellation.  Pure gauge
// (forces unchanged).  Mirrors phase (D) of mtl_poisson_level.  Call only for the
// base level; refined levels are Dirichlet (no null space).
void mtl_mg_gauge_pin(int ilevel) {
    int nF = MG.n_fine, head = MG.fine_head;
    if (nF <= 0) return;
    // Host-side, double-precision mean of phi over ACTIVE (mask>0) fine cells, then
    // subtract.  Host double IS the extended precision the hybrid plan wants for the
    // periodic-base null-space removal (replaces the deleted mg_phi_sum/mg_phi_shift
    // kernels; same gauge pin, no fp32 cancellation in the reduction).
    mtl_drain();
    float* phi = (float*)B.phi.contents;
    const float* f = (const float*)B.f.contents;
    double sum = 0.0; long cnt = 0;
    for (int o=head; o<head+nF; ++o) for (int c=1;c<=TWOTONDIM;++c)
        if (f[IDX3(c,3,o)] > 0.0f) { sum += (double)phi[IDX2(c,o)]; ++cnt; }
    if (cnt == 0) return;
    float mean = (float)(sum / (double)cnt);
    for (int o=head; o<head+nF; ++o) for (int c=1;c<=TWOTONDIM;++c)
        if (f[IDX3(c,3,o)] > 0.0f) phi[IDX2(c,o)] -= mean;
}

// Project the null space out of a coarse level's RHS f(:,2): subtract its mean
// over the active (mask>0) cells.  REQUIRED on the PERIODIC base hierarchy: that
// operator is singular (constant null space), so a nonzero-mean RHS is
// inconsistent and Gauss-Seidel AMPLIFIES the constant mode (the restricted
// residual is zero-mean only in exact arithmetic; fp32 cancellation leaves a
// small mean that grows each V-cycle -> base over-energization).  Refined
// (Dirichlet) coarse levels are non-singular and must NOT be zero-meaned, so the
// caller gates this on the base solve only.
void mtl_mg_zeromean_rhs(int ilevel, int ifine) {
    int n = MGNUM(ifine); if (n <= 0) return;
    mtl_drain();
    float* f = (float*)MGF(ifine).contents;
    int head = MGHEAD(ifine);
    double sum = 0.0; long cnt = 0;
    for (int o = head; o < head + n; ++o)
        for (int c = 1; c <= TWOTONDIM; ++c) {
            if (f[IDX3(c,3,o)] > 0.0f) { sum += (double)f[IDX3(c,2,o)]; ++cnt; }
        }
    if (cnt == 0) return;
    float mean = (float)(sum / (double)cnt);
    for (int o = head; o < head + n; ++o)
        for (int c = 1; c <= TWOTONDIM; ++c)
            if (f[IDX3(c,3,o)] > 0.0f) f[IDX3(c,2,o)] -= mean;
}

// PROBE: residual L2 norm (sum r^2 over unmasked cells) at ANY MG level ifine,
// reading that level's f(:,1)/f(:,3).  Used to localise where the V-cycle stalls.
double mtl_mg_norm_at(int ilevel, int ifine) {
    int n = MGNUM(ifine); if (n <= 0) return 0.0;
    int ntg = (n + 255) / 256;
    @autoreleasepool {
        MgParams P{}; P.head_idx=MGHEAD(ifine); P.num_octs=n;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("residual_norm")];
        [e setBuffer:MGF(ifine) offset:0 atIndex:0]; [e setBuffer:B.mgnorm offset:0 atIndex:1];
        [e setBytes:&P length:sizeof(P) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; submit_async(cb);
    }
    mtl_drain();
    const float* part=(const float*)B.mgnorm.contents;
    double s=0.0; for (int i=0;i<ntg;++i) s+=(double)part[2*i]+(double)part[2*i+1];  // df64 hi+lo
    return s;
}

// Fine-level residual L2 norm (sum r^2 over unmasked cells); host sums partials.
double mtl_mg_residual_norm2(int ilevel) {
    int n = MG.n_fine, ntg = (n + 255) / 256;
    @autoreleasepool {
        MgParams P{}; P.head_idx=MG.fine_head; P.num_octs=n;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso("residual_norm")];
        [e setBuffer:B.f offset:0 atIndex:0]; [e setBuffer:B.mgnorm offset:0 atIndex:1];
        [e setBytes:&P length:sizeof(P) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; submit_async(cb);
    }
    mtl_drain();
    const float* part = (const float*)B.mgnorm.contents;
    double s = 0.0; for (int i=0;i<ntg;++i) s += (double)part[2*i]+(double)part[2*i+1];  // df64 hi+lo
    return s;
}

void mtl_finalize() { g_pso.clear(); g_lib=nil; g_queue=nil; g_dev=nil; g_mg.clear(); g_mg_nl=-1; }

} // extern "C"

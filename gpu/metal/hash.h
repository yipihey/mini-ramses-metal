//============================================================================
// hash.h  —  Metal analog of gpu_hash.cuf (device-side READ path).
//
// Mirrors the gpu_hash.cuf logical steps that run on the GPU: the FNV-1a key
// hash and the read-only linear-probe lookup.  Hash INSERTION (hash_set/free/
// update, insert_hash_kernel, update_nbor) is performed on the HOST in the
// runner (metal_bridge.mm) because this Apple GPU/toolchain has no 64-bit
// atomics; the GPU kernels only ever READ the table (immutable during them).
//============================================================================
#ifndef RAMSES_MSL_HASH_H
#define RAMSES_MSL_HASH_H

#include <metal_stdlib>
#include "../ramses_metal.h"
using namespace metal;

// 64-bit FNV-1a hash.  Returns the SIGNED bit pattern so the bucket modulo
// reproduces the CUDA-Fortran signed MOD + fixup EXACTLY.
inline long fnv64(long key_signed) {
    ulong key = as_type<ulong>(key_signed);
    ulong h   = 14695981039346656037UL;   // fnv64_basis
    for (int j = 0; j < 8; ++j) {
        ulong k = (key >> (8 * j)) & 0xFFUL;
        h ^= k;
        h *= 1099511628211UL;              // fnv64_prime
    }
    return as_type<long>(h);
}

// 1-based bucket index, matching Fortran:
//   ibucket = MOD(fnv64(key), hash_size); if(<0) +hash_size; ibucket+1
inline int hash_bucket(long key, int hash_size) {
    long ib = fnv64(key) % (long)hash_size;   // C '%' truncates toward zero == Fortran MOD
    if (ib < 0) ib += hash_size;
    return (int)ib + 1;
}

// hash_get: read-only lookup (table immutable during gravity kernels).
// Returns 0 on miss (empty slot), matching the Fortran sentinel.
// hkey/hval are 1-based logical arrays -> deref [idx-1].
inline int hash_get(device const long* hkey, device const int* hval,
                    int hash_size, long key) {
    int b = hash_bucket(key, hash_size);
    for (;;) {
        long cur = hkey[b - 1];
        if (cur == key) return hval[b - 1];
        if (cur == 0)   return 0;
        b = (b % hash_size) + 1;            // linear probe, wraps 1..hash_size
    }
}

#endif // RAMSES_MSL_HASH_H

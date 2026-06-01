//============================================================================
// hilbert.h  —  Metal analog of gpu_hilbert.cuf.
//
// Hilbert space-filling key: exact integer transcription of gpu_hilbert.cuf
// (left_shift=4, right_shift=-1; amr/hilbert.f90).  NDIM=1 -> identity (the
// line is its own space-filling curve).
//============================================================================
#ifndef RAMSES_MSL_HILBERT_H
#define RAMSES_MSL_HILBERT_H

#include <metal_stdlib>
#include "../ramses_metal.h"
using namespace metal;

constant long hilbert_next_digits[96] = {
     0, 1, 3, 2, 7, 6, 4, 5,
     0, 7, 1, 6, 3, 4, 2, 5,
     0, 3, 7, 4, 1, 2, 6, 5,
     2, 3, 1, 0, 5, 4, 6, 7,
     4, 3, 5, 2, 7, 0, 6, 1,
     6, 5, 1, 2, 7, 4, 0, 3,
     4, 7, 3, 0, 5, 6, 2, 1,
     6, 7, 5, 4, 1, 0, 2, 3,
     2, 5, 3, 4, 1, 6, 0, 7,
     2, 1, 5, 6, 3, 0, 4, 7,
     4, 5, 7, 6, 3, 2, 0, 1,
     6, 1, 7, 0, 5, 2, 4, 3
};
constant int hilbert_next_state[96] = {
     1, 2, 3, 2, 4, 5, 3, 5,
     2, 6, 0, 7, 8, 8, 0, 7,
     0, 9,10, 9, 1, 1,11,11,
     6, 0, 6,11, 9, 0, 9, 8,
    11,11, 0, 7, 5, 9, 0, 7,
     4, 4, 8, 8, 0, 6,10, 6,
     5, 7, 5, 3, 1, 1,11,11,
     6, 1, 6,10, 9, 4, 9,10,
    10, 3, 1, 1,10, 3, 5, 9,
     4, 4, 8, 8, 2, 7, 2, 3,
     7, 2,11, 2, 7, 5, 8, 5,
    10, 3, 2, 6,10, 3, 4, 4
};

inline long hilbert_key(int3 ix, int level) {
#if NDIM == 1
    // 1D space-filling curve is the identity (the line itself): key = coordinate.
    (void)level;
    return (long)ix.x;
#else
    int  cstate = 0;
    long hkey   = 0;
    int  ixv[3] = { ix.x, ix.y, ix.z };
    for (int ibit = level - 1; ibit >= 0; --ibit) {
        hkey = hkey << 4;                       // ISHFT(.,left_shift=+4)
        hkey = (long)((ulong)hkey >> 1);        // ISHFT(.,right_shift=-1)
        int sdigit = 0;
        for (int idim = 1; idim <= NDIM; ++idim) {
            int add_digit = 1 << (NDIM - idim); // 2^(ndim-idim)
            if ((ixv[idim - 1] >> ibit) & 1) sdigit += add_digit;
        }
        int ind = cstate * TWOTONDIM + sdigit;
        cstate  = hilbert_next_state[ind];
        hkey    = hkey + hilbert_next_digits[ind];
    }
    return hkey;
#endif
}

#endif // RAMSES_MSL_HILBERT_H

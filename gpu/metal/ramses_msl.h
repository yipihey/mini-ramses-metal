//============================================================================
// ramses_msl.h
//
// Umbrella include for the mini-ramses DM-gravity Metal kernels.  The device-
// side leaf primitives are split into per-module headers NAMED AFTER the CUDA-
// Fortran source they transliterate, so the Metal tree mirrors the .cuf tree
// and is easy to cross-reference:
//
//   hash.h     <- gpu_hash.cuf      (fnv64 / hash_get; read path)
//   hilbert.h  <- gpu_hilbert.cuf   (hilbert_key)
//   utils.h    <- gpu_utils.cuf     (NDIM-generic parent-cell / oct-key / CIC
//                                    father index+weight / fixed-point pos)
//   reduce.h   <- gpu_reduce.cuf    (fixed-point atomics + simd/tg reductions)
//   nbor.h     <- gpu_nbor.cuf      (3^NDIM father-cell gather + mg_nbor)
//
// Each gpu/metal/*.metal kernel file may #include this umbrella (or just the
// per-module headers it needs).  Every .metal file is a separate translation
// unit (own .air linked into the .metallib), so file-scope `constant` tables in
// these headers are duplicated per TU, which is fine.
//============================================================================
#ifndef RAMSES_MSL_H
#define RAMSES_MSL_H

#include <metal_stdlib>
#include "../ramses_metal.h"   // Oct, index macros, param structs, fixed-point
using namespace metal;

#include "hash.h"      // gpu_hash.cuf    : fnv64, hash_bucket, hash_get
#include "hilbert.h"   // gpu_hilbert.cuf : hilbert_key
#include "utils.h"     // gpu_utils.cuf   : POW3, floor_div2, mg_parent_cell, mg_oct_key, CIC, fix-pos
#include "reduce.h"    // gpu_reduce.cuf  : fixed-point atomics + reductions
#include "nbor.h"      // gpu_nbor.cuf    : nbor_father_cells_mg, mg_nbor, MG_CUBE_CENTER

#endif // RAMSES_MSL_H

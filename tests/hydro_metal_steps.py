#!/usr/bin/env python3
# Multi-step CPU-vs-Metal parity for the hydro Godunov update.  Two independent
# states from the SAME uniform periodic 1D namelist evolve in lockstep:
#   state A: CPU  newdt -> set_unew -> godunov_fine -> set_uold   (fp64 umuscl)
#   state B: Metal newdt -> ramses_metal_godunov_fine             (fp32 hydro.metal)
# After each step we diff m%uold by ckey.  Advection is non-chaotic, so a faithful
# fp32 port must track the fp64 CPU to ~fp32 accumulation (a slow O(sqrt(N)*eps)
# drift), not diverge.  Reports the per-step and final L2-rel drift.
import ctypes as C, os, sys, math

LIB = sys.argv[1] if len(sys.argv) > 1 else "bin/libramses1d_metal.dylib"
NML = sys.argv[2] if len(sys.argv) > 2 else "namelist/advect1d_uniform.nml"
NSTEP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
NDIM = 1
TWOTONDIM = 1 << NDIM

lib = C.CDLL(os.path.abspath(LIB))
lib.ramses_init.restype = C.c_int
lib.ramses_init.argtypes = [C.c_char_p, C.c_int]
lib.ramses_nvar.restype = C.c_int
lib.ramses_get_hydro.restype = C.c_int
lib.ramses_get_hydro.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                 C.POINTER(C.c_int), C.POINTER(C.c_double)]
for fn in ("ramses_set_unew", "ramses_godunov_fine", "ramses_set_uold",
           "ramses_metal_godunov_fine", "ramses_newdt_fine"):
    getattr(lib, fn).argtypes = [C.c_int, C.c_int]
lib.ramses_get_dt.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_double),
                              C.POINTER(C.c_double), C.POINTER(C.c_double)]

A = lib.ramses_init(NML.encode(), -1)   # CPU state
B = lib.ramses_init(NML.encode(), -1)   # Metal state
if A <= 0 or B <= 0:
    print("FAIL: ramses_init", A, B); sys.exit(2)

NMAX = 20000
ckey = (C.c_int * (NDIM * NMAX))()
val  = (C.c_double * (TWOTONDIM * NMAX))()
def get(h, ivar, L):
    n = lib.ramses_get_hydro(h, 0, ivar, L, NMAX, ckey, val)
    return n, [val[i] for i in range(TWOTONDIM * n)]

L = next((lv for lv in range(1, 21) if lib.ramses_get_hydro(A, 0, 1, lv, NMAX, ckey, val) > 0), -1)
noct, _ = get(A, 1, L)
print(f"level {L}: {noct} octs ({noct*TWOTONDIM} cells), {NSTEP} steps")

def drift(L):
    num = den = 0.0
    for iv in range(1, 6):
        _, a = get(A, iv, L); _, b = get(B, iv, L)
        num += sum((x - y) ** 2 for x, y in zip(a, b))
        den += sum(x * x for x in a)
    return math.sqrt(num / (den or 1.0))

dtn = C.c_double(); dto = C.c_double(); ax = C.c_double()
worst = 0.0
for step in range(1, NSTEP + 1):
    lib.ramses_newdt_fine(A, L)
    lib.ramses_set_unew(A, L); lib.ramses_godunov_fine(A, L); lib.ramses_set_uold(A, L)
    lib.ramses_newdt_fine(B, L)
    lib.ramses_metal_godunov_fine(B, L)
    d = drift(L); worst = max(worst, d)
    if step % 5 == 0 or step == 1:
        lib.ramses_get_dt(A, L, dtn, dto, ax)
        print(f"  step {step:3d}  dt={dtn.value:.4e}  L2rel(CPU,GPU)={d:.3e}")

TOL = 5e-5   # fp32 Godunov accumulation over NSTEP steps (advection, non-chaotic)
print("PASS" if worst < TOL else "FAIL", f"(worst L2 rel = {worst:.3e}, tol {TOL:.0e})")
sys.exit(0 if worst < TOL else 1)

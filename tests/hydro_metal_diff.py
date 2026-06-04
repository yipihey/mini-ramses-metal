#!/usr/bin/env python3
# CPU-vs-Metal parity diff for the Godunov hydro step, through the RamsesNG C-API.
# Loads libramses1d_metal.dylib (which contains BOTH the CPU Fortran routines and
# the Metal GPU routines), inits one state from a uniform periodic 1D namelist,
# then on the SAME mesh compares:
#   CPU:   set_unew -> godunov_fine -> set_uold   (host umuscl)
#   Metal: ramses_metal_godunov_fine              (GPU hydro.metal port)
# reading m%uold (field=0) both times and diffing per cell.  A faithful fp32 port
# matches the fp64 CPU to the discretization/round-off floor.
import ctypes as C, os, sys, math

LIB = sys.argv[1] if len(sys.argv) > 1 else "bin/libramses1d_metal.dylib"
NML = sys.argv[2] if len(sys.argv) > 2 else "namelist/advect1d_uniform.nml"
WANT_L = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 0 = auto (first populated)
NDIM = 1
TWOTONDIM = 1 << NDIM

lib = C.CDLL(os.path.abspath(LIB))
lib.ramses_init.restype = C.c_int
lib.ramses_init.argtypes = [C.c_char_p, C.c_int]
lib.ramses_nvar.restype = C.c_int
lib.ramses_get_hydro.restype = C.c_int
lib.ramses_get_hydro.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                 C.POINTER(C.c_int), C.POINTER(C.c_double)]
lib.ramses_set_hydro.restype = C.c_int
lib.ramses_set_hydro.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                 C.POINTER(C.c_int), C.POINTER(C.c_double)]
for fn in ("ramses_set_unew", "ramses_godunov_fine", "ramses_set_uold",
           "ramses_metal_godunov_fine", "ramses_newdt_fine"):
    getattr(lib, fn).argtypes = [C.c_int, C.c_int]
lib.ramses_get_dt.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_double),
                              C.POINTER(C.c_double), C.POINTER(C.c_double)]

h = lib.ramses_init(NML.encode(), -1)
if h <= 0:
    print("FAIL: ramses_init returned", h); sys.exit(2)
nvar = lib.ramses_nvar()

NMAX = 20000
ckey = (C.c_int * (NDIM * NMAX))()
val  = (C.c_double * (TWOTONDIM * NMAX))()

def get(field, ivar, L):
    n = lib.ramses_get_hydro(h, field, ivar, L, NMAX, ckey, val)
    return n, [val[i] for i in range(TWOTONDIM * n)], [ckey[i] for i in range(NDIM * n)]

def set_uold(ivar, L, ck, data):
    n = len(data) // TWOTONDIM
    cb = (C.c_int * len(ck))(*ck)
    vb = (C.c_double * len(data))(*data)
    return lib.ramses_set_hydro(h, 0, ivar, L, n, cb, vb)

# report all populated levels, then pick the target (WANT_L, or first populated)
pop = [(lv, lib.ramses_get_hydro(h, 0, 1, lv, NMAX, ckey, val)) for lv in range(1, 21)]
pop = [(lv, n) for lv, n in pop if n > 0]
print("populated levels:", ", ".join(f"L{lv}={n}" for lv, n in pop))
L = WANT_L if WANT_L else (pop[0][0] if pop else -1)
if L < 0 or all(lv != L for lv, _ in pop):
    print(f"FAIL: target level {L} not populated"); sys.exit(2)
noct, _, _ = get(0, 1, L)
# compute a real timestep so the Godunov update is non-trivial (both paths read g%dtnew)
lib.ramses_newdt_fine(h, L)
dtn, dto, ax = C.c_double(), C.c_double(), C.c_double()
lib.ramses_get_dt(h, L, dtn, dto, ax)
print(f"level {L}: {noct} octs ({noct*TWOTONDIM} cells), nvar={nvar}, dt={dtn.value:.6e}")
if not (dtn.value > 0):
    print("FAIL: dt is not positive -> the step would be a no-op"); sys.exit(2)

# snapshot uold0 (all vars) + ckey for reset
uold0, cks = {}, None
for iv in range(1, 6):
    n, d, ck = get(0, iv, L); uold0[iv] = d; cks = ck

# --- CPU path: set_unew -> godunov_fine -> set_uold ---
lib.ramses_set_unew(h, L); lib.ramses_godunov_fine(h, L); lib.ramses_set_uold(h, L)
cpu = {iv: get(0, iv, L)[1] for iv in range(1, 6)}

# reset uold to the snapshot
for iv in range(1, 6):
    set_uold(iv, L, cks, uold0[iv])

# --- Metal path: ramses_metal_godunov_fine ---
lib.ramses_metal_godunov_fine(h, L)
gpu = {iv: get(0, iv, L)[1] for iv in range(1, 6)}

names = {1: "rho", 2: "rho*u", 3: "rho*v", 4: "rho*w", 5: "E"}
worst = 0.0
print(f"{'var':8} {'max|abs|':>12} {'max rel':>12} {'L2 rel':>12}")
for iv in range(1, 6):
    a, b = cpu[iv], gpu[iv]
    num = sum((x - y) ** 2 for x, y in zip(a, b))
    den = sum(x * x for x in a) or 1.0
    l2 = math.sqrt(num / den)
    mx_abs = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    mx_rel = max((abs(x - y) / max(1e-30, abs(x)) for x, y in zip(a, b)), default=0.0)
    worst = max(worst, l2)
    print(f"{names[iv]:8} {mx_abs:12.3e} {mx_rel:12.3e} {l2:12.3e}")

TOL = 2e-4   # fp32 vs fp64 Godunov floor over one step
print("PASS" if worst < TOL else "FAIL", f"(worst L2 rel = {worst:.3e}, tol {TOL:.0e})")
sys.exit(0 if worst < TOL else 1)

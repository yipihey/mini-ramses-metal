#!/usr/bin/env python3
# CPU-vs-Metal coarse-fine REFLUX parity.  On a 2-level mesh, the CPU
# godunov_fine(fine) adds a flux correction onto the COARSE level's unew.  We diff
# that coarse unew:
#   CPU:   set_unew(coarse) + set_unew(fine) + godunov_fine(fine) -> unew(coarse)
#   Metal: ramses_metal_godunov_reflux(fine)  (unew=uold both levels, fine Godunov
#          scatters the reflux, finalize -> unew(coarse))
# read unew (field=1) at the coarse level by ckey.  The reflux only touches the
# refined coarse cells (those with fine children); elsewhere unew==uold (diff 0).
import ctypes as C, os, sys, math

LIB = sys.argv[1] if len(sys.argv) > 1 else "bin/libramses1d_metal.dylib"
NML = sys.argv[2] if len(sys.argv) > 2 else "namelist/advect1d_2lvl.nml"
FINE = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NDIM = 1; TWOTONDIM = 1 << NDIM
COARSE = FINE - 1

lib = C.CDLL(os.path.abspath(LIB))
lib.ramses_init.restype = C.c_int
lib.ramses_init.argtypes = [C.c_char_p, C.c_int]
lib.ramses_get_hydro.restype = C.c_int
lib.ramses_get_hydro.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                 C.POINTER(C.c_int), C.POINTER(C.c_double)]
for fn in ("ramses_set_unew", "ramses_godunov_fine", "ramses_newdt_fine",
           "ramses_metal_godunov_reflux"):
    getattr(lib, fn).argtypes = [C.c_int, C.c_int]
lib.ramses_get_dt.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_double),
                              C.POINTER(C.c_double), C.POINTER(C.c_double)]

h = lib.ramses_init(NML.encode(), -1)
if h <= 0: print("FAIL init", h); sys.exit(2)
NMAX = 20000
ckey = (C.c_int * (NDIM * NMAX))(); val = (C.c_double * (TWOTONDIM * NMAX))()
def get(field, ivar, L):
    n = lib.ramses_get_hydro(h, field, ivar, L, NMAX, ckey, val)
    return n, [val[i] for i in range(TWOTONDIM * n)]

ncoarse, _ = get(0, 1, COARSE); nfine, _ = get(0, 1, FINE)
print(f"coarse L{COARSE}={ncoarse} octs, fine L{FINE}={nfine} octs")
if ncoarse == 0 or nfine == 0: print("FAIL: need both levels populated"); sys.exit(2)

for lv in range(1, FINE + 1):   # set dt at every level (subcycle hierarchy)
    lib.ramses_newdt_fine(h, lv)
dtn = C.c_double(); dto = C.c_double(); ax = C.c_double()
lib.ramses_get_dt(h, FINE, dtn, dto, ax)
print(f"dt(L{FINE}) = {dtn.value:.6e}")
uold_f0 = {iv: get(0, iv, FINE)[1] for iv in range(1, 6)}
# CPU: set_unew both, godunov(fine), read coarse unew(field=1)
lib.ramses_set_unew(h, COARSE); lib.ramses_set_unew(h, FINE); lib.ramses_godunov_fine(h, FINE)
cpu = {iv: get(1, iv, COARSE)[1] for iv in range(1, 6)}
unew_f = {iv: get(1, iv, FINE)[1] for iv in range(1, 6)}
chg_f = max(max((abs(a-b) for a,b in zip(unew_f[iv], uold_f0[iv])), default=0.0) for iv in range(1,6))
print(f"fine-level unew change (godunov ran?): max|unew-uold| = {chg_f:.3e}")
# Metal reflux path -> coarse unew
lib.ramses_metal_godunov_reflux(h, FINE)
gpu = {iv: get(1, iv, COARSE)[1] for iv in range(1, 6)}

# also report how much the reflux actually changed vs uold (to confirm it's non-trivial)
uold0 = {iv: get(0, iv, COARSE)[1] for iv in range(1, 6)}
names = {1:"rho",2:"rho*u",3:"rho*v",4:"rho*w",5:"E"}
worst = 0.0; refluxmag = 0.0
print(f"{'var':8} {'max|d|':>12} {'L2 rel':>12} {'|reflux|':>12}")
for iv in range(1, 6):
    a, b, u = cpu[iv], gpu[iv], uold0[iv]
    num = sum((x-y)**2 for x,y in zip(a,b)); den = sum(x*x for x in a) or 1.0
    l2 = math.sqrt(num/den); mx = max((abs(x-y) for x,y in zip(a,b)), default=0.0)
    rfx = max((abs(x-z) for x,z in zip(a,u)), default=0.0)   # CPU reflux size vs uold
    worst = max(worst, l2); refluxmag = max(refluxmag, rfx)
    print(f"{names[iv]:8} {mx:12.3e} {l2:12.3e} {rfx:12.3e}")

if refluxmag < 1e-12:
    print("INCONCLUSIVE: CPU reflux ~0 (no coarse-fine flux this step / dt=0)"); sys.exit(2)
TOL = 2e-4   # fp32 vs fp64 floor for the coarse-fine reflux correction
if worst < TOL:
    print(f"PASS (worst L2 rel = {worst:.3e}, reflux mag {refluxmag:.2e}, tol {TOL:.0e})"); sys.exit(0)
print(f"FAIL (worst L2 rel = {worst:.3e}, reflux mag {refluxmag:.2e}, tol {TOL:.0e})")
print("NOTE: needs RAMSES_METAL_CACHE=1 so the fine octs' coarse neighbours are")
print("materialized as cache octs (nbor>ngridmax) and filled by interpol_hydro --")
print("else the Metal reflux scatter never fires and diff == the full reflux.")
sys.exit(1)

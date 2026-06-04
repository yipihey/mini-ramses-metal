#!/bin/zsh
# Build + run the full Metal-port unit-test suite (no Ramses runs).
# Each test: compile the needed .metal (-fno-fast-math, the right -DNDIM) into a
# metallib, build the self-contained .mm host harness (SAME -DNDIM for struct
# layout parity), run it, collect PASS/FAIL.  Usage: ./run_tests.sh
set -u
cd "$(dirname "$0")/.."            # gpu/metal
SDK=$(xcrun -sdk macosx --show-sdk-path 2>/dev/null)
TMP=/tmp/mtltests; mkdir -p "$TMP"
INC=(-I. -I..)
pass=0; fail=0; failed=()

buildlib() {  # buildlib <ndim> <out.metallib> <src.metal...>
  local ndim=$1 out=$2; shift 2
  local airs=()
  for s in "$@"; do
    local a="$TMP/$(basename ${s%.metal})_${ndim}.air"
    xcrun -sdk macosx metal -c "$s" -o "$a" -fno-fast-math -DNDIM=$ndim "${INC[@]}" 2>"$TMP/m.log" \
      || { echo "  [metal FAIL] $s"; cat "$TMP/m.log"; return 1; }
    airs+=("$a")
  done
  xcrun -sdk macosx metallib "${airs[@]}" -o "$out" 2>"$TMP/l.log" || { echo "  [metallib FAIL]"; cat "$TMP/l.log"; return 1; }
}

runtest() {   # runtest <name> <ndim> <libpath> <metal-src...>  (.mm = tests/<name>.mm)
  local name=$1 ndim=$2 lib=$3; shift 3
  printf "%-20s " "$name"
  buildlib "$ndim" "$lib" "$@" || { echo "BUILD-FAIL(metal)"; fail=$((fail+1)); failed+=($name); return; }
  local bin="$TMP/$name"
  clang++ -std=c++17 -ObjC++ -fobjc-arc -DNDIM=$ndim "tests/$name.mm" -I.. \
    -framework Metal -framework Foundation -o "$bin" 2>"$TMP/c.log" \
    || { echo "BUILD-FAIL(host)"; cat "$TMP/c.log"; fail=$((fail+1)); failed+=($name); return; }
  if "$bin" "$lib" >"$TMP/$name.out" 2>&1; then
    echo "PASS"; pass=$((pass+1))
  else
    echo "FAIL"; sed 's/^/    /' "$TMP/$name.out"; fail=$((fail+1)); failed+=($name)
  fi
}

echo "=== foundation / reduce ==="
runtest test_df64       1 "$TMP/test_df64.metallib"    tests/test_df64.metal
runtest test_reduce     1 "$TMP/test_reduce.metallib"  tests/test_reduce.metal
runtest test_resnorm    3 "$TMP/test_red.metallib"     reduce.metal

echo "=== mg solver (NDIM=1) ==="
runtest test_gauss      1 "$TMP/test_mg1d.metallib"    mg.metal
runtest test_residual   1 "$TMP/test_mg1d.metallib"    mg.metal
runtest test_gradient   1 "$TMP/test_mg1d.metallib"    mg.metal
runtest test_resetrhs   1 "$TMP/test_mg1d.metallib"    mg.metal
echo "=== mg solver (NDIM=3) ==="
runtest test_epot       3 "$TMP/test_mg.metallib"      mg.metal
runtest test_mask       3 "$TMP/test_mg.metallib"      mg.metal

echo "=== rho / part ==="
runtest test_newdt      1 "$TMP/test_part1d.metallib"  part.metal
runtest test_cicdeposit 1 "$TMP/test_rho1d.metallib"   rho.metal hash.metal
runtest test_gatherkick 1 "$TMP/test_pk1d.metallib"    part.metal hash.metal
runtest test_adjoint_boundary 1 "$TMP/test_adj1d.metallib" rho.metal part.metal hash.metal

echo "=== flag / refine ==="
runtest test_flag        1 "$TMP/test_flag1d.metallib"   flag.metal
runtest test_cachecompact 1 "$TMP/test_refine1d.metallib" refine.metal
runtest test_makecache   1 "$TMP/test_rc1d.metallib"     refine.metal hash.metal

echo "=== hydro (core: EOS / Riemann / slopes / state copy / integrator) ==="
runtest test_hydro       3 "$TMP/test_hydro.metallib"   hydro.metal tests/test_hydro.metal
runtest test_hydro       1 "$TMP/test_hydro1d.metallib" hydro.metal tests/test_hydro.metal

echo "=== integrated MG-solve (#31: per-leaf V-cycle vs convergence) ==="
# Builds the full NDIM=3 metallib + the bridge object, links the integrated
# V-cycle harness (gpu/metal_bridge_h6vtest.mm), and checks the residual drops
# many orders -> the per-leaf mtl_mg_* operators compose into a converging solve.
{
  GPU=..; LIB=$TMP/ramses_kernels.metallib    # runner cwd is gpu/metal; bridge lives in gpu/
  airs=(); for f in *.metal; do
    a="$TMP/$(basename ${f%.metal})_int3.air"
    xcrun -sdk macosx metal -c "$f" -o "$a" -fno-fast-math -DNDIM=3 "${INC[@]}" 2>/dev/null && airs+=("$a")
  done
  xcrun -sdk macosx metallib "${airs[@]}" -o "$LIB" 2>/dev/null
  clang++ -std=c++17 -ObjC++ -fobjc-arc -DNDIM=3 -I"$GPU" -c "$GPU/metal_bridge.mm" -o "$TMP/mb3.o" 2>/dev/null
  printf "%-20s " "test_mg_vcycle"
  if clang++ -std=c++17 -ObjC++ -fobjc-arc -DNDIM=3 -I"$GPU" "$GPU/metal_bridge_h6vtest.mm" "$TMP/mb3.o" \
        -framework Metal -framework Foundation -o "$TMP/test_vcycle" 2>"$TMP/vc.log"; then
    if "$TMP/test_vcycle" "$LIB" >"$TMP/vc.out" 2>&1; then echo "PASS"; pass=$((pass+1));
    else echo "FAIL"; sed 's/^/    /' "$TMP/vc.out"; fail=$((fail+1)); failed+=(test_mg_vcycle); fi
  else echo "BUILD-FAIL(host)"; sed 's/^/    /' "$TMP/vc.log" | head; fail=$((fail+1)); failed+=(test_mg_vcycle); fi
}

echo "=== cache-oct host orchestration (#30: mtl_make_cache on a coarse-fine patch) ==="
{
  GPU=..; LIB1=$TMP/test_bridge1d.metallib
  airs=(); for f in *.metal; do
    a="$TMP/$(basename ${f%.metal})_co1.air"
    xcrun -sdk macosx metal -c "$f" -o "$a" -fno-fast-math -DNDIM=1 "${INC[@]}" 2>/dev/null && airs+=("$a")
  done
  xcrun -sdk macosx metallib "${airs[@]}" -o "$LIB1" 2>/dev/null
  clang++ -std=c++17 -ObjC++ -fobjc-arc -DNDIM=1 -I"$GPU" -c "$GPU/metal_bridge.mm" -o "$TMP/mb1.o" 2>/dev/null
  printf "%-20s " "test_cache_orch"
  if clang++ -std=c++17 -ObjC++ -fobjc-arc -DNDIM=1 -I"$GPU" "$GPU/metal_bridge_cacheorch_test.mm" "$TMP/mb1.o" \
        -framework Metal -framework Foundation -o "$TMP/test_cacheorch" 2>"$TMP/co.log"; then
    if "$TMP/test_cacheorch" "$LIB1" >"$TMP/co.out" 2>&1; then echo "PASS"; pass=$((pass+1));
    else echo "FAIL"; sed 's/^/    /' "$TMP/co.out"; fail=$((fail+1)); failed+=(test_cache_orch); fi
  else echo "BUILD-FAIL(host)"; sed 's/^/    /' "$TMP/co.log" | head; fail=$((fail+1)); failed+=(test_cache_orch); fi
}

echo "================================================"
echo "TOTAL: $pass passed, $fail failed"
[ $fail -gt 0 ] && { echo "FAILED: ${failed[*]}"; exit 1; }
echo "ALL PASS"; exit 0

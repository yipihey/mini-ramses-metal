#!/usr/bin/env python3
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_acoustic as uniform

HERE = Path(__file__).resolve().parent
LEVELS = (6, 7)
TFINAL = 5.0


def add_api(lib):
    lib.ramses_get_field.restype = C.c_int
    lib.ramses_get_field.argtypes = [
        C.c_int, C.c_int, C.c_int, C.c_int,
        C.POINTER(C.c_int), C.POINTER(C.c_double),
    ]
    lib.ramses_flag_fine.argtypes = [C.c_int, C.c_int, C.c_int]
    lib.ramses_refine_fine.argtypes = [C.c_int, C.c_int]
    return lib


def level_oct_data(lib, handle, level, ivar):
    nmax = 1000
    keys = (C.c_int * nmax)()
    vals = (C.c_double * (2 * nmax))()
    noct = lib.ramses_get_hydro(handle, 0, ivar, level, nmax, keys, vals)
    return (
        np.ctypeslib.as_array(keys)[:noct].copy(),
        np.ctypeslib.as_array(vals)[:2*noct].copy(),
    )


def level_nref(lib, handle, level):
    nmax = 1000
    keys = (C.c_int * nmax)()
    vals = (C.c_double * (2 * nmax))()
    noct = lib.ramses_get_field(handle, 6, level, nmax, keys, vals)
    return (
        np.ctypeslib.as_array(keys)[:noct].copy(),
        np.ctypeslib.as_array(vals)[:2*noct].copy(),
    )


def coordinates(keys, level):
    n = 2**level
    x = np.empty(2 * len(keys))
    for j, key in enumerate(keys):
        for cell in range(2):
            x[2*j+cell] = (2*int(key) + cell + 0.5) / n
    return x


def set_level_state(lib, handle, level):
    keys, _ = level_oct_data(lib, handle, level, 1)
    x = coordinates(keys, level)
    state = uniform.initial_state(x)
    ckeys = np.ascontiguousarray(keys, dtype=np.int32)
    for ivar, values in enumerate(state, 1):
        cvals = np.ascontiguousarray(values, dtype=np.float64)
        for field in (0, 1):
            nset = lib.ramses_set_hydro(
                handle, field, ivar, level, len(keys),
                ckeys.ctypes.data_as(C.POINTER(C.c_int)),
                cvals.ctypes.data_as(C.POINTER(C.c_double)),
            )
            if nset != len(keys):
                raise RuntimeError(f"L{level} ivar={ivar} field={field}: {nset}/{len(keys)}")


def build_central_patch(lib, handle):
    keys, _ = level_oct_data(lib, handle, LEVELS[0], 1)
    x = coordinates(keys, LEVELS[0])
    rho = np.ones_like(x)
    inside = (x >= 0.3125) & (x < 0.6875)
    cell_index = np.floor(x * 2**LEVELS[0]).astype(int)
    rho[inside] = 1.0 + (cell_index[inside] & 1)
    velocity = np.ones_like(x)
    pressure = np.full_like(x, uniform.P0)
    energy = pressure/(uniform.GAMMA-1) + 0.5*rho*velocity**2
    state = (rho, rho*velocity, np.zeros_like(rho), np.zeros_like(rho), energy)
    ckeys = np.ascontiguousarray(keys, dtype=np.int32)
    for ivar, values in enumerate(state, 1):
        cvals = np.ascontiguousarray(values, dtype=np.float64)
        for field in (0, 1):
            nset = lib.ramses_set_hydro(
                handle, field, ivar, LEVELS[0], len(keys),
                ckeys.ctypes.data_as(C.POINTER(C.c_int)),
                cvals.ctypes.data_as(C.POINTER(C.c_double)),
            )
            if nset != len(keys):
                raise RuntimeError(f"patch seed ivar={ivar}: {nset}/{len(keys)}")
    lib.ramses_flag_fine(handle, LEVELS[0], 1)
    lib.ramses_refine_fine(handle, LEVELS[0])
    fine_keys, _ = level_oct_data(lib, handle, LEVELS[1], 1)
    if not len(fine_keys):
        raise RuntimeError("central patch construction produced no fine cells")


def composite(lib, handle):
    raster = np.full(2**LEVELS[-1], np.nan)
    fine_mask = np.zeros_like(raster, dtype=bool)
    bounds = []
    for level in LEVELS:
        keys, rho = level_oct_data(lib, handle, level, 1)
        nkeys, nref = level_nref(lib, handle, level)
        if not np.array_equal(keys, nkeys):
            raise RuntimeError(f"L{level} hydro/nref key mismatch")
        x = coordinates(keys, level)
        width = 2**(LEVELS[-1]-level)
        for xc, value, refined in zip(x, rho, nref):
            if refined > 0:
                continue
            start = int(round((xc - 0.5/2**level) * 2**LEVELS[-1]))
            raster[start:start+width] = value
            if level == LEVELS[-1]:
                fine_mask[start:start+width] = True
    if np.isnan(raster).any():
        raise RuntimeError("composite profile has uncovered cells")
    ids = np.flatnonzero(fine_mask)
    if ids.size:
        jumps = np.flatnonzero(np.diff(ids) > 1)
        groups = np.split(ids, jumps+1)
        bounds = [(g[0]/len(raster), (g[-1]+1)/len(raster)) for g in groups]
    x = (np.arange(len(raster)) + 0.5) / len(raster)
    return x, raster, bounds


def run_solver(lib, nml, coarse_dt, nsteps):
    handle = lib.ramses_init(str(nml).encode(), -1)
    if handle <= 0:
        raise RuntimeError(f"RAMSES initialization failed: {nml}")
    build_central_patch(lib, handle)
    for level in LEVELS:
        set_level_state(lib, handle, level)
    x, initial, bounds = composite(lib, handle)
    print(f"{nml.stem}: fine leaf patches {bounds}", flush=True)
    for step in range(1, nsteps+1):
        lib.ramses_set_dt(handle, LEVELS[0], coarse_dt, coarse_dt)
        lib.ramses_set_dt(handle, LEVELS[1], coarse_dt/2, coarse_dt/2)
        lib.ramses_amr_step(handle, LEVELS[0], step)
        if step % 250 == 0:
            print(f"{nml.stem}: {step}/{nsteps}", flush=True)
    xf, final, final_bounds = composite(lib, handle)
    if not np.allclose(x, xf) or bounds != final_bounds:
        raise RuntimeError("static AMR mesh changed")
    return x, initial, final, bounds


def plot(path, x, initial, usual, local, bounds):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    fig.patch.set_facecolor("#10151d")
    axes[0].plot(x, 1e3*(initial-1), "--", color="#aab4be", lw=1.4, label="analytic/composite at t=5")
    axes[0].plot(x, 1e3*(usual-1), color="#65a9ff", lw=1.7, label="usual MC + HLLC")
    axes[0].plot(x, 1e3*(local-1), color="#ffb45e", lw=1.7, label="Local PPM + two-shock")
    axes[0].set_ylabel(r"$10^3(\rho-1)$")
    axes[0].set_title("Acoustic wave crossing a fixed fine patch: 10 box crossings, matched t=5")
    axes[0].legend(frameon=False, ncol=3, labelcolor="#dce5ed")
    axes[1].plot(x, 1e3*(usual-initial), color="#65a9ff", lw=1.4, label="usual - analytic")
    axes[1].plot(x, 1e3*(local-initial), color="#ffb45e", lw=1.4, label="Local PPM - analytic")
    axes[1].axhline(0, color="#8995a2", lw=0.8)
    axes[1].set(xlabel="x", ylabel=r"$10^3\Delta\rho$")
    axes[1].legend(frameon=False, labelcolor="#dce5ed")
    for ax in axes:
        for left, right in bounds:
            ax.axvspan(left, right, color="#9b7cff", alpha=0.08)
            ax.axvline(left, color="#b69cff", alpha=0.45, lw=0.8)
            ax.axvline(right, color="#b69cff", alpha=0.45, lw=0.8)
        ax.set_facecolor("#10151d")
        ax.grid(color="#607080", alpha=0.17)
        ax.tick_params(colors="#c8d2dc")
        for spine in ax.spines.values():
            spine.set_color("#46515e")
        ax.xaxis.label.set_color("#dce5ed")
        ax.yaxis.label.set_color("#dce5ed")
        ax.title.set_color("#f4f7fa")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="#10151d")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--metallib", required=True)
    parser.add_argument("--output", default=str(HERE / "acoustic_amr_results.npz"))
    args = parser.parse_args()
    os.environ.update(
        RAMSES_GPU_HYDRO="1",
        RAMSES_METAL_CACHE="1",
        RAMSES_METALLIB=str(Path(args.metallib).resolve()),
    )
    max_speed = uniform.U0 + uniform.CS0 + uniform.CS0*uniform.AMP
    nsteps = math.ceil(TFINAL / (0.3 / 2**LEVELS[0] / max_speed))
    coarse_dt = TFINAL / nsteps
    lib = add_api(uniform.api(args.library))
    x, initial, usual, bounds = run_solver(
        lib, HERE/"acoustic_amr_usual.nml", coarse_dt, nsteps
    )
    x2, initial2, local, bounds2 = run_solver(
        lib, HERE/"acoustic_amr_localppm.nml", coarse_dt, nsteps
    )
    if not np.allclose(x, x2) or not np.allclose(initial, initial2) or bounds != bounds2:
        raise RuntimeError("solver initial meshes or profiles differ")
    result = {
        "configuration": {
            "base_level": LEVELS[0], "fine_level": LEVELS[1],
            "wavenumber": uniform.KW, "amplitude": uniform.AMP,
            "tfinal": TFINAL, "coarse_steps": nsteps,
            "coarse_dt": coarse_dt, "fine_dt": coarse_dt/2,
            "coarse_cfl": coarse_dt*max_speed*2**LEVELS[0],
            "fine_patch_bounds": bounds,
        },
        "usual": uniform.metrics(usual, initial),
        "localppm": uniform.metrics(local, initial),
    }
    output = Path(args.output)
    np.savez_compressed(output, x=x, initial=initial, usual=usual, localppm=local,
                        fine_patch_bounds=np.asarray(bounds), metadata=json.dumps(result))
    plot(output.with_suffix(".png"), x, initial, usual, local, bounds)
    print(json.dumps(result, indent=2))
    print(f"wrote {output} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()

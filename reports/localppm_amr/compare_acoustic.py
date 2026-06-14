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

HERE = Path(__file__).resolve().parent
GAMMA = 1.4
NX = 128
LEVEL = 7
KW = 4
AMP = 1.0e-3
RHO0 = 1.0
CS0 = 1.0
U0 = 1.0
P0 = CS0**2 * RHO0 / GAMMA
TFINAL = 5.0


def api(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.ramses_init.restype = C.c_int
    lib.ramses_init.argtypes = [C.c_char_p, C.c_int]
    lib.ramses_get_hydro.restype = C.c_int
    lib.ramses_get_hydro.argtypes = [
        C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
        C.POINTER(C.c_int), C.POINTER(C.c_double),
    ]
    lib.ramses_set_hydro.restype = C.c_int
    lib.ramses_set_hydro.argtypes = [
        C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
        C.POINTER(C.c_int), C.POINTER(C.c_double),
    ]
    lib.ramses_amr_step.argtypes = [C.c_int, C.c_int, C.c_int]
    lib.ramses_set_time_cap.argtypes = [C.c_int, C.c_int, C.c_double]
    lib.ramses_get_time.argtypes = [
        C.c_int,
        C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
        C.POINTER(C.c_int),
    ]
    return lib


def oct_data(lib, handle, ivar):
    nmax = 1000
    keys = (C.c_int * nmax)()
    vals = (C.c_double * (2 * nmax))()
    noct = lib.ramses_get_hydro(handle, 0, ivar, LEVEL, nmax, keys, vals)
    if noct <= 0:
        raise RuntimeError("no level-7 hydro cells")
    return (
        np.ctypeslib.as_array(keys)[:noct].copy(),
        np.ctypeslib.as_array(vals)[:2 * noct].copy(),
    )


def cell_coordinates(keys):
    x = np.empty(2 * len(keys))
    for j, key in enumerate(keys):
        for c in range(2):
            x[2*j+c] = (2 * int(key) + c + 0.5) / NX
    return x


def initial_state(x):
    wave = AMP * np.sin(2 * np.pi * KW * x)
    rho = RHO0 + wave
    vel = U0 + CS0 * wave / RHO0
    pressure = P0 + CS0**2 * wave
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * vel**2
    return rho, rho * vel, np.zeros_like(rho), np.zeros_like(rho), energy


def set_state(lib, handle, keys, state):
    ckeys = np.ascontiguousarray(keys, dtype=np.int32)
    for ivar, values in enumerate(state, 1):
        cvals = np.ascontiguousarray(values, dtype=np.float64)
        for field in (0, 1):
            nset = lib.ramses_set_hydro(
                handle, field, ivar, LEVEL, len(keys),
                ckeys.ctypes.data_as(C.POINTER(C.c_int)),
                cvals.ctypes.data_as(C.POINTER(C.c_double)),
            )
            if nset != len(keys):
                raise RuntimeError(f"set ivar={ivar} field={field}: {nset}/{len(keys)} octs")


def profile(lib, handle):
    keys, rho = oct_data(lib, handle, 1)
    x = cell_coordinates(keys)
    order = np.argsort(x)
    return x[order], rho[order]


def fmode(values, mode):
    phase = 2 * np.pi * mode * np.arange(len(values)) / len(values)
    return 2.0 * np.sum(values * np.exp(-1j * phase)) / len(values)


def metrics(final, initial):
    amp = np.ptp(final) / np.ptp(initial)
    c0, cf = fmode(initial, KW), fmode(final, KW)
    phase = np.angle(cf / c0) / (2 * np.pi)
    harm = math.sqrt(sum(abs(fmode(final, m))**2 for m in (2*KW, 3*KW, 4*KW))) / abs(cf)
    return {
        "amplitude_retention": float(amp),
        "phase_error_wavelengths": float(phase),
        "harmonic_distortion": float(harm),
        "l1_density_error": float(np.mean(np.abs(final - initial))),
        "mass_change": float(np.mean(final) - np.mean(initial)),
    }


def get_time(lib, handle):
    t, texp, aexp = C.c_double(), C.c_double(), C.c_double()
    nstep = C.c_int()
    lib.ramses_get_time(handle, C.byref(t), C.byref(texp), C.byref(aexp), C.byref(nstep))
    return t.value, nstep.value


def run_solver(lib, nml, target_time):
    handle = lib.ramses_init(str(nml).encode(), -1)
    if handle <= 0:
        raise RuntimeError(f"RAMSES initialization failed: {nml}")
    keys, _ = oct_data(lib, handle, 1)
    x = cell_coordinates(keys)
    state = initial_state(x)
    set_state(lib, handle, keys, state)
    xi, initial = profile(lib, handle)
    lib.ramses_set_time_cap(handle, 1, target_time)
    step = 0
    while True:
        t, nstep = get_time(lib, handle)
        if t >= target_time - 1.0e-12:
            break
        step += 1
        lib.ramses_amr_step(handle, LEVEL, step)
        if step % 500 == 0:
            print(f"{nml.stem}: step={step} t={get_time(lib, handle)[0]:.8g}", flush=True)
        if step > 200000:
            raise RuntimeError(f"{nml.stem}: exceeded step limit before t={target_time}")
    lib.ramses_set_time_cap(handle, 0, 0.0)
    xf, final = profile(lib, handle)
    if not np.allclose(xi, xf):
        raise RuntimeError("cell ordering changed on the static mesh")
    return xi, initial, final, get_time(lib, handle)[0], step


def make_plot(path, x, initial, usual, local, results):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    fig.patch.set_facecolor("#10151d")
    colors = {"usual": "#65a9ff", "local": "#ffb45e"}
    axes[0].plot(x, 1e3*(initial-1), "--", color="#aab4be", lw=1.5, label="analytic at t=5")
    axes[0].plot(x, 1e3*(usual-1), color=colors["usual"], lw=1.8, label="usual MC + HLLC")
    axes[0].plot(x, 1e3*(local-1), color=colors["local"], lw=1.8, label="Local PPM + two-shock")
    axes[0].set_ylabel(r"$10^3(\rho-1)$")
    axes[0].set_title("Translating acoustic eigenmode: 10 box crossings, matched t=5")
    axes[0].legend(frameon=False, ncol=3, labelcolor="#dce5ed")
    axes[1].plot(x, 1e3*(usual-initial), color=colors["usual"], lw=1.5, label="usual - analytic")
    axes[1].plot(x, 1e3*(local-initial), color=colors["local"], lw=1.5, label="Local PPM - analytic")
    axes[1].axhline(0, color="#8995a2", lw=0.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$10^3\Delta\rho$")
    axes[1].legend(frameon=False, labelcolor="#dce5ed")
    for ax in axes:
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
    parser.add_argument("--output", default=str(HERE / "acoustic_results.npz"))
    args = parser.parse_args()
    os.environ.update(
        RAMSES_GPU_HYDRO="1",
        RAMSES_METAL_CACHE="1",
        RAMSES_METALLIB=str(Path(args.metallib).resolve()),
    )
    lib = api(args.library)
    x, initial, usual, t_usual, steps_usual = run_solver(lib, HERE / "acoustic_usual.nml", TFINAL)
    x2, initial2, local, t_local, steps_local = run_solver(lib, HERE / "acoustic_localppm.nml", TFINAL)
    if not np.allclose(x, x2) or not np.allclose(initial, initial2):
        raise RuntimeError("solver initial conditions differ")
    results = {
        "configuration": {
            "nx": NX, "wavenumber": KW, "amplitude": AMP, "u0": U0,
            "cs0": CS0, "tfinal": TFINAL,
            "usual_steps": steps_usual, "localppm_steps": steps_local,
            "usual_time": t_usual, "localppm_time": t_local,
        },
        "usual": metrics(usual, initial),
        "localppm": metrics(local, initial),
    }
    output = Path(args.output)
    np.savez_compressed(output, x=x, initial=initial, usual=usual, localppm=local,
                        metadata=json.dumps(results))
    make_plot(output.with_suffix(".png"), x, initial, usual, local, results)
    print(json.dumps(results, indent=2))
    print(f"wrote {output} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()

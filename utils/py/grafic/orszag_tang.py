"""
Orszag-Tang vortex initial condition generator for mini-ramses (GRAFIC format).

Writes the unformatted Fortran "grafic" records the mini-ramses reader expects
(hydro/input_hydro_grafic.f90). The Orszag-Tang vortex (Orszag & Tang 1979;
Fromang, Teyssier & Hennebelle 2006) is a 2D MHD problem in the x-y plane that
develops into MHD turbulence. This script lays it out in a 1D-uniform (along z)
3D box, so the 2D problem validates a 3D (e.g. GPU) MHD build: the fields vary
only in x,y and are uniform along z (z-symmetry).

Standard state (code units, gamma = 5/3, periodic box of size 1):
    rho = 25/(36*pi)              (uniform)
    p   = 5/(12*pi)               (uniform)
    vx  = -v_amp * sin(2*pi*y)
    vy  =  v_amp * sin(2*pi*x)
    vz  =  0
    B0  = b_amp / sqrt(4*pi)
    Bx  = -B0 * sin(2*pi*y)
    By  =  B0 * sin(4*pi*x)
    Bz  =  0
B is the curl of A_z = B0*(cos(4*pi*x)/(4*pi) + cos(2*pi*y)/(2*pi)), so div B = 0.

Files written (primitive hydro + face-centred B; the reader builds the conserved
state and adds the magnetic energy itself):
    ic_d ic_u ic_v ic_w ic_p
    ic_bxleft ic_byleft ic_bzleft   (low  x/y/z faces -> bold 1,2,3)
    ic_bxright ic_byright ic_bzright (high x/y/z faces -> bold 4,5,6)

divB = 0 by construction: Bx depends only on y (uniform along its own normal x),
By depends only on x (uniform along its own normal y), and Bz = 0, so
bxleft==bxright, byleft==byright, bzleft==bzright in every cell.

Examples:
    # 2D, level 8 (256^2), default Orszag-Tang:
    python3 orszag_tang.py 8 --ndim 2
    # 3D box for the GPU MHD validation, level 6 (64^3):
    python3 orszag_tang.py 6 --ndim 3 --outdir ic_orszag_tang_6_3d

Namelist reminder (the IC carries no gamma / solver choice): set gamma=1.6666667,
mhd=.true., riemann='hlld', riemann2d='hlld', boxlen=<--size>, periodic
boundaries, and levelmin=levelmax for the level-min comparison.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import grafic


def write_array(filename: str, array: np.ndarray, box_size_cu: float) -> None:
    """Write a 3D numpy array (n1, n2, n3) to a GRAFIC file (float32, sliced)."""
    g = grafic.Grafic()
    g.set_data(np.ascontiguousarray(array, dtype=np.float32))
    g.make_header(box_size_cu)
    g.write_float(filename)


def generate_orszag_tang_fields(
    lvl: int,
    box_size_cu: float,
    *,
    ndim: int = 3,
    rho0: float = 25.0 / (36.0 * np.pi),
    p0: float = 5.0 / (12.0 * np.pi),
    v_amp: float = 1.0,
    b_amp: float = 1.0,
):
    """Build the Orszag-Tang primitive + face-B fields, shaped (n1, n2, n3).

    The profile varies only in x,y (the vortex plane) and is broadcast uniformly
    along z (z-symmetry). Returns
        (d, u, v, w, p, bxl, byl, bzl, bxr, byr, bzr),
    where the b*l / b*r arrays are the low/high face values along each component's
    own axis (bx on x-faces, by on y-faces, bz on z-faces).
    """
    n = 2 ** int(lvl)
    L = float(box_size_cu)
    if ndim == 3:
        n1, n2, n3 = n, n, n
    elif ndim == 2:
        n1, n2, n3 = n, n, 1
    else:
        raise ValueError("ndim must be 2 or 3 (the Orszag-Tang vortex is a 2D problem)")

    # Cell-centred coordinates as a fraction of the box (period-1 trig below).
    xc = (np.arange(n1, dtype=np.float64) + 0.5) / n1
    yc = (np.arange(n2, dtype=np.float64) + 0.5) / n2
    X, Y = np.meshgrid(xc, yc, indexing="ij")   # shape (n1, n2)

    two_pi = 2.0 * np.pi
    four_pi = 4.0 * np.pi
    B0 = float(b_amp) / np.sqrt(four_pi)

    # 2D (x,y) profiles; broadcast uniformly along z.
    u_2d = -float(v_amp) * np.sin(two_pi * Y)   # vx = -sin(2*pi*y)
    v_2d = float(v_amp) * np.sin(two_pi * X)    # vy = +sin(2*pi*x)
    bx_2d = -B0 * np.sin(two_pi * Y)            # Bx, uniform along x -> bxleft==bxright
    by_2d = B0 * np.sin(four_pi * X)            # By, uniform along y -> byleft==byright

    def bcast(profile_2d):
        out = np.empty((n1, n2, n3), dtype=np.float32)
        out[:, :, :] = profile_2d[:, :, None]
        return out

    def const(val):
        return np.full((n1, n2, n3), float(val), dtype=np.float32)

    d = const(rho0)
    p = const(p0)
    u = bcast(u_2d)
    v = bcast(v_2d)
    w = const(0.0)

    # Face-centred B. Each component is uniform along its own normal direction for
    # this problem, so the low and high face arrays are identical (-> divB = 0):
    #   Bx depends only on y -> uniform in x -> bxleft  == bxright  == Bx(y)
    #   By depends only on x -> uniform in y -> byleft  == byright  == By(x)
    #   Bz = 0               -> bzleft  == bzright  == 0
    bxl = bcast(bx_2d)
    bxr = bcast(bx_2d)
    byl = bcast(by_2d)
    byr = bcast(by_2d)
    bzl = const(0.0)
    bzr = const(0.0)

    return d, u, v, w, p, bxl, byl, bzl, bxr, byr, bzr


def main():
    parser = argparse.ArgumentParser(description="Generate Orszag-Tang vortex MHD ICs (GRAFIC format)")
    parser.add_argument("lvl", type=int, help="Refinement level (grid size is 2^lvl along each axis)")
    parser.add_argument("--size", type=float, default=1.0, help="Box size in code units (default: 1.0). Set boxlen to match.")
    parser.add_argument("--ndim", type=int, default=3, choices=[2, 3], help="Grid dimensionality: 2 (n,n,1) or 3 (n,n,n) (default: 3)")
    parser.add_argument("--rho0", type=float, default=25.0 / (36.0 * np.pi), help="Uniform density (default: 25/(36*pi))")
    parser.add_argument("--p0", type=float, default=5.0 / (12.0 * np.pi), help="Uniform pressure (default: 5/(12*pi))")
    parser.add_argument("--v_amp", type=float, default=1.0, help="Velocity amplitude (default: 1.0)")
    parser.add_argument("--b_amp", type=float, default=1.0, help="Magnetic field amplitude (default: 1.0; B0 = b_amp/sqrt(4*pi))")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for ICs. Default: ./ic_orszag_tang/ic_orszag_tang_<lvl>_<ndim>d",
    )

    args = parser.parse_args()

    lvl = int(args.lvl)
    size = float(args.size)

    d, u, v, w, p, bxl, byl, bzl, bxr, byr, bzr = generate_orszag_tang_fields(
        lvl,
        size,
        ndim=args.ndim,
        rho0=args.rho0,
        p0=args.p0,
        v_amp=args.v_amp,
        b_amp=args.b_amp,
    )

    # Determine output dir.
    tag = f"ic_orszag_tang_{lvl}_{args.ndim}d"
    outdir = Path(args.outdir) if args.outdir is not None else Path("ic_orszag_tang") / tag
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)

    # Primitive hydro fields (the reader forms the conserved state + total energy).
    write_array("ic_d", d, size)
    write_array("ic_u", u, size)
    write_array("ic_v", v, size)
    write_array("ic_w", w, size)
    write_array("ic_p", p, size)

    # Face-centred B (low faces -> bold 1,2,3 ; high faces -> bold 4,5,6).
    write_array("ic_bxleft", bxl, size)
    write_array("ic_byleft", byl, size)
    write_array("ic_bzleft", bzl, size)
    write_array("ic_bxright", bxr, size)
    write_array("ic_byright", byr, size)
    write_array("ic_bzright", bzr, size)

    print(f"Orszag-Tang ICs ({args.ndim}D, {2**lvl} cells along x,y) written to: {outdir}")
    print("Namelist: gamma=1.6666667, mhd=.true., riemann='hlld', riemann2d='hlld',")
    print(f"          boxlen={size}, periodic boundaries, levelmin=levelmax={lvl} for the level-min test.")


if __name__ == "__main__":
    main()

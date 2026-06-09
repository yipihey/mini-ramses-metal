"""
Brio & Wu MHD shock-tube initial condition generator for mini-ramses (GRAFIC format).

Writes the unformatted Fortran "grafic" records the mini-ramses reader expects
(hydro/input_hydro_grafic.f90). The classic Brio & Wu (1988, JCP 75, 400) test is
a 1D coplanar MHD Riemann problem; this script lays it out along x in a 1D, 2D, or
3D box (uniform in the transverse directions), so the same IC validates a 1D, 2D,
or NDIM=3 (e.g. GPU) MHD build.

Standard left/right states (code units; magnetic pressure = B^2/2, gamma = 2):
    left  (x < x0):  rho=1.0,   p=1.0,  vx=vy=vz=0,  Bx=0.75,  By= 1.0,  Bz=0
    right (x > x0):  rho=0.125,  p=0.1,  vx=vy=vz=0,  Bx=0.75,  By=-1.0,  Bz=0
Bx (normal to the interface) is uniform, By (transverse) flips sign across it.

Files written (primitive hydro + face-centred B; the reader builds the conserved
state and adds the magnetic energy itself):
    ic_d ic_u ic_v ic_w ic_p
    ic_bxleft ic_byleft ic_bzleft   (low  x/y/z faces -> bold 1,2,3)
    ic_bxright ic_byright ic_bzright (high x/y/z faces -> bold 4,5,6)

divB = 0 by construction: each field is uniform along its own normal direction, so
bxleft=bxright, byleft=byright, bzleft=bzright in every cell.

Examples:
    # 1D, level 8 (256 cells), default Brio-Wu:
    python3 brio_wu.py 8 --ndim 1
    # 3D box for the GPU MHD validation, level 6 (64^3):
    python3 brio_wu.py 6 --ndim 3 --outdir ic_brio_wu_6_3d

Namelist reminder (the IC carries no gamma / solver choice): set gamma=2.0,
mhd=.true., riemann='hlld' (or 'hll'), riemann2d='hllf', boxlen=<--size>,
periodic boundaries, and levelmin=levelmax for the level-min comparison.
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


def generate_brio_wu_fields(
    lvl: int,
    box_size_cu: float,
    *,
    ndim: int = 3,
    x0: float = 0.5,
    rho_l: float = 1.0,
    rho_r: float = 0.125,
    p_l: float = 1.0,
    p_r: float = 0.1,
    bx: float = 0.75,
    by_l: float = 1.0,
    by_r: float = -1.0,
    bz: float = 0.0,
):
    """Build the Brio-Wu primitive + face-B fields, shaped (n1, n2, n3).

    The profile varies only along x (axis 0) and is broadcast uniformly over the
    transverse axes. Returns
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
    elif ndim == 1:
        n1, n2, n3 = n, 1, 1
    else:
        raise ValueError("ndim must be 1, 2, or 3")

    # Cell-centred x coordinates (code units) and the left/right mask.
    xc = (np.arange(n1, dtype=np.float64) + 0.5) * (L / n1)
    left = xc < (float(x0) * L)   # shape (n1,)

    # 1D profiles along x.
    rho_1d = np.where(left, rho_l, rho_r)
    p_1d = np.where(left, p_l, p_r)
    by_1d = np.where(left, by_l, by_r)   # transverse B (uniform along y -> byleft==byright)

    def bcast(profile_1d):
        out = np.empty((n1, n2, n3), dtype=np.float32)
        out[:, :, :] = profile_1d[:, None, None]
        return out

    def const(val):
        return np.full((n1, n2, n3), float(val), dtype=np.float32)

    d = bcast(rho_1d)
    p = bcast(p_1d)
    u = const(0.0)
    v = const(0.0)
    w = const(0.0)

    # Face-centred B. Each component is uniform along its own normal direction for
    # this problem, so the low and high face arrays are identical (-> divB = 0):
    #   Bx (normal)     uniform in x  -> bxleft  == bxright  == bx
    #   By (transverse) uniform in y  -> byleft  == byright  == By(x)
    #   Bz              = 0           -> bzleft  == bzright  == bz
    bxl = const(bx)
    bxr = const(bx)
    byl = bcast(by_1d)
    byr = bcast(by_1d)
    bzl = const(bz)
    bzr = const(bz)

    return d, u, v, w, p, bxl, byl, bzl, bxr, byr, bzr


def main():
    parser = argparse.ArgumentParser(description="Generate Brio-Wu MHD shock-tube ICs (GRAFIC format)")
    parser.add_argument("lvl", type=int, help="Refinement level (grid size is 2^lvl along x)")
    parser.add_argument("--size", type=float, default=1.0, help="Box size in code units (default: 1.0). Set boxlen to match.")
    parser.add_argument("--ndim", type=int, default=3, choices=[1, 2, 3], help="Grid dimensionality: 1 (n,1,1), 2 (n,n,1), 3 (n,n,n) (default: 3)")
    parser.add_argument("--x0", type=float, default=0.5, help="Discontinuity position as a fraction of the box (default: 0.5)")
    parser.add_argument("--rho_l", type=float, default=1.0, help="Left density")
    parser.add_argument("--rho_r", type=float, default=0.125, help="Right density")
    parser.add_argument("--p_l", type=float, default=1.0, help="Left pressure")
    parser.add_argument("--p_r", type=float, default=0.1, help="Right pressure")
    parser.add_argument("--bx", type=float, default=0.75, help="Normal field Bx (uniform)")
    parser.add_argument("--by_l", type=float, default=1.0, help="Left transverse field By")
    parser.add_argument("--by_r", type=float, default=-1.0, help="Right transverse field By")
    parser.add_argument("--bz", type=float, default=0.0, help="Field Bz (uniform)")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for ICs. Default: ./ic_brio_wu/ic_brio_wu_<lvl>_<ndim>d",
    )

    args = parser.parse_args()

    lvl = int(args.lvl)
    size = float(args.size)

    d, u, v, w, p, bxl, byl, bzl, bxr, byr, bzr = generate_brio_wu_fields(
        lvl,
        size,
        ndim=args.ndim,
        x0=args.x0,
        rho_l=args.rho_l,
        rho_r=args.rho_r,
        p_l=args.p_l,
        p_r=args.p_r,
        bx=args.bx,
        by_l=args.by_l,
        by_r=args.by_r,
        bz=args.bz,
    )

    # Determine output dir.
    tag = f"ic_brio_wu_{lvl}_{args.ndim}d"
    outdir = Path(args.outdir) if args.outdir is not None else Path("ic_brio_wu") / tag
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

    print(f"Brio-Wu ICs ({args.ndim}D, {2**lvl} cells along x) written to: {outdir}")
    print("Namelist: gamma=2.0, mhd=.true., riemann='hlld' (or 'hll'), riemann2d='hllf',")
    print(f"          boxlen={size}, periodic boundaries, levelmin=levelmax={lvl} for the level-min test.")


if __name__ == "__main__":
    main()

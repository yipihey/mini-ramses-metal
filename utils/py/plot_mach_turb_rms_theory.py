#!/usr/bin/env python3
"""
Mach number vs turb_rms for mini-ramses MHD turbulence driving.

Primary scaling (empirical, Stellar L8 a200 beta=0.1 run):
  M ≈ C * sqrt(turb_rms),  C ≈ 1.658  (calibrated: turb_rms=200 → M≈23.45)

Reference OU linear model (turb/turb_commons.f90, turb/turb_hydro.f90):
  v_rms ~ a_rms * turb_T  =>  M = turb_rms * turb_T / c_s
  Not fitted to simulation; shown as dashed reference only.

Uniform ICs: rho=1, Bz=1, beta=0.1 => p=0.05, c_s=sqrt(gamma*p/rho).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Code constants (turb/turb_commons.f90, turb/turb_init.f90) ----------------
TURB_GS = 64
GAMMA = 1.4
TURB_T = 0.1
BETA = 0.1
RHO = 1.0
BZ = 1.0
COMP_FRAC = 0.5
FORCING = "parabolic"

# Empirical calibration: Stellar L8 a200, beta=0.1, turb_rms=200 → M≈23.45
L8_TURB_RMS = 200.0
L8_MACH = 23.45
EMP_C = L8_MACH / np.sqrt(L8_TURB_RMS)

HARNESS_TURB_RMS = 5.094074502271999
M10_TURB_RMS = (10.0 / EMP_C) ** 2


def proj_rms_norm(sol_frac: float) -> float:
    """turb_commons.f90 :: proj_rms_norm (NDIM=3 fit)."""
    return 0.797 * sol_frac**2 - 0.529 * sol_frac + 0.568


def parabolic_power(kx: int, ky: int, kz: int) -> float:
    """turb_commons.f90 :: calc_power_spectrum, case('parabolic')."""
    k_mag = float(np.sqrt(kx * kx + ky * ky + kz * kz))
    if 1.0 < k_mag < 3.0:
        return 1.0 - (k_mag - 2.0) ** 2
    return 0.0


def power_rms_norm() -> float:
    """Match turb_commons.f90 :: power_rms_norm for parabolic spectrum."""
    half = TURB_GS // 2
    power = np.zeros((TURB_GS, TURB_GS, TURB_GS), dtype=np.float64)
    for i in range(TURB_GS):
        kx = i if i <= half else i - TURB_GS
        for j in range(TURB_GS):
            ky = j if j <= half else j - TURB_GS
            for k in range(TURB_GS):
                kz = k if k <= half else k - TURB_GS
                power[i, j, k] = parabolic_power(kx, ky, kz)

    real_fields = []
    for _ in range(3):
        field = np.fft.ifftn(power).real / (TURB_GS**3)
        real_fields.append(field)
    stacked = np.stack(real_fields)
    return float(np.sqrt(np.mean(stacked**2)))


def thermodynamics(gamma: float, beta: float, rho: float, bz: float) -> dict[str, float]:
    """Uniform plasma with beta = 2*p/B^2 (RAMSES emag = B^2/2 in code units)."""
    pressure = 0.5 * beta * bz * bz
    cs = np.sqrt(gamma * pressure / rho)
    va = bz / np.sqrt(rho)
    return {
        "rho": rho,
        "p": pressure,
        "Bz": bz,
        "beta": beta,
        "c_s": cs,
        "v_A": va,
    }


def mach_empirical(turb_rms: np.ndarray, c: float = EMP_C) -> np.ndarray:
    """Empirical: M = C * sqrt(turb_rms)."""
    return c * np.sqrt(np.maximum(turb_rms, 0.0))


def mach_ou_linear(turb_rms: np.ndarray, turb_t: float, cs: float) -> np.ndarray:
    """Reference OU model: M = turb_rms * turb_T / c_s."""
    return turb_rms * turb_t / cs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("mach_vs_turb_rms_theory.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    thermo = thermodynamics(GAMMA, BETA, RHO, BZ)
    cs = thermo["c_s"]

    sol_frac = 1.0 - COMP_FRAC
    p_norm = power_rms_norm()
    ou_norm = np.sqrt(TURB_T / 2.0)
    turb_norm = 1.0 / (p_norm * proj_rms_norm(sol_frac) * ou_norm)

    turb_rms = np.linspace(0.0, 250.0, 500)
    mach_emp = mach_empirical(turb_rms)
    mach_ou = mach_ou_linear(turb_rms, TURB_T, cs)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    ax.plot(
        turb_rms,
        mach_emp,
        color="#1f77b4",
        lw=2.4,
        label=rf"empirical: $M = C\sqrt{{\mathrm{{turb\_rms}}}}$ ($C={EMP_C:.3f}$)",
    )
    ax.plot(
        turb_rms,
        mach_ou,
        color="#888888",
        lw=1.4,
        ls="--",
        alpha=0.85,
        label=r"reference OU: $M = \mathrm{turb\_rms}\,\mathrm{turb\_T}/c_s$",
    )

    for m_ref, color, ls in [(1.0, "#444444", "--"), (5.0, "#888888", ":"), (10.0, "#666666", "-.")]:
        ax.axhline(m_ref, color=color, ls=ls, lw=1.0, alpha=0.75)
        ax.text(252.0, m_ref, f" M={m_ref:g}", va="center", ha="left", fontsize=9, color=color)

    for tr, label, color in [
        (HARNESS_TURB_RMS, "harness default\n(5.09)", "#d62728"),
        (M10_TURB_RMS, "M≈10 namelist\n(36.37)", "#9467bd"),
        (L8_TURB_RMS, "L8 calib.\n(200)", "#2ca02c"),
    ]:
        m_pt = mach_empirical(np.array([tr]))[0]
        ax.axvline(tr, color=color, ls="--", lw=1.0, alpha=0.75)
        ax.plot(tr, m_pt, "o", color=color, ms=7, zorder=5)
        ax.annotate(
            f"{label}\nM={m_pt:.2f}",
            xy=(tr, m_pt),
            xytext=(8, 12 if tr < 100 else -18),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    ax.set_xlim(0.0, 250.0)
    ax.set_ylim(0.0, max(mach_emp[-1] * 1.05, 12.0))
    ax.set_xlabel(r"turb_rms  (RMS forcing acceleration, code units)")
    ax.set_ylabel(r"Mach number  $M = v_{\rm rms} / c_s$")
    ax.set_title(
        "Mach vs turb_rms — mini-ramses MHD driven turbulence\n"
        rf"$\gamma={GAMMA}$, $\beta={BETA}$, $\rho={RHO}$, $B_z={BZ}$, "
        rf"$turb_T={TURB_T}$, parabolic $k$-band, $f_{{\rm comp}}={COMP_FRAC}$"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    formula = (
        rf"empirical (L8 a200): $M = C\sqrt{{\mathrm{{turb\_rms}}}},\ "
        rf"C = {L8_MACH:.2g}/\sqrt{{{L8_TURB_RMS:.0f}}} = {EMP_C:.4f}$" + "\n"
        + rf"$M=10 \Rightarrow \mathrm{{turb\_rms}} = (10/C)^2 = {M10_TURB_RMS:.3f}$" + "\n"
        + r"reference OU: $M = \mathrm{turb\_rms}\,\mathrm{turb\_T}/c_s$, "
        + rf"$c_s={cs:.4f}$" + "\n"
        + rf"OU at turb_rms=200: $M={mach_ou_linear(np.array([L8_TURB_RMS]), TURB_T, cs)[0]:.2f}$ "
        + rf"(vs empirical {L8_MACH:.2g})"
    )
    ax.text(
        0.02,
        0.98,
        formula,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")
    print(f"C (empirical) = {EMP_C:.6f}")
    print(f"turb_rms for M=10 = {M10_TURB_RMS:.6f}")
    print(f"c_s = {cs:.6f}")
    print(f"turb_norm = {turb_norm:.6g}")
    print(f"M(harness turb_rms={HARNESS_TURB_RMS}) empirical = {mach_empirical(np.array([HARNESS_TURB_RMS]))[0]:.3f}")
    print(f"M(L8 turb_rms={L8_TURB_RMS}) empirical = {mach_empirical(np.array([L8_TURB_RMS]))[0]:.3f}")

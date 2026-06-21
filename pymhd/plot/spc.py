# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/plot/spc.py
-----------------

Plotting tools for turbulent energy spectra:
    - plotShell(): plot shell-integrated spectra.
    - plotAnisotropic(): plot anisotropic spectra along x, y, z.
    - plotAxisymmetric(): plot axisymmetric spectra along Bx/Bz.
    - plot(): plot spectra based on the type of the EnergySpectra object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

from matplotlib.axes import Axes

from ..spectra import EnergySpectra, Spectrum1D, get1D

# Font: Computer Modern
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True


def plotShell(spc: EnergySpectra, outdir: Path) -> None:
    """Plot a single shell-integrated spectra.

    Currently hard-coded to plot a k^{5/3}-compensated spectrum.

    Parameters
    ----------
    spc   : EnergySpectra, the energy spectra object.
    outdir: Path, the output directory.
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    slope = 5.0 / 3.0

    kin1D = get1D(spc.totEkin, mode="shell")
    k = kin1D.k
    ax.loglog(k, kin1D.Ek * k**slope, label="Kinetic", lw=2.0, color="k")

    if spc.totEmag is not None:
        mag1D = get1D(spc.totEmag, mode="shell")
        ax.loglog(k, mag1D.Ek * k**slope, label="Magnetic", lw=2.0, color="r")

    ax.set_xlabel(r"$k$", fontsize=14)
    ax.set_ylabel(r"$k^{5/3}E(k)$", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "shell.pdf", bbox_inches="tight")
    plt.close(fig)


def plotAnisotropic(spc: EnergySpectra, outdir: Path) -> None:
    """Plot anisotropic spectra along x, y, z.

    Currently hard-coded to plot a k^{3/2}-compensated spectrum for MRI-driven turbulence.

    Two rows:
        - first row : averaged spectra E(k_i) for i = x, y, z;
        - second row: shell-integrated spectra E_i(|k|) for i = x, y, z.
    """
    assert spc.totEmag is not None
    assert spc.xEmag is not None
    assert spc.yEmag is not None
    assert spc.zEmag is not None

    axes: tuple[Literal["x"], Literal["y"], Literal["z"]] = ("x", "y", "z")
    Ekins = {"x": spc.xEkin, "y": spc.yEkin, "z": spc.zEkin}
    Emags = {"x": spc.xEmag, "y": spc.yEmag, "z": spc.zEmag}

    fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharey="row")
    slope = 3 / 2

    # First row: E(k_i) for i = x, y, z.
    for i, axis in enumerate(axes):
        kin1D = get1D(spc.totEkin, mode="avg", axis=axis)
        k = kin1D.k
        axs[0, i].loglog(k, kin1D.Ek * k**slope, label=r"$E_{\mathrm{kin}}$", lw=2.5, color="k")

        mag1D = get1D(spc.totEmag, mode="avg", axis=axis)
        axs[0, i].loglog(k, mag1D.Ek * k**slope, label=r"$E_{\mathrm{mag}}$", lw=2.5, color="r")

        axs[0, i].set_xlabel(r"$k_{%s}$" % axis)
        axs[0, i].grid(True, which="both", ls="--", alpha=0.4)
        axs[0, i].tick_params(axis="both", direction="in", which="both", pad=6.5)

    axs[0, 0].set_ylabel(r"$k_i^{3/2} E(k_i)$")
    for i in range(3):
        axs[0, i].legend(loc="lower left", fontsize=14)

    # Second row: shell-integrated spectra E_i(|k|) for i = x, y, z.
    for i, axis in enumerate(axes):

        shellEkin = get1D(Ekins[axis], mode="shell")
        k, Ek = shellEkin.k, shellEkin.Ek
        axs[1, i].loglog(k, Ek * k**slope, label=r"$E_{\mathrm{kin}}$", lw=2.5, color="k")

        shellEmag = get1D(Emags[axis], mode="shell")
        k, Ek = shellEmag.k, shellEmag.Ek
        axs[1, i].loglog(k, Ek * k**slope, label=r"$E_{\mathrm{mag}}$", lw=2.5, color="r")

        axs[1, i].set_xlabel(r"$k$")
        axs[1, i].grid(True, which="both", ls="--", alpha=0.4)
        axs[1, i].tick_params(axis="both", direction="in", which="both", pad=6.5)
        axs[1, i].legend(loc="lower left", fontsize=14)

    axs[1, 0].set_ylabel(r"$k^{3/2}E_i(k)$")
    fig.tight_layout()
    fig.savefig(outdir / "anisotropic.pdf", bbox_inches="tight")
    plt.close(fig)


def plotAxisymmetric(spc: EnergySpectra, outdir: Path) -> None:
    r"""Plot axisymmetric spectra for Afvénic turbulence.

    The background field direction is derived from spc.type:
        - spc.type == 'Bx' -> axis='x';
        - spc.type == 'Bz' -> axis='z'.

    Two rows:
        - first row : k_\perp (left) and k_\parallel (right) spectra E(k_\perp)/E(k_\parallel);
        - second row: shell-integrated spectra E_perp(|k|) and E_para(|k|).
    """
    assert spc.totEmag is not None
    assert spc.xEmag is not None
    assert spc.yEmag is not None
    assert spc.zEmag is not None

    axismap: dict[Literal["Bx", "Bz"], Literal["x", "z"]] = {
        "Bx": "x",
        "Bz": "z",
    }
    if spc.type not in axismap:
        raise ValueError(f"plotAxisymmetric requires Bx or Bz, got {spc.type!r}")
    axis = axismap[spc.type]

    # Component energy sums for the second row.
    # perp: sum of the two directions perpendicular to the background field.
    # para: the background-field direction.
    if axis == "z":
        perpEkin = spc.xEkin + spc.yEkin
        paraEkin = spc.zEkin
        perpEmag = spc.xEmag + spc.yEmag
        paraEmag = spc.zEmag
    else:  # axis == "x"
        perpEkin = spc.yEkin + spc.zEkin
        paraEkin = spc.xEkin
        perpEmag = spc.yEmag + spc.zEmag
        paraEmag = spc.xEmag

    shellPerpEkin = get1D(perpEkin, mode="shell")
    shellParaEkin = get1D(paraEkin, mode="shell")
    shellPerpEmag = get1D(perpEmag, mode="shell")
    shellParaEmag = get1D(paraEmag, mode="shell")

    fig, axs = plt.subplots(2, 2, figsize=(12.0, 10.0))

    # ===== First row =====
    # anisotropic spectra of total kinetic and magnetic energy

    first: list[tuple[Literal["perp", "para"], float]] = [
        ("perp", 5.0 / 3.0),
        ("para", 2.0      )
    ]

    for ax, (mode, slope) in zip(axs[0], first):

        kin1D = get1D(spc.totEkin, mode=mode, axis=axis)
        mag1D = get1D(spc.totEmag, mode=mode, axis=axis)

        ax.loglog(kin1D.k, kin1D.Ek * kin1D.k**slope, label=r"$E_{\mathrm{kin}}$", lw=2.0, color="k")
        ax.loglog(mag1D.k, mag1D.Ek * mag1D.k**slope, label=r"$E_{\mathrm{mag}}$", lw=2.0, color="r")

        ax.grid(True, which="both", ls="--", alpha=0.4)

        ticksize = 12
        ax.tick_params(axis="both", which="major", direction="in", labelsize=ticksize)
        ax.tick_params(axis="both", which="minor", direction="in", labelsize=ticksize - 1)

    fontsize = 14
    axs[0, 0].set_xlabel(r"$k_{\perp}$"                         , fontsize=fontsize)
    axs[0, 0].set_ylabel(r"$k_{\perp}^{5/3} E(k_{\perp})$"      , fontsize=fontsize)
    axs[0, 1].set_xlabel(r"$k_{\parallel}$"                     , fontsize=fontsize)
    axs[0, 1].set_ylabel(r"$k_{\parallel}^{2} E(k_{\parallel})$", fontsize=fontsize)

    axs[0, 0].legend(loc="lower left", fontsize=12)
    axs[0, 1].legend(loc="lower left", fontsize=12)

    # ===== Second row =====
    # shell-integrated spectra of perp/para components

    ticksize = 12
    slope = 5.0 / 3.0
    second: list[tuple[Axes, Spectrum1D, Spectrum1D, str, str]] = [
        (axs[1, 0], shellPerpEkin, shellPerpEmag, r"$k$", r"$k^{5/3} E_{\perp}(k)$"),
        (axs[1, 1], shellParaEkin, shellParaEmag, r"$k$", r"$k^{5/3} E_{\parallel}(k)$"),
    ]
    for ax, kin, mag, xlabel, ylabel in second:

        ax.loglog(kin.k, kin.Ek * kin.k**slope, label=r"$E_{\mathrm{kin}}$", lw=2.0, color="k")
        ax.loglog(mag.k, mag.Ek * mag.k**slope, label=r"$E_{\mathrm{mag}}$", lw=2.0, color="r")

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.tick_params(axis="both", which="major", direction="in", labelsize=ticksize)
        ax.tick_params(axis="both", which="minor", direction="in", labelsize=ticksize - 1)
        ax.legend(loc="lower left", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)
    fig.savefig(outdir / "axisymmetric.pdf", bbox_inches="tight")
    plt.close(fig)


def plot(spc: EnergySpectra) -> None:
    """Plot 1D spectra based on the type of the EnergySpectra object.

    Parameters
    ----------
    spc : EnergySpectra, the energy spectra object.
    """
    outdir = Path("spectra")
    outdir.mkdir(parents=True, exist_ok=True)

    # plot shell-integrated spectra for all types
    plotShell(spc, outdir=outdir)

    # plot axisymmetric spectra for Afvénic turbulence
    if spc.type in ("Bx", "Bz"):
        plotAxisymmetric(spc, outdir=outdir)

    # plot anisotropic spectra for MRI-driven turbulence
    if spc.type == "MRI":
        plotAnisotropic(spc, outdir=outdir)

    print(f"Visualization of turbulent spectra completed! Results saved to ./{outdir}/")
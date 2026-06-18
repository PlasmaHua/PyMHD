# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/plot/nd.py
----------------

Plotting tools for numerical dissipation analysis in (M)HD turbulence, including:
    - 2D slices of numerical and physical dissipation terms
    - Numerical and physical dissipation spectra

For 2D slices, similar to pymhd/plot/slc.py, currently:
    - Shearing-box simulations: the box ratio is hard coded to be Lx : Ly : Lz = 2 : 4 : 1
    - Forced turbulence: the box ratio is hard coded to be Lx : Ly : Lz = 1 : 1 : 1
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine

from pathlib import Path
import pickle

from typing import Callable, Literal

from functools import partial

from KDEpy import FFTKDE

from scipy.ndimage import gaussian_filter1d as Gaussian

from ..turbulence import ScalarField, VectorField
from ..spectra import Spectrum, Spectrum1D, EnergySpectra, get1D
from ..numdiss import NumericalDissipation, DissipationSpectra

# Set flush=True to avoid buffer output
print = partial(print, flush=True)

# Font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif' ] = ['cmr10']

def float2LaTeX(value: float, ndigits: int = 2) -> str:
    r"""Format a float as LaTeX scientific notation (no outer $), e.g. 3.1\times 10^{-4}."""
    if not np.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    mantissa, exp = f"{value:.{ndigits}e}".split("e")
    exp = int(exp)
    return rf"{mantissa}\times 10^{{{exp}}}"

class Curve:
    """1D spectrum curve for dissipation spectra plots.

    Attributes
    ----------
    spc1d : Spectrum1D, the 1D spectrum object.
    color : str, the color of the curve.
    label : str, the label of the curve.
    peak  : float | None, the peak wavenumber of the curve, or None if not plotted.
    lw    : float, line width for plotDissipation (defaults to 2.0).
    dashed: bool, dashed linestyle for plotDissipation (default False).

    Methods
    -------
    getPeak         : peak wavenumber from smoothed |E(k)|.
    plotDissipation : Plot dissipation spectrum.
    plotEnergy      : Plot energy spectrum.
    """
    def __init__(
        self,
        spc1d : Spectrum1D,
        color : str,
        label : str,
        peak  : bool  = False,
        mask  : bool  = False,
        lw    : float = 2.0,
        dashed: bool  = False,
    ) -> None:
        self.spc1d  = spc1d
        self.color  = color
        self.label  = label
        self.lw     = lw
        self.dashed = dashed

        if peak:
            self.peak = Curve.getPeak(spc1d, mask=mask)
        else:
            self.peak = None

    @staticmethod
    def getPeak(spc1d: Spectrum1D, mask: bool = False) -> float | None:
        """Peak wavenumber of smoothed |E(k)|, or None if unavailable."""

        k, Ek = spc1d.k, np.abs(spc1d.Ek)
        sigma = 2.0  # kernel scale: radius 2*dk in units of dk-spaced samples
        smoothEk = Gaussian(Ek, sigma=sigma, mode="nearest")
        if mask:
            valid = (k > 5.0) & np.isfinite(Ek)
        else:
            valid = np.isfinite(Ek)
        if not np.any(valid):
            return None
        k, smoothEk = k[valid], smoothEk[valid]
        idx = int(np.argmax(smoothEk))
        kpeak = float(k[idx])
        kmin  = float(np.min(k))
        if np.isclose(kpeak, kmin, rtol=1e-12, atol=1e-12 * max(1.0, abs(kmin))):
            return None
        return kpeak

    def plotDissipation(self, ax: Axes) -> None:
        if self.peak is not None:
            ax.axvline(self.peak, color=self.color, ls="--", lw=1.8, zorder=1)
        k, Ek = self.spc1d.k, self.spc1d.Ek
        ls = "--" if self.dashed else "-"
        ax.semilogx(k, Ek, color=self.color, lw=self.lw, ls=ls, label=self.label, zorder=3)

    def plotEnergy(self, ax: Axes, slope: float = 5.0 / 3.0) -> None:
        k = self.spc1d.k
        y = self.spc1d.Ek * k**slope
        ax.loglog(k, y, color=self.color, ls="--", lw=1.8, alpha=0.8, label=self.label)


def plotSlices(nd: NumericalDissipation, fraction: float = 1.0) -> None:
    """Plot 2D slices of dissipation terms

    Outputs to nd.outputdir/slices/:
    - For type in ('SSD', 'Bx', 'Bz'):
        slice.phy.res.pdf, slice.phy.vis.pdf,
        slice.num.res.pdf, slice.num.vis.pdf
        all.slice.pdf, all.slice.pdf (for 'Bx' and 'Bz')
    - For type 'hydro':
        slice.phy.vis.pdf, slice.num.vis.pdf,
        slice.rate.phy.vis.pdf, slice.rate.num.vis.pdf
    - For type 'MRI':
        'term.{num,phy}.{res,vis}.{x,y,z}.pdf' (12 files)

    Parameters
    ----------
    nd       : NumericalDissipation object
    fraction : float in (0, 1], colormap parameter
    """
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    path = Path(nd.outputdir) / "slices"
    path.mkdir(parents=True, exist_ok=True)

    if nd.type == 'hydro':
        terms = [
            ('term.phy.vis', nd.phyVisTerm if nd.nu != 0 else nd.divStressT),
            ('term.num.vis', nd.numVisTerm),
            ('rate.phy.vis', nd.phyVisRate if nd.nu != 0 else nd.VdotStress), # ScalarField
            ('rate.num.vis', nd.numVisRate),                                  # ScalarField
        ]
    else:
        terms = [
            ('term.phy.res', nd.phyResTerm if nd.eta != 0 else nd.LaplacianB),
            ('term.phy.vis', nd.phyVisTerm if nd.nu  != 0 else nd.divStressT),
            ('term.num.res', nd.numResTerm),
            ('term.num.vis', nd.numVisTerm),
            ('rate.phy.res', nd.phyResRate if nd.eta != 0 else nd.BdotLaplaB), # ScalarField
            ('rate.num.res', nd.numResRate),                                   # ScalarField
            ('rate.phy.vis', nd.phyVisRate if nd.nu  != 0 else nd.VdotStress), # ScalarField
            ('rate.num.vis', nd.numVisRate),                                   # ScalarField
        ]

    Nx, Ny, Nz = nd.Nx, nd.Ny, nd.Nz
    Lx, Ly, Lz = nd.Lx, nd.Ly, nd.Lz

    cmap = plt.colormaps['RdBu']

    def getRange(data: np.ndarray, frac: float | None = None) -> tuple[float, float]:

        f = fraction if frac is None else frac
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return -1.0, 1.0
        r = float(np.percentile(np.abs(valid), f * 100))
        return -r, r

    for name, term in terms:
        if term is None:
            continue

        if isinstance(term, VectorField):
            slices = [
                term.x[Nx // 2, :, :],
                term.y[:, Ny // 2, :],
                term.z[:, :, Nz // 2],
            ]
        elif isinstance(term, ScalarField):
            slices = [
                term.data[Nx // 2, :, :],
                term.data[:, Ny // 2, :],
                term.data[:, :, Nz // 2],
            ]
        else:
            continue

        if nd.type == 'MRI':
            if not isinstance(term, VectorField) or not name.startswith('term.'):
                continue

            nu  = r'\nu '  if nd.nu  != 0 else ''
            eta = r'\eta ' if nd.eta != 0 else ''

            linewidth = 1.5
            pad = 7

            for comp, vardata in (
                ('x', term.x),
                ('y', term.y),
                ('z', term.z),
            ):
                merged = np.concatenate(
                    [
                        vardata[:, :, Nz // 2].ravel(),
                        vardata[:, Ny // 2, :].ravel(),
                        vardata[Nx // 2, :, :].ravel(),
                    ]
                )
                vmin, vmax = getRange(merged)

                fig = plt.figure(figsize=(14, 8), constrained_layout=False)
                gs = plt.GridSpec(
                    2, 3,
                    figure=fig,
                    width_ratios=[4, 1, 0.2],
                    height_ratios=[2, 1],
                    left=0.1,
                    right=0.9,
                    top=0.9,
                    bottom=0.1,
                    wspace=0.08,
                    hspace=0.08,
                )

                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, 0])
                cax = fig.add_subplot(gs[:, 2])

                im1 = ax1.imshow(
                    vardata[:, :, Nz // 2],
                    origin='upper',
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Ly / 2, Ly / 2, Lx / 2, -Lx / 2),
                )
                im2 = ax2.imshow(
                    vardata[:, Ny // 2, :],
                    origin='upper',
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Lz / 2, Lz / 2, Lx / 2, -Lx / 2),
                )
                im3 = ax3.imshow(
                    vardata[Nx // 2, :, :].T,
                    origin='lower',
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Ly / 2, Ly / 2, -Lz / 2, Lz / 2),
                )

                ax1.set_aspect('equal')
                ax2.set_aspect('equal')
                ax3.set_aspect('equal')

                ax1.xaxis.set_ticklabels([])

                ax2.yaxis.set_ticklabels([])
                ax2.set_ylabel('')

                ax1.tick_params(direction='in')
                ax2.tick_params(direction='in')
                ax3.tick_params(direction='in')

                ax1.set_ylabel(r'$x/H$', labelpad=0)
                ax2.set_xlabel(r'$z/H$', labelpad=10)
                ax3.set_xlabel(r'$y/H$', labelpad=10)
                ax3.set_ylabel(r'$z/H$', labelpad=0)

                ax1.tick_params(width=linewidth, pad=pad)
                ax2.tick_params(width=linewidth, pad=pad)
                ax3.tick_params(width=linewidth, pad=pad)

                for ax in (ax1, ax2, ax3):
                    for spine in ax.spines.values():
                        spine.set_linewidth(linewidth)

                cbar = fig.colorbar(im1, cax=cax, shrink=0.9)

                # Physical terms: same notation as vislabel / reslabel (Bx/Bz composite).
                # Numerical terms: D^{\mathrm{num}}_{\mathrm{vis/res}, comp}.
                if name == 'term.phy.res':
                    cbarlabel = r'$' + eta + r'\nabla^2 B_' + comp + r'$'
                elif name == 'term.num.res':
                    cbarlabel = (r'$D^{\mathrm{num}}_{\mathrm{res}, ' + comp + r'}$')
                elif name == 'term.phy.vis':
                    cbarlabel = r'$' + nu + r'(\nabla \cdot \mathbb{T})_' + comp + r'$'
                elif name == 'term.num.vis':
                    cbarlabel = r'$D^{\mathrm{num}}_{\mathrm{vis}, ' + comp + r'}$'
                else:
                    raise ValueError(f'Unexpected MRI term name: {name!r}')
                cbar.set_label(cbarlabel, labelpad=10)

                outline_spine = cbar.ax.spines.get('outline')
                if outline_spine is not None:
                    outline_spine.set_linewidth(linewidth)

                plt.savefig(path / f'{name}.{comp}.pdf', bbox_inches='tight')
                plt.close()

            continue

        elif nd.type in ('SSD', 'Bx', 'Bz', 'hydro'):
            merged = np.concatenate([arr.flatten() for arr in slices])
            vmin, vmax = getRange(merged)

            fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
            ax1, ax2, ax3 = axes

            im1 = ax1.imshow(
                slices[0].T, origin='lower',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=(-Ly / 2, Ly / 2, -Lz / 2, Lz / 2), aspect='auto'
            )
            im2 = ax2.imshow(
                slices[1].T, origin='lower',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=(-Lz / 2, Lz / 2, -Lx / 2, Lx / 2), aspect='auto'
            )
            im3 = ax3.imshow(
                slices[2].T, origin='lower',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2), aspect='auto'
            )

            ax1.set_xlabel(r'$y/H$')
            ax1.set_ylabel(r'$z/H$')
            ax2.set_xlabel(r'$z/H$')
            ax2.set_ylabel(r'$x/H$')
            ax3.set_xlabel(r'$x/H$')
            ax3.set_ylabel(r'$y/H$')

            for ax in [ax1, ax2, ax3]:
                ax.tick_params(direction='in', width=1.5, pad=7)
                ax.set_box_aspect(1)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            cbar_bottom = 1.035
            cbar_height = 0.06
            for ax, im in [(ax1, im1), (ax2, im2), (ax3, im3)]:
                cax = ax.inset_axes((0, cbar_bottom, 1, cbar_height))
                cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.tick_params(labelsize=12, pad=2)
                outline_spine = cbar.ax.spines.get("outline")
                if outline_spine is not None:
                    outline_spine.set_linewidth(1.5)

            filename = f'{name}.pdf'
            plt.savefig(path / filename, bbox_inches='tight')
            plt.close()

        else:
            raise ValueError(
                f"Unsupported turbulence {nd.type!r}; expected 'SSD', 'Bx', 'Bz', 'MRI', or 'hydro'."
            )

    # ===== Composite slice plots: all.vis.pdf and all.res.pdf =====
    # Each figure is a 2x3 grid. Rows are (plane, component) pairs.
    # all.vis.pdf columns: velocity, physical viscosity, numerical viscosity
    # all.res.pdf columns: magnetic field, physical resistivity, numerical resistivity
    # For type = 'Bz', row 1 = z-plane & z-component; row 2 = x-plane & x-component.
    # For type = 'Bx', row 1 = x-plane & x-component; row 2 = z-plane & z-component.
    if nd.type in ('Bx', 'Bz'):

        phyVisField = nd.phyVisTerm if nd.nu  != 0 else nd.divStressT
        phyResField = nd.phyResTerm if nd.eta != 0 else nd.LaplacianB
        assert phyVisField is not None and phyResField is not None
        assert nd.numVisTerm is not None and nd.numResTerm is not None
        assert nd.V is not None and nd.B is not None

        def getslice(arr: np.ndarray, plane: str) -> np.ndarray:
            if plane == 'x':
                return arr[Nx // 2, :, :]
            elif plane == 'z':
                return arr[:, :, Nz // 2]
            else:
                raise ValueError(f"Unsupported plane: {plane!r}; expected 'x' or 'z'.")

        def getextent(plane: str) -> tuple[float, float, float, float]:
            if plane == 'x':
                return (-Ly / 2, Ly / 2, -Lz / 2, Lz / 2)
            elif plane == 'z':
                return (-Lx / 2, Lx / 2, -Ly / 2, Ly / 2)
            else:
                raise ValueError(f"Unsupported plane: {plane!r}; expected 'x' or 'z'.")

        def getlabels(plane: str) -> tuple[str, str]:
            if plane == 'x':
                return r'$y$', r'$z$'
            elif plane == 'z':
                return r'$x$', r'$y$'
            else:
                raise ValueError(f"Unsupported plane: {plane!r}; expected 'x' or 'z'.")

        planes = {
            'Bz': [('z', 'z'), ('x', 'x')],
            'Bx': [('x', 'x'), ('z', 'z')],
        }[nd.type]

        nu  = r'\nu '  if nd.nu  != 0 else ''
        eta = r'\eta ' if nd.eta != 0 else ''

        def vislabel(kind: str, comp: str) -> str:
            return {
                'V'     : r'$u_' + comp + r'$',
                'phyVis': r'$' + nu + r'(\nabla \cdot \mathbb{T})_' + comp + r'$',
                'numVis': r'$D^{\mathrm{num}}_{\mathrm{vis}, ' + comp + r'}$',
            }[kind]

        def reslabel(kind: str, comp: str) -> str:
            return {
                'B'     : r'$B_' + comp + r'$',
                'phyRes': r'$' + eta + r'\nabla^2 B_' + comp + r'$',
                'numRes': r'$D^{\mathrm{num}}_{\mathrm{res}, ' + comp + r'}$',
            }[kind]

        visCols = [('V', nd.V), ('phyVis', phyVisField), ('numVis', nd.numVisTerm)]
        resCols = [('B', nd.B), ('phyRes', phyResField), ('numRes', nd.numResTerm)]

        def plot(
            filename: str,
            columns : list[tuple[str, VectorField]],
            labelFn : Callable[[str, str], str],
        ) -> None:
            nrows, ncols = 2, 3
            fig = plt.figure(figsize=(15.0, 11))
            gs  = fig.add_gridspec(
                nrows, ncols,
                wspace=0.24, hspace=0.04,
                left=0.0, right=1.0, top=0.94, bottom=0.06,
            )
            axes = np.empty((nrows, ncols), dtype=object)
            for r in range(nrows):
                for c in range(ncols):
                    axes[r, c] = fig.add_subplot(gs[r, c])

            for r, (plane, comp) in enumerate(planes):
                extent         = getextent(plane)
                xlabel, ylabel = getlabels(plane)
                for c, (kind, field) in enumerate(columns):
                    ax   = axes[r, c]
                    data = getslice(getattr(field, comp), plane)
                    vmin, vmax = getRange(data, frac=1.0 if c == 0 else None)

                    im = ax.imshow(
                        data.T, origin='lower',
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=extent, aspect='auto'
                    )

                    isLeft = (c == 0)
                    ax.set_xlabel(xlabel)
                    if isLeft:
                        ax.set_ylabel(ylabel, labelpad=0)
                    ax.tick_params(direction='in', width=1.5, pad=5, labelleft=isLeft, labelsize=14)
                    ax.set_box_aspect(1)
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)

                    ax.text(
                        0.06, 0.94, labelFn(kind, comp),
                        transform=ax.transAxes,
                        ha='left', va='top',
                        fontsize=18,
                        bbox=dict(
                            boxstyle='round,pad=0.5',
                            facecolor='white',
                            edgecolor='black',
                            linewidth=1.2,
                        ),
                        zorder=10,
                    )

                    cax = ax.inset_axes((1.04, 0.0, 0.06, 1.0))
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.yaxis.set_ticks_position('right')
                    cbar.ax.yaxis.set_label_position('right')
                    cbar.ax.tick_params(labelsize=14, pad=4)
                    outline_spine = cbar.ax.spines.get("outline")
                    if outline_spine is not None:
                        outline_spine.set_linewidth(1.5)

            plt.savefig(path / filename, bbox_inches='tight')
            plt.close()

        plot('all.vis.pdf', visCols, vislabel)
        plot('all.res.pdf', resCols, reslabel)


def plotShellSpectra(ds: DissipationSpectra, spc: EnergySpectra, outdir: Path) -> None:
    """Plot shell-integrated dissipation and energy spectra

    Parameters
    ----------
    ds    : DissipationSpectra
    spc   : EnergySpectra
    outdir: Path, the output directory.
    """
    totEkin = spc.totEkin
    totEmag = spc.totEmag

    Ekin1D = get1D(totEkin, mode="shell")
    Emag1D = get1D(totEmag, mode="shell") if totEmag is not None else None

    def plotCurves(
        filename: str,
        curves1 : list[Curve],
        curves2 : list[Curve],
        ylabel  : str,
    ) -> None:

        fig, ax1 = plt.subplots(figsize=(9.0, 6.0))
        ax2 = ax1.twinx()

        # Draw left-axis lines above right-axis lines
        ax2.set_zorder(0)
        ax1.set_zorder(1)
        ax1.patch.set_visible(False)

        for curve in curves1:
            curve.plotDissipation(ax1)

        for curve in curves2:
            curve.plotEnergy(ax2)

        # Keep low-k forcing from dominating y-max by using only k > 10 for the upper bound.
        all  = np.concatenate([curve.spc1d.Ek for curve in curves1])
        high = np.concatenate([curve.spc1d.Ek[curve.spc1d.k > 10.0] for curve in curves1])

        ymin = float(np.min(all))
        ymax = float(np.max(high))

        # Auto margin following the matplotlib default.
        ymargin = float(plt.rcParams.get("axes.ymargin", 0.05))
        pad = ymargin * (ymax - ymin)
        ax1.set_ylim(ymin - pad, ymax + pad)
        ax1.axhline(0.0, color="k", ls="-", lw=1.0, zorder=-10)

        labelsize = 16
        ticksize = 12

        ax1.set_xlabel(r"$k$", fontsize=labelsize)
        ax1.set_ylabel(ylabel, fontsize=labelsize)
        ax2.set_ylabel(r"$k^{5/3}E(k)$", fontsize=labelsize)

        ax1.tick_params(axis="both", direction="in", which="both", labelsize=ticksize, pad=5)
        ax2.tick_params(axis="y", direction="in", which="both", labelsize=ticksize, pad=5)

        ax1.grid(True, which="both", ls="--", alpha=0.3)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles, labels = handles1 + handles2, labels1 + labels2
        ax1.legend(handles, labels, loc="lower left", fontsize=14, framealpha=1.0)

        fig.tight_layout()
        fig.savefig(outdir / filename, bbox_inches="tight")
        plt.close(fig)

    # ===== shell.num.pdf =====
    # numerical viscous + numerical resistive dissipation spectra
    num1: list[Curve] = []
    spectrum1d = get1D(ds.eNumVis, mode="shell", negative=True)
    num1.append(
        Curve(spectrum1d, "k", r"$\varepsilon_{\mathrm{vis}}^{\mathrm{num}}$", peak=True, mask=True)
    )
    if spc.totEmag is not None:
        spectrum1d = get1D(ds.eNumRes, mode="shell", negative=True)
        num1.append(
            Curve(spectrum1d, "r", r"$\varepsilon_{\mathrm{res}}^{\mathrm{num}}$", peak=True, mask=True)
        )
    if num1:
        num2: list[Curve] = []
        curve = Curve(Ekin1D, "k", r"$E_{\mathrm{kin}}$")
        num2.append(curve)
        if Emag1D is not None:
            curve = Curve(Emag1D, "r", r"$E_{\mathrm{mag}}$")
            num2.append(curve)
        ylabel = r"$-\varepsilon_{\mathrm{diss}}^{\mathrm{num}}(k)$"
        plotCurves("shell.num.pdf", curves1 = num1, curves2 = num2, ylabel = ylabel)

    # ===== shell.all.pdf =====
    if Emag1D is not None and spc.totEmag is not None:

        # vis: numerical/physical/total viscous dissipation spectrum
        vis: list[Curve] = []

        # numerical viscous dissipation spectrum
        spectrum1d = get1D(ds.eNumVis, mode="shell", negative=True)
        label      = r"$\varepsilon_{\mathrm{vis}}^{\mathrm{num}}$"
        vis.append(Curve(spectrum1d, "b", label, peak=True, mask=True, lw=2.5, dashed=True))

        # physical viscous dissipation spectrum
        if ds.nu != 0.0:
            spectrum1d = get1D(ds.ePhyVis, mode="shell", negative=True)
            label      = r"$\varepsilon_{\mathrm{vis}}^{\mathrm{phy}}$"
            vis.append(Curve(spectrum1d, "r", label, peak=True, mask=True, lw=2.5, dashed=True))

        # total viscous dissipation spectrum
        spectrum1d = get1D(ds.eTotVis, mode="shell", negative=True)
        label      = r"$\varepsilon_{\mathrm{vis}}^{\mathrm{tot}}$"
        vis.append(Curve(spectrum1d, "k", label, peak=True, mask=True, lw=2.5, dashed=False))

        # kin: kinetic energy spectrum
        kin: list[Curve] = [Curve(Ekin1D, "k", r"$E_{\mathrm{kin}}$")]

        # res: numerical/physical/total resistive dissipation spectrum
        res: list[Curve] = []

        # numerical resistive dissipation spectrum
        spectrum1d = get1D(ds.eNumRes, mode="shell", negative=True)
        label      = r"$\varepsilon_{\mathrm{res}}^{\mathrm{num}}$"
        res.append(Curve(spectrum1d, "b", label, peak=True, mask=True, lw=2.5, dashed=True))

        # physical resistive dissipation spectrum
        if ds.eta != 0.0:
            spectrum1d = get1D(ds.ePhyRes, mode="shell", negative=True)
            label      = r"$\varepsilon_{\mathrm{res}}^{\mathrm{phy}}$"
            res.append(Curve(spectrum1d, "r", label, peak=True, mask=True, lw=2.5, dashed=True))

        # total resistive dissipation spectrum
        spectrum1d = get1D(ds.eTotRes, mode="shell", negative=True)
        label      = r"$\varepsilon_{\mathrm{res}}^{\mathrm{tot}}$"
        res.append(Curve(spectrum1d, "k", label, peak=True, mask=True, lw=2.5, dashed=False))

        # mag: magnetic energy spectrum
        mag: list[Curve] = [Curve(Emag1D, "k", r"$E_{\mathrm{mag}}$")]

        yleftlabel  = r"$-\varepsilon(k)$"
        yrightlabel = r"$k^{5/3}E(k)$"

        xlabelsize = 16
        ylabelsize = 14
        ticksize = 12
        pad = 5.5

        fig, (ax1vis, ax1res) = plt.subplots(1, 2, figsize=(12.0, 5.5), sharey=True)
        ax2vis = ax1vis.twinx()
        ax2res = ax1res.twinx()
        ax2res.sharey(ax2vis)

        for ax1, ax2 in ((ax1vis, ax2vis), (ax1res, ax2res)):
            ax2.set_zorder(0)
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)

        for curve in vis:
            curve.plotDissipation(ax1vis)
        for curve in kin:
            curve.plotEnergy(ax2vis)
        for curve in res:
            curve.plotDissipation(ax1res)
        for curve in mag:
            curve.plotEnergy(ax2res)

        totvis1d = get1D(ds.eTotVis, mode="shell", negative=True)
        totres1d = get1D(ds.eTotRes, mode="shell", negative=True)
        diss_Ek = np.concatenate([totvis1d.Ek, totres1d.Ek])
        diss_Ek_highk = np.concatenate(
            [
                totvis1d.Ek[totvis1d.k > 10.0],
                totres1d.Ek[totres1d.k > 10.0],
            ]
        )
        ymin = float(np.min(diss_Ek))
        ymax = float(np.max(diss_Ek_highk))
        ymargin = float(plt.rcParams.get("axes.ymargin", 0.05))
        ypad = ymargin * (ymax - ymin)
        ylow, yhigh = ymin - ypad, ymax + ypad
        ax1vis.set_ylim(ylow, yhigh)
        ax1vis.axhline(0.0, color="k", ls="-", lw=1.0, zorder=-10)
        ax1res.axhline(0.0, color="k", ls="-", lw=1.0, zorder=-10)

        ax2vis.relim()
        ax2res.relim()
        ax2vis.autoscale_view()

        ax1vis.set_xlabel(r"$k$", fontsize=xlabelsize)
        ax1res.set_xlabel(r"$k$", fontsize=xlabelsize)
        ax1vis.set_ylabel(yleftlabel, fontsize=ylabelsize)
        ax2res.set_ylabel(yrightlabel, fontsize=ylabelsize)

        ax1vis.tick_params(axis="both", direction="in", which="both", labelsize=ticksize, pad=pad)
        ax1res.tick_params(axis="x"   , direction="in", which="both", labelsize=ticksize, pad=pad)
        ax1res.tick_params(axis="y"   , direction="in", which="both", labelsize=ticksize, pad=pad, labelleft=False)
        ax2vis.tick_params(axis="y"   , direction="in", which="both", labelsize=ticksize, pad=pad, labelright=False)
        ax2res.tick_params(axis="y"   , direction="in", which="both", labelsize=ticksize, pad=pad)

        ax1vis.grid(True, which="both", ls="--", alpha=0.3)
        ax1res.grid(True, which="both", ls="--", alpha=0.3)

        handles1, labels1 = ax1vis.get_legend_handles_labels()
        handles2, labels2 = ax2vis.get_legend_handles_labels()
        frame = ax1vis.legend(
            handles1 + handles2, labels1 + labels2, loc="lower left", bbox_to_anchor=(0.007, 0.007), fontsize=14,
            framealpha=1.0, fancybox=True, facecolor="white", edgecolor="k",
        ).get_frame()
        frame.set_edgecolor("k")
        frame.set_linewidth(0.9)
        frame.set_boxstyle("round", pad=0.15, rounding_size=0.7)

        handles1, labels1 = ax1res.get_legend_handles_labels()
        handles2, labels2 = ax2res.get_legend_handles_labels()
        frame = ax1res.legend(
            handles1 + handles2, labels1 + labels2, loc="lower left", bbox_to_anchor=(0.007, 0.007), fontsize=14,
            framealpha=1.0, fancybox=True, facecolor="white", edgecolor="k",
        ).get_frame()
        frame.set_edgecolor("k")
        frame.set_linewidth(0.9)
        frame.set_boxstyle("round", pad=0.15, rounding_size=0.7)

        bbox = dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="1.0",
            linewidth=0.8,
            alpha=0.0,
        )

        nu  = float2LaTeX(ds.nu, ndigits=1)
        eta = float2LaTeX(ds.eta, ndigits=1)
        ax1vis.text(
            0.98, 0.97, rf"$\nu = {nu}$", transform=ax1vis.transAxes,
            ha="right", va="top", fontsize=14, bbox=bbox,
        )
        ax1res.text(
            0.98, 0.97, rf"$\eta = {eta}$", transform=ax1res.transAxes,
            ha="right", va="top", fontsize=14, bbox=bbox,
        )

        fig.tight_layout()
        fig.savefig(outdir / "shell.all.pdf", bbox_inches="tight")
        plt.close(fig)


def plotAxisymmetricSpectra(ds: DissipationSpectra, spc: EnergySpectra, outdir: Path) -> None:
    """Plot axisymmetric dissipation and energy spectra for Bx/Bz turbulence.

    Parameters
    ----------
    ds    : DissipationSpectra
    spc   : EnergySpectra
    outdir: Path, the output directory.
    """
    if spc.type == "Bx":
        axis = "x"
    elif spc.type == "Bz":
        axis = "z"
    else:
        raise ValueError(f"plotAxisymmetricSpectra requires type 'Bx' or 'Bz', got '{spc.type}'")

    if spc.totEmag is None:
        raise ValueError("Magnetic energy spectrum cache is required for axisymmetric spectrum plotting.")

    totEkin = ds.Ek.totEkin
    totEmag = ds.Ek.totEmag
    assert totEmag is not None
    perpEkin1D = get1D(totEkin, mode="perp", axis=axis)
    paraEkin1D = get1D(totEkin, mode="para", axis=axis)
    perpEmag1D = get1D(totEmag, mode="perp", axis=axis)
    paraEmag1D = get1D(totEmag, mode="para", axis=axis)

    def getDissipationCurves(
        phy  : Spectrum,
        num  : Spectrum,
        tot  : Spectrum,
        mode : Literal["perp", "para"],
        diss : Literal["vis", "res"],
    ) -> list[Curve]:

        # Order: numerical, physical, total (legend: num on top). Num = blue dashed, phy = red dashed.
        return [
            Curve(
                get1D(num, mode=mode, axis=axis, negative=True),
                "b", rf"$\varepsilon_{{\mathrm{{{diss}}}}}^{{\mathrm{{num}}}}$", lw=2.5, dashed=True,
            ),
            Curve(
                get1D(phy, mode=mode, axis=axis, negative=True),
                "r", rf"$\varepsilon_{{\mathrm{{{diss}}}}}^{{\mathrm{{phy}}}}$", lw=2.5, dashed=True,
            ),
            Curve(
                get1D(tot, mode=mode, axis=axis, negative=True),
                "k", rf"$\varepsilon_{{\mathrm{{{diss}}}}}^{{\mathrm{{tot}}}}$", lw=2.5, dashed=False,
            ),
        ]

    def getShellDissipationCurves(
        phy      : Spectrum,
        num      : Spectrum,
        tot      : Spectrum,
        diss     : Literal["vis", "res"],
        direction: Literal["perp", "para"],
    ) -> list[Curve]:

        directionLabel = r"\perp" if direction == "perp" else r"\parallel"
        curves: list[Curve] = []
        # Plot / legend order: num, phy, tot. Num = blue dashed, phy = red dashed.
        for spectrum, color, label, dashed, lw, use_peak in (
            (num, "b", "num", True , 2.5, True ),
            (phy, "r", "phy", True , 2.5, True ),
            (tot, "k", "tot", False, 2.5, False),
        ):
            spectrum1D = get1D(spectrum, mode="shell", negative=True)
            curveLabel = rf"$\mathscr{{D}}_{{\mathrm{{{diss}}},{directionLabel}}}^{{\mathrm{{{label}}}}}$"
            curves.append(
                Curve(
                    spectrum1D, color, curveLabel,
                    lw=lw, dashed=dashed, peak=use_peak, mask=use_peak,
                ),
            )
        return curves

    def plotCurves(
        filename: str,
        panels  : list[tuple[list[Curve], list[Curve], float, str, str, str]],
        xlim    : tuple[float, float] | None = None,
    ) -> None:

        fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.5), sharey="row")
        ax2s = [[axs[i, j].twinx() for j in range(2)] for i in range(2)]

        ax2s[0][1].sharey(ax2s[0][0])
        ax2s[1][1].sharey(ax2s[1][0])

        labelsize = 14
        ticksize  = 12
        pad       = 5.5

        for i, (curves1, curves2, slope, xlabel, ylabelLeft, ylabelRight) in enumerate(panels):
            row, col = divmod(i, 2)
            ax1 = axs[row, col]
            ax2 = ax2s[row][col]

            ax2.set_zorder(0)
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)

            for curve in curves1:
                curve.plotDissipation(ax1)
            for curve in curves2:
                curve.plotEnergy(ax2, slope=slope)

            ax1.axhline(0.0, color="k", ls="-", lw=1.0, zorder=-10)
            # ax1.grid(True, which="both", ls="--", alpha=0.3)

            # Omit bottom-row x labels on the first row when x matches the panel below (same column).
            samexlabel = panels[col][3] == panels[2 + col][3]
            showbottom = (row == 1) or not samexlabel
            if showbottom:
                ax1.set_xlabel(xlabel, fontsize=labelsize)

            if col == 0:
                ax1.set_ylabel(ylabelLeft, fontsize=labelsize)
            else:
                ax1.tick_params(axis="y", labelleft=False)
                ax2.set_ylabel(ylabelRight, fontsize=labelsize)

            ax1.tick_params(
                axis="both", direction="in", which="both",
                labelsize=ticksize, pad=pad, labelbottom=showbottom,
            )
            ax2.tick_params(axis="y"   , direction="in", which="both", labelsize=ticksize, pad=pad)

            if col == 0:
                ax2.tick_params(axis="y", labelright=False)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            frame = ax1.legend(
                handles1 + handles2, labels1 + labels2,
                loc="lower left", bbox_to_anchor=(0.007, 0.007), fontsize=12,
                framealpha=1.0, fancybox=True, facecolor="white", edgecolor="k",
            ).get_frame()
            frame.set_edgecolor("k")
            frame.set_linewidth(0.9)
            frame.set_boxstyle("round", pad=0.15, rounding_size=0.5)

        if xlim is not None:
            k0, k1 = xlim
            ym = float(plt.rcParams.get("axes.ymargin", 0.05))
            for row in (0, 1):
                parts: list[np.ndarray] = []
                for j in (0, 1):
                    for c in panels[2 * row + j][0]:
                        k, ek = c.spc1d.k, c.spc1d.Ek
                        m = (k >= k0) & (k <= k1) & np.isfinite(ek)
                        if np.any(m):
                            parts.append(ek[m])
                if parts:
                    y = np.concatenate(parts)
                    lo, hi = float(np.min(y)), float(np.max(y))
                    p = ym * ((hi - lo) if hi > lo else max(abs(hi), abs(lo), 1.0))
                    axs[row, 0].set_ylim(lo - p, hi + p)
            for ax1 in axs.flat:
                ax1.set_xlim(xlim)

        # Semi-transparent band for energy injection scales in [k_min, k_inj]. Draw on twin ax2 only
        # (lower axes zorder) so axvspan stays below energy curves and ax1 dissipation draws on top.
        injection_k_hi = 6.5
        span_kw = dict(facecolor="0.5", alpha=0.24, zorder=-15, linewidth=0)
        for row in range(2):
            for col in range(2):
                ax2 = ax2s[row][col]
                k_lo, k_hi_axis = ax2.get_xlim()
                k_inj_right = min(injection_k_hi, k_hi_axis)
                if k_inj_right > k_lo:
                    ax2.axvspan(k_lo, k_inj_right, **span_kw)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.07, wspace=0.06)
        fig.savefig(outdir / filename, bbox_inches="tight")
        plt.close(fig)

    # ===== axisymmetric.pdf =====
    # Dissipation spectra: \varepsilon_{vis,res}(k_{\perp,\parallel})
    # Energy spectra: E_{kin,mag}(k_{\perp,\parallel})
    perpVis: list[Curve] = getDissipationCurves(
        ds.ePhyVis, ds.eNumVis, ds.eTotVis, mode="perp", diss="vis",
    )
    perpRes: list[Curve] = getDissipationCurves(
        ds.ePhyRes, ds.eNumRes, ds.eTotRes, mode="perp", diss="res",
    )
    paraVis: list[Curve] = getDissipationCurves(
        ds.ePhyVis, ds.eNumVis, ds.eTotVis, mode="para", diss="vis",
    )
    paraRes: list[Curve] = getDissipationCurves(
        ds.ePhyRes, ds.eNumRes, ds.eTotRes, mode="para", diss="res",
    )

    perpKin: list[Curve] = [Curve(perpEkin1D, "0.5", r"$E_{\mathrm{kin}}$")]
    perpMag: list[Curve] = [Curve(perpEmag1D, "0.5", r"$E_{\mathrm{mag}}$")]
    paraKin: list[Curve] = [Curve(paraEkin1D, "0.5", r"$E_{\mathrm{kin}}$")]
    paraMag: list[Curve] = [Curve(paraEmag1D, "0.5", r"$E_{\mathrm{mag}}$")]

    panels = [
        # top-left panel: \varepsilon_{vis}(k_{\perp})
        (
            perpVis, perpKin, 5.0 / 3.0,
            r"$k_{\perp}$", r"$-\varepsilon(k_{\perp})$", r"$k_{\perp}^{5/3}E(k_{\perp})$",
        ),
        # top-right panel: \varepsilon_{res}(k_{\perp})
        (
            perpRes, perpMag, 5.0 / 3.0,
            r"$k_{\perp}$", r"$-\varepsilon(k_{\perp})$", r"$k_{\perp}^{5/3}E(k_{\perp})$",
        ),
        # bottom-left panel: \varepsilon_{vis}(k_{\parallel})
        (
            paraVis, paraKin, 2.0,
            r"$k_{\parallel}$", r"$-\varepsilon(k_{\parallel})$", r"$k_{\parallel}^{2}E(k_{\parallel})$",
        ),
        # bottom-right panel: \varepsilon_{res}(k_{\parallel})
        (
            paraRes, paraMag, 2.0,
            r"$k_{\parallel}$", r"$-\varepsilon(k_{\parallel})$", r"$k_{\parallel}^{2}E(k_{\parallel})$",
        ),
    ]
    plotCurves("axisymmetric.pdf", panels)

    # ===== components.pdf =====
    # Shell-integrated component-wise dissipation spectra: \mathscr{D}_{vis,res}(k)
    # Shell-integrated component-wise energy spectra: E_{kin,mag}(k)
    Ek = ds.Ek

    if axis == "z":
        perpEkin, paraEkin = Ek.xEkin + Ek.yEkin, Ek.zEkin
        if Ek.xEmag is None or Ek.yEmag is None or Ek.zEmag is None:
            raise ValueError("Per-component magnetic energy spectra are required for components.pdf.")
        perpEmag, paraEmag = Ek.xEmag + Ek.yEmag, Ek.zEmag

        numPerpVis, numParaVis = ds.xNumVis + ds.yNumVis, ds.zNumVis
        numPerpRes, numParaRes = ds.xNumRes + ds.yNumRes, ds.zNumRes
        phyPerpVis, phyParaVis = ds.xPhyVis + ds.yPhyVis, ds.zPhyVis
        phyPerpRes, phyParaRes = ds.xPhyRes + ds.yPhyRes, ds.zPhyRes
    else:
        perpEkin, paraEkin = Ek.yEkin + Ek.zEkin, Ek.xEkin
        if Ek.xEmag is None or Ek.yEmag is None or Ek.zEmag is None:
            raise ValueError("Per-component magnetic energy spectra are required for components.pdf.")
        perpEmag, paraEmag = Ek.yEmag + Ek.zEmag, Ek.xEmag

        numPerpVis, numParaVis = ds.yNumVis + ds.zNumVis, ds.xNumVis
        numPerpRes, numParaRes = ds.yNumRes + ds.zNumRes, ds.xNumRes
        phyPerpVis, phyParaVis = ds.yPhyVis + ds.zPhyVis, ds.xPhyVis
        phyPerpRes, phyParaRes = ds.yPhyRes + ds.zPhyRes, ds.xPhyRes

    totPerpVis = phyPerpVis + numPerpVis
    totParaVis = phyParaVis + numParaVis
    totPerpRes = phyPerpRes + numPerpRes
    totParaRes = phyParaRes + numParaRes

    perpVis = getShellDissipationCurves(phyPerpVis, numPerpVis, totPerpVis, diss="vis", direction="perp")
    perpRes = getShellDissipationCurves(phyPerpRes, numPerpRes, totPerpRes, diss="res", direction="perp")
    paraVis = getShellDissipationCurves(phyParaVis, numParaVis, totParaVis, diss="vis", direction="para")
    paraRes = getShellDissipationCurves(phyParaRes, numParaRes, totParaRes, diss="res", direction="para")

    perpKin = [Curve(get1D(perpEkin, mode="shell"), "0.5", r"$E_{\mathrm{kin},\perp}$")]
    perpMag = [Curve(get1D(perpEmag, mode="shell"), "0.5", r"$E_{\mathrm{mag},\perp}$")]
    paraKin = [Curve(get1D(paraEkin, mode="shell"), "0.5", r"$E_{\mathrm{kin},\parallel}$")]
    paraMag = [Curve(get1D(paraEmag, mode="shell"), "0.5", r"$E_{\mathrm{mag},\parallel}$")]

    panels = [
        (
            perpVis, perpKin, 5.0 / 3.0,
            r"$k$", r"$-\mathscr{D}_{\perp}(k)$", r"$k^{5/3}E_{\perp}(k)$",
        ),
        (
            perpRes, perpMag, 5.0 / 3.0,
            r"$k$", r"$-\mathscr{D}_{\perp}(k)$", r"$k^{5/3}E_{\perp}(k)$",
        ),
        (
            paraVis, paraKin, 5.0 / 3.0,
            r"$k$", r"$-\mathscr{D}_{\parallel}(k)$", r"$k^{5/3}E_{\parallel}(k)$",
        ),
        (
            paraRes, paraMag, 5.0 / 3.0,
            r"$k$", r"$-\mathscr{D}_{\parallel}(k)$", r"$k^{5/3}E_{\parallel}(k)$",
        ),
    ]
    kmax = float(np.max(perpKin[0].spc1d.k))
    plotCurves("components.pdf", panels, xlim=(2.0, kmax))


def plotAnisotropicSpectra(ds: DissipationSpectra, spc: EnergySpectra, outdir: Path) -> None:
    """Plot anisotropic dissipation and energy spectra for x/y/z components.

    Outputs components.pdf to nd.outputdir.

    Three panels (left to right: x, y, z component), shared y-axis.
    Left axis : physical, numerical, and total resistive dissipation spectra.
    Right axis: k^{3/2}-compensated component-wise kinetic and magnetic energy spectra.

    Parameters
    ----------
    ds    : DissipationSpectra, the precomputed dissipation spectra.
    spc   : EnergySpectra, the cached energy spectra.
    outdir: Path, the output directory.
    """
    Ek = ds.Ek
    if Ek.xEmag is None or Ek.yEmag is None or Ek.zEmag is None:
        raise ValueError("Per-component magnetic energy spectra are required for components.pdf.")

    directions = ("x", "y", "z")

    # Shell-integrated component-wise resistive dissipation spectra
    phyRes = {
        "x": get1D(ds.xPhyRes, mode="shell", negative=True),
        "y": get1D(ds.yPhyRes, mode="shell", negative=True),
        "z": get1D(ds.zPhyRes, mode="shell", negative=True),
    }
    numRes = {
        "x": get1D(ds.xNumRes, mode="shell", negative=True),
        "y": get1D(ds.yNumRes, mode="shell", negative=True),
        "z": get1D(ds.zNumRes, mode="shell", negative=True),
    }
    totRes = {
        "x": get1D(ds.xTotRes, mode="shell", negative=True),
        "y": get1D(ds.yTotRes, mode="shell", negative=True),
        "z": get1D(ds.zTotRes, mode="shell", negative=True),
    }

    # Shell-integrated component-wise energy spectra
    ekin = {
        "x": get1D(Ek.xEkin, mode="shell"),
        "y": get1D(Ek.yEkin, mode="shell"),
        "z": get1D(Ek.zEkin, mode="shell"),
    }
    emag = {
        "x": get1D(Ek.xEmag, mode="shell"),
        "y": get1D(Ek.yEmag, mode="shell"),
        "z": get1D(Ek.zEmag, mode="shell"),
    }

    # ===== components.pdf =====
    fig, ax1s = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    ax2s: list[Axes] = []
    for ax1 in ax1s:
        ax2 = ax1.twinx()
        ax2.set_zorder(0)
        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        ax2s.append(ax2)
    ax2s[1].sharey(ax2s[0])
    ax2s[2].sharey(ax2s[0])

    for i, direction in enumerate(directions):
        ax1 = ax1s[i]
        ax2 = ax2s[i]

        # Left axis: numerical / physical / total (legend: num on top). Num = blue dashed, phy = red dashed.
        ax1.semilogx(
            numRes[direction].k, numRes[direction].Ek,
            color="b", ls="--", lw=2.5,
            label=r"${\mathscr{D}}^{\mathrm{num}}_{\mathrm{res},%s}$" % direction,
        )
        ax1.semilogx(
            phyRes[direction].k, phyRes[direction].Ek,
            color="r", ls="--", lw=2.5,
            label=r"${\mathscr{D}}^{\mathrm{phy}}_{\mathrm{res},%s}$" % direction,
        )
        ax1.semilogx(
            totRes[direction].k, totRes[direction].Ek,
            color="k", ls="-", lw=2.5,
            label=r"${\mathscr{D}}^{\mathrm{tot}}_{\mathrm{res},%s}$" % direction,
        )

        # Right axis: component kinetic and magnetic energy spectra
        k = ekin[direction].k
        slope = 3.0 / 2.0
        ax2.loglog(
            k, ekin[direction].Ek * k**slope,
            color="k", ls="--", lw=1.5, alpha=0.8,
            label=r"$E_{\mathrm{kin},%s}$" % direction,
        )
        ax2.loglog(
            k, emag[direction].Ek * k**slope,
            color="r", ls="--", lw=1.5, alpha=0.8,
            label=r"$E_{\mathrm{mag},%s}$" % direction,
        )

        ks = (
            phyRes[direction].k,
            numRes[direction].k,
            totRes[direction].k,
            ekin[direction].k,
            emag[direction].k,
        )
        kmin = float(min(float(np.min(ka)) for ka in ks))
        kmax = float(max(float(np.max(ka)) for ka in ks))
        ax1.set_xlim(kmin, kmax)

        ax1.set_xlabel(r"$k$", fontsize=14)
        ax1.tick_params(axis="x", direction="in", which="both", labelsize=12, pad=5)
        ax1.tick_params(
            axis="y", direction="in", which="both", labelsize=12, pad=5,
            labelleft=(i == 0),
        )
        ax2.tick_params(
            axis="y", direction="in", which="both", labelsize=12, pad=5,
            labelright=(i == 2),
        )

        # ax1.grid(True, which="both", ls="--", alpha=0.3)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        frame = ax1.legend(
            h1 + h2, l1 + l2,
            loc="lower left", bbox_to_anchor=(0.007, 0.007), fontsize=12,
            framealpha=1.0, fancybox=True, facecolor="white", edgecolor="k",
        ).get_frame()
        frame.set_edgecolor("k")
        frame.set_linewidth(0.9)
        frame.set_boxstyle("round", pad=0.15, rounding_size=0.5)

    _, ytop = ax1s[0].get_ylim()
    ax1s[0].set_ylim(0.0, ytop)

    ax1s[0].set_ylabel(r"$-{\mathscr{D}}_{\mathrm{res}}(k)$", fontsize=16)
    ax2s[2].set_ylabel(r"$k^{3/2}E_i(k)$", fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.07)
    fig.savefig(outdir / "components.pdf", bbox_inches="tight")
    plt.close(fig)


def plotSpectra(nd: NumericalDissipation, spc: EnergySpectra) -> None:
    """Plot dissipation spectra.

    Extracts the energy spectra and computes the dissipation spectra only once,
    then shares them across all dissipation spectrum plotting functions.

    Parameters
    ----------
    nd : NumericalDissipation, the numerical dissipation object.
    spc: EnergySpectra, the cached energy spectra.
    """
    ds = DissipationSpectra(nd, spc)
    outdir = Path(nd.outputdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plotShellSpectra(ds, spc, outdir)

    if nd.type in ("Bx", "Bz"):
        plotAxisymmetricSpectra(ds, spc, outdir)

    if nd.type == "MRI":
        plotAnisotropicSpectra(ds, spc, outdir)


def plotHistogram(
    nd        : NumericalDissipation,
    xcoverage : float = 0.95,
    ycoverage : float = 0.99,
    resolution: int   = 400
) -> None:
    """Plot 2D histograms of dissipation terms (combined xyz only)

    Outputs to nd.outputdir/histograms/:
        hist.num.res.pdf, hist.num.vis.pdf

    Two rows:
        - first row  = 2D JPDF from KDE;
        - second row = conditional mean and 68% interval per x bin

    Parameters
    ----------
    nd        : NumericalDissipation object
    xcoverage : fraction of data enclosed by symmetric range for x data (lapl)
    ycoverage : fraction of data enclosed by symmetric range for y data (diss)
    resolution: histogram resolution (grid points per dimension for JPDF)
    """
    outputdir = Path(nd.outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    path = outputdir / "histograms"
    path.mkdir(parents=True, exist_ok=True)

    ticksize = 14
    plt.rcParams['xtick.labelsize'] = ticksize
    plt.rcParams['ytick.labelsize'] = ticksize
    xlabelpad, ylabelpad = 5, 2
    xlabelsize = 22
    ylabelsize = 24
    cbaroutlinewidth = 1.5

    def xlabel(comp: str, term: str) -> str:
        if term == 'vis':
            return r'$u_' + comp + r'(\nabla \cdot \mathrm{\mathbb{T}})_' + comp + r'$'
        if term == 'res':
            return r'$B_' + comp + r'\nabla^2 B_' + comp + r'$'
        raise ValueError(f"Unsupported term {term!r}; expected 'vis' or 'res'.")

    def ylabel(comp: str, term: str, mode: str | None) -> str:
        sub = r'_{\mathrm{' + term + r'}, ' + comp + r'}'
        if mode is not None:
            return r'${\mathscr{D}}^{\mathrm{' + mode + r'}}' + sub + r'$'
        return r'${\mathscr{D}}' + sub + r'$'

    def flatten(diss: np.ndarray, lapl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diss = diss.flatten()
        lapl = lapl.flatten()
        valid = np.isfinite(diss) & np.isfinite(lapl)
        return diss[valid], lapl[valid]

    def KDE(lapl: np.ndarray, diss: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate JPDF using 2D KDE with KDEpy.FFTKDE

        KDEpy docs: https://kdepy.readthedocs.io/en/latest/index.html

        Returns hist on (resolution, resolution) grid, with xgrid, ygrid from KDEpy.
        grid: (resolution^2, 2) mesh points; points: density at each grid point.
        """
        if len(lapl) == 0 or len(diss) == 0:
            raise ValueError("No valid finite data for KDE after flatten.")

        xrange = float(np.percentile(np.abs(lapl), xcoverage * 100))
        yrange = float(np.percentile(np.abs(diss), ycoverage * 100))
        mask = (np.abs(lapl) <= xrange) & (np.abs(diss) <= yrange)
        lapl = lapl[mask]
        diss = diss[mask]

        if len(lapl) == 0:
            raise ValueError("No data within coverage range for KDE.")

        # KDEpy 2D: data shape (obs, dims), grid_points tuple = (n_x, n_y) per dimension
        # Normalize each dimension by its own sigma so a scalar bw applies uniformly.
        # Reference:
        # Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and Visualization.
        # Silverman, Bernard W. Density estimation for statistics and data analysis. Routledge, 2018.
        def Silverman(data: np.ndarray) -> float:
            sigma = float(np.std(data, ddof=1))
            IQR   = float(np.percentile(data, 75) - np.percentile(data, 25))
            N  = len(data)
            bw = min(sigma, IQR/1.34) * N ** (-1.0 / 5)
            return float(bw)

        bwx = Silverman(lapl)
        bwy = Silverman(diss)
        data = np.column_stack([lapl / bwx, diss / bwy])
        kde  = FFTKDE(kernel='gaussian', bw=1).fit(data)

        # Use an explicit grid so the plotting domain matches coverage range.
        # FFTKDE requires all data points to lie strictly inside the grid;
        # expand endpoints by a tiny epsilon to satisfy this when points land on boundary.
        epsilon = 1e-6
        xgrid = np.linspace(-xrange - epsilon, xrange + epsilon, resolution)
        ygrid = np.linspace(-yrange - epsilon, yrange + epsilon, resolution)
        # Custom FFTKDE grids must be sorted in cartesian-product order.
        grid = np.column_stack([
            np.repeat(xgrid / bwx, resolution),
            np.tile(ygrid / bwy, resolution),
        ])
        points = kde.evaluate(grid)

        hist  = np.asarray(points).reshape(resolution, resolution).T / (bwx * bwy)

        return hist, xgrid, ygrid

    cmap = plt.get_cmap('inferno')

    if nd.type == 'hydro':
        configs = [
            ('vis', 'num', nd.V, nd.numVisTerm, nd.divStressT, nd.nu),
        ]
    else:
        configs = [
            ('res', 'num', nd.B, nd.numResTerm, nd.LaplacianB, nd.eta),
            ('vis', 'num', nd.V, nd.numVisTerm, nd.divStressT, nd.nu),
        ]

    for term, mode, field, dissterm, laplacian, coeff in configs:

        if dissterm is None or laplacian is None:
            continue

        components = [
            ('x', field.x * dissterm.x, field.x * laplacian.x),
            ('y', field.y * dissterm.y, field.y * laplacian.y),
            ('z', field.z * dissterm.z, field.z * laplacian.z)
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 11.75), constrained_layout=True)
        layout_engine = fig.get_layout_engine()
        assert isinstance(layout_engine, ConstrainedLayoutEngine)
        layout_engine.set(wspace=0.04)

        # First pass: compute KDE and conditional stats for all three components
        jpdflist: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        meanlist: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for comp, diss, lapl in components:
            diss, lapl = flatten(diss, lapl)
            hist, xgrid, ygrid = KDE(lapl, diss)

            totals = hist.sum(axis=0)
            totals = np.where(totals > 0, totals, 1.0)
            conditional = hist / totals

            cdf = np.cumsum(conditional, axis=0)
            cdf_last = cdf[-1, :]
            cdf_last = np.where(cdf_last > 0, cdf_last, 1.0)
            cdf = cdf / cdf_last

            ymean = np.sum(conditional * ygrid[:, None], axis=0)

            lower = np.zeros(cdf.shape[1])
            upper = np.zeros(cdf.shape[1])
            for i in range(cdf.shape[1]):
                lower[i] = np.interp(0.1585, cdf[:, i], ygrid)
                upper[i] = np.interp(0.8415, cdf[:, i], ygrid)

            valid = (totals > 0) & np.isfinite(ymean) & (upper > lower)
            x     = xgrid[valid]
            ymean = ymean[valid]
            lower = lower[valid]
            upper = upper[valid]

            jpdflist.append((hist, xgrid, ygrid))
            meanlist.append((x, ymean, lower, upper))

        # Compute unified y-axis range for second row
        xranges     = [float(xgrid[-1]) for (_, xgrid, _) in jpdflist]
        ymeanranges = [
            float(np.max(np.abs(ymean))) if len(ymean) > 0 else 0.0
            for (_, ymean, _, _) in meanlist
        ]
        kranges = [
            ymeanranges[i] / xranges[i] if xranges[i] > 0 else 0.0
            for i in range(3)
        ]
        kmax  = float(np.max(kranges))
        ymaxs = [xranges[i] * kmax * 1.2 for i in range(3)]

        # Second pass: plot
        for idx, (comp, _, _) in enumerate(components):
            ax0 = axes[0, idx]
            ax1 = axes[1, idx]

            hist, xgrid, ygrid = jpdflist[idx]
            x, ymean, lower, upper = meanlist[idx]
            ymax = ymaxs[idx]

            extent = (float(xgrid[0]), float(xgrid[-1]), float(ygrid[0]), float(ygrid[-1]))

            # ===== First row: JPDF =====
            visible  = hist[(ygrid >= -ymax) & (ygrid <= ymax), :]
            positive = visible[visible > 0]
            if positive.size == 0:
                raise ValueError(
                    f'JPDF color scale: no positive values in y-range '
                    f'(term={term!r}, mode={mode!r}, comp={comp!r}, ymax={ymax})'
                )
            vmin = float(np.percentile(positive, 5))
            vmax = float(np.max(positive))
            if not (
                np.isfinite(vmin)
                and np.isfinite(vmax)
                and vmin > 0
                and vmax > 0
                and vmax > vmin
            ):
                raise ValueError(
                    f'LogNorm needs finite vmin, vmax with 0 < vmin < vmax; '
                    f'got vmin={vmin}, vmax={vmax} (term={term!r}, mode={mode!r}, comp={comp!r})'
                )
            norm = LogNorm(vmin=vmin, vmax=vmax)

            im = ax0.imshow(hist, origin='lower', extent=extent, cmap=cmap, aspect='auto', norm=norm)
            ax0.set_xlim(xgrid[0], xgrid[-1])
            ax0.set_ylim(-ymax, ymax)

            ax0.set_xlabel(xlabel(comp, term), labelpad=xlabelpad, fontsize=xlabelsize)
            ax0.set_ylabel(ylabel(comp, term, mode), labelpad=ylabelpad, fontsize=ylabelsize)
            ax0.tick_params(direction='in', width=1.5, pad=7, labelsize=ticksize)
            for spine in ax0.spines.values():
                spine.set_linewidth(1.5)
            ax0.set_box_aspect(1)

            cax = ax0.inset_axes((0, 1.035, 1, 0.06))
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(labelsize=12)
            cbar.outline.set_linewidth(cbaroutlinewidth)  # type: ignore[union-attr]

            # ===== Second row: conditional mean + 68% interval =====
            if len(x) > 0:
                ax1.scatter(
                    x, ymean, marker='o', facecolors='k', edgecolors='k', s=24, alpha=1.0, zorder=2,
                    label=r'mean ' + ylabel(comp, term, mode)
                )
                ax1.fill_between(
                    x, lower, upper, alpha=0.2, color='gray', zorder=1,
                    label=r'$68.3\%$ interval'
                )
                ax1.plot(x, lower, color='gray', linewidth=1, linestyle='--', zorder=1)
                ax1.plot(x, upper, color='gray', linewidth=1, linestyle='--', zorder=1)

                if coeff != 0.0:

                    coefficient = {
                        'vis': r'\nu',
                        'res': r'\eta',
                    }[term]

                    ax1.plot(
                        x, coeff * x, color='r', linestyle='--', linewidth=3, zorder=3,
                        label=rf'${coefficient} = {float2LaTeX(coeff)}$',
                    )

                frame = ax1.legend(
                    loc='lower right', fontsize=16, bbox_to_anchor=(0.995, 0.005),
                    framealpha=1.0, fancybox=True, facecolor="white", edgecolor="k",
                ).get_frame()
                frame.set_edgecolor("k")
                frame.set_linewidth(0.8)
                frame.set_boxstyle("round", pad=0.15, rounding_size=0.4)

            ax1.set_xlim(xgrid[0], xgrid[-1])
            ax1.set_ylim(-ymax, ymax)
            ax1.set_xlabel(xlabel(comp, term), labelpad=xlabelpad, fontsize=xlabelsize)
            ax1.set_ylabel(ylabel(comp, term, None), labelpad=ylabelpad, fontsize=ylabelsize)
            ax1.tick_params(direction='in', width=1.5, pad=7, labelsize=ticksize)
            for spine in ax1.spines.values():
                spine.set_linewidth(1.5)
            ax1.set_box_aspect(1)

        plt.savefig(path / f'hist.{mode}.{term}.pdf', bbox_inches='tight')
        plt.close()


def plot(
    nd       : NumericalDissipation,
    fraction : float = 1.0,
) -> None:
    """Plot numerical dissipation slices and histograms

    Parameters
    ----------
    nd        : NumericalDissipation object
    xcoverage : fraction of data enclosed by symmetric range for x (lapl)
    ycoverage : fraction of data enclosed by symmetric range for y (diss)
    fraction  : float in (0, 1], passed to slice colormap scaling (see plotSlices).
    """
    print("═════════ Result Visualization ═════════\n")
    print(f"Plotting dissipation term slices ...")
    plotSlices(nd, fraction=fraction)

    print(f"Plotting histograms of numerical dissipation ...")
    plotHistogram(nd)

    print("Plotting dissipation spectra ...")
    path = Path("spectra") / "spectra.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Energy spectra cache not found: {path}")

    with path.open("rb") as f:
        spc = pickle.load(f)

    plotSpectra(nd, spc)

    print(f"All plots completed! Numerical dissipation analysis done.")
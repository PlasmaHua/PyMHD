# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/plot/slc.py
-----------------

Implements the tools for plotting 2D slices of turbulence.

Currently supports MRI-driven turbulence, forced MHD turbulence, and hydrodynamic turbulence.
    - Shearing-box simulations: the box ratio is hard coded to be Lx : Ly : Lz = 2 : 4 : 1
    - Forced turbulence: the box ratio is hard coded to be Lx : Ly : Lz = 1 : 1 : 1

TODO: Support arbitrary box ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from pathlib import Path

from ..turbulence import Turbulence

# Font: Computer Modern
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True

plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['figure.titlesize'] = 16

def getrange(
    variables: list[tuple[str, np.ndarray]],
    fraction : float = 1.0,
) -> dict[str, tuple[float, float]]:
    """Compute the colormap ranges for each variable

    Parameters
    ----------
    variables : list of tuples, each containing a variable name and its data
    fraction  : float in (0, 1]. Proportion of data to include in [vmin, vmax].
        Default 1.0 = full range. For non-rho, range is symmetric: vmax = percentile(|data|, fraction*100).

    Returns
    -------
    ranges : list of tuples, variable names and colormap ranges (vmin, vmax)
    """
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    ranges: dict[str, tuple[float, float]] = {}

    for varname, vardata in variables:

        data = vardata.flatten()

        if varname == 'rho':
            mean = np.mean(data)
            delta = max(np.max(data) - mean, mean - np.min(data))
            ranges[varname] = (mean - delta, mean + delta)
        else:
            vmax = float(np.percentile(np.abs(data), fraction * 100))
            ranges[varname] = (-vmax, vmax)

    return ranges


def plotForcedTurbulence(
    turbulence: Turbulence,
    fraction  : float = 1.0,
) -> None:
    """Plot 2D slices of forced turbulence

    Supports both forced MHD and hydrodynamic turbulence.

    Parameters
    ----------
    turbulence: Turbulence object
    fraction  : float in (0, 1]. Proportion of data in color range; default 1.0 = full range.
    """
    basedir = Path('slices')
    outputdirs = {
        'rho': basedir / 'rho',
        'V'  : basedir / 'V',
        'B'  : basedir / 'B',
        'J'  : basedir / 'J',
        'all': basedir / 'all',
    }
    for outdir in outputdirs.values():
        outdir.mkdir(parents=True, exist_ok=True)

    pct = fraction * 100

    def get_slice_range(slices: list[tuple[str, np.ndarray]]) -> tuple[float, float]:
        """Get a shared linear color range from the three plotted slices."""
        data = np.concatenate([arr.flatten() for _, arr in slices])
        vmax = float(np.percentile(np.abs(data), pct))
        return -vmax, vmax

    for index, time in enumerate(turbulence.times):

        rho = turbulence.rhos[index]
        V   = turbulence.Vs[index]
        B   = turbulence.Bs[index]
        J   = turbulence.Js[index]

        Nx, Ny, Nz = rho.data.shape
        Lx, Ly, Lz = rho.box

        # Each tuple: (output_dir_key, variables_for_range, slices_for_x_y_z, colorbar_labels_for_x_y_z)
        groups: list[tuple[str, list[tuple[str, np.ndarray]], list[str]]] = [
            (
                'rho',
                [('x', rho.data[Nx // 2, :, :]), ('y', rho.data[:, Ny // 2, :]), ('z', rho.data[:, :, Nz // 2])],
                [r'$\rho$', r'$\rho$', r'$\rho$'],
            ),
            (
                'V',
                [('x', V.x[Nx // 2, :, :]), ('y', V.y[:, Ny // 2, :]), ('z', V.z[:, :, Nz // 2])],
                [r'$u_x$', r'$u_y$', r'$u_z$'],
            ),
            (
                'B',
                [('x', B.x[Nx // 2, :, :]), ('y', B.y[:, Ny // 2, :]), ('z', B.z[:, :, Nz // 2])],
                [r'$B_x$', r'$B_y$', r'$B_z$'],
            ),
            (
                'J',
                [('x', J.x[Nx // 2, :, :]), ('y', J.y[:, Ny // 2, :]), ('z', J.z[:, :, Nz // 2])],
                [r'$J_x$', r'$J_y$', r'$J_z$'],
            ),
        ]

        for outkey, slices, cbarlabels in groups:

            fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
            ax1, ax2, ax3 = axes

            if outkey == 'rho':
                cmap = 'Blues'
                rho_values = np.concatenate([arr.flatten() for _, arr in slices])
                if np.any(rho_values < 0):
                    raise ValueError("rho slice data contains negative values, cannot use LogNorm.")
                vmin = float(np.min(rho_values))
                vmax = float(np.max(rho_values))
                useLog = (vmin > 0) and (vmax / vmin > 10)
                norm = LogNorm(vmin=vmin, vmax=vmax) if useLog else Normalize(vmin=vmin, vmax=vmax)
                im1 = ax1.imshow(
                    slices[0][1].T, origin='lower', cmap=cmap, norm=norm,
                    extent=(-Ly / 2, Ly / 2, -Lz / 2, Lz / 2), aspect='auto'
                )
                im2 = ax2.imshow(
                    slices[1][1].T, origin='lower', cmap=cmap, norm=norm,
                    extent=(-Lz / 2, Lz / 2, -Lx / 2, Lx / 2), aspect='auto'
                )
                im3 = ax3.imshow(
                    slices[2][1].T, origin='lower', cmap=cmap, norm=norm,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2), aspect='auto'
                )

            else:
                cmap = 'RdBu'
                vmin, vmax = get_slice_range(slices)
                im1 = ax1.imshow(
                    slices[0][1].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Ly / 2, Ly / 2, -Lz / 2, Lz / 2), aspect='auto'
                )
                im2 = ax2.imshow(
                    slices[1][1].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Lz / 2, Lz / 2, -Lx / 2, Lx / 2), aspect='auto'
                )
                im3 = ax3.imshow(
                    slices[2][1].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2), aspect='auto'
                )

            ax1.set_xlabel(r'$y$')
            ax1.set_ylabel(r'$z$')
            ax2.set_xlabel(r'$z$')
            ax2.set_ylabel(r'$x$')
            ax3.set_xlabel(r'$x$')
            ax3.set_ylabel(r'$y$')

            for ax in [ax1, ax2, ax3]:
                ax.tick_params(direction='in', width=1.5, pad=7)
                ax.set_box_aspect(1)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            # Manually control colorbar gap/size; width always equals subplot width
            cbar_bottom = 1.035
            cbar_height = 0.06
            cax1 = ax1.inset_axes((0, cbar_bottom, 1, cbar_height))
            cax2 = ax2.inset_axes((0, cbar_bottom, 1, cbar_height))
            cax3 = ax3.inset_axes((0, cbar_bottom, 1, cbar_height))

            cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar1.set_label(cbarlabels[0], labelpad=8)
            cbar1.ax.xaxis.set_ticks_position('top')
            cbar1.ax.xaxis.set_label_position('top')
            cbar1.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar1.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

            cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
            cbar2.set_label(cbarlabels[1], labelpad=8)
            cbar2.ax.xaxis.set_ticks_position('top')
            cbar2.ax.xaxis.set_label_position('top')
            cbar2.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar2.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

            cbar3 = fig.colorbar(im3, cax=cax3, orientation='horizontal')
            cbar3.set_label(cbarlabels[2], labelpad=8)
            cbar3.ax.xaxis.set_ticks_position('top')
            cbar3.ax.xaxis.set_label_position('top')
            cbar3.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar3.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

            plt.savefig(outputdirs[outkey] / f't={time:.2f}.pdf', bbox_inches='tight')
            plt.close()

        all: list[tuple[str, np.ndarray, str, str]] = [
            ('rho', rho.data[:, :, Nz // 2], r'$\rho$', 'Blues'),
            ('Vz' ,      V.z[:, :, Nz // 2], r'$u_z$' , 'RdBu' ),
            ('Bz' ,      B.z[:, :, Nz // 2], r'$B_z$' , 'RdBu' ),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

        for ax, (varname, slicedata, cbarlabel, cmap) in zip(axes, all):
            if varname == 'rho':
                flattened = slicedata.flatten()
                if np.any(flattened < 0):
                    raise ValueError("rho slice data contains negative values, cannot use LogNorm.")
                vmin = float(np.min(flattened))
                vmax = float(np.max(flattened))
                useLog = (vmin > 0) and (vmax / vmin > 10)
                norm = LogNorm(vmin=vmin, vmax=vmax) if useLog else Normalize(vmin=vmin, vmax=vmax)
                image = ax.imshow(
                    slicedata.T,
                    origin='lower',
                    cmap=cmap, norm=norm,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
                    aspect='auto',
                )
            else:
                vmax = float(np.percentile(np.abs(slicedata.flatten()), pct))
                image = ax.imshow(
                    slicedata.T,
                    origin='lower',
                    cmap=cmap, vmin=-vmax, vmax=vmax,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
                    aspect='auto',
                )

            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.tick_params(direction='in', width=1.5, pad=7)
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            cax = ax.inset_axes((0, 1.035, 1, 0.06))
            cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
            cbar.set_label(cbarlabel, labelpad=8)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

        plt.savefig(outputdirs['all'] / f't={time:.2f}.pdf', bbox_inches='tight')
        plt.close()


def plotHydroTurbulence(
    turbulence: Turbulence,
    fraction: float = 1.0,
) -> None:
    """Plot 2D slices of hydrodynamic turbulence

    Parameters
    ----------
    turbulence: Turbulence object
    fraction  : float in (0, 1]. Proportion of data in color range; default 1.0 = full range.
    """
    basedir = Path('slices')
    outputdirs = {
        'rho': basedir / 'rho',
        'V'  : basedir / 'V',
        'all': basedir / 'all',
    }
    for outdir in outputdirs.values():
        outdir.mkdir(parents=True, exist_ok=True)

    pct = fraction * 100

    def get_slice_range(slices: list[tuple[str, np.ndarray]]) -> tuple[float, float]:
        """Get a shared linear color range from the three plotted slices."""
        data = np.concatenate([arr.flatten() for _, arr in slices])
        vmax = float(np.percentile(np.abs(data), pct))
        return -vmax, vmax

    for index, time in enumerate(turbulence.times):

        rho = turbulence.rhos[index]
        V   = turbulence.Vs[index]

        Nx, Ny, Nz = rho.data.shape
        Lx, Ly, Lz = rho.box

        groups: list[tuple[str, list[tuple[str, np.ndarray]], list[str]]] = [
            (
                'rho',
                [('x', rho.data[Nx // 2, :, :]), ('y', rho.data[:, Ny // 2, :]), ('z', rho.data[:, :, Nz // 2])],
                [r'$\rho$', r'$\rho$', r'$\rho$'],
            ),
            (
                'V',
                [('x', V.x[Nx // 2, :, :]), ('y', V.y[:, Ny // 2, :]), ('z', V.z[:, :, Nz // 2])],
                [r'$u_x$', r'$u_y$', r'$u_z$'],
            ),
        ]

        for outkey, slices, cbarlabels in groups:

            fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
            ax1, ax2, ax3 = axes

            if outkey == 'rho':
                cmap = 'Blues'
                rho_values = np.concatenate([arr.flatten() for _, arr in slices])
                if np.any(rho_values < 0):
                    raise ValueError("rho slice data contains negative values, cannot use LogNorm.")
                vmin = float(np.min(rho_values))
                vmax = float(np.max(rho_values))
                useLog = (vmin > 0) and (vmax / vmin > 10)
                norm = LogNorm(vmin=vmin, vmax=vmax) if useLog else Normalize(vmin=vmin, vmax=vmax)
                im1 = ax1.imshow(
                    slices[0][1].T, origin='lower', cmap=cmap, norm=norm,
                    extent=(-Ly / 2, Ly / 2, -Lz / 2, Lz / 2), aspect='auto'
                )
                im2 = ax2.imshow(
                    slices[1][1].T, origin='lower', cmap=cmap, norm=norm,
                    extent=(-Lz / 2, Lz / 2, -Lx / 2, Lx / 2), aspect='auto'
                )
                im3 = ax3.imshow(
                    slices[2][1].T, origin='lower', cmap=cmap, norm=norm,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2), aspect='auto'
                )

            else:
                cmap = 'RdBu'
                vmin, vmax = get_slice_range(slices)
                im1 = ax1.imshow(
                    slices[0][1].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Ly / 2, Ly / 2, -Lz / 2, Lz / 2), aspect='auto'
                )
                im2 = ax2.imshow(
                    slices[1][1].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Lz / 2, Lz / 2, -Lx / 2, Lx / 2), aspect='auto'
                )
                im3 = ax3.imshow(
                    slices[2][1].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2), aspect='auto'
                )

            ax1.set_xlabel(r'$y$')
            ax1.set_ylabel(r'$z$')
            ax2.set_xlabel(r'$z$')
            ax2.set_ylabel(r'$x$')
            ax3.set_xlabel(r'$x$')
            ax3.set_ylabel(r'$y$')

            for ax in [ax1, ax2, ax3]:
                ax.tick_params(direction='in', width=1.5, pad=7)
                ax.set_box_aspect(1)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            cbar_bottom = 1.035
            cbar_height = 0.06
            cax1 = ax1.inset_axes((0, cbar_bottom, 1, cbar_height))
            cax2 = ax2.inset_axes((0, cbar_bottom, 1, cbar_height))
            cax3 = ax3.inset_axes((0, cbar_bottom, 1, cbar_height))

            cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar1.set_label(cbarlabels[0], labelpad=8)
            cbar1.ax.xaxis.set_ticks_position('top')
            cbar1.ax.xaxis.set_label_position('top')
            cbar1.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar1.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

            cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
            cbar2.set_label(cbarlabels[1], labelpad=8)
            cbar2.ax.xaxis.set_ticks_position('top')
            cbar2.ax.xaxis.set_label_position('top')
            cbar2.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar2.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

            cbar3 = fig.colorbar(im3, cax=cax3, orientation='horizontal')
            cbar3.set_label(cbarlabels[2], labelpad=8)
            cbar3.ax.xaxis.set_ticks_position('top')
            cbar3.ax.xaxis.set_label_position('top')
            cbar3.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar3.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

            plt.savefig(outputdirs[outkey] / f't={time:.2f}.pdf', bbox_inches='tight')
            plt.close()

        # all: two subplots, left=rho z=0, right=Vz z=0
        all_slices: list[tuple[str, np.ndarray, str, str]] = [
            ('rho', rho.data[:, :, Nz // 2], r'$\rho$', 'Blues'),
            ('Vz', V.z[:, :, Nz // 2], r'$u_z$', 'RdBu'),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)

        for ax, (varname, slicedata, cbarlabel, cmap) in zip(axes, all_slices):
            if varname == 'rho':
                flattened = slicedata.flatten()
                if np.any(flattened < 0):
                    raise ValueError("rho slice data contains negative values, cannot use LogNorm.")
                vmin = float(np.min(flattened))
                vmax = float(np.max(flattened))
                useLog = (vmin > 0) and (vmax / vmin > 10)
                norm = LogNorm(vmin=vmin, vmax=vmax) if useLog else Normalize(vmin=vmin, vmax=vmax)
                image = ax.imshow(
                    slicedata.T,
                    origin='lower',
                    cmap=cmap,
                    norm=norm,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
                    aspect='auto',
                )
            else:
                vmax = float(np.percentile(np.abs(slicedata.flatten()), pct))
                image = ax.imshow(
                    slicedata.T,
                    origin='lower',
                    cmap=cmap,
                    vmin=-vmax,
                    vmax=vmax,
                    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
                    aspect='auto',
                )

            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.tick_params(direction='in', width=1.5, pad=7)
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            cax = ax.inset_axes((0, 1.035, 1, 0.06))
            cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
            cbar.set_label(cbarlabel, labelpad=8)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(labelsize=12, pad=2)
            outline_spine = cbar.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(1.5)

        plt.savefig(outputdirs['all'] / f't={time:.2f}.pdf', bbox_inches='tight')
        plt.close()


def plotMRITurbulence(
    turbulence: Turbulence,
    fraction: float = 1.0,
) -> None:
    """Plot 2D slices of MRI-driven turbulence

    Plot a figure for each physical component, containing three slices in the following directions:
        - upper left : yx plane (z=0)
        - upper right: zx plane (y=0)
        - lower left : yz plane (x=0)

    Parameters
    ----------
    turbulence: Turbulence object
    fraction  : float in (0, 1]. Passed to getrange; default 1.0 = full range.
    """
    basedir = Path('slices')
    basedir.mkdir(parents=True, exist_ok=True)

    for index, time in enumerate(turbulence.times):

        rho = turbulence.rhos[index]
        V   = turbulence.Vs[index]
        B   = turbulence.Bs[index]

        Nx, Ny, Nz = rho.data.shape

        variables: list[tuple[str, np.ndarray]] = [
            ('rho', rho.data),
            ('Vx', V.x), ('Vy', V.y), ('Vz', V.z),
            ('Bx', B.x), ('By', B.y), ('Bz', B.z),
        ]

        ranges = getrange(variables, fraction=fraction)

        for varname, vardata in variables:

            vardir = basedir / varname
            vardir.mkdir(parents=True, exist_ok=True)

            fig = plt.figure(figsize=(14, 8), constrained_layout=False)

            # Create grid layout, ensure the height of im1 and im2 is the same
            # The first row occupies 2/3 height, the second row occupies 1/3 height
            # The first column occupies 2/3 width, the second column occupies 1/3 width
            gs = plt.GridSpec(
                2, 3, figure  = fig,
                width_ratios  = [4, 1, 0.2],
                height_ratios = [2, 1],
                left   = 0.1,
                right  = 0.9,
                top    = 0.9,
                bottom = 0.1,
                wspace = 0.08,
                hspace = 0.08
            )

            ax1 = fig.add_subplot(gs[0, 0])    # yx plane
            ax2 = fig.add_subplot(gs[0, 1])    # zx plane
            ax3 = fig.add_subplot(gs[1, 0])    # yz plane

            cax = fig.add_subplot(gs[:, 2])    # colorbar subplot

            vmin, vmax = ranges[varname]

            Lx, Ly, Lz = rho.box

            cmap = 'RdBu'

            im1 = ax1.imshow(
                vardata[:, :, Nz//2]  , origin='upper',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=(-Ly/2, Ly/2, Lx/2, -Lx/2)
            )
            im2 = ax2.imshow(
                vardata[:, Ny//2, :]  , origin='upper',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=(-Lz/2, Lz/2, Lx/2, -Lx/2)
            )
            im3 = ax3.imshow(
                vardata[Nx//2, :, :].T, origin='lower',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=(-Ly/2, Ly/2, -Lz/2, Lz/2)
            )

            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')

            # Set the position of the ticklabels
            # Remove the bottom axis ticklabels of im1
            ax1.xaxis.set_ticklabels([])

            # Remove the left axis ticklabels of im2
            ax2.yaxis.set_ticklabels([])
            ax2.set_ylabel('')

            ax1.tick_params(direction='in')
            ax2.tick_params(direction='in')
            ax3.tick_params(direction='in')

            ax1.set_ylabel(r'$x$', labelpad=0)
            ax2.set_xlabel(r'$z$', labelpad=10)
            ax3.set_xlabel(r'$y$', labelpad=10)
            ax3.set_ylabel(r'$z$', labelpad=0)

            linewidth = 1.5
            pad = 7

            ax1.tick_params(width=linewidth, pad=pad)
            ax2.tick_params(width=linewidth, pad=pad)
            ax3.tick_params(width=linewidth, pad=pad)

            # set the width of the axes borders
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_linewidth(linewidth)

            cbar = fig.colorbar(im1, cax=cax, shrink=0.9)

            cbarlabel = {
                'rho': r'$\rho$',
                'Vx' : r'$V_x$',
                'Vy' : r'$V_y$',
                'Vz' : r'$V_z$',
                'Bx' : r'$B_x$',
                'By' : r'$B_y$',
                'Bz' : r'$B_z$'
            }[varname]

            cbar.set_label(f'{cbarlabel}', labelpad=10)

            outline_spine = cbar.ax.spines.get("outline")
            if outline_spine is not None:
                outline_spine.set_linewidth(linewidth)  # width of the colorbar border

            plt.savefig(vardir / f't={time:.2f}.pdf', bbox_inches='tight')
            plt.close()

def plot2dslice(
    turbulence: Turbulence,
    fraction: float = 1.0,
) -> None:
    """Plot 2D slices of turbulence

    Route to the corresponding implementation based on turbulence.type.

    Parameters
    ----------
    turbulence: Turbulence object
    fraction  : float in (0, 1]. Proportion of data in color range; default 1.0 = full range.
    """
    if turbulence.type == 'MRI':
        plotMRITurbulence(turbulence, fraction=fraction)
    elif turbulence.type in ('SSD', 'Bx', 'Bz'):
        plotForcedTurbulence(turbulence, fraction=fraction)
    elif turbulence.type == 'hydro':
        plotHydroTurbulence(turbulence, fraction=fraction)
    else:
        raise ValueError(f"Unsupported turbulence type: {turbulence.type}")
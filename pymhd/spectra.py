# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/spectra.py
----------------

Implements the tools for turbulent energy spectra calculations:
    - Spectrum class: data container for a single 3D spectrum.
    - Spectrum1D class: data container for a 1D spectrum.
    - EnergySpectra class: data container for magnetic and kinetic energy spectra.
    - get1D() function: project a 3D spectrum to a 1D spectrum.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

import pickle

from collections import UserList
from typing import Literal

import time

from .turbulence import Turbulence, VectorField

class Spectrum:
    """Container for a single 3D spectrum.

    Parameters
    ----------
    EK  : np.ndarray with shape (Nx, Ny, Nz), the 3D spectrum data.
    box : tuple[float, float, float], the box size in x, y, z directions (Lx, Ly, Lz).

    Attributes
    ----------
    EK           : np.ndarray
    box          : tuple[float, float, float]
    Nx, Ny, Nz   : int, the number of grid points in x, y, z directions.
    dx, dy, dz   : float, the grid spacings in x, y, z directions.
    dkx, dky, dkz: float, the grid spacings in Fourier space.
    dVk          : float, the volume of a grid cell in Fourier space.
    kx, ky, kz   : np.ndarray, the wavenumbers.
    """
    def __init__(self, EK: np.ndarray, box: tuple[float, float, float]):
        if EK.ndim != 3:
            raise ValueError("Spectrum.EK must be a 3D array")

        self.EK  = EK
        self.box = box
        self.Lx, self.Ly, self.Lz = box

        self.Nx, self.Ny, self.Nz = EK.shape
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz

        self.dkx = 1.0 / self.Lx
        self.dky = 1.0 / self.Ly
        self.dkz = 1.0 / self.Lz
        self.dVk = self.dkx * self.dky * self.dkz

        self.kx = np.fft.fftfreq(self.Nx, d=self.dx)
        self.ky = np.fft.fftfreq(self.Ny, d=self.dy)
        self.kz = np.fft.fftfreq(self.Nz, d=self.dz)

    def __add__(self, other) -> Spectrum:
        """Addition

        Supported operations
        --------------------
        Spectrum + Spectrum -> Spectrum
        Spectrum + None     -> Spectrum
        """
        if other is None:
            return self
        if isinstance(other, Spectrum) and assertMatchSpectra(self, other):
            return Spectrum(self.EK + other.EK, self.box)

        return NotImplemented

    def __radd__(self, other) -> Spectrum:
        """Right addition

        Supported operations
        --------------------
        None + Spectrum -> Spectrum
        """
        if other is None:
            return self

        return NotImplemented

    def __sub__(self, other) -> Spectrum:
        """Subtraction

        Supported operations
        --------------------
        Spectrum - Spectrum -> Spectrum
        """
        if isinstance(other, Spectrum) and assertMatchSpectra(self, other):
            return Spectrum(self.EK - other.EK, self.box)

        return NotImplemented


def assertMatchSpectra(spectrum1: Spectrum, spectrum2: Spectrum) -> bool:
    """Assert that two spectra have matching box and data shape.

    Returns True by default; raises ValueError on mismatch.

    Raises
    ------
    ValueError : if box or array shapes do not match.
    """
    if spectrum1.box != spectrum2.box:
        raise ValueError("Box must match")
    if spectrum1.EK.shape != spectrum2.EK.shape:
        raise ValueError("Spectrum data shapes must match")

    return True


class Spectrum1D:
    """Container for a 1D spectrum.

    Parameters
    ----------
    k    : np.ndarray, shape (N,), the 1D wavenumbers.
    Ek   : np.ndarray, shape (N,), the 1D spectrum data.

    Attributes
    ----------
    k    : np.ndarray
    Ek   : np.ndarray
    dk   : float, the grid spacing in wavenumber space.
    kmax : float, the maximum wavenumber.
    """
    def __init__(self, k: np.ndarray, Ek: np.ndarray):

        if k.ndim != 1 or Ek.ndim != 1:
            raise ValueError("Spectrum1D must contain 1D arrays")

        self.k    = k
        self.Ek   = Ek
        self.dk   = k[1] - k[0]
        self.kmax = np.max(k)


class SpectraList(UserList):
    """List of Spectrum objects.

    Attributes
    ----------
    data : list[Spectrum]
    avg  : Spectrum, the arithmetic mean of the spectra.
    """
    @property
    def avg(self) -> Spectrum:
        if len(self.data) == 0:
            raise ValueError("Cannot compute avg on an empty SpectraList")

        box   = self.data[0].box
        stack = np.stack([spc.EK for spc in self.data], axis=0)

        return Spectrum(np.mean(stack, axis=0), box)


class EnergySpectra:
    """Magnetic and kinetic energy spectra.

    Parameters
    ----------
    turbulence: Turbulence, the Turbulence object.

    Attributes
    ----------
    type    : Literal['SSD', 'Bx', 'Bz', 'MRI', 'hydro'], inherited from Turbulence.
    nu      : float, kinematic viscosity, inherited from Turbulence.
    eta     : float | None, magnetic diffusivity, inherited from Turbulence.

    totEkin : Spectrum, total kinetic energy spectrum.
    totEmag : Spectrum | None, total magnetic energy spectrum.

    xEkin   : Spectrum, x-component kinetic energy spectrum.
    yEkin   : Spectrum, y-component kinetic energy spectrum.
    zEkin   : Spectrum, z-component kinetic energy spectrum.

    xEmag   : Spectrum | None, x-component magnetic energy spectrum.
    yEmag   : Spectrum | None, y-component magnetic energy spectrum.
    zEmag   : Spectrum | None, z-component magnetic energy spectrum.
    """
    def cache(self, path: Path) -> None:
        """Cache computed result with pickle serialization."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"Failed to save cache: {exc}\n")
            return

        print(f"EnergySpectra cache saved to ./{path}\n")

    def __init__(self, turbulence: Turbulence) -> None:

        start_time = time.time()

        self.type = turbulence.type
        self.nu   = turbulence.nu
        self.eta  = turbulence.eta

        def computeEnergySpectra(
            fields: list[VectorField],
        ) -> tuple[Spectrum, Spectrum, Spectrum, Spectrum]:

            xlist, ylist, zlist, totlist = [], [], [], []

            for field in fields:
                dV = field.dxdydz
                fx = np.fft.fftn(field.x) * dV
                fy = np.fft.fftn(field.y) * dV
                fz = np.fft.fftn(field.z) * dV
                Ex = Spectrum(0.5 * np.abs(fx) ** 2, field.box)
                Ey = Spectrum(0.5 * np.abs(fy) ** 2, field.box)
                Ez = Spectrum(0.5 * np.abs(fz) ** 2, field.box)
                xlist.append(Ex)
                ylist.append(Ey)
                zlist.append(Ez)
                totlist.append(Ex + Ey + Ez)

            return (
                SpectraList(xlist).avg,
                SpectraList(ylist).avg,
                SpectraList(zlist).avg,
                SpectraList(totlist).avg,
            )

        self.xEkin, self.yEkin, self.zEkin, self.totEkin = computeEnergySpectra(turbulence.wVs)

        if turbulence.type != "hydro":
            self.xEmag, self.yEmag, self.zEmag, self.totEmag = computeEnergySpectra(turbulence.Bs)
        else:
            self.xEmag, self.yEmag, self.zEmag, self.totEmag = None, None, None, None

        end_time = time.time()
        print(f"Computation of turbulent spectra completed! Elapsed time: {end_time - start_time:.2f} s")

        self.cache(Path("spectra") / "spectra.pkl")

def get1D(
    spc     : Spectrum,
    mode    : Literal["shell", "perp", "para", "avg"],
    axis    : Literal["x", "y", "z"] | None = None,
    negative: bool = False,
) -> Spectrum1D:
    r"""Project a 3D Spectrum to a Spectrum1D.

    Parameters
    ----------
    spc : Spectrum, the 3D spectrum object.
    mode: the mode of the 3D to 1D projection.
        - "shell": shell-integrated spectrum E(k), where k = |\bm{k}|;
        - "perp" : E(k_\perp), axisymmetric around the background field;
        - "para" : E(k_\parallel), where k_\parallel is parallel to the background field;
        - "avg"  : integrate out the two axes perpendicular to `axis`, giving E(k_{axis}).
    axis: the axis of the 3D to 1D projection for certain modes:
        - "shell": should be None;
        - "perp" : specifies the background field direction;
        - "para" : specifies the background field direction;
        - "avg"  : specifies the retained axis.
    negative: bool
        - False: return the absolute value of the spectrum (default);
        - True : return the negative of the spectrum (mainly for dissipation spectra).

    Returns
    -------
    Spectrum1D, the 1D spectrum object.
    """
    if mode == "shell":

        kx3, ky3, kz3 = np.meshgrid(spc.kx, spc.ky, spc.kz, indexing="ij")
        K = np.sqrt(kx3**2 + ky3**2 + kz3**2)

        # Integrate up to the maximum complete spherical shell fully in the Fourier space.
        kxmax, kymax, kzmax = 1.0 / (2.0 * spc.dx), 1.0 / (2.0 * spc.dy), 1.0 / (2.0 * spc.dz)
        Kmax = min(kxmax, kymax, kzmax)
        dK   = max(spc.dkx, spc.dky, spc.dkz)

        bins = np.arange(0.0, Kmax + dK, dK)
        ctrs = bins[:-1] + 0.5 * dK
        Ek   = np.zeros_like(ctrs)

        for i, (kl, kr) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (K >= kl) & (K < kr)
            Ek[i] = np.sum(spc.EK[mask]) * spc.dVk / dK

        valid = np.isfinite(Ek) & (ctrs >= dK)
        k  = ctrs[valid]
        Ek = Ek[valid]

        if Ek.size == 0:
            raise ValueError("No valid shell-integrated spectrum values found.")

        Ek = -Ek if negative else np.abs(Ek)
        return Spectrum1D(k, Ek)

    # Map the parallel axis to (axis index, dk, k array),
    # and the two perpendicular axes to their indices and dk values.
    if axis == "x":
        para , Kpara , dKpara  = 0, spc.kx, spc.dkx
        perp1, Kperp1, dKperp1 = 1, spc.ky, spc.dky
        perp2, Kperp2, dKperp2 = 2, spc.kz, spc.dkz
    elif axis == "y":
        para , Kpara , dKpara  = 1, spc.ky, spc.dky
        perp1, Kperp1, dKperp1 = 0, spc.kx, spc.dkx
        perp2, Kperp2, dKperp2 = 2, spc.kz, spc.dkz
    elif axis == "z":
        para , Kpara , dKpara = 2, spc.kz, spc.dkz
        perp1, Kperp1, dKperp1 = 0, spc.kx, spc.dkx
        perp2, Kperp2, dKperp2 = 1, spc.ky, spc.dky
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z' for mode='{mode}'")

    if mode == "avg":

        Ek = np.sum(spc.EK, axis=(perp1, perp2)) * dKperp1 * dKperp2
        k  = np.fft.fftshift(Kpara)
        Ek = np.fft.fftshift(Ek)

        mask  = k > 0.0
        k, Ek = k[mask], 2.0 * Ek[mask]
        Ek = -Ek if negative else np.abs(Ek)
        return Spectrum1D(k, Ek)

    if mode in ("perp", "para"):

        mesh = np.meshgrid(Kperp1, Kperp2, indexing="ij")
        Kperp2D = np.sqrt(mesh[0]**2 + mesh[1]**2)

        # Shell binning for the perpendicular direction
        Kperpmax = min(max(Kperp1), max(Kperp2))
        dKperp   = max(dKperp1, dKperp2)

        bins = np.arange(0.0, Kperpmax + dKperp, dKperp)
        ctrs = bins[:-1] + 0.5 * dKperp

        # E2D[i_perp, i_para]: shell-integrated 2D spectrum E(k_\perp, k_\parallel).
        # For each perpendicular ring and each parallel index, sum E * dKperp1 * dKperp2,
        # then divide by dKperp to get a density in k_perp.
        Npara = spc.EK.shape[para]
        Ek2D  = np.zeros((ctrs.size, Npara))
        data  = np.moveaxis(spc.EK, para, -1)  # shape: (perp1, perp2, para)
        for i, (kl, kr) in enumerate(zip(bins[:-1], bins[1:])):
            ring = (Kperp2D >= kl) & (Kperp2D < kr)
            if not np.any(ring):
                continue
            Ek2D[i, :] = data[ring, :].sum(axis=0) * dKperp1 * dKperp2 / dKperp

        if mode == "perp":
            # E(k_\perp) = int E(k_\perp, k_\parallel) dKparallel
            k  = ctrs
            Ek = np.sum(Ek2D, axis=1) * dKpara
            valid = np.isfinite(Ek) & (k >= dKperp)
            k, Ek = k[valid], Ek[valid]
        else:
            # E(k_\parallel) = int E(k_\perp, k_\parallel) dKperp
            Ek = np.sum(Ek2D, axis=0) * dKperp
            k  = np.fft.fftshift(Kpara)
            Ek = np.fft.fftshift(Ek)
            mask  = k > 0.0
            k, Ek = k[mask], 2.0 * Ek[mask]

        if Ek.size == 0:
            raise ValueError(f"No valid {mode}-projected spectrum values found.")

        Ek = -Ek if negative else np.abs(Ek)
        return Spectrum1D(k, Ek)

    raise ValueError("mode must be 'shell', 'perp', 'para', or 'avg'")
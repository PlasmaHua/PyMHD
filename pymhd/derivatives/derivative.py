# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/derivatives/derivative.py
-------------------------------

Implements the numerical schemes for derivatives of ScalarField and VectorField, including:
    - Dx, Dy, Dz: partial derivatives of ScalarField and VectorField;
    - grad      : gradient of ScalarField;
    - div       : divergence of VectorField;
    - curl      : curl of VectorField;
    - laplacian : Laplacian of ScalarField and VectorField.

Supported algorithms:
    - 'WENO'    : WENO5-Z and WENO7-Z;
    - 'TENO'    : TENO7-M;
    - 'TCS'     : targeted compact scheme (TCS7-M);
    - 'CENTRAL' : 2nd, 4th, and 8th order central difference;
    - 'SPECTRAL': Fourier spectral difference.
"""

import numpy as np

from typing import TypeVar, Literal

from ..turbulence import ScalarField, VectorField

# Define generic type for Fields (ScalarField or VectorField)
Field = TypeVar('Field', ScalarField, VectorField)

# ========== Algorithm class ==========
# supported algorithms and parameters:
# 'WENO'    : stencil width (5 or 7)
# 'TENO'    : CT (discontinuity threshold)
# 'TCS'     : CT (discontinuity threshold)
# 'CENTRAL' : stencil width (accuracy order + 1, i.e. order = stencil - 1)
# 'SPECTRAL': no parameters

class Algorithm:
    """Algorithm class

    Attributes
    ----------
    method  : type of algorithm, options: 'WENO', 'TENO', 'TCS', 'CENTRAL', 'SPECTRAL'
    stencil : stencil width for 'WENO', 'TENO', and 'CENTRAL'
    CT      : discontinuity threshold for 'TENO' and 'TCS'
    """
    method  : str
    stencil : int | None
    CT      : float | None

    def __init__(self, method: str, stencil: int | None = None, CT: float | None = None):
        self.method = method.upper()

        if self.method == 'WENO':
            self.stencil = 7 if stencil is None else stencil
            self.CT      = None
            if self.stencil not in (5, 7):
                raise ValueError("WENO stencil must be 5 or 7.")

        elif self.method == 'CENTRAL':
            # CENTRAL uses stencil = order + 1, supporting orders of 2, 4, and 8.
            self.stencil = 9 if stencil is None else stencil
            self.CT      = None
            if self.stencil not in (3, 5, 9):
                raise ValueError("CENTRAL stencil must be 3, 5, or 9 (order 2, 4, 8).")

        elif self.method == 'TENO':
            if stencil not in (None, 7):
                raise ValueError(f"{self.method} stencil must be 7 or None.")
            self.stencil = 7 if stencil is None else stencil
            self.CT      = 0.01 if CT is None else CT

        elif self.method == 'TCS':
            if stencil not in (None, 7):
                raise ValueError(f"{self.method} stencil must be 7 or None.")
            self.stencil = None
            self.CT      = 0.01 if CT is None else CT

        elif self.method == 'SPECTRAL':
            self.stencil = None
            self.CT      = None

        else:
            raise ValueError(f"Unsupported method: {self.method}.")


# ========== WENO derivatives ==========

from .WENO import WENOx, WENOy, WENOz

def wenoDx(field: Field, stencil: int = 7) -> Field:
    """x derivative of Field using WENO-Z

    Parameters
    ----------
    field   : Field
    stencil : stencil width (5 or 7), default: 7

    Returns
    -------
    x derivative of Field
    """
    if isinstance(field, ScalarField):
        data = field.data
        dx   = field.dx

        # WENO reconstructed cell interfaces
        uL, uR = WENOx(data, stencil)
        DuDx = (uR - uL) / dx

        return ScalarField(DuDx, field.box)

    if isinstance(field, VectorField):
        DxVx = wenoDx(ScalarField(field.x, field.box), stencil=stencil)
        DxVy = wenoDx(ScalarField(field.y, field.box), stencil=stencil)
        DxVz = wenoDx(ScalarField(field.z, field.box), stencil=stencil)

        return VectorField(DxVx.data, DxVy.data, DxVz.data, field.box)

    raise TypeError("Input of 'wenoDx()' must be ScalarField or VectorField.")


def wenoDy(field: Field, stencil: int = 7) -> Field:
    """y derivative of Field using WENO-Z

    Parameters
    ----------
    field   : Field
    stencil : width of stencil (5 or 7), default: 7

    Returns
    -------
    y derivative of Field
    """
    if isinstance(field, ScalarField):
        data = field.data
        dy   = field.dy

        # WENO reconstructed cell interfaces
        uL, uR = WENOy(data, stencil)
        DuDy = (uR - uL) / dy

        return ScalarField(DuDy, field.box)

    if isinstance(field, VectorField):
        DyVx = wenoDy(ScalarField(field.x, field.box), stencil=stencil)
        DyVy = wenoDy(ScalarField(field.y, field.box), stencil=stencil)
        DyVz = wenoDy(ScalarField(field.z, field.box), stencil=stencil)

        return VectorField(DyVx.data, DyVy.data, DyVz.data, field.box)

    raise TypeError("Input of 'wenoDy()' must be ScalarField or VectorField.")


def wenoDz(field: Field, stencil: int = 7) -> Field:
    """z derivative of Field using WENO-Z

    Parameters
    ----------
    field   : Field
    stencil : width of stencil (5 or 7), default: 7

    Returns
    -------
    z derivative of Field
    """
    if isinstance(field, ScalarField):
        data = field.data
        dz = field.dz

        # WENO reconstructed cell interfaces
        uL, uR = WENOz(data, stencil)
        DuDz = (uR - uL) / dz

        return ScalarField(DuDz, field.box)

    if isinstance(field, VectorField):
        DzVx = wenoDz(ScalarField(field.x, field.box), stencil=stencil)
        DzVy = wenoDz(ScalarField(field.y, field.box), stencil=stencil)
        DzVz = wenoDz(ScalarField(field.z, field.box), stencil=stencil)

        return VectorField(DzVx.data, DzVy.data, DzVz.data, field.box)

    raise TypeError("Input of 'wenoDz()' must be ScalarField or VectorField.")

# ========== TENO derivatives ==========

from .TENO import TENO7Mx, TENO7My, TENO7Mz

def tenoDx(field: Field, CT: float = 0.01) -> Field:
    """x derivative of Field using TENO-M scheme

    TENO-M: TENO scheme with multi-stencil discontinuity detector

    Parameters
    ----------
    field : Field
    CT    : smoothness threshold, default: 0.01

    Returns
    -------
    x derivative of Field
    """
    if isinstance(field, ScalarField):
        data = field.data
        dx   = field.dx

        # TENO reconstructed cell interfaces
        uL, uR = TENO7Mx(data, mode='hybrid', CT=CT)

        DuDx = np.asarray((uR - uL) / dx)

        return ScalarField(DuDx, field.box)

    if isinstance(field, VectorField):
        DxVx = tenoDx(ScalarField(field.x, field.box), CT)
        DxVy = tenoDx(ScalarField(field.y, field.box), CT)
        DxVz = tenoDx(ScalarField(field.z, field.box), CT)

        return VectorField(DxVx.data, DxVy.data, DxVz.data, field.box)

    raise TypeError("Input of 'tenoDx()' must be ScalarField or VectorField.")


def tenoDy(field: Field, CT: float = 0.01) -> Field:
    """y derivative of Field using TENO-M

    TENO-M: TENO scheme with multi-stencil discontinuity detector

    Parameters
    ----------
    field : Field
    CT    : smoothness threshold, default: 0.01

    Returns
    -------
    y derivative of Field
    """
    if isinstance(field, ScalarField):
        data = field.data
        dy   = field.dy

        # TENO reconstructed cell interfaces
        uL, uR = TENO7My(data, mode='hybrid', CT=CT)

        DuDy = np.asarray((uR - uL) / dy)

        return ScalarField(DuDy, field.box)

    if isinstance(field, VectorField):
        DyVx = tenoDy(ScalarField(field.x, field.box), CT)
        DyVy = tenoDy(ScalarField(field.y, field.box), CT)
        DyVz = tenoDy(ScalarField(field.z, field.box), CT)

        return VectorField(DyVx.data, DyVy.data, DyVz.data, field.box)

    raise TypeError("Input of 'tenoDy()' must be ScalarField or VectorField.")


def tenoDz(field: Field, CT: float = 0.01) -> Field:
    """z derivative of Field using TENO-M

    TENO-M: TENO scheme with multi-stencil discontinuity detector

    Parameters
    ----------
    field : Field
    CT    : smoothness threshold, default: 0.01

    Returns
    -------
    z derivative of Field
    """
    if isinstance(field, ScalarField):
        data = field.data
        dz   = field.dz

        # TENO reconstructed cell interfaces
        uL, uR = TENO7Mz(data, mode='hybrid', CT=CT)

        DuDz = np.asarray((uR - uL) / dz)

        return ScalarField(DuDz, field.box)

    if isinstance(field, VectorField):
        DzVx = tenoDz(ScalarField(field.x, field.box), CT)
        DzVy = tenoDz(ScalarField(field.y, field.box), CT)
        DzVz = tenoDz(ScalarField(field.z, field.box), CT)

        return VectorField(DzVx.data, DzVy.data, DzVz.data, field.box)

    raise TypeError("Input of 'tenoDz()' must be ScalarField or VectorField.")


# ========== TCS (Targeted Compact Scheme) derivatives ==========

from .compact import TCS7Mx, TCS7My, TCS7Mz

def tcsDx(field: Field, CT: float = 0.01) -> Field:
    """x derivative of Field using TCS7-M

    TCS7-M: targeted compact scheme with multi-stencil discontinuity detector

    Parameters
    ----------
    field : Field
    CT    : smoothness threshold, default: 0.01

    Returns
    -------
    x derivative of Field
    """
    if isinstance(field, ScalarField):
        DuDx = np.asarray(TCS7Mx(field.data, CT=CT, L=field.Lx))
        return ScalarField(DuDx, field.box)

    if isinstance(field, VectorField):
        DxVx = tcsDx(ScalarField(field.x, field.box), CT)
        DxVy = tcsDx(ScalarField(field.y, field.box), CT)
        DxVz = tcsDx(ScalarField(field.z, field.box), CT)

        return VectorField(DxVx.data, DxVy.data, DxVz.data, field.box)

    raise TypeError("Input of 'tcsDx()' must be ScalarField or VectorField.")


def tcsDy(field: Field, CT: float = 0.01) -> Field:
    """y derivative of Field using TCS7-M

    TCS7-M: targeted compact scheme with multi-stencil discontinuity detector

    Parameters
    ----------
    field : Field
    CT    : smoothness threshold, default: 0.01

    Returns
    -------
    y derivative of Field
    """
    if isinstance(field, ScalarField):
        DuDy = np.asarray(TCS7My(field.data, CT=CT, L=field.Ly))
        return ScalarField(DuDy, field.box)

    if isinstance(field, VectorField):
        DyVx = tcsDy(ScalarField(field.x, field.box), CT)
        DyVy = tcsDy(ScalarField(field.y, field.box), CT)
        DyVz = tcsDy(ScalarField(field.z, field.box), CT)

        return VectorField(DyVx.data, DyVy.data, DyVz.data, field.box)

    raise TypeError("Input of 'tcsDy()' must be ScalarField or VectorField.")


def tcsDz(field: Field, CT: float = 0.01) -> Field:
    """z derivative of Field using TCS7-M

    TCS7-M: targeted compact scheme with multi-stencil discontinuity detector

    Parameters
    ----------
    field : Field
    CT    : smoothness threshold, default: 0.01

    Returns
    -------
    z derivative of Field
    """
    if isinstance(field, ScalarField):
        DuDz = np.asarray(TCS7Mz(field.data, CT=CT, L=field.Lz))
        return ScalarField(DuDz, field.box)

    if isinstance(field, VectorField):
        DzVx = tcsDz(ScalarField(field.x, field.box), CT)
        DzVy = tcsDz(ScalarField(field.y, field.box), CT)
        DzVz = tcsDz(ScalarField(field.z, field.box), CT)

        return VectorField(DzVx.data, DzVy.data, DzVz.data, field.box)

    raise TypeError("Input of 'tcsDz()' must be ScalarField or VectorField.")


# ========== Central difference derivatives ==========

def central(
    u: np.ndarray, dx: float, direction: Literal['x', 'y', 'z'], order: int = 8
) -> np.ndarray:
    """derivative of array using central difference scheme

    Parameters
    ----------
    u         : 3D array
    dx        : grid spacing
    direction : derivative direction ('x', 'y', 'z')
    order     : order of accuracy (2, 4, 8), default: 8

    Returns
    -------
    derivative of u
    """
    # determine the axis of roll based on direction
    directions = {'x': 0, 'y': 1, 'z': 2}
    axis = directions[direction]

    if order == 2:
        # second order: f'(x) = (f(x+h) - f(x-h)) / (2h)
        uR1  = np.roll(u, -1, axis=axis)  # f(x + h)
        uL1  = np.roll(u,  1, axis=axis)  # f(x - h)
        DuDx = (1 / 2 * uR1 - 1 / 2 * uL1) / dx

    elif order == 4:
        # fourth order: f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        uR2  = np.roll(u, -2, axis=axis)  # f(x + 2h)
        uR1  = np.roll(u, -1, axis=axis)  # f(x +  h)
        uL1  = np.roll(u,  1, axis=axis)  # f(x -  h)
        uL2  = np.roll(u,  2, axis=axis)  # f(x - 2h)
        DuDx = (
            1 / 12 * uL2 - 2 / 3 * uL1 + 2 / 3 * uR1 - 1 / 12 * uR2
        ) / dx

    elif order == 8:
        # eighth order:
        # f'(x) = (1/280*f(x-4h) - 4/105*f(x-3h) +   1/5*f(x-2h) -   4/5*f(x-h)
        #          + 4/5*f(x+h)  -   1/5*f(x+2h) + 4/105*f(x+3h) - 1/280*f(x+4h)) / h
        uR4 = np.roll(u, -4, axis=axis)  # f(x + 4h)
        uR3 = np.roll(u, -3, axis=axis)  # f(x + 3h)
        uR2 = np.roll(u, -2, axis=axis)  # f(x + 2h)
        uR1 = np.roll(u, -1, axis=axis)  # f(x +  h)
        uL1 = np.roll(u,  1, axis=axis)  # f(x -  h)
        uL2 = np.roll(u,  2, axis=axis)  # f(x - 2h)
        uL3 = np.roll(u,  3, axis=axis)  # f(x - 3h)
        uL4 = np.roll(u,  4, axis=axis)  # f(x - 4h)
        DuDx = (
              (1 / 280) * uL4
            - (4 / 105) * uL3
            + (1 / 5)   * uL2
            - (4 / 5)   * uL1
            + (4 / 5)   * uR1
            - (1 / 5)   * uR2
            + (4 / 105) * uR3
            - (1 / 280) * uR4
        ) / dx
    else:
        raise ValueError(f"Unsupported order: {order}. Supported orders are 2, 4, 8.")

    return DuDx

def centralDx(field: Field, order: int = 8) -> Field:
    """x derivative of Field using central difference scheme

    Parameters
    ----------
    field : Field
    order : order of accuracy (2, 4, 8), default: 8

    Returns
    -------
    x derivative of Field
    """
    if isinstance(field, ScalarField):

        data = field.data
        dx   = field.dx

        DuDx = central(data, dx, 'x', order)

        return ScalarField(DuDx, field.box)

    if isinstance(field, VectorField):

        DxVx = centralDx(ScalarField(field.x, field.box), order)
        DxVy = centralDx(ScalarField(field.y, field.box), order)
        DxVz = centralDx(ScalarField(field.z, field.box), order)

        return VectorField(DxVx.data, DxVy.data, DxVz.data, field.box)

    raise TypeError("Input of 'centralDx()' must be ScalarField or VectorField.")


def centralDy(field: Field, order: int = 8) -> Field:
    """y derivative of Field using central difference scheme

    Parameters
    ----------
    field : Field
    order : order of accuracy (2, 4, 8), default: 8

    Returns
    -------
    y derivative of Field
    """
    if isinstance(field, ScalarField):

        data = field.data
        dy   = field.dy

        DuDy = central(data, dy, 'y', order)

        return ScalarField(DuDy, field.box)

    if isinstance(field, VectorField):

        DyVx = centralDy(ScalarField(field.x, field.box), order)
        DyVy = centralDy(ScalarField(field.y, field.box), order)
        DyVz = centralDy(ScalarField(field.z, field.box), order)

        return VectorField(DyVx.data, DyVy.data, DyVz.data, field.box)

    raise TypeError("Input of 'centralDy()' must be ScalarField or VectorField.")


def centralDz(field: Field, order: int = 8) -> Field:
    """z derivative of Field using central difference scheme

    Parameters
    ----------
    field : Field
    order : order of accuracy (2, 4, 8), default: 8

    Returns
    -------
    z derivative of Field
    """
    if isinstance(field, ScalarField):

        data = field.data
        dz   = field.dz

        DuDz = central(data, dz, 'z', order)

        return ScalarField(DuDz, field.box)

    if isinstance(field, VectorField):

        DzVx = centralDz(ScalarField(field.x, field.box), order)
        DzVy = centralDz(ScalarField(field.y, field.box), order)
        DzVz = centralDz(ScalarField(field.z, field.box), order)

        return VectorField(DzVx.data, DzVy.data, DzVz.data, field.box)

    raise TypeError("Input of 'centralDz()' must be ScalarField or VectorField.")


# ========== Spectral difference ==========

def spectralDx(field: Field) -> Field:
    """x derivative of Field using spectral difference

    Parameters
    ----------
    field : Field

    Returns
    -------
    x derivative of Field
    """
    if isinstance(field, ScalarField):
        data, Nx = field.data, field.Nx

        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=field.dx)

        # 1D FFT along x
        uhat = np.fft.fft(data, axis=0)
        DuDx = np.real(np.fft.ifft(1j * kx[:, np.newaxis, np.newaxis] * uhat, axis=0))

        return ScalarField(DuDx, field.box)

    if isinstance(field, VectorField):

        DxVx = spectralDx(ScalarField(field.x, field.box))
        DxVy = spectralDx(ScalarField(field.y, field.box))
        DxVz = spectralDx(ScalarField(field.z, field.box))

        return VectorField(DxVx.data, DxVy.data, DxVz.data, field.box)

    raise TypeError("Input of 'spectralDx()' must be ScalarField or VectorField.")


def spectralDy(field: Field) -> Field:
    """y derivative of Field using spectral difference

    Parameters
    ----------
    field : Field

    Returns
    -------
    y derivative of Field

    """
    if isinstance(field, ScalarField):
        data, Ny = field.data, field.Ny

        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=field.dy)

        # 1D FFT along y
        uhat = np.fft.fft(data, axis=1)
        DuDy = np.real(np.fft.ifft(1j * ky[np.newaxis, :, np.newaxis] * uhat, axis=1))

        return ScalarField(DuDy, field.box)

    if isinstance(field, VectorField):
        DyVx = spectralDy(ScalarField(field.x, field.box))
        DyVy = spectralDy(ScalarField(field.y, field.box))
        DyVz = spectralDy(ScalarField(field.z, field.box))

        return VectorField(DyVx.data, DyVy.data, DyVz.data, field.box)

    raise TypeError("Input of 'spectralDy()' must be ScalarField or VectorField.")

def spectralDz(field: Field) -> Field:
    """z derivative of Field using spectral difference

    Parameters
    ----------
    field : Field

    Returns
    -------
    z derivative of Field
    """
    if isinstance(field, ScalarField):
        data, Nz = field.data, field.Nz

        kz = 2 * np.pi * np.fft.fftfreq(Nz, d=field.dz)

        # 1D FFT along z
        uhat = np.fft.fft(data, axis=2)
        DuDz = np.real(np.fft.ifft(1j * kz[np.newaxis, np.newaxis, :] * uhat, axis=2))

        return ScalarField(DuDz, field.box)

    if isinstance(field, VectorField):
        DzVx = spectralDz(ScalarField(field.x, field.box))
        DzVy = spectralDz(ScalarField(field.y, field.box))
        DzVz = spectralDz(ScalarField(field.z, field.box))

        return VectorField(DzVx.data, DzVy.data, DzVz.data, field.box)

    raise TypeError("Input of 'spectralDz()' must be ScalarField or VectorField.")


# ========== Derivatives calculation ==========

def Dx(field: Field, algorithm: Algorithm) -> Field:
    """x derivative of Field

    Parameters
    ----------
    field     : Field
    algorithm : Algorithm

    Returns
    -------
    x derivative of Field
    """
    if not isinstance(field, (ScalarField, VectorField)):
        raise TypeError("Input of 'Dx()' must be ScalarField or VectorField.")

    if algorithm.method == 'WENO':
        if algorithm.stencil is None:
            raise ValueError("WENO requires stencil.")
        return wenoDx(field, stencil=algorithm.stencil)

    if algorithm.method == 'TENO':
        if algorithm.CT is None:
            raise ValueError("TENO requires CT.")
        return tenoDx(field, CT=algorithm.CT)

    if algorithm.method == 'TCS':
        if algorithm.CT is None:
            raise ValueError("TCS requires CT.")
        return tcsDx(field, CT=algorithm.CT)

    if algorithm.method == 'CENTRAL':
        if algorithm.stencil is None:
            raise ValueError("CENTRAL requires stencil.")
        return centralDx(field, order=algorithm.stencil - 1)

    if algorithm.method == 'SPECTRAL':
        return spectralDx(field)

    raise ValueError(f"Unsupported method: {algorithm.method}.")


def Dy(field: Field, algorithm: Algorithm) -> Field:
    """y derivative of Field

    Parameters
    ----------
    field     : Field
    algorithm : Algorithm

    Returns
    -------
    y derivative of Field
    """
    if not isinstance(field, (ScalarField, VectorField)):
        raise TypeError("Input of 'Dy()' must be ScalarField or VectorField.")

    if algorithm.method == 'WENO':
        if algorithm.stencil is None:
            raise ValueError("WENO requires stencil.")
        return wenoDy(field, stencil=algorithm.stencil)

    if algorithm.method == 'TENO':
        if algorithm.CT is None:
            raise ValueError("TENO requires CT.")
        return tenoDy(field, CT=algorithm.CT)

    if algorithm.method == 'TCS':
        if algorithm.CT is None:
            raise ValueError("TCS requires CT.")
        return tcsDy(field, CT=algorithm.CT)

    if algorithm.method == 'CENTRAL':
        if algorithm.stencil is None:
            raise ValueError("CENTRAL requires stencil.")
        return centralDy(field, order=algorithm.stencil - 1)

    if algorithm.method == 'SPECTRAL':
        return spectralDy(field)

    raise ValueError(f"Unsupported method: {algorithm.method}.")

def Dz(field: Field, algorithm: Algorithm) -> Field:
    """z derivative of Field

    Parameters
    ----------
    field     : Field
    algorithm : Algorithm

    Returns
    -------
    z derivative of Field
    """
    if not isinstance(field, (ScalarField, VectorField)):
        raise TypeError("Input of 'Dz()' must be ScalarField or VectorField.")

    if algorithm.method == 'WENO':
        if algorithm.stencil is None:
            raise ValueError("WENO requires stencil.")
        return wenoDz(field, stencil=algorithm.stencil)

    if algorithm.method == 'TENO':
        if algorithm.CT is None:
            raise ValueError("TENO requires CT.")
        return tenoDz(field, CT=algorithm.CT)

    if algorithm.method == 'TCS':
        if algorithm.CT is None:
            raise ValueError("TCS requires CT.")
        return tcsDz(field, CT=algorithm.CT)

    if algorithm.method == 'CENTRAL':
        if algorithm.stencil is None:
            raise ValueError("CENTRAL requires stencil.")
        return centralDz(field, order=algorithm.stencil - 1)

    if algorithm.method == 'SPECTRAL':
        return spectralDz(field)

    raise ValueError(f"Unsupported method: {algorithm.method}.")


# ========== nabla operators ==========

def grad(field: ScalarField, algorithm: Algorithm) -> VectorField:
    """gradient of ScalarField

    Gradient: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)

    Parameters
    ----------
    field     : ScalarField
    algorithm : Algorithm

    Returns
    -------
    gradient of ScalarField
    """
    if not isinstance(field, ScalarField):
        raise TypeError("Input of 'grad()' must be ScalarField.")

    DfDx: ScalarField = Dx(field, algorithm)
    DfDy: ScalarField = Dy(field, algorithm)
    DfDz: ScalarField = Dz(field, algorithm)

    return VectorField(DfDx.data, DfDy.data, DfDz.data, field.box)


def div(field: VectorField, algorithm: Algorithm) -> ScalarField:
    """divergence of VectorField

    Divergence: ∇·V = ∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z

    Parameters
    ----------
    field     : VectorField
    algorithm : Algorithm

    Returns
    -------
    divergence of VectorField
    """
    if not isinstance(field, VectorField):
        raise TypeError("Input of 'div()' must be VectorField.")

    DVxDx: ScalarField = Dx(ScalarField(field.x, field.box), algorithm)
    DVyDy: ScalarField = Dy(ScalarField(field.y, field.box), algorithm)
    DVzDz: ScalarField = Dz(ScalarField(field.z, field.box), algorithm)

    return DVxDx + DVyDy + DVzDz


def curl(field: VectorField, algorithm: Algorithm) -> VectorField:
    """curl of VectorField

    Curl: ∇ × V = (∂Vz/∂y - ∂Vy/∂z, ∂Vx/∂z - ∂Vz/∂x, ∂Vy/∂x - ∂Vx/∂y)

    Parameters
    ----------
    field     : VectorField
    algorithm : Algorithm

    Returns
    -------
    curl of VectorField
    """
    if not isinstance(field, VectorField):
        raise TypeError("Input of 'curl()' must be VectorField.")

    Vx = ScalarField(field.x, field.box)
    Vy = ScalarField(field.y, field.box)
    Vz = ScalarField(field.z, field.box)

    curlVx = Dy(Vz, algorithm) - Dz(Vy, algorithm)
    curlVy = Dz(Vx, algorithm) - Dx(Vz, algorithm)
    curlVz = Dx(Vy, algorithm) - Dy(Vx, algorithm)

    return VectorField(curlVx.data, curlVy.data, curlVz.data, field.box)

def laplacian(field: Field, algorithm: Algorithm) -> Field:
    """laplacian of Field

    Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²

    Parameters
    ----------
    field     : Field
    algorithm : Algorithm

    Returns
    -------
    Laplacian of Field
    """
    if not isinstance(field, (ScalarField, VectorField)):
        raise TypeError("Input of 'laplacian()' must be ScalarField or VectorField.")

    if isinstance(field, ScalarField):

        D2fDx2: ScalarField = Dx(Dx(field, algorithm), algorithm)
        D2fDy2: ScalarField = Dy(Dy(field, algorithm), algorithm)
        D2fDz2: ScalarField = Dz(Dz(field, algorithm), algorithm)

        return D2fDx2 + D2fDy2 + D2fDz2

    if isinstance(field, VectorField):

        laplacianVx = laplacian(ScalarField(field.x, field.box), algorithm)
        laplacianVy = laplacian(ScalarField(field.y, field.box), algorithm)
        laplacianVz = laplacian(ScalarField(field.z, field.box), algorithm)

        return VectorField(laplacianVx.data, laplacianVy.data, laplacianVz.data, field.box)

    raise TypeError("Input of 'laplacian()' must be ScalarField or VectorField.")

# ========== cell averages to cell centers ==========

def average2center(field: Field, algorithm: Algorithm) -> Field:
    """convert cell averages to cell centers

    Convert cell averages to cell centers for 3D FVM with sixth order accuracy:
        u_ctr = u_avg
            - (1/24)[dx²∂²u_avg/∂x² + dy²∂²u_avg/∂y² + dz²∂²u_avg/∂z²]
            + (7/5760)[dx⁴∂⁴u_avg/∂x⁴ + dy⁴∂⁴u_avg/∂y⁴ + dz⁴∂⁴u_avg/∂z⁴]
            + (1/576)[(dx*dy)²∂⁴u_avg/∂x²∂y² + (dx*dz)²∂⁴u_avg/∂x²∂z² + (dy*dz)²∂⁴u_avg/∂y²∂z²]

    Parameters
    ----------
    field      : Field, cell averages
    algorithm  : Algorithm

    Returns
    -------
    cellcenters: Field, cell centers
    """
    if not isinstance(field, (ScalarField, VectorField)):
        raise TypeError("Input of 'average2center()' must be ScalarField or VectorField.")

    dx, dy, dz = field.dx, field.dy, field.dz

    D2uDx2: Field = Dx(Dx(field, algorithm), algorithm)  # ∂²u/∂x²
    D2uDy2: Field = Dy(Dy(field, algorithm), algorithm)  # ∂²u/∂y²
    D2uDz2: Field = Dz(Dz(field, algorithm), algorithm)  # ∂²u/∂z²

    D4uDx4: Field = Dx(Dx(D2uDx2, algorithm), algorithm)  # ∂⁴u/∂x⁴
    D4uDy4: Field = Dy(Dy(D2uDy2, algorithm), algorithm)  # ∂⁴u/∂y⁴
    D4uDz4: Field = Dz(Dz(D2uDz2, algorithm), algorithm)  # ∂⁴u/∂z⁴

    D4uDx2Dy2: Field = Dx(Dx(D2uDy2, algorithm), algorithm)  # ∂⁴u/∂x²∂y²
    D4uDx2Dz2: Field = Dx(Dx(D2uDz2, algorithm), algorithm)  # ∂⁴u/∂x²∂z²
    D4uDy2Dz2: Field = Dy(Dy(D2uDz2, algorithm), algorithm)  # ∂⁴u/∂y²∂z²

    cellcenters: Field = (
        field \
        - (1/24)   * (dx**2 * D2uDx2 + dy**2 * D2uDy2 + dz**2 * D2uDz2) \
        + (7/5760) * (dx**4 * D4uDx4 + dy**4 * D4uDy4 + dz**4 * D4uDz4) \
        + (1/576)  * ((dx*dy)**2 * D4uDx2Dy2 + (dx*dz)**2 * D4uDx2Dz2 + (dy*dz)**2 * D4uDy2Dz2)
    )

    return cellcenters
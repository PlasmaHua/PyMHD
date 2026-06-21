# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/__init__.py
-----------------

Unified interface for PyMHD.
"""

from .turbulence import Scalar, Vector, ScalarField, VectorField, Turbulence, avg, std
from .derivatives.derivative import Dx, Dy, Dz, grad, div, curl, laplacian, Algorithm

from .spectra import Spectrum, Spectrum1D, EnergySpectra
from .numdiss import NumericalDissipation
from .preprocess import output2turbulence

from .plot import plot, plot2dslice

__all__ = [
    "Scalar", "Vector",
    "ScalarField", "VectorField", "avg", "std", "Turbulence",
    "Algorithm",
    "Dx", "Dy", "Dz", "grad", "div", "curl", "laplacian",
    "Spectrum", "Spectrum1D",
    "EnergySpectra",
    "NumericalDissipation",
    "output2turbulence",
    "plot", "plot2dslice",
]
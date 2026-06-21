# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/plot/__init__.py
----------------------

Plotting tools for PyMHD.
"""

from __future__ import annotations

from ..numdiss import NumericalDissipation
from ..spectra import EnergySpectra

from .nd import plot as plotNumericalDissipation
from .spc import plot as plotSpectra
from .slc import plot2dslice

def plot(obj: NumericalDissipation | EnergySpectra, **kwargs) -> None:
    """Unified plot function for PyMHD

    Route to the appropriate plot function for each object type.

    Parameters
    ----------
    obj     : NumericalDissipation | EnergySpectra, a plottable PyMHD object
    **kwargs: passed to the plot function of the object (e.g. fraction for colormap scaling).

    Examples
    --------
    >>> from pymhd import plot, NumericalDissipation, EnergySpectra
    >>> spectra = EnergySpectra(...)
    >>> plot(spectra)
    >>> nd = NumericalDissipation(...)
    >>> plot(nd)
    """
    if isinstance(obj, NumericalDissipation):
        plotNumericalDissipation(obj, **kwargs)
        return
    if isinstance(obj, EnergySpectra):
        plotSpectra(obj)
        return

    raise TypeError(
        f"plot() does not support type '{type(obj).__name__}'. "
        f"Supported types: NumericalDissipation, EnergySpectra"
    )

# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/preprocess/__init__.py
----------------------------

Unified interface for extracting turbulence data from simulation outputs. Currently supports:
    - Athena++/AthenaK/AthenaPK output files.
"""

from __future__ import annotations

from typing import Sequence
from pathlib import Path

from ..turbulence import Turbulence

def output2turbulence(
    code     : str,
    outputs  : str | Path | Sequence[str | Path] | None = None,
    outn     : str | None = None,
    inputfile: str | Path | None = None,
    t1       : float | None = None,
    t2       : float | None = None,
    casename : str   | None = None,
) -> Turbulence:
    """Extract simulation output data and build a Turbulence object

    Unified interface that dispatches to code-specific implementations.

    Parameters
    ----------
    code      : simulation code name. Use 'Athena' for Athena++/AthenaK/AthenaPK.
    outputs   : output file paths or glob pattern string
                    e.g., ['HGB.prim.00000.athdf', 'HGB.prim.00001.athdf'] or './outputs/*.prim.*.athdf';
                    If None, a default pattern inferred from the input file is adopted.
    outn      : output ID for resolving output file pattern when 'outputs' is None.
                    e.g., 'prim' for slice and spectra plots, 'nd' for numerical dissipation analysis.
    inputfile : path to input parameter file, e.g. 'athinput.hgb', 'turb.athinput';
                    If None, automatically detect default Athena(++/K/PK) input files.
    t1        : start time for filtering, defaults to None (no lower limit)
    t2        : end time for filtering, defaults to None (no upper limit)
    casename  : case name, defaults to None (use parent directory name)

    Returns
    -------
    Turbulence: Turbulence object containing extracted field data

    Examples
    --------
    >>> from pymhd import output2turbulence
    >>> turbulence = output2turbulence(code='Athena', t1=10.0, t2=20.0)

    """
    if code != "Athena":
        raise ValueError(
            f"Unsupported code: '{code}'. Use 'Athena' as unified entry for Athena++/AthenaK/AthenaPK."
        )

    from .Athena import output2turbulence as o2t

    return o2t(casename=casename, outputs=outputs, outn=outn, inputfile=inputfile, t1=t1, t2=t2)


__all__ = [
    "output2turbulence"
]

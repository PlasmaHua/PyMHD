# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
tests/spectra.test.py
---------------------

Test script for plotting 1D spectra.

Test command: run in the `PyMHD/examples/{case}` directory, time range is optional
    python ../../tests/spectra.test.py -t1 [t1] -t2 [t2]

Arguments:
    -t1 : start time for filtering (optional, default: None)
    -t2 : end time for filtering (optional, default: None)
"""

from __future__ import annotations

import argparse

from pymhd import EnergySpectra, Turbulence, plot, output2turbulence as o2t

def test(args: argparse.Namespace) -> None:
    t1 = args.t1
    t2 = args.t2

    turbulence: Turbulence = o2t(code="Athena", t1=t1, t2=t2)
    spectra = EnergySpectra(turbulence=turbulence)

    plot(spectra)

    print("Tests completed! Spectra plotting done.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="1D spectra plotting test script")
    parser.add_argument(
        "-t1",
        type=float,
        default=None,
        help="Start time for filtering (optional).",
    )
    parser.add_argument(
        "-t2",
        type=float,
        default=None,
        help="End time for filtering (optional).",
    )

    test(parser.parse_args())


if __name__ == "__main__":
    main()

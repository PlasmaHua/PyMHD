# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
tests/numdiss.test.py
---------------------

Test the numerical dissipation analysis module in PyMHD.

Test command: run in the `PyMHD/examples/{case}` directory.
    python ../../tests/numdiss.test.py
    python ../../tests/numdiss.test.py -o nd -m TENO -C 0.00001 -f 0.99
    python ../../tests/numdiss.test.py -o nd -m TCS -C 0.00001
    python ../../tests/numdiss.test.py -o nd -m WENO
note that numdiss.py supports parallel acceleration, in Slurm this may require, e.g.,
    srun -N 1 -n 1 -c [core-per-node] python ../../tests/numdiss.test.py

Arguments:
    -o, --outn     : output basename, defaults to 'nd'
    -m, --method   : options: TENO, TCS, WENO, SPECTRAL
    -C, --CT       : threshold for TENO/TCS schemes
    -f, --fraction : proportion of data in slice colormap range (0, 1], default: 1 (full range)
"""

import argparse
from pathlib import Path

from pymhd import NumericalDissipation, plot, Algorithm
from pymhd import output2turbulence as o2t

def test(args):

    outn     = args.outn
    method   = args.method
    CT       = args.CT
    fraction = args.fraction
    stencil  = 7

    try:
        algorithm = Algorithm(method, stencil, CT)
        cache = Path(NumericalDissipation.alg2dir(algorithm)) / "nd.pkl"

        if cache.is_file():
            try:
                nd = NumericalDissipation(
                    turbulence = None,
                    algorithm  = algorithm,
                )
            except ValueError:
                turbulence = o2t(code = 'Athena', outn = outn)
                nd = NumericalDissipation(
                    turbulence = turbulence,
                    algorithm  = algorithm,
                )
        else:
            turbulence = o2t(code = 'Athena', outn = outn)
            nd = NumericalDissipation(
                turbulence = turbulence,
                algorithm  = algorithm,
            )

        plot(nd, fraction=fraction)

    except FileNotFoundError as e:
        print(f"Error: 'ndiss' output file not found {e}")

    except Exception:
        import traceback
        traceback.print_exc()


def main():

    parser = argparse.ArgumentParser(description='Numerical dissipation analysis test script')

    parser.add_argument(
        '-o', '--outn',
        type=str,
        default='nd',
        help="Output ID, defaults to 'nd'",
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        default='TENO',
        choices=['TENO', 'TCS', 'WENO', 'SPECTRAL'],
        help='Differentiation method, options: TENO, TCS, WENO, SPECTRAL',
    )
    parser.add_argument(
        '-C', '--CT',
        type=float,
        default=0.01,
        help='Threshold for TENO/TCS schemes.',
    )
    parser.add_argument(
        '-f', '--fraction',
        type=float,
        default=1.0,
        help='Proportion of data in slice colormap range (0, 1], default: 1 (full range)',
    )

    test(parser.parse_args())


if __name__ == "__main__":
    main()

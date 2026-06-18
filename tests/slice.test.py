# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
tests/slice.test.py
-------------------

2D slice plotting test script

Test command: run in the `PyMHD/examples/{case}` directory, time range is optional
    python ../../tests/slice.test.py -t1 [t1] -t2 [t2]

Arguments:
    -t1 : start time for filtering (optional, default: None)
    -t2 : end time for filtering (optional, default: None)
    -f, --fraction : proportion of data in color range (0, 1], default: 1 (full range)
"""

import argparse

from pymhd import Turbulence, output2turbulence as o2t, plot2dslice

def test(t1: float | None, t2: float | None, fraction: float) -> None:
    try:
        turbulence: Turbulence = o2t(code="Athena", t1=t1, t2=t2)
        plot2dslice(turbulence, fraction=fraction)
        print("Tests completed! 2D slice plotting done.\n")

    except FileNotFoundError as e:
        print(f"Error: output file not found: {e}")

    except Exception:
        import traceback
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="2D slice plotting test script")

    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=1.0,
        help="Proportion of data in color range (0, 1], default: 1 (full range)",
    )
    parser.add_argument(
        "-t1",
        type=float,
        default=None,
        help="Start time for snapshot filtering (default: None, no lower limit).",
    )
    parser.add_argument(
        "-t2",
        type=float,
        default=None,
        help="End time for snapshot filtering (default: None, no upper limit).",
    )
    args = parser.parse_args()

    test(t1=args.t1, t2=args.t2, fraction=args.fraction)

if __name__ == "__main__":
    main()
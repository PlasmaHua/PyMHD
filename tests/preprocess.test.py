# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
tests/preprocess.test.py
------------------------

Test the preprocessing module for Athena++/AthenaK/AthenaPK output files.

Test command: run in the `PyMHD/examples/{case}` directory, time range is optional
    python ../../tests/preprocess.test.py -t1 [t1] -t2 [t2]
"""

import argparse
import numpy as np

from pymhd import avg
from pymhd import output2turbulence

def test(t1: float | None, t2: float | None):

    turbulence = output2turbulence(
        code = 'Athena',
        t1   = t1,
        t2   = t2,
    )

    print(f'Number of snapshots : {len(turbulence.times)}')
    print(f'Number of rho fields: {len(turbulence.rhos)}')
    print(f'Number of V   fields: {len(turbulence.Vs)}')

    Bs = getattr(turbulence, "Bs", None)
    if Bs is not None:
        print(f'Number of B   fields: {len(Bs)}')
    accs = getattr(turbulence, "accs", None)
    if accs is not None:
        print(f'Number of acc fields: {len(accs)}')
    print(f'time list: {turbulence.times}\n')

    print(f'case   = {turbulence.case}')
    print(f'solver = {turbulence.solver}')
    print(f'type   = {turbulence.type}')
    print(f'EoS    = {turbulence.EoS}')

    if turbulence.EoS == 'isothermal':
        print(f'Cs     = {turbulence.Cs}')
    elif turbulence.EoS == 'adiabatic':
        print(f'gamma  = {turbulence.gamma}')

    if turbulence.type == 'MRI':
        print(f'Omega  = {turbulence.Omega}')
        print(f'q      = {turbulence.q}')

    print(f'nu     = {turbulence.nu}')
    print(f'eta    = {turbulence.eta}')
    print(f'Pm     = {turbulence.Pm}\n')

    rho = avg(turbulence.rhos)
    print(f'Averaged density: {np.mean(rho.data)}')

    p = avg(turbulence.ps)
    print(f'Averaged pressure: {np.mean(p.data)}')

    print(f'Total kinetic energy: {np.mean(turbulence.KEs)} ± {np.std(turbulence.KEs)}')

    if turbulence.type != 'hydro':
        avgBys = [avgB.y for avgB in turbulence.avgBs]
        avgBzs = [avgB.z for avgB in turbulence.avgBs]
        print(f'Averaged By: {np.mean(avgBys)} ± {np.std(avgBys)}')
        print(f'Averaged Bz: {np.mean(avgBzs)} ± {np.std(avgBzs)}')
        print(f'Total magnetic energy: {np.mean(turbulence.MEs)} ± {np.std(turbulence.MEs)}')
    else:
        print('Averaged By: N/A for hydrodynamic turbulence')
        print('Averaged Bz: N/A for hydrodynamic turbulence')
        print('Total magnetic energy: N/A for hydrodynamic turbulence')

    drhos = turbulence.drhos
    print(f'density fluctuation: {np.mean(drhos)} ± {np.std(drhos)}')

def main():
    parser = argparse.ArgumentParser(
        description="Test preprocessing module with auto-detected input/output files."
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

    try:
        test(t1=args.t1, t2=args.t2)
    except FileNotFoundError as e:
        print(f"Error: Required files not found: {e}")
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
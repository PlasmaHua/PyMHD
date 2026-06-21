# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

import pymhd as mhd

from pymhd import Turbulence, EnergySpectra, NumericalDissipation, Algorithm
from pymhd import output2turbulence as o2t

# AthenaK input file
inputfile = "turb.athinput"

# AthenaK output files for visualization of 2D slices and turbulent energy spectra
outputs = "outputs/bin/Turb.prim.*.bin"

# Construct Turbulence object from output files
turbulence: Turbulence = o2t(
    code      = "Athena", 
    outputs   = outputs, 
    inputfile = inputfile,
    t1        = 8.0,
    t2        = 10.0,
)

# Calculate turbulent energy spectra
spectra = EnergySpectra(turbulence=turbulence)

# Visualize 2D slices of MHD turbulence and turbulent energy spectra
mhd.plot2dslice(turbulence, fraction=1.0)
mhd.plot(spectra)

# AthenaK output files for numerical dissipation analysis
outputs = "outputs/bin/Turb.nd.*.*.bin"

# Construct Turbulence object from output files for numerical dissipation analysis
turbulence: Turbulence = o2t(
    code      = "Athena", 
    outputs   = outputs, 
    inputfile = inputfile,
)

# Construct NumericalDissipation object from turbulence
nd = NumericalDissipation(
    turbulence = turbulence,
    algorithm  = Algorithm(method="TENO", stencil=7, CT=0.01),
)

# Visualize numerical dissipation analysis
mhd.plot(nd, fraction=0.99)
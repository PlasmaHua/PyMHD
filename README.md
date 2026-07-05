# PyMHD

[![PyPI](https://img.shields.io/pypi/v/pymhd.svg)](https://pypi.org/project/pymhd/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20749062.svg)](https://doi.org/10.5281/zenodo.20749062)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/PlasmaHua/PyMHD/blob/main/LICENSE)


A Python package for post-processing magnetohydrodynamic (MHD) turbulence simulations.

## Features

- [x] High-fidelity ***a posteriori* estimation and visualization of numerical dissipation** in MHD turbulence simulations, see our method paper ([arXiv:2606.22506](https://arxiv.org/abs/2606.22506)) for details
- [x] Computation and visualization of turbulent energy spectra
- [x] Visualization of 2D slices of MHD variables
- [x] Built-in support for simulations from the Athena family, including [Athena++](https://github.com/PrincetonUniversity/athena), [AthenaK](https://github.com/IAS-Astrophysics/athenak), and [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk)
- [x] Support for **driven MHD turbulence** (e.g., Alfvénic turbulence and small-scale dynamos), driven hydrodynamic turbulence, and **MRI-driven turbulence** (from shearing-box simulations) with triply periodic boundary conditions
- [x] Support for adiabatic and isothermal equations of state (EoS); TODO: add support for incompressible EoS
- [x] Parallel acceleration on (single-node) multi-core CPUs powered by JAX; TODO: support acceleration on GPUs and multi-node CPUs
- [ ] TODO: Computation and visualization of correlation functions
- [ ] TODO: Spectral energy transfer analysis following [Philipp Grete et al. Physics of Plasmas 24.9 (2017)](https://doi.org/10.1063/1.4990613)


## Quick Start

### Installation

Install PyMHD from [PyPI](https://pypi.org/project/pymhd/) with:

```bash
pip install pymhd
```

This command installs PyMHD and its required dependencies, including:
- [NumPy](https://github.com/numpy/numpy) for core computations
- [JAX](https://github.com/jax-ml/jax) for parallel acceleration on single-node, multi-core CPUs; therefore, [the CPU version of JAX](https://docs.jax.dev/en/latest/installation.html#pip-installation-cpu) is required
- [yt](https://github.com/yt-project/yt) and [h5py](https://github.com/h5py/h5py) for extracting data from HDF5 output files, specifically Athena++ `.athdf` files, AthenaK `.bin` files, and AthenaPK `.phdf` files
- [Matplotlib](https://github.com/matplotlib/matplotlib) for plotting functionality in the `pymhd.plot` modules
- [KDEpy](https://github.com/tommyod/KDEpy) for plotting smoothed histograms of numerical dissipation data in the `pymhd.plot.nd` module

### Built-in Examples

For a quick start, clone the source repository:

```bash
git clone https://github.com/PlasmaHua/PyMHD.git
```

Note that cloning the repository may take a while, as the `examples` directory contains roughly 2.4 GB of AthenaK `.bin` output files. Alternatively, you can manually download the source code from the [GitHub releases page](https://github.com/PlasmaHua/PyMHD/releases). Then navigate to the `examples` directory with `cd PyMHD/examples`, where two AthenaK simulation examples are provided.

Each subdirectory under [`PyMHD/examples`](https://github.com/PlasmaHua/PyMHD/tree/main/examples) includes a `quick.py` script, such as [`Bz128beta10Cs10nu1e-4Pm1PLM/quick.py`](https://github.com/PlasmaHua/PyMHD/tree/main/examples/Bz128beta10Cs10nu1e-4Pm1PLM/quick.py):
```python
import pymhd as mhd

from pymhd import Turbulence, EnergySpectra, NumericalDissipation, Algorithm
from pymhd import output2turbulence as o2t

inputfile = "turb.athinput"
outputs = "outputs/bin/Turb.prim.*.bin"

turbulence: Turbulence = o2t(
    code      = "Athena", 
    outputs   = outputs, 
    inputfile = inputfile,
    t1        = 8.0,
    t2        = 10.0,
)

spectra = EnergySpectra(turbulence=turbulence)

mhd.plot2dslice(turbulence, fraction=1.0)
mhd.plot(spectra)

outputs = "outputs/bin/Turb.nd.*.*.bin"

turbulence: Turbulence = o2t(
    code      = "Athena", 
    outputs   = outputs, 
    inputfile = inputfile,
)

nd = NumericalDissipation(
    turbulence = turbulence,
    algorithm  = Algorithm(method="TENO", stencil=7, CT=0.01),
)

mhd.plot(nd, fraction=0.99)
```
This script demonstrates the basic usage of PyMHD, including:
- Extracting a PyMHD `Turbulence` object from AthenaK output files with the `pymhd.output2turbulence` function
- Calculating turbulent energy spectra using the PyMHD `EnergySpectra` class
- Estimating numerical dissipation *a posteriori* using the PyMHD `NumericalDissipation` class
- Visualizing turbulent spectra and numerical dissipation with the `pymhd.plot` module

### Analyze Your Own Athena Simulations

PyMHD natively supports simulations from Athena++, AthenaK, and AthenaPK. Following the above examples, use the `pymhd.output2turbulence` function to  automatically extract output data and input-file metadata, then construct a PyMHD `Turbulence` object for further analysis.

## API Documentation

PyMHD adopts an object-oriented design, providing the `ScalarField` and `VectorField` classes to represent scalar and vector fields, respectively. Built on top of these classes, the `Turbulence` class organizes data from MHD turbulence simulations and serves as the foundation for PyMHD's post-processing pipeline.

### `ScalarField` and `VectorField`

The core attributes of the `ScalarField` class are:
- `box`: a tuple of three floating-point values `(Lx, Ly, Lz)` that defines the simulation box size
- `data`: a 3D NumPy array containing the scalar values on the numerical grid
- `Nx, Ny, Nz`: the simulation resolution, inferred from the shape of `data`

A `ScalarField` instance can be created as follows:
```python
scalarfield = ScalarField(data: np.ndarray, box: tuple[float, float, float])
```

The core attributes of the `VectorField` class are:
- `box`: a tuple of three floating-point values `(Lx, Ly, Lz)` that defines the simulation box size
- `x, y, z`: three 3D NumPy arrays containing the vector components
- `Nx, Ny, Nz`: the simulation resolution, inferred from the shapes of `x`, `y`, and `z`

A `VectorField` instance can be created as follows:
```python
vectorfield = VectorField(
  x  : np.ndarray,
  y  : np.ndarray,
  z  : np.ndarray,
  box: tuple[float, float, float],
)
```

PyMHD implements common operations on `ScalarField` and `VectorField` objects, including dot product $\cdot$, cross product $\times$, gradient $\nabla$, divergence $\nabla\cdot$, and curl $\nabla\times$, among other operations. The following example illustrates the basic usage:

```python
from pymhd import ScalarField, VectorField
from pymhd import grad, div, curl, laplacian

box = 1.0, 1.0, 1.0
B = VectorField(Bx, By, Bz, box) # Arrays such as Bx are omitted for brevity.
V = VectorField(Vx, Vy, Vz, box) # Arrays such as Vx are omitted for brevity.
Bdot = VectorField(Bdotx, Bdoty, Bdotz, box) # Arrays such as Bdotx are omitted for brevity.

Faraday    = curl(V ** B) # Faraday term in the MHD induction equation
LaplacianB = laplacian(B) 

phyResTerm = eta * LaplacianB # Physical resistive dissipation term
numResTerm = Bdot - Faraday - phyResTerm # Estimated numerical resistive dissipation term
```

### `Turbulence`

Under construction.

## Citation

If PyMHD contributes to your work, please consider citing the package using the following BibTeX entry:

```bibtex
@misc{Hua2026PyMHD,
  doi = {10.5281/zenodo.20749062},
  url = {https://zenodo.org/doi/10.5281/zenodo.20749062},
  author = {Hua, Yuyang},
  title = {PyMHD: Python package for post-processing MHD turbulence simulations},
  publisher = {Zenodo},
  year = {2026},
  copyright = {MIT License}
}
```

If your work employs the proposed framework for numerical dissipation estimation, please cite our [method paper](https://arxiv.org/abs/2606.22506):

```bibtex
@misc{Hua2026nd,
  title = {Characterization of Numerical Dissipation in Simulations of Magnetohydrodynamic Turbulence}, 
  author = {Hua, Yuyang and Zhao, Zhonghai and Qiao, Bin},
  year = {2026},
  eprint = {2606.22506},
  archivePrefix = {arXiv},
  primaryClass = {physics.plasm-ph},
  url = {https://arxiv.org/abs/2606.22506}, 
}
```
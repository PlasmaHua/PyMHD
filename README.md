# PyMHD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20749062.svg)](https://doi.org/10.5281/zenodo.20749062)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/PlasmaHua/PyMHD/blob/main/LICENSE)


A Python package for post-processing magnetohydrodynamic (MHD) turbulence simulations.

## Features

- [x] High-fidelity ***a posteriori* estimation and visualization of numerical dissipation** in MHD turbulence simulations
- [x] Computation and visualization of turbulent energy spectra
- [x] Visualization of 2D slices of MHD variables
- [x] Built-in support for simulations from the Athena family, including [Athena++](https://github.com/PrincetonUniversity/athena), [AthenaK](https://github.com/IAS-Astrophysics/athenak), and [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk)
- [x] Support for **driven MHD turbulence** (e.g., Alfvénic turbulence and small-scale dynamos), driven hydrodynamic turbulence, and **MRI-driven turbulence** (from shearing-box simulations) with triply periodic boundary conditions
- [x] Support for adiabatic and isothermal equations of state (EoS); TODO: add support for incompressible EoS
- [ ] TODO: Computation and visualization of correlation functions
- [ ] TODO: Spectral energy transfer analysis following [Grete, Philipp, et al. Physics of Plasmas 24.9 (2017)](https://doi.org/10.1063/1.4990613)
- [ ] TODO: JAX acceleration on GPUs and multi-node CPUs (currently, certain algorithms support parallelism on single-node, multi-core CPUs)


## Quick Start

### Installation

Install PyMHD from [PyPI](https://pypi.org/) with:

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

Alternatively, download the source code from the [GitHub releases page](https://github.com/PlasmaHua/PyMHD/releases). Then navigate to the `./examples` subdirectory with `cd PyMHD/examples`, where two AthenaK simulation examples are provided.

Each directory in [`PyMHD/examples`](https://github.com/PlasmaHua/PyMHD/tree/main/examples) contains a Python script, [`quick.py`](https://github.com/PlasmaHua/PyMHD/tree/main/examples/Bz128beta10Cs10nu1e-4Pm1PLM/quick.py), that demonstrates basic usage of PyMHD, including:
- Extracting a PyMHD `Turbulence` object from output files with the `pymhd.output2turbulence` function
- Calculating turbulent energy spectra with the PyMHD `EnergySpectra` class
- Estimating numerical dissipation *a posteriori* with the PyMHD `NumericalDissipation` class
- Visualizing turbulent spectra and numerical dissipation with the `pymhd.plot` module

### Analyze Your Own Athena Simulations

PyMHD natively supports simulations from Athena++, AthenaK, and AthenaPK. Following the examples in [`PyMHD/examples`](https://github.com/PlasmaHua/PyMHD/tree/main/examples), use the `pymhd.output2turbulence` function to extract output data and input-file metadata, then construct a PyMHD `Turbulence` object for further analysis.

## API Documentation

Under construction.

## Citation

If PyMHD contributes to your work, please consider citing the package using the following BibTeX entry:

```bibtex
@misc{Hua2026,
  doi = {10.5281/zenodo.20749062},
  url = {https://zenodo.org/doi/10.5281/zenodo.20749062},
  author = {Hua, Yuyang},
  title = {PyMHD: Python package for post-processing MHD turbulence simulations},
  publisher = {Zenodo},
  year = {2026},
  copyright = {MIT License}
}
```
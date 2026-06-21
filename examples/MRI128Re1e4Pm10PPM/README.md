## Simulation Setup

This directory provides a small AthenaK shearing-box simulation of a turbulent MRI-driven dynamo with:

- Box size $1.0 \times 2.0 \times 0.5$
- Resolution $128 \times 128 \times 64$
- Isothermal EoS with $c_{\mathrm{s}}=1.0$ and angular velocity $\Omega_{0} = 1.0$, corresponding to a characteristic scale height $H=c_{\mathrm{s}}/\Omega_{0}=1.0$
- Keplerian shearing profile with $q=3/2$
- Zero-net-flux initial magnetic field configuration
- Physical viscosity $\nu=10^{-4}$ and resistivity $\eta=10^{-5}$, giving a magnetic Prandtl number of $\mathrm{Pm}=10$
- PPM reconstruction, RK3 time integrator, and HLLD Riemann solver

See `mri3d_unstratified.athinput` for the complete simulation configuration.

## Post-processing with PyMHD

The `quick.py` script provides a quick start demo of post-processing MHD turbulence simulations with PyMHD. It reads AthenaK outputs from `outputs/` and writes processed results to directories such as `spectra/`, `slices/`, and `numdiss(...)/`.

### AthenaK Outputs in `outputs/`

The `outputs/` directory contains standard AthenaK output files:

- History file `HGB.mhd.hst` (corresponding plots are in `hst/`)
- Binary outputs in `outputs/bin/`, including:
  - `HGB.prim.*.bin` for 2D slice visualization and turbulent spectra (configured by `mri3d_unstratified.athinput`)
  - Consecutive `HGB.nd.*.bin` files for *a posteriori* estimation of numerical dissipation (configured by `nd.athinput`)

### Turbulent Spectra in `spectra/`

The `spectra/` directory stores spectra computed with `pymhd.spectra` and visualized with `pymhd.plot.spc`:

- `shell.pdf`: shell-integrated kinetic and magnetic energy spectra
- `anisotropic.pdf`:
  - Top row: spectral energy distributions with respect to $k_{x}$, $k_{y}$, and $k_{z}$, respectively
  - Bottom row: shell-integrated component-wise spectra
- `spectra.pkl`: cache file containing the computed `EnergySpectra` object (generated after running `quick.py`)

### 2D Slice Plots in `slices/`

The `slices/` directory stores 2D slices visualized with `pymhd.plot.slc`, including:

- Density $\rho$ in `slices/rho/`
- Components of velocity $\bm{u}$ in `slices/{Vx, Vy, Vz}/`
- Components of magnetic field $\bm{B}$ in `slices/{Bx, By, Bz}/`

Each subdirectory contains snapshots at multiple output times. The slices clearly show the anisotropy induced by the background shear flow.

### Numerical Dissipation Analysis in `numdiss(...)/`

The `numdiss(...)/` directory stores results from the *a posteriori* analysis of numerical dissipation:

- `slices/`: 2D slices of physical and numerical dissipation terms/rates
- `histograms/`: histograms of numerical dissipation statistics
- `*.pdf`: physical and numerical dissipation spectra
- `{nd, ds}.pkl`: cache files containing the computed `NumericalDissipation` and `DissipationSpectra` objects (generated after running `quick.py`)

It is worth noting that results obtained with two different schemes (i.e., TENO7-M and TCS7-M) are nearly identical, indicating that both methods are sufficiently accurate for the high-fidelity estimation of numerical dissipation.
## Simulation Setup

This directory provides a small AthenaK simulation of driven MHD turbulence with:

- Box size $1.0^3$
- Resolution $128^3$
- Isothermal EoS with $c_{\mathrm{s}}=10.0$
- Background magnetic field $\bm{B}_0 = B_0 \bm{e}_z$ with $\beta_0=10.0$
- Physical viscosity and resistivity $\nu=\eta=10^{-4}$
- 2nd-order PLM reconstruction, RK2 time integrator, and HLLD Riemann solver

See `turb.athinput` for the complete simulation configuration.

## Post-processing with PyMHD

The `quick.py` script provides a quick start demo of post-processing MHD turbulence simulations with PyMHD. It reads AthenaK outputs from `outputs/` and writes processed results to directories such as `spectra/`, `slices/`, and `numdiss(...)/`.

### AthenaK Outputs in `outputs/`

The `outputs/` directory contains standard AthenaK output files:

- History file `Turb.mhd.hst` (corresponding plots are in `hst/`)
- Binary outputs in `outputs/bin/`, including:
  - `Turb.prim.*.bin` for 2D slice visualization and turbulent spectra (configured by `turb.athinput`)
  - Consecutive `Turb.nd.{acc,prim}.*.bin` files for *a posteriori* estimation of numerical dissipation (configured by `nd.athinput`)

### Turbulent Spectra in `spectra/`

The `spectra/` directory stores spectra computed with `pymhd.spectra` and visualized with `pymhd.plot.spc`:

- `shell.pdf`: shell-integrated kinetic and magnetic energy spectra
- `axisymmetric.pdf`:
  - Top row: spectral energy distributions with respect to $k_{\parallel}$ and $k_{\perp}$, respectively
  - Bottom row: shell-integrated component-wise spectra (parallel and perpendicular to $\bm{B}_0$)
- `spectra.pkl`: cache file containing the computed `EnergySpectra` object (generated after running `quick.py`)

### 2D Slice Plots in `slices/`

The `slices/` directory stores 2D slices visualized with `pymhd.plot.slc`, including:

- Density $\rho$ in `slices/rho/`
- Velocity $\bm{u}$ in `slices/V/`
- Magnetic field $\bm{B}$ in `slices/B/`
- Current density $\bm{J}$ in `slices/J/`
- Combined plots in `slices/all/`

Each subdirectory contains snapshots at multiple output times. The slices, especially those of $\bm{J}$, clearly show the anisotropy induced by the background $\bm{B}_0$ field.

### Numerical Dissipation Analysis in `numdiss(...)/`

The `numdiss(...)/` directory stores results from the *a posteriori* analysis of numerical dissipation:

- `slices/`: 2D slices of physical and numerical dissipation terms/rates
- `histograms/`: histograms of numerical dissipation statistics
- `*.pdf`: physical and numerical dissipation spectra
- `{nd, ds}.pkl`: cache files containing the computed `NumericalDissipation` and `DissipationSpectra` objects (generated after running `quick.py`)

It is worth noting that results obtained with two different schemes (i.e., TENO7-M and TCS7-M) are nearly identical, indicating that both methods are sufficiently accurate for the high-fidelity estimation of numerical dissipation.
# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/numdiss.py
----------------

Implements the framework for the analysis of numerical dissipation in (M)HD turbulence.
    - NumericalDissipation class: computes and stores numerical dissipation terms
    - DissipationSpectra class: computes and stores physical and numerical dissipation spectra
"""

import numpy as np

from pathlib import Path
import pickle

from typing import Any, Sequence

from functools import partial

import time

from .turbulence import ScalarField, VectorField, Vector, Turbulence
from .derivatives import derivative, Algorithm
from .spectra import EnergySpectra, Spectrum

# Set flush=True to avoid buffer output
print = partial(print, flush=True)

def calculateFornbergWeights(
    times: Sequence[float], t0: float, M: int
) -> np.ndarray:
    """Calculate Fornberg finite difference weights

    Calculate finite difference weights on non-uniform grid with Fornberg's algorithm

    References
    ----------
    [1] Fornberg, Bengt. Mathematics of computation 51.184 (1988): 699-706.

    Parameters
    ----------
    times : time sequence
    t0    : target time
    M     : maximum derivative order

    Returns
    -------
    array of weights c[m, n, i], where:
        - m: derivative order (0 to M)
        - n: number of data points - 1 (0 to N)
        - i: sum index (0 to n)

    Usage
    -----
    f^(m)(t0) ≈ sum_i=0^n c[m, n, i] * f(times[i])
    """
    N = len(times) - 1
    c = np.zeros((M + 1, N + 1, N + 1))
    c[0, 0, 0] = 1.0
    c1 = 1.0

    for n in range(1, N + 1):
        c2 = 1.0
        for i in range(n):
            c3 = times[n] - times[i]
            c2 *= c3

            if n <= M:
                c[n, n - 1, i] = 0

            for m in range(0, min(n, M) + 1):
                c[m, n, i] = ((times[n] - t0) * c[m, n - 1, i] - m * c[m - 1, n - 1, i]) / c3

        for m in range(0, min(n, M) + 1):
            c[m, n, n] = c1 / c2 * (m * c[m - 1, n - 1, n - 1] - (times[n - 1] - t0) * c[m, n - 1, n - 1])

        c1 = c2

    return c


def computeTimeDerivative(
    turbulence: Turbulence, t0: float
) -> tuple[VectorField, VectorField | None]:
    """Calculate time derivative of fields at t0

    Parameters
    ----------
    turbulence : Turbulence object containing consecutive time steps
    t0         : target time for calculating derivative

    Returns
    -------
    Vdot : time derivative of velocity field, dV/dt
    Bdot : time derivative of magnetic field, dB/dt; None for hydro
    """
    times = turbulence.times
    N = len(times) - 1  # Order of accuracy

    coeffs = calculateFornbergWeights(times, t0, 1)[1, N, :]

    Vdot = sum((c * V for c, V in zip(coeffs, turbulence.Vs)), turbulence.Vs[0] * 0)
    Bdot = None
    if turbulence.type != 'hydro':
        Bdot = sum((c * B for c, B in zip(coeffs, turbulence.Bs)), turbulence.Bs[0] * 0)

    return Vdot, Bdot


class NumericalDissipation:
    """Data container for numerical dissipation analysis

    Computes and stores numerical dissipation terms in MHD turbulence.

    Attributes
    ----------
    type       : turbulence type ('SSD', 'Bx', 'Bz', 'MRI', or 'hydro')
    nu         : kinematic viscosity
    eta        : resistivity (magnetic diffusivity)
    outputdir  : output directory name

    phyVisTerm : VectorField, ν∇·T, where T = ρ[∇u+(∇u)ᵀ-(2/3)(∇·u)I] is the stress tensor
    divStressT : VectorField, ∇·T (divergence of stress tensor)
    phyResTerm : VectorField, η∇²B
    LaplacianB : VectorField, ∇²B
    numVisTerm : VectorField, D^num_vis
    numResTerm : VectorField, D^num_res

    rho        : ScalarField, density field
    V          : VectorField, velocity field
    B          : VectorField, magnetic field

    numVisRate : ScalarField, numerical viscous   dissipation rate, V @ numVisTerm
    numResRate : ScalarField, numerical resistive dissipation rate, B @ numResTerm
    VdotStress : ScalarField, V·(∇·T), V @ divStressT
    BdotLaplaB : ScalarField, B·(∇²B), B @ LaplacianB
    phyVisRate : ScalarField, physical viscous   dissipation rate, V @ phyVisTerm
    phyResRate : ScalarField, physical resistive dissipation rate, B @ phyResTerm
    """
    @staticmethod
    def alg2dir(algorithm: Algorithm) -> str:

        method  = algorithm.method.upper()
        stencil = algorithm.stencil
        CT      = algorithm.CT

        return {
            'WENO'    : f'numdiss(WENO{stencil})',
            'TENO'    : f'numdiss(TENO{stencil}-M,CT={CT})',
            'TCS'     : f'numdiss(TCS7-M,CT={CT})',
            'CENTRAL' : 'numdiss(CENTRAL)',
            'SPECTRAL': 'numdiss(SPECTRAL)'
        }[method]

    def load(self, path: Path) -> bool:
        """Load cached result if metadata matches current request."""
        if not path.is_file():
            return False

        print(f"Loading NumericalDissipation cache from {path} ...")

        try:
            with path.open('rb') as f:
                obj = pickle.load(f)
        except Exception as exc:
            print(f"Failed to load cache: {exc}. Recomputing...\n")
            return False

        self.__dict__.update(obj.__dict__)
        print("NumericalDissipation cache loaded.\n")
        return True

    def cache(self, path: Path) -> None:
        """Save computed result with low-overhead pickle serialization."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"Failed to save cache: {exc}\n")
            return

        print(f"NumericalDissipation cache saved to ./{path}\n")

    def __init__(
        self,
        turbulence: Turbulence | None = None,
        algorithm : Algorithm = Algorithm(method = 'TENO', stencil = 7, CT = 0.01)
    ) -> None:
        """Compute numerical dissipation

        Parameters
        ----------
        turbulence: Turbulence or None, input Turbulence object
        algorithm : high-order scheme in L^{OH}, defaults to TENO7-M
        """
        self.outputdir = self.alg2dir(algorithm)

        # if cache exists, load it
        path = Path(self.outputdir) / "cache.pkl"
        if self.load(path):
            return

        if turbulence is None:
            raise ValueError("NumericalDissipation requires turbulence when cache is not valid.")

        self.type = turbulence.type
        self.Nx, self.Ny, self.Nz = turbulence.Nx, turbulence.Ny, turbulence.Nz
        self.Lx, self.Ly, self.Lz = turbulence.Lx, turbulence.Ly, turbulence.Lz

        self.nu  = turbulence.nu
        self.eta = turbulence.eta

        print("┌──────────────────────────────────────┐")
        print("│                                      │")
        print("│    Numerical Dissipation Analysis    │")
        print("│                                      │")
        print("└──────────────────────────────────────┘")

        print("")
        print("═════════════ Computation ══════════════")
        print("")

        start_time = time.time()
        results = self.compute(turbulence, algorithm)
        self.phyResTerm = results["phyResTerm"]
        self.LaplacianB = results["LaplacianB"]
        self.phyVisTerm = results["phyVisTerm"]
        self.divStressT = results["divStressT"]
        self.numVisTerm = results["numVisTerm"]
        self.numResTerm = results["numResTerm"]
        self.rho        = results["rho"]
        self.V          = results["V"]
        self.B          = results["B"]
        self.numVisRate = results["numVisRate"]
        self.numResRate = results["numResRate"]
        self.VdotStress = results["VdotStress"]
        self.BdotLaplaB = results["BdotLaplaB"]
        self.phyVisRate = results["phyVisRate"]
        self.phyResRate = results["phyResRate"]
        self.cache(path)
        end_time = time.time()

    @staticmethod
    def compute(
        turbulence: Turbulence,
        algorithm : Algorithm,
    ) -> dict[str, Any]:

        center = len(turbulence.times) // 2
        t0 = turbulence.times[center]

        rho = turbulence.rhos[center]
        p   = turbulence.ps[center]
        V   = turbulence.Vs[center]
        Vdot, Bdot = computeTimeDerivative(turbulence, t0)

        acc = None
        if turbulence.type != 'MRI':
            accs = getattr(turbulence, 'accs', None)
            if accs is None or len(accs) == 0:
                raise ValueError("NumericalDissipation for forced turbulence requires acceleration field.")
            acc = accs[center]

        nu  = turbulence.nu
        eta = turbulence.eta

        if turbulence.solver == 'FVM':
            print(f"Converting cell averages to cell centers...")
            average2center = partial(derivative.average2center, algorithm=algorithm)
            start_time = time.time()

            rho  = average2center(rho)
            p    = average2center(p)
            V    = average2center(V)
            Vdot = average2center(Vdot)

            end_time = time.time()
            print(f"rho, p, V, Vdot converted! Time: {end_time - start_time:.2f} s\n")

        print(f"Computing numerical dissipation terms...")
        start_time = time.time()

        Dx        = partial(derivative.Dx, algorithm=algorithm)
        Dy        = partial(derivative.Dy, algorithm=algorithm)
        Dz        = partial(derivative.Dz, algorithm=algorithm)
        grad      = partial(derivative.grad, algorithm=algorithm)
        div       = partial(derivative.div, algorithm=algorithm)
        curl      = partial(derivative.curl, algorithm=algorithm)
        laplacian = partial(derivative.laplacian, algorithm=algorithm)

        Nx, Ny, Nz = rho.data.shape
        box = rho.box
        Lx, Ly, Lz = box
        dx, dy, dz = rho.dx, rho.dy, rho.dz

        xs = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False) + dx / 2
        ys = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False) + dy / 2
        zs = np.linspace(-Lz / 2, Lz / 2, Nz, endpoint=False) + dz / 2
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        x = ScalarField(X, box)

        # Unit vectors
        Ex = Vector(1, 0, 0)
        Ey = Vector(0, 1, 0)
        Ez = Vector(0, 0, 1)

        Vx = ScalarField(V.x, V.box)
        Vy = ScalarField(V.y, V.box)
        Vz = ScalarField(V.z, V.box)

        B: VectorField | None = None
        if turbulence.type != 'hydro':
            B = turbulence.Bs[center]

        if turbulence.type in ('SSD', 'Bx', 'Bz'):

            assert Bdot is not None
            assert B is not None
            J = curl(B)

            assert acc is not None  # guaranteed by __init__

            Faraday    = curl(V ** B)
            LaplacianB = laplacian(B)

            convective = rho * (Vx * Dx(V) + Vy * Dy(V) + Vz * Dz(V))
            pressure   = -grad(p)
            Lorentz    = J ** B

            Tx = rho * (grad(Vx) + Dx(V) - (2 / 3) * div(V) * Ex)
            Ty = rho * (grad(Vy) + Dy(V) - (2 / 3) * div(V) * Ey)
            Tz = rho * (grad(Vz) + Dz(V) - (2 / 3) * div(V) * Ez)

            divStressT = div(Tx) * Ex + div(Ty) * Ey + div(Tz) * Ez
            phyResTerm = eta * LaplacianB
            phyVisTerm = nu * divStressT

            # ===== Numerical dissipation =====
            NavierStokesLHS = rho * Vdot + convective
            NavierStokesRHS = pressure + Lorentz + phyVisTerm + rho * acc

            numVisTerm = NavierStokesLHS - NavierStokesRHS
            numResTerm = Bdot - Faraday - phyResTerm

        elif turbulence.type == 'MRI':

            assert Bdot is not None
            assert B is not None
            Bx = ScalarField(B.x, B.box)
            J = curl(B)

            Omega = turbulence.Omega
            q     = turbulence.q

            # ===== Magnetic induction equation =====
            Faraday1   = -q * Omega * Bx * Ey
            Faraday2   = q * Omega * x * Dy(B)
            Faraday3   = curl(V ** B)

            Faraday    = Faraday1 + Faraday2 + Faraday3
            phyResTerm = eta * laplacian(B)
            LaplacianB = laplacian(B)

            # ===== Navier-Stokes equation =====
            convective = rho * (Vx * Dx(V) + Vy * Dy(V) + Vz * Dz(V) - q * Omega * x * Dy(V) - q * Omega * Vx * Ey)
            pressure   = -grad(p)
            Coriolis   = -2 * rho * Omega * (Ez ** V)
            Lorentz    = J ** B

            Tx = rho * (grad(Vx) + Dx(V) - (2 / 3) * div(V) * Ex)
            Ty = rho * (grad(Vy) + Dy(V) - (2 / 3) * div(V) * Ey)
            Tz = rho * (grad(Vz) + Dz(V) - (2 / 3) * div(V) * Ez)
            divTx, divTy, divTz = div(Tx), div(Ty), div(Tz)
            divT       = divTx * Ex + divTy * Ey + divTz * Ez
            divStressT = divT
            phyVisTerm = nu * divT

            # ===== Numerical dissipation =====
            NavierStokesLHS = rho * Vdot + convective
            NavierStokesRHS = pressure + Coriolis + Lorentz + phyVisTerm

            numVisTerm = NavierStokesLHS - NavierStokesRHS
            numResTerm = Bdot - Faraday - phyResTerm


        elif turbulence.type == 'hydro':

            convective = rho * (Vx * Dx(V) + Vy * Dy(V) + Vz * Dz(V))
            pressure   = -grad(p)

            Tx = rho * (grad(Vx) + Dx(V) - (2 / 3) * div(V) * Ex)
            Ty = rho * (grad(Vy) + Dy(V) - (2 / 3) * div(V) * Ey)
            Tz = rho * (grad(Vz) + Dz(V) - (2 / 3) * div(V) * Ez)

            divStressT = div(Tx) * Ex + div(Ty) * Ey + div(Tz) * Ez
            phyVisTerm = nu * divStressT

            assert acc is not None
            NavierStokesLHS = rho * Vdot + convective
            NavierStokesRHS = pressure + phyVisTerm + rho * acc

            numVisTerm = NavierStokesLHS - NavierStokesRHS

            phyResTerm = None
            LaplacianB = None
            numResTerm = None

        else:
            raise ValueError(
                f"Unsupported type: {turbulence.type!r}; expected 'SSD', 'Bx', 'Bz', 'MRI', or 'hydro'."
            )

        end_time = time.time()
        print(f"Numerical dissipation computation completed! Time: {end_time - start_time:.2f} s")

        numVisRate = V @ numVisTerm
        VdotStress = V @ divStressT
        phyVisRate = V @ phyVisTerm

        if B is not None:
            assert numResTerm is not None
            assert phyResTerm is not None
            assert LaplacianB is not None
            numResRate = B @ numResTerm
            BdotLaplaB = B @ LaplacianB
            phyResRate = B @ phyResTerm
        else:
            numResRate = None
            BdotLaplaB = None
            phyResRate = None

        results: dict[str, Any] = {
            "phyResTerm": phyResTerm,
            "LaplacianB": LaplacianB,
            "phyVisTerm": phyVisTerm,
            "divStressT": divStressT,
            "numVisTerm": numVisTerm,
            "numResTerm": numResTerm,
            "rho"       : rho,
            "V"         : V,
            "B"         : B,
            "numVisRate": numVisRate,
            "numResRate": numResRate,
            "VdotStress": VdotStress,
            "BdotLaplaB": BdotLaplaB,
            "phyVisRate": phyVisRate,
            "phyResRate": phyResRate,
        }
        return results


def computeSpectra(
    field1: VectorField | None, field2: VectorField | None
) -> tuple[Spectrum, Spectrum, Spectrum, Spectrum]:
    r"""Compute component-wise and total spectra

    Parameters
    ----------
    field1: VectorField, the first field
    field2: VectorField, the second field

    Returns
    -------
    tuple[Spectrum, Spectrum, Spectrum, Spectrum]: the component-wise and total spectra

    Raises
    ------
    ValueError: if field1 or field2 is None or if field1 and field2 do not match
    """
    if field1 is None or field2 is None:
        raise ValueError("computeSpectra(): field1 and field2 must not be None.")

    Nx1, Ny1, Nz1 = field1.x.shape
    Nx2, Ny2, Nz2 = field2.x.shape

    if (Nx1, Ny1, Nz1) != (Nx2, Ny2, Nz2):
        raise ValueError(
            "computeSpectra(): field1 and field2 must have the same resolution, "
            f"got {(Nx1, Ny1, Nz1)} and {(Nx2, Ny2, Nz2)}."
        )
    if field1.box != field2.box:
        raise ValueError(
            "computeSpectra(): field1 and field2 must share the same box, "
            f"got {field1.box} and {field2.box}."
        )

    dV  = field1.dxdydz
    box = field1.box

    f1x = np.fft.fftn(field1.x) * dV
    f1y = np.fft.fftn(field1.y) * dV
    f1z = np.fft.fftn(field1.z) * dV

    f2x = np.fft.fftn(field2.x) * dV
    f2y = np.fft.fftn(field2.y) * dV
    f2z = np.fft.fftn(field2.z) * dV

    Sx  = Spectrum(np.real(np.conjugate(f1x) * f2x), box)
    Sy  = Spectrum(np.real(np.conjugate(f1y) * f2y), box)
    Sz  = Spectrum(np.real(np.conjugate(f1z) * f2z), box)
    Stot = Sx + Sy + Sz

    return (Sx, Sy, Sz, Stot)


class DissipationSpectra:
    """Container for dissipation spectra used by numerical dissipation plots.

    Attributes
    ----------
    Lx, Ly, Lz: float, size of the box in x, y, z directions
    nu, eta   : float, kinematic viscosity and resistivity (magnetic diffusivity)

    Ek: EnergySpectra, cached energy spectra (component and total E_{kin} / E_{mag}).

    For nd.type == 'SSD', only the summed spectra are specified:
        - eNumVis, eNumRes, ePhyVis, ePhyRes, eTotVis, eTotRes.

    Otherwise (e.g. 'Bx', 'Bz', 'MRI'):
        - xNumVis, yNumVis, zNumVis: component-wise numerical viscous   dissipation spectra
        - xNumRes, yNumRes, zNumRes: component-wise numerical resistive dissipation spectra
        - xPhyVis, yPhyVis, zPhyVis: component-wise physical  viscous   dissipation spectra
        - xPhyRes, yPhyRes, zPhyRes: component-wise physical  resistive dissipation spectra

        - eNumVis, eNumRes: sums of the three num components
        - ePhyVis, ePhyRes: sums of the three phy components

        - xTotVis, yTotVis, zTotVis: per-component viscous  total  (phy + num)
        - xTotRes, yTotRes, zTotRes: per-component resistive total (phy + num)
        - eTotVis, eTotRes: summed total dissipation

        - xUdivT, yUdivT, zUdivT, eUdivT: physical viscous   dissipation spectra without nu
        - xBLapB, yBLapB, zBLapB, eBLapB: physical resistive dissipation spectra without eta
    """
    def cache(self, path: Path) -> None:

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"Failed to save dissipation spectra cache: {exc}\n")
            return

        print(f"DissipationSpectra cache saved to ./{path}\n")

    def __init__(
        self,
        nd: NumericalDissipation,
        Ek: EnergySpectra,
    ) -> None:

        self.box = nd.V.box
        self.Lx, self.Ly, self.Lz = self.box

        self.nu  = nd.nu
        self.eta = nd.eta
        self.Ek  = Ek

        if nd.type == "SSD":
            *_, self.eNumVis = computeSpectra(nd.V, nd.numVisTerm)
            *_, self.eNumRes = computeSpectra(nd.B, nd.numResTerm)
            *_, self.ePhyVis = computeSpectra(nd.V, nd.phyVisTerm)
            *_, self.ePhyRes = computeSpectra(nd.B, nd.phyResTerm)

            self.eTotVis, self.eTotRes = self.ePhyVis + self.eNumVis, self.ePhyRes + self.eNumRes

        else:
            xNumVis, yNumVis, zNumVis, eNumVis = computeSpectra(nd.V, nd.numVisTerm)
            xNumRes, yNumRes, zNumRes, eNumRes = computeSpectra(nd.B, nd.numResTerm)
            xPhyVis, yPhyVis, zPhyVis, ePhyVis = computeSpectra(nd.V, nd.phyVisTerm)
            xPhyRes, yPhyRes, zPhyRes, ePhyRes = computeSpectra(nd.B, nd.phyResTerm)

            self.xNumVis, self.yNumVis, self.zNumVis = xNumVis, yNumVis, zNumVis
            self.xNumRes, self.yNumRes, self.zNumRes = xNumRes, yNumRes, zNumRes
            self.xPhyVis, self.yPhyVis, self.zPhyVis = xPhyVis, yPhyVis, zPhyVis
            self.xPhyRes, self.yPhyRes, self.zPhyRes = xPhyRes, yPhyRes, zPhyRes

            self.eNumVis, self.eNumRes = eNumVis, eNumRes
            self.ePhyVis, self.ePhyRes = ePhyVis, ePhyRes

            self.xTotVis, self.yTotVis, self.zTotVis = xPhyVis + xNumVis, yPhyVis + yNumVis, zPhyVis + zNumVis
            self.xTotRes, self.yTotRes, self.zTotRes = xPhyRes + xNumRes, yPhyRes + yNumRes, zPhyRes + zNumRes
            self.eTotVis, self.eTotRes = ePhyVis + eNumVis, ePhyRes + eNumRes

            xUdivT, yUdivT, zUdivT, eUdivT = computeSpectra(nd.V, nd.divStressT)
            xBLapB, yBLapB, zBLapB, eBLapB = computeSpectra(nd.B, nd.LaplacianB)

            self.xUdivT, self.yUdivT, self.zUdivT = xUdivT, yUdivT, zUdivT
            self.xBLapB, self.yBLapB, self.zBLapB = xBLapB, yBLapB, zBLapB
            self.eUdivT, self.eBLapB = eUdivT, eBLapB

        self.cache(Path(nd.outputdir) / "nd.spectra.pkl")
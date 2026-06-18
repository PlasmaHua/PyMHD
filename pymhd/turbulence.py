# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/turbulence.py
-------------------

Implements the basic data containers for PyMHD:
    - ScalarField class: data container for scalar fields, e.g. density, pressure, etc.
    - VectorField class: data container for vector fields, e.g. velocity, magnetic field, etc.
    - Turbulence  class: data container for turbulence data from (M)HD simulations.
"""

from __future__ import annotations

import numpy as np

from typing import Any, Sequence, Literal, TypeGuard, TypeAlias, overload

# Simple implementation of Scalar and Vector classes
Scalar: TypeAlias = float | int

class Vector:
    """Vector class

    Data container for vectors, e.g. unit vector e_z = Vector(0, 0, 1).

    Attributes
    ----------
    x, y, z (float) : x, y, z components
    """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other) -> VectorField:
        """Multiplication

        Supported operations
        --------------------
        Vector * ScalarField -> VectorField
        """
        if isinstance(other, ScalarField):
            return VectorField(
                self.x * other.data,
                self.y * other.data,
                self.z * other.data,
                other.box
            )

        return NotImplemented

    def __rmul__(self, other) -> VectorField:
        """Right multiplication

        Supported operations
        --------------------
        ScalarField * Vector -> VectorField
        """
        if isinstance(other, ScalarField):
            return VectorField(
                other.data * self.x,
                other.data * self.y,
                other.data * self.z,
                other.box
            )

        return NotImplemented

# ========== ScalarField ==========

class ScalarField:
    """ScalarField class

    Data container for scalar fields, e.g. density, pressure, etc.

    Attributes
    ----------
    data  (np.ndarray) : scalar field data
    box        (tuple) : box sizes
    Lx, Ly, Lz (float) : box size in x, y, z directions
    Nx, Ny, Nz   (int) : resolutions in x, y, z directions
    dx, dy, dz (float) : grid size in x, y, z directions
    dxdydz     (float) : volume of a grid cell

    Operations
    ----------
    1. NumPy functions : e.g. np.sqrt(ScalarField) -> ScalarField
    2. Plus / Minus    : ScalarField + ScalarField -> ScalarField
    3. Multiplication  : ScalarField * ScalarField -> ScalarField
                         ScalarField * Scalar      -> ScalarField
                         Scalar      * ScalarField -> ScalarField
    4. Division        : ScalarField / ScalarField -> ScalarField
                         ScalarField / Scalar      -> ScalarField
                         Scalar      / ScalarField -> ScalarField
    5. Power           : ScalarField ** Scalar     -> ScalarField
    """
    def __init__(self, data: np.ndarray, box: tuple[float, float, float]):
        self.data = data
        self.box  = box
        self.Lx, self.Ly, self.Lz = box

        self.Nx, self.Ny, self.Nz = data.shape
        self.dx, self.dy, self.dz = self.Lx/self.Nx, self.Ly/self.Ny, self.Lz/self.Nz
        self.dxdydz = self.dx * self.dy * self.dz

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support NumPy functions, e.g. np.sqrt"""
        processed_inputs = []
        for input in inputs:
            if isinstance(input, ScalarField):
                processed_inputs.append(input.data)
            else:
                processed_inputs.append(input)

        results = getattr(ufunc, method)(*processed_inputs, **kwargs)

        if isinstance(results, tuple):
            return tuple(
                ScalarField(result, self.box)
                if isinstance(result, np.ndarray) else result
                for result in results
            )
        elif isinstance(results, np.ndarray):
            return ScalarField(results, self.box)
        else:
            return results

    def __add__(self, other) -> ScalarField:
        """Addition

        Supported operations
        --------------------
        ScalarField + ScalarField -> ScalarField
        """
        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            return ScalarField(self.data + other.data, self.box)

        return NotImplemented

    def __neg__(self) -> ScalarField:
        """Negation

        Supported operations
        --------------------
        -ScalarField -> ScalarField
        """
        return ScalarField(-self.data, self.box)

    def __sub__(self, other) -> ScalarField:
        """Subtraction

        Supported operations
        --------------------
        ScalarField - ScalarField -> ScalarField
        """
        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            return ScalarField(self.data - other.data, self.box)

        return NotImplemented

    def __mul__(self, other: Scalar | ScalarField) -> ScalarField:
        """Left multiplication

        Supported operations
        --------------------
        ScalarField * Scalar      -> ScalarField
        ScalarField * ScalarField -> ScalarField
        """
        if isinstance(other, Scalar):
            return ScalarField(self.data * other, self.box)

        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            return ScalarField(self.data * other.data, self.box)

        # Let VectorField handle ScalarField * VectorField
        return NotImplemented

    def __rmul__(self, other: Scalar) -> ScalarField:
        """Right multiplication

        Supported operations
        --------------------
        Scalar * ScalarField -> ScalarField
        """
        if isinstance(other, Scalar):
            return ScalarField(other * self.data, self.box)

        # Let VectorField handle VectorField * ScalarField
        return NotImplemented

    def __truediv__(self, other: Scalar | ScalarField) -> ScalarField:
        """Left division

        Supported operations
        --------------------
        1. ScalarField / Scalar      -> ScalarField
        2. ScalarField / ScalarField -> ScalarField
        """
        if isinstance(other, Scalar):
            return ScalarField(self.data / other, self.box)

        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            return ScalarField(self.data / other.data, self.box)

        return NotImplemented

    def __rtruediv__(self, other: Scalar) -> ScalarField:
        """Right division

        Supported operations
        --------------------
        Scalar / ScalarField -> ScalarField
        """
        if isinstance(other, Scalar):
            return ScalarField(other / self.data, self.box)

        # Let VectorField handle VectorField / ScalarField
        return NotImplemented

    def __pow__(self, other: Scalar) -> ScalarField:
        """Power

        Supported operations
        --------------------
        ScalarField ** Scalar -> ScalarField
        """
        if isinstance(other, Scalar):
            return ScalarField(self.data ** other, self.box)

        return NotImplemented

    @property
    def mean(self) -> float:
        """Volume average of the scalar field."""
        return float(np.mean(self.data))

    @property
    def std(self) -> float:
        """Volume standard deviation of the scalar field."""
        return float(np.std(self.data))

    @property
    def total(self) -> float:
        """Volume integral of the scalar field."""
        return float(np.sum(self.data) * self.dxdydz)


# ========== VectorField ==========

class VectorField:
    """VectorField class

    Data container for vector fields, e.g. velocity, magnetic field, etc.

    Attributes
    ----------
    x, y, z (np.ndarray) : x, y, z components
    box          (tuple) : box sizes
    Lx, Ly, Lz   (float) : box size in x, y, z directions
    Nx, Ny, Nz     (int) : resolutions in x, y, z directions
    dx, dy, dz   (float) : grid size in x, y, z directions
    dxdydz       (float) : volume of a grid cell
    norm   (ScalarField) : magnitude of the vector field

    Operations
    ----------
    1. Plus / Minus   : VectorField + VectorField -> VectorField,
                        VectorField - VectorField -> VectorField;
    2. Multiplication : VectorField * ScalarField -> VectorField,
                        ScalarField * VectorField -> VectorField,
                        Scalar      * VectorField -> VectorField,
                        VectorField * Scalar      -> VectorField;
    3. Dot Product    : VectorField @ VectorField -> ScalarField;
    4. Cross Product  : VectorField ** VectorField -> VectorField;
    5. Division       : VectorField / ScalarField -> VectorField,
                        VectorField / Scalar      -> VectorField.
    """
    def __init__(
        self,
        Vx : np.ndarray,
        Vy : np.ndarray,
        Vz : np.ndarray,
        box: tuple[float, float, float]
    ):

        if not Vx.shape == Vy.shape == Vz.shape:
            raise ValueError("x, y, z components must have the same shape")

        self.x = Vx
        self.y = Vy
        self.z = Vz
        self.box = box
        self.Lx, self.Ly, self.Lz = box

        self.Nx, self.Ny, self.Nz = Vx.shape
        self.dx, self.dy, self.dz = self.Lx/self.Nx, self.Ly/self.Ny, self.Lz/self.Nz
        self.dxdydz = self.dx * self.dy * self.dz

    @property
    def norm(self) -> ScalarField:
        return ScalarField(np.sqrt(self.x**2 + self.y**2 + self.z**2), self.box)

    def __add__(self, other: VectorField) -> VectorField:
        """Addition

        Supported operations
        --------------------
        VectorField + VectorField -> VectorField
        """
        if isinstance(other, VectorField) and assertMatchFields(self, other):
            return VectorField(self.x + other.x, self.y + other.y, self.z + other.z, self.box)

        return NotImplemented

    def __neg__(self) -> VectorField:
        """Negation

        Supported operations
        --------------------
        -VectorField -> VectorField
        """
        return VectorField(-self.x, -self.y, -self.z, self.box)

    def __sub__(self, other: VectorField) -> VectorField:
        """Subtraction

        Supported operations
        --------------------
        VectorField - VectorField -> VectorField
        """
        if isinstance(other, VectorField) and assertMatchFields(self, other):
            return VectorField(self.x - other.x, self.y - other.y, self.z - other.z, self.box)

        return NotImplemented

    def __mul__(self, other: Scalar | ScalarField) -> VectorField:
        """Left multiplication

        Supported operations
        --------------------
        1. VectorField * ScalarField -> VectorField
        2. VectorField * Scalar      -> VectorField
        """
        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            # Scalar product for ScalarField
            return VectorField(
                self.x * other.data,
                self.y * other.data,
                self.z * other.data,
                self.box
            )
        if isinstance(other, Scalar):
            # Scalar product for scalar
            return VectorField(
                self.x * other,
                self.y * other,
                self.z * other,
                self.box
            )

        return NotImplemented

    def __rmul__(self, other: Scalar | ScalarField) -> VectorField:
        """Right multiplication

        Supported operations
        --------------------
        1. ScalarField * VectorField -> VectorField
        2. Scalar      * VectorField -> VectorField
        """
        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            # Scalar product for ScalarField
            return VectorField(
                other.data * self.x,
                other.data * self.y,
                other.data * self.z,
                self.box
            )
        if isinstance(other, Scalar):
            # Scalar product for scalar
            return VectorField(
                other * self.x,
                other * self.y,
                other * self.z,
                self.box
            )

        return NotImplemented

    def __matmul__(self, other: Vector | VectorField) -> ScalarField:
        """Dot product

        Supported operations
        --------------------
        1. VectorField @ VectorField -> ScalarField
        2. VectorField @ Vector      -> ScalarField
        """
        if isinstance(other, VectorField) and assertMatchFields(self, other):
            return ScalarField(
                self.x * other.x + self.y * other.y + self.z * other.z,
                self.box
            )
        if isinstance(other, Vector):
            return ScalarField(
                self.x * other.x + self.y * other.y + self.z * other.z,
                self.box
            )

        return NotImplemented

    def __rmatmul__(self, other: Vector) -> ScalarField:
        """Right dot product

        Supported operations
        --------------------
        1. Vector @ VectorField -> ScalarField
        """
        if isinstance(other, Vector):
            return ScalarField(
                other.x * self.x + other.y * self.y + other.z * self.z,
                self.box
            )

        return NotImplemented

    def __truediv__(self, other: Scalar | ScalarField) -> VectorField:
        """Left division

        Supported operations
        --------------------
        1. VectorField / ScalarField -> VectorField
        2. VectorField / Scalar      -> VectorField
        """
        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            return VectorField(
                self.x / other.data,
                self.y / other.data,
                self.z / other.data,
                self.box
            )
        if isinstance(other, Scalar):
            return VectorField(
                self.x / other,
                self.y / other,
                self.z / other,
                self.box
            )

        return NotImplemented

    def __pow__(self, other: Vector | VectorField) -> VectorField:
        """Cross product

        Supported operations
        --------------------
        1. VectorField ** VectorField -> VectorField
        2. VectorField ** Vector      -> VectorField
        """
        if isinstance(other, VectorField) and assertMatchFields(self, other):
            return VectorField(
                self.y * other.z - self.z * other.y,
                self.z * other.x - self.x * other.z,
                self.x * other.y - self.y * other.x,
                self.box
            )
        elif isinstance(other, Vector):
            return VectorField(
                self.y * other.z - self.z * other.y,
                self.z * other.x - self.x * other.z,
                self.x * other.y - self.y * other.x,
                self.box
            )

        return NotImplemented

    def __rpow__(self, other: Vector) -> VectorField:
        """Right cross product

        Supported operations
        --------------------
        Vector ** VectorField -> VectorField
        """
        if isinstance(other, Vector):
            return VectorField(
                other.y * self.z - other.z * self.y,
                other.z * self.x - other.x * self.z,
                other.x * self.y - other.y * self.x,
                self.box
            )

        return NotImplemented


def assertMatchFields(
    field1: ScalarField | VectorField,
    field2: ScalarField | VectorField
) -> bool:
    """Assert that two fields have matching box dimensions and array shapes.

    Always returns True on success; raises ValueError on mismatch.
    Designed for use in operator guards, e.g.:
        ```python
        if isinstance(other, ScalarField) and assertMatchFields(self, other):
            ...
        ```

    Parameters
    ----------
    field1 : ScalarField | VectorField
    field2 : ScalarField | VectorField

    Returns
    -------
    bool : always True if the assertion passes

    Raises
    ------
    ValueError : if box dimensions or array shapes do not match
    """
    if field1.box != field2.box:
        raise ValueError("Box must match")

    matched = False

    if isinstance(field1, ScalarField) and isinstance(field2, ScalarField):
        matched = field1.data.shape == field2.data.shape

    elif isinstance(field1, VectorField) and isinstance(field2, VectorField):
        matched = (
            field1.x.shape == field2.x.shape and
            field1.y.shape == field2.y.shape and
            field1.z.shape == field2.z.shape
        )

    elif isinstance(field1, ScalarField) and isinstance(field2, VectorField):
        matched = field1.data.shape == field2.x.shape == field2.y.shape == field2.z.shape

    elif isinstance(field1, VectorField) and isinstance(field2, ScalarField):
        matched = (
            field1.x.shape == field2.data.shape and
            field1.y.shape == field2.data.shape and
            field1.z.shape == field2.data.shape
        )

    if not matched:
        raise ValueError("Array shapes must match")

    return True

def isMatchScalarFieldList(
    fields: Sequence[ScalarField | VectorField]
) -> TypeGuard[Sequence[ScalarField]]:
    """Check if all elements are ScalarField. Raise TypeError for mixed types."""
    if not fields:
        return False
    count = sum(isinstance(f, ScalarField) for f in fields)
    if 0 < count < len(fields):
        raise TypeError(
            f"Mixed field types: {count} ScalarField(s) and "
            f"{len(fields) - count} VectorField(s). All elements must be the same type."
        )
    if count > 0:
        for field in fields[1:]:
            assertMatchFields(fields[0], field)
    return count > 0

def isMatchVectorFieldList(
    fields: Sequence[ScalarField | VectorField]
) -> TypeGuard[Sequence[VectorField]]:
    """Check if all elements are VectorField. Raise TypeError for mixed types."""
    if not fields:
        return False
    count = sum(isinstance(f, VectorField) for f in fields)
    if 0 < count < len(fields):
        raise TypeError(
            f"Mixed field types: {count} VectorField(s) and "
            f"{len(fields) - count} ScalarField(s). All elements must be the same type."
        )
    if count > 0:
        for field in fields[1:]:
            assertMatchFields(fields[0], field)
    return count > 0


def sqrt(field: ScalarField) -> ScalarField:
    """Square root of a ScalarField

    Parameters
    ----------
    field: ScalarField

    Returns
    -------
    ScalarField, square root of the input ScalarField
    """
    if not isinstance(field, ScalarField):
        raise TypeError("sqrt() only supports ScalarField")

    return ScalarField(np.sqrt(field.data), field.box)

@overload
def avg(fields: Sequence[ScalarField]) -> ScalarField: ...

@overload
def avg(fields: Sequence[VectorField]) -> VectorField: ...

def avg(fields: Sequence[ScalarField | VectorField]) -> ScalarField | VectorField:
    """Average of fields

    Parameters
    ----------
    fields: Sequence[ScalarField | VectorField]

    Returns
    -------
    ScalarField | VectorField: Average of the input fields
    """
    if not fields:
        raise ValueError("Field list cannot be empty")

    box = fields[0].box

    if isMatchScalarFieldList(fields):
        stacked = np.stack([field.data for field in fields])
        avgdata = np.mean(stacked, axis=0)
        return ScalarField(avgdata, box)

    if isMatchVectorFieldList(fields):
        xs = np.stack([field.x for field in fields])
        ys = np.stack([field.y for field in fields])
        zs = np.stack([field.z for field in fields])

        avgx = np.mean(xs, axis=0)
        avgy = np.mean(ys, axis=0)
        avgz = np.mean(zs, axis=0)

        return VectorField(avgx, avgy, avgz, box)

    raise TypeError("Unsupported type, must be ScalarField or VectorField")

@overload
def std(fields: Sequence[ScalarField]) -> ScalarField: ...

@overload
def std(fields: Sequence[VectorField]) -> VectorField: ...

def std(fields: Sequence[ScalarField | VectorField]) -> ScalarField | VectorField:
    """Standard deviation of fields

    Parameters
    ----------
    fields: Sequence[ScalarField | VectorField]

    Returns
    -------
    ScalarField | VectorField: Standard deviation of the input fields
    """
    if not fields:
        raise ValueError("Field list cannot be empty")

    box = fields[0].box

    if isMatchScalarFieldList(fields):
        stacked = np.stack([field.data for field in fields])
        stddata = np.std(stacked, axis=0)
        return ScalarField(stddata, box)

    if isMatchVectorFieldList(fields):
        xs = np.stack([field.x for field in fields])
        ys = np.stack([field.y for field in fields])
        zs = np.stack([field.z for field in fields])

        stdx = np.std(xs, axis=0)
        stdy = np.std(ys, axis=0)
        stdz = np.std(zs, axis=0)

        return VectorField(stdx, stdy, stdz, box)

    raise TypeError("Unsupported type, must be ScalarField or VectorField")


# ========== Turbulence ==========

class Turbulence:
    """(M)HD Turbulence

    Currently supported types of turbulence:
        - SSD  : Small-scale dynamo, forced MHD turbulence with zero-net-flux B field
        - Bx   : Forced MHD turbulence with a background Bx field.
        - Bz   : Forced MHD turbulence with a background Bz field.
        - MRI  : MRI-driven turbulence from shearing box simulations.
        - hydro: Forced hydrodynamic turbulence.

    Currently supported equation of state (EoS):
        - isothermal EoS with sound speed Cs
        - adiabatic EoS with adiabatic index gamma

    Attributes
    ----------
    case  (str)               : case name
    type  (str)               : turbulence type, options: 'SSD', 'Bx', 'Bz', 'MRI', 'hydro'
    solver(str)               : solver scheme, options: 'FVM', 'FDM', 'SPECTRAL'

    EoS   (str)               : equation of state, options: 'isothermal', 'adiabatic', 'incompressible'
    Cs    (float | None)      : isothermal sound speed (for isothermal EoS)
    gamma (float | None)      : adiabatic index (for adiabatic EoS)

    times (list[float])       : time list
    rhos  (list[ScalarField]) : list of density fields
    ps    (list[ScalarField]) : list of pressure fields
    Vs    (list[VectorField]) : list of velocity fields
    Bs    (list[VectorField]) : list of magnetic fields
    accs  (list[VectorField]) : list of acceleration fields (driving forces)

    nu    (float)             : kinematic viscosity
    eta   (float)             : magnetic diffusivity (resistivity)
    Pm    (float)             : magnetic Prandtl number, Pm = nu / eta

    Omega (float)             : angular velocity [rad/s]
    q     (float)             : shear parameter

    Derived quantities
    ------------------
    Nx, Ny, Nz (int)          : grid resolution
    Lx, Ly, Lz (float)        : box size
    wVs   (list[VectorField]) : density-weighted velocity fields
    avgBs (list[Vector])      : mean magnetic fields
    KEs   (list[float])       : total kinetic energies
    MEs   (list[float])       : total magnetic energies
    drhos (list[float])       : relative density fluctuations
    """
    type  : Literal['SSD', 'Bx', 'Bz', 'MRI', 'hydro']
    solver: Literal["FVM", "FDM", "SPECTRAL"]
    EoS   : Literal["isothermal", "adiabatic", "incompressible"]

    def __init__(
        self,
        case  : str,
        type  : Literal['SSD', 'Bx', 'Bz', 'MRI', 'hydro'],
        solver: Literal["FVM", "FDM", "SPECTRAL"],
        EoS   : Literal["isothermal", "adiabatic", "incompressible"],
        Cs    : float | None,
        gamma : float | None,
        rhos  : Sequence[ScalarField | None],
        ps    : Sequence[ScalarField | None],
        Vs    : Sequence[VectorField],
        Bs    : Sequence[VectorField | None],
        accs  : Sequence[VectorField | None],
        times : Sequence[float],
        nu    : float = 0,
        eta   : float = 0,
        Omega : float = 0,
        q     : float = 0,
    ):
        self.case   = case
        self.type   = type
        self.solver = solver
        self.Vs     = list(Vs)
        self.times  = list(times)
        self.Omega  = Omega
        self.q      = q
        self.EoS    = EoS

        if type not in ['SSD', 'Bx', 'Bz', 'MRI', 'hydro']:
            raise ValueError("Unsupported type, available options: SSD, Bx, Bz, MRI, hydro")

        if solver not in ['FVM', 'FDM', 'SPECTRAL']:
            raise ValueError("Unsupported scheme, available options: FVM, FDM, SPECTRAL")

        snapshots = len(self.times)
        if snapshots == 0:
            raise ValueError("No data found.")

        def assertMatchListLength(name: str, seq: Sequence[Any]):
            if len(seq) != snapshots:
                raise ValueError(f"Length mismatch: {snapshots} snapshots, {len(seq)} {name}.")

        assertMatchListLength("Vs", self.Vs)

        # Assert that all Fields have matching box and shape.
        V0 = self.Vs[0]
        for V in self.Vs[1:]:
            assertMatchFields(V0, V)

        if type != 'hydro':
            assertMatchListLength("Bs", Bs)
            for V, B in zip(self.Vs, Bs):
                if B is not None:
                    assertMatchFields(V, B)

        if EoS in ['isothermal', 'adiabatic']:
            assertMatchListLength("rhos", rhos)
            for V, rho in zip(self.Vs, rhos):
                if rho is not None:
                    assertMatchFields(V, rho)

        if EoS == 'adiabatic':
            assertMatchListLength("ps", ps)
            for V, p in zip(self.Vs, ps):
                if p is not None:
                    assertMatchFields(V, p)

        if type != 'MRI':
            if len(accs) not in [0, snapshots]:
                raise ValueError(f"Length mismatch: expected 0 or {snapshots} accs, got {len(accs)}.")
            for V, acc in zip(self.Vs, accs):
                if acc is not None:
                    assertMatchFields(V, acc)

        if EoS == 'isothermal':
            if Cs is None:
                raise ValueError("Isothermal EoS requires Cs value.")
            if any(rho is None for rho in rhos):
                raise ValueError("Isothermal EoS requires all rho snapshots to be provided.")

            self.rhos  = [rho for rho in rhos if rho is not None]
            self.Cs    = Cs
            self.ps    = [self.Cs**2 * rho for rho in self.rhos]

        elif EoS == 'adiabatic':
            if gamma is None:
                raise ValueError("Adiabatic EoS requires gamma value.")
            if any(rho is None for rho in rhos) or any(p is None for p in ps):
                raise ValueError("Adiabatic EoS requires all rho and p snapshots to be provided.")

            self.rhos  = [rho for rho in rhos if rho is not None]
            self.ps    = [p for p in ps if p is not None]
            self.gamma = gamma

        elif EoS == 'incompressible':
            # TODO: add support for incompressible EoS
            raise NotImplementedError("Incompressible EoS is not supported yet.")

        else:
            raise ValueError("Unsupported EoS, available options: isothermal, adiabatic, incompressible")

        if type != 'hydro':
            if any(B is None for B in Bs):
                raise ValueError("MHD turbulence requires all B snapshots to be provided.")
            self.Bs = [B for B in Bs if B is not None]

        if type != 'MRI':
            self.accs = [acc for acc in accs if acc is not None]

        self.nu  = nu
        self.eta = eta
        self.Pm  = nu / eta if eta != 0 else np.inf
        if nu == 0 and eta == 0:
            self.Pm = np.nan

        self.Nx, self.Ny, self.Nz = V0.x.shape
        self.Lx, self.Ly, self.Lz = V0.box

    def __getattr__(self, name: str):
        """Provide informative errors for EoS/type-specific attributes."""
        if name == 'Cs' and self.EoS != 'isothermal':
            raise AttributeError("'Cs' is only available for isothermal EoS.")
        if name == 'gamma' and self.EoS != 'adiabatic':
            raise AttributeError("'gamma' is only available for adiabatic EoS.")
        if name == 'Bs' and self.type == 'hydro':
            raise AttributeError("'Bs' is not available for hydrodynamic turbulence.")
        if name == 'accs' and self.type == 'MRI':
            raise AttributeError("'accs' is not available for MRI turbulence.")

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def Js(self) -> list[VectorField]:
        """Current density fields: J = ∇ × B."""
        if self.type == 'hydro':
            raise AttributeError("'Js' is not available for hydrodynamic turbulence.")

        from .derivatives.derivative import Algorithm, curl

        if self.solver == 'SPECTRAL':
            algorithm = Algorithm('SPECTRAL')
        else:
            algorithm = Algorithm(method='TENO', stencil=7, CT=0.01)

        return [curl(B, algorithm) for B in self.Bs]

    @property
    def wVs(self) -> list[VectorField]:
        """Density-weighted velocity fields: sqrt(rho) * V."""
        if self.EoS == 'incompressible':
            return self.Vs

        return [sqrt(rho) * V for rho, V in zip(self.rhos, self.Vs)]

    @property
    def avgBs(self) -> list[Vector]:
        """Mean magnetic field at each time."""
        if self.type == 'hydro':
            raise AttributeError("'avgBs' is not available for hydrodynamic turbulence.")
        return [Vector(float(np.mean(B.x)), float(np.mean(B.y)), float(np.mean(B.z))) for B in self.Bs]

    @property
    def KEs(self) -> list[float]:
        """Total kinetic energy at each time."""
        return [ 0.5 * (wV.norm**2).total for wV in self.wVs ]

    @property
    def MEs(self) -> list[float]:
        """Total magnetic energy at each time."""
        if self.type == 'hydro':
            raise AttributeError("'MEs' is not available for hydrodynamic turbulence.")
        return [ 0.5 * (B.norm**2).total  for B  in self.Bs  ]

    @property
    def drhos(self) -> list[float]:
        """Relative density fluctuation δρ / <ρ> at each time."""
        if self.EoS == 'incompressible':
            raise AttributeError("'drhos' is not available for incompressible EoS.")
        return [rho.std / rho.mean for rho in self.rhos]
# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/derivatives/WENO.py
--------------------------

Implements the WENO5-Z and WENO7-Z reconstruction schemes with two backends:
    - NumPy (default): more compatible, but limited to a single CPU core
    - JAX: requires JAX, but leverages multi-core CPU and GPU acceleration
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

hasJAX = False

try:
    import jax
    jax.config.update("jax_enable_x64", True)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    jax.config.update("jax_num_cpu_devices", 8)

    mesh: Mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=("X", "Y"))

    sharding = {
        "x": NamedSharding(mesh, P(None, "X", "Y")),
        "y": NamedSharding(mesh, P("X", None, "Y")),
        "z": NamedSharding(mesh, P("X", "Y", None)),
    }

    hasJAX = True

except ImportError:
    pass

# ===== WENO5 =====

def coreWENO5(u: ArrayLike, axis: int) -> tuple[Any, Any]:
    """WENO5-Z reconstruction along a single axis."""
    if hasJAX:
        import jax.numpy as jnp
        xp = jnp
    else:
        xp = np
    u = xp.asarray(u)

    uR2 = xp.roll(u, -2, axis=axis)
    uR1 = xp.roll(u, -1, axis=axis)
    u0  = u
    uL1 = xp.roll(u,  1, axis=axis)
    uL2 = xp.roll(u,  2, axis=axis)

    beta1 = 13/12 * (uL2 - 2 * uL1 + u0 )**2 + 1/4 * (uL2 - 4 * uL1 + 3 * u0)**2
    beta2 = 13/12 * (uL1 - 2 * u0  + uR1)**2 + 1/4 * (uL1 - uR1)**2
    beta3 = 13/12 * (u0  - 2 * uR1 + uR2)**2 + 1/4 * (3 * u0 - 4 * uR1 + uR2)**2

    tau5 = xp.abs(beta1 - beta3)
    epsilon = 1e-6
    p = 4

    gamma1, gamma2, gamma3 = 1/10, 3/5, 3/10

    bL1, aL1, c1, aR1, bR1 =  1/3, -7/6, 11/6,    0,    0
    bL2, aL2, c2, aR2, bR2 =    0, -1/6,  5/6,  1/3,    0
    bL3, aL3, c3, aR3, bR3 =    0,    0,  1/3,  5/6, -1/6

    weights1 = gamma1 * (1 + (tau5/(beta1 + epsilon)) ** p)
    weights2 = gamma2 * (1 + (tau5/(beta2 + epsilon)) ** p)
    weights3 = gamma3 * (1 + (tau5/(beta3 + epsilon)) ** p)

    w1 = weights1 / (weights1 + weights2 + weights3)
    w2 = weights2 / (weights1 + weights2 + weights3)
    w3 = weights3 / (weights1 + weights2 + weights3)

    bL = w1 * bL1 + w2 * bL2 + w3 * bL3
    aL = w1 * aL1 + w2 * aL2 + w3 * aL3
    c  = w1 * c1  + w2 * c2  + w3 * c3
    aR = w1 * aR1 + w2 * aR2 + w3 * aR3
    bR = w1 * bR1 + w2 * bR2 + w3 * bR3

    uR = bL * uL2 + aL * uL1 + c * u0 + aR * uR1 + bR * uR2

    gamma1, gamma2, gamma3 = 3/10, 3/5, 1/10

    bL1, aL1, c1, aR1, bR1 = -1/6,  5/6,  1/3,    0,    0
    bL2, aL2, c2, aR2, bR2 =    0,  1/3,  5/6, -1/6,    0
    bL3, aL3, c3, aR3, bR3 =    0,    0, 11/6, -7/6,  1/3

    weights1 = gamma1 * (1 + (tau5/(beta1 + epsilon)) ** p)
    weights2 = gamma2 * (1 + (tau5/(beta2 + epsilon)) ** p)
    weights3 = gamma3 * (1 + (tau5/(beta3 + epsilon)) ** p)

    w1 = weights1 / (weights1 + weights2 + weights3)
    w2 = weights2 / (weights1 + weights2 + weights3)
    w3 = weights3 / (weights1 + weights2 + weights3)

    bL = w1 * bL1 + w2 * bL2 + w3 * bL3
    aL = w1 * aL1 + w2 * aL2 + w3 * aL3
    c  = w1 * c1  + w2 * c2  + w3 * c3
    aR = w1 * aR1 + w2 * aR2 + w3 * aR3
    bR = w1 * bR1 + w2 * bR2 + w3 * bR3

    uL = bL * uL2 + aL * uL1 + c * u0 + aR * uR1 + bR * uR2

    return uL, uR

if hasJAX:
    import jax
    coreWENO5 = jax.jit(coreWENO5, static_argnames=("axis",))

def WENO5(u: ArrayLike, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z reconstruction on a 3D array along the given axis."""
    if hasJAX:
        import jax.numpy as jnp
        xp: Any = jnp
    else:
        xp = np
    u = xp.asarray(u)

    if axis == 2:
        u = xp.moveaxis(u, 2, 0)
        uL, uR = coreWENO5(u, axis=0)
        return np.asarray(xp.moveaxis(uL, 0, 2)), np.asarray(xp.moveaxis(uR, 0, 2))

    uL, uR = coreWENO5(u, axis=axis)
    return np.asarray(uL), np.asarray(uR)

def WENO5x(u: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z reconstruction in x direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["x"])
    return WENO5(u, axis=0)

def WENO5y(u: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z reconstruction in y direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["y"])
    return WENO5(u, axis=1)

def WENO5z(u: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z reconstruction in z direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["z"])
    return WENO5(u, axis=2)

# ===== WENO7 =====

def coreWENO7(u: ArrayLike, axis: int) -> tuple[Any, Any]:
    """WENO7-Z reconstruction along a single axis."""
    if hasJAX:
        import jax.numpy as jnp
        xp = jnp
    else:
        xp = np
    u = xp.asarray(u)

    uR3 = xp.roll(u, -3, axis=axis)
    uR2 = xp.roll(u, -2, axis=axis)
    uR1 = xp.roll(u, -1, axis=axis)
    u0  = u
    uL1 = xp.roll(u,  1, axis=axis)
    uL2 = xp.roll(u,  2, axis=axis)
    uL3 = xp.roll(u,  3, axis=axis)

    beta1 = 1 / 240 * (
        uL3 * (  547 * uL3 -  3882 * uL2 + 4642 * uL1 - 1854 * u0) + \
        uL2 * ( 7043 * uL2 - 17246 * uL1 + 7042 * u0) + \
        uL1 * (11003 * uL1 -  9402 * u0) + \
        u0  * ( 2107 * u0)
    )
    beta2 = 1 / 240 * (
        uL2 * ( 267 * uL2 - 1642 * uL1 + 1602 * u0 - 494 * uR1) + \
        uL1 * (2843 * uL1 - 5966 * u0  + 1922 * uR1) + \
        u0  * (3443 * u0  - 2522 * uR1) + \
        uR1 * ( 547 * uR1)
    )
    beta3 = 1 / 240 * (
        uL1 * ( 547 * uL1 - 2522 * u0  + 1922 * uR1 - 494 * uR2) + \
        u0  * (3443 * u0  - 5966 * uR1 + 1602 * uR2) + \
        uR1 * (2843 * uR1 - 1642 * uR2) + \
        uR2 * ( 267 * uR2)
    )
    beta4 = 1 / 240 * (
        u0  * ( 2107 * u0  -  9402 * uR1 + 7042 * uR2 - 1854 * uR3) + \
        uR1 * (11003 * uR1 - 17246 * uR2 + 4642 * uR3) + \
        uR2 * ( 7043 * uR2 -  3882 * uR3) + \
        uR3 * (  547 * uR3)
    )

    tau = xp.abs(beta1 - beta4)
    epsilon = 1e-6
    p = 4

    gamma1, gamma2, gamma3, gamma4 = 1/35, 12/35, 18/35, 4/35

    bL1, aL1, c1, aR1, bR1, cR1, dR1 = -1/4,  13/12, -23/12,  25/12,      0,      0,      0
    bL2, aL2, c2, aR2, bR2, cR2, dR2 =    0,   1/12,  -5/12,  13/12,    1/4,      0,      0
    bL3, aL3, c3, aR3, bR3, cR3, dR3 =    0,      0,  -1/12,   7/12,   7/12,  -1/12,      0
    bL4, aL4, c4, aR4, bR4, cR4, dR4 =    0,      0,      0,    1/4,  13/12,  -5/12,   1/12

    weights1 = gamma1 * (1 + (tau/(beta1 + epsilon)) ** p)
    weights2 = gamma2 * (1 + (tau/(beta2 + epsilon)) ** p)
    weights3 = gamma3 * (1 + (tau/(beta3 + epsilon)) ** p)
    weights4 = gamma4 * (1 + (tau/(beta4 + epsilon)) ** p)

    w1 = weights1 / (weights1 + weights2 + weights3 + weights4)
    w2 = weights2 / (weights1 + weights2 + weights3 + weights4)
    w3 = weights3 / (weights1 + weights2 + weights3 + weights4)
    w4 = weights4 / (weights1 + weights2 + weights3 + weights4)

    bL = w1 * bL1 + w2 * bL2 + w3 * bL3 + w4 * bL4
    aL = w1 * aL1 + w2 * aL2 + w3 * aL3 + w4 * aL4
    c  = w1 * c1  + w2 * c2  + w3 * c3  + w4 * c4
    aR = w1 * aR1 + w2 * aR2 + w3 * aR3 + w4 * aR4
    bR = w1 * bR1 + w2 * bR2 + w3 * bR3 + w4 * bR4
    cR = w1 * cR1 + w2 * cR2 + w3 * cR3 + w4 * cR4
    dR = w1 * dR1 + w2 * dR2 + w3 * dR3 + w4 * dR4
    uR = bL * uL3 + aL * uL2 + c * uL1 + aR * u0 + bR * uR1 + cR * uR2 + dR * uR3

    gamma1, gamma2, gamma3, gamma4 = 4/35, 18/35, 12/35, 1/35

    bL1, aL1, c1, aR1, bR1, cR1, dR1 =  1/12,  -5/12,  13/12,    1/4,       0,      0,      0
    bL2, aL2, c2, aR2, bR2, cR2, dR2 =     0,  -1/12,   7/12,   7/12,   -1/12,      0,      0
    bL3, aL3, c3, aR3, bR3, cR3, dR3 =     0,      0,    1/4,  13/12,   -5/12,   1/12,      0
    bL4, aL4, c4, aR4, bR4, cR4, dR4 =     0,      0,      0,  25/12,  -23/12,  13/12,   -1/4

    weights1 = gamma1 * (1 + (tau/(beta1 + epsilon)) ** p)
    weights2 = gamma2 * (1 + (tau/(beta2 + epsilon)) ** p)
    weights3 = gamma3 * (1 + (tau/(beta3 + epsilon)) ** p)
    weights4 = gamma4 * (1 + (tau/(beta4 + epsilon)) ** p)

    w1 = weights1 / (weights1 + weights2 + weights3 + weights4)
    w2 = weights2 / (weights1 + weights2 + weights3 + weights4)
    w3 = weights3 / (weights1 + weights2 + weights3 + weights4)
    w4 = weights4 / (weights1 + weights2 + weights3 + weights4)

    bL = w1 * bL1 + w2 * bL2 + w3 * bL3 + w4 * bL4
    aL = w1 * aL1 + w2 * aL2 + w3 * aL3 + w4 * aL4
    c  = w1 * c1  + w2 * c2  + w3 * c3  + w4 * c4
    aR = w1 * aR1 + w2 * aR2 + w3 * aR3 + w4 * aR4
    bR = w1 * bR1 + w2 * bR2 + w3 * bR3 + w4 * bR4
    cR = w1 * cR1 + w2 * cR2 + w3 * cR3 + w4 * cR4
    dR = w1 * dR1 + w2 * dR2 + w3 * dR3 + w4 * dR4
    uL = bL * uL3 + aL * uL2 + c * uL1 + aR * u0 + bR * uR1 + cR * uR2 + dR * uR3

    return uL, uR

if hasJAX:
    import jax
    coreWENO7 = jax.jit(coreWENO7, static_argnames=("axis",))

def WENO7(u: ArrayLike, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """WENO7-Z reconstruction on a 3D array along the given axis."""
    if hasJAX:
        import jax.numpy as jnp
        xp: Any = jnp
    else:
        xp = np
    u = xp.asarray(u)

    if axis == 2:
        u = xp.moveaxis(u, 2, 0)
        uL, uR = coreWENO7(u, axis=0)
        return np.asarray(xp.moveaxis(uL, 0, 2)), np.asarray(xp.moveaxis(uR, 0, 2))

    uL, uR = coreWENO7(u, axis=axis)
    return np.asarray(uL), np.asarray(uR)

def WENO7x(u: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """WENO7-Z reconstruction in x direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["x"])
    return WENO7(u, axis=0)

def WENO7y(u: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """WENO7-Z reconstruction in y direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["y"])
    return WENO7(u, axis=1)

def WENO7z(u: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """WENO7-Z reconstruction in z direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["z"])
    return WENO7(u, axis=2)

def WENOx(u: ArrayLike, stencils: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z or WENO7-Z reconstruction in x direction."""
    if stencils == 5:
        return WENO5x(u)
    if stencils == 7:
        return WENO7x(u)

    raise ValueError(f"Invalid stencils: {stencils}. Supported stencils: 5, 7.")

def WENOy(u: ArrayLike, stencils: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z or WENO7-Z reconstruction in y direction."""
    if stencils == 5:
        return WENO5y(u)
    if stencils == 7:
        return WENO7y(u)

    raise ValueError(f"Invalid stencils: {stencils}. Supported stencils: 5, 7.")

def WENOz(u: ArrayLike, stencils: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z or WENO7-Z reconstruction in z direction."""
    if stencils == 5:
        return WENO5z(u)
    if stencils == 7:
        return WENO7z(u)

    raise ValueError(f"Invalid stencils: {stencils}. Supported stencils: 5, 7.")

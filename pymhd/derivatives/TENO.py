# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/derivatives/TENO.py
--------------------------

Implements the TENO7-M reconstruction scheme with two backends:
    - NumPy (default): more compatible, but limited to a single CPU core
    - JAX: requires JAX, but leverages multi-core CPU and (future) GPU acceleration

See the method paper (https://arxiv.org/abs/2606.22506) for details
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

hasJAX = False

try:
    import jax
    jax.config.update("jax_enable_x64", True)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    try:
        jax.config.update("jax_num_cpu_devices", 8)
    except RuntimeError:
        # In some JAX versions, re-setting this after backend init raises.
        pass

    mesh: Mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=("X", "Y"))

    sharding = {
        "x": NamedSharding(mesh, P(None, "X", "Y")),
        "y": NamedSharding(mesh, P("X", None, "Y")),
        "z": NamedSharding(mesh, P("X", "Y", None)),
    }

    hasJAX = True

except ImportError:
    pass

def coreTENO7M(
    u: ArrayLike, direction: str, axis: int, CT: float = 0.01
) -> tuple[Any, Any]:
    """TENO7-M reconstruction.

    Reconstruct the cell interface values from the cell averages

    Parameters
    ----------
    u        : npt.ArrayLike, cell-centered or cell-averaged values
    direction: str, reconstruction direction, 'L' (for i - 1/2) or 'R' (for i + 1/2)
    axis     : int, the axis direction (0, 1, 2 corresponds to x, y, z)
    CT       : float, smoothness threshold, defaults to 0.01

    Returns
    -------
    uHalf    : np.ndarray or jax.Array, reconstructed value at the cell interfaces (i ± 1/2)
    shockflag: np.ndarray or jax.Array, shock detection flag
    """
    if hasJAX:
        import jax.numpy as jnp
        xp = jnp
    else:
        xp = np
    u = xp.asarray(u)

    def roll(arr, offset: int):
        shift = -offset if direction == "R" else offset
        return xp.roll(arr, shift, axis=axis)

    uR3 = roll(u, 3)
    uR2 = roll(u, 2)
    uR1 = roll(u, 1)
    u0  = u
    uL1 = roll(u, -1)
    uL2 = roll(u, -2)
    uL3 = roll(u, -3)

    # Calculate the smoothness indicators for (7, 4) stencil
    # Ref: Balsara & Shu, Journal of Computational Physics 160, no. 2 (May 2000): 405–52
    beta0 = 1 / 240 * (
        uL3 * (  547 * uL3 -  3882 * uL2 + 4642 * uL1 - 1854 * u0) + \
        uL2 * ( 7043 * uL2 - 17246 * uL1 + 7042 * u0) + \
        uL1 * (11003 * uL1 -  9402 * u0) + \
        u0  * ( 2107 * u0)
    )
    beta1 = 1 / 240 * (
        uL2 * ( 267 * uL2 - 1642 * uL1 + 1602 * u0 - 494 * uR1) + \
        uL1 * (2843 * uL1 - 5966 * u0  + 1922 * uR1) + \
        u0  * (3443 * u0  - 2522 * uR1) + \
        uR1 * ( 547 * uR1)
    )
    beta2 = 1 / 240 * (
        uL1 * ( 547 * uL1 - 2522 * u0  + 1922 * uR1 - 494 * uR2) + \
        u0  * (3443 * u0  - 5966 * uR1 + 1602 * uR2) + \
        uR1 * (2843 * uR1 - 1642 * uR2) + \
        uR2 * ( 267 * uR2)
    )
    beta3 = 1 / 240 * (
        u0  * ( 2107 * u0  -  9402 * uR1 + 7042 * uR2 - 1854 * uR3) + \
        uR1 * (11003 * uR1 - 17246 * uR2 + 4642 * uR3) + \
        uR2 * ( 7043 * uR2 -  3882 * uR3) + \
        uR3 * (  547 * uR3)
    )

    epsilon = 1e-40
    q = 6

    gamma0 = 1 / (beta0 + epsilon) ** q
    gamma1 = 1 / (beta1 + epsilon) ** q
    gamma2 = 1 / (beta2 + epsilon) ** q
    gamma3 = 1 / (beta3 + epsilon) ** q

    chi0mask = (gamma0 / (gamma0 + gamma1 + gamma2 + gamma3)) < CT
    chi1mask = (gamma1 / (gamma0 + gamma1 + gamma2 + gamma3)) < CT
    chi2mask = (gamma2 / (gamma0 + gamma1 + gamma2 + gamma3)) < CT
    chi3mask = (gamma3 / (gamma0 + gamma1 + gamma2 + gamma3)) < CT

    delta0 = ~(chi0mask & roll(chi1mask, -1) & roll(chi2mask, -2) & roll(chi3mask, -3))
    delta1 = ~(chi1mask & roll(chi2mask, -1) & roll(chi3mask, -2) & roll(chi0mask,  1))
    delta2 = ~(chi2mask & roll(chi3mask, -1) & roll(chi1mask,  1) & roll(chi0mask,  2))
    delta3 = ~(chi3mask & roll(chi2mask,  1) & roll(chi1mask,  2) & roll(chi0mask,  3))

    # ----- Dispersion-relation-preserving (DRP) -----
    # Ref: Tam. Computational aeroacoustics: A wave number approach, 2012.
    a71 = 0.77088238051822552
    a72 = -0.166705904414580469
    a73 = 0.02084314277031176

    a91 = 0.8301178834769906875382633360472085
    a92 = -0.23175338776901819008451262109655756
    a93 = 0.05287205020483696423592156502901203
    a94 = -6.306814638366300019250697235282424e-3

    cL3 = 2 * a94
    cL2 = 2 * a93 - a73
    cL1 = 2 * a92 + 2 * a94 - a72 + a73
    c0  = 2 * a91 + 2 * a93 - a71 + a72 - a73
    cR1 = 2 * a92 + 2 * a94 - a72 + a73 + a71
    cR2 = 2 * a93 - a73 + a72
    cR3 = 2 * a94 + a73

    # Optimal reconstruction polynomial for smooth stencils
    uHalf = cL3 * uL3 + cL2 * uL2 + cL1 * uL1 + c0 * u0 + cR1 * uR1 + cR2 * uR2 + cR3 * uR3

    # a single discontinuity lies in (uL3, uL2)
    uHalf = xp.where(
        (~delta0) & delta1 & delta2 & delta3,
        1 / 60 * uL2 - 2 / 15 * uL1 + 37 / 60 * u0 + 37 / 60 * uR1 - 2 / 15 * uR2 + 1 / 60 * uR3,
        uHalf,
    )

    # a single discontinuity lies in (uL2, uL1)
    uHalf = xp.where(
        (~delta0) & (~delta1) & delta2 & delta3,
        -1 / 20 * uL1 + 9 / 20 * u0 + 47 / 60 * uR1 - 13 / 60 * uR2 + 1 / 30 * uR3,
        uHalf,
    )

    # a single discontinuity lies in (uL1, u0)
    uHalf = xp.where(
        (~delta0) & (~delta1) & (~delta2) & delta3,
        1 / 4 * u0 + 13 / 12 * uR1 - 5 / 12 * uR2 + 1 / 12 * uR3,
        uHalf,
    )

    # a single discontinuity lies in (u0, uR1)
    uHalf = xp.where(
        delta0 & (~delta1) & (~delta2) & (~delta3),
        -1 / 4 * uL3 + 13 / 12 * uL2 - 23 / 12 * uL1 + 25 / 12 * u0,
        uHalf,
    )

    # a single discontinuity lies in (uR1, uR2)
    uHalf = xp.where(
        delta0 & delta1 & (~delta2) & (~delta3),
        -1 / 20 * uL3 + 17 / 60 * uL2 - 43 / 60 * uL1 + 77 / 60 * u0 + 1 / 5 * uR1,
        uHalf,
    )

    # a single discontinuity lies in (uR2, uR3)
    uHalf = xp.where(
        delta0 & delta1 & delta2 & (~delta3),
        -1 / 60 * uL3 + 7 / 60 * uL2 - 23 / 60 * uL1 + 19 / 20 * u0 + 11 / 30 * uR1 - 1 / 30 * uR2,
        uHalf,
    )

    # two discontinuities lie in (uL3, uL2) and (uR2, uR3)
    uHalf = xp.where(
        (~delta0) & delta1 & delta2 & (~delta3),
        2 / 60 * uL2 - 13 / 60 * uL1 + 47 / 60 * u0 + 27 / 60 * uR1 - 3 / 60 * uR2,
        uHalf,
    )

    # Any stencil other than fully smooth is considered discontinuous.
    shockflag = ~(delta0 & delta1 & delta2 & delta3)

    return uHalf, shockflag

if hasJAX:
    import jax
    coreTENO7M = jax.jit(coreTENO7M, static_argnames=("direction", "axis", "CT"))

def TENO7M(
    u: ArrayLike, axis: int, mode: str = "hybrid", CT: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """TENO7-M reconstruction."""
    if hasJAX:
        import jax.numpy as jnp
        xp: Any = jnp
    else:
        xp = np
    u = xp.asarray(u)

    if axis == 2:
        u = xp.moveaxis(u, 2, 0)

        uL, shockFlagL = coreTENO7M(u, direction="L", axis=0, CT=CT)
        uR, shockFlagR = coreTENO7M(u, direction="R", axis=0, CT=CT)

        if mode == "inner":
            return np.asarray(xp.moveaxis(uL, 0, 2)), np.asarray(xp.moveaxis(uR, 0, 2))

        if mode == "upwind":
            uLupwind = xp.roll(uR, 1, axis=0)
            return np.asarray(xp.moveaxis(uLupwind, 0, 2)), np.asarray(xp.moveaxis(uR, 0, 2))

        if mode == "hybrid":
            uLhybrid = xp.where(~shockFlagL, (uL + xp.roll(uR, 1, axis=0)) / 2, uL)
            uRhybrid = xp.where(~shockFlagR, (uR + xp.roll(uL, -1, axis=0)) / 2, uR)
            return np.asarray(xp.moveaxis(uLhybrid, 0, 2)), np.asarray(xp.moveaxis(uRhybrid, 0, 2))

        raise ValueError(f"Invalid mode: {mode}. Options: 'inner', 'upwind', and 'hybrid'.")

    uL, shockFlagL = coreTENO7M(u, direction="L", axis=axis, CT=CT)
    uR, shockFlagR = coreTENO7M(u, direction="R", axis=axis, CT=CT)

    if mode == "inner":
        return np.asarray(uL), np.asarray(uR)

    if mode == "upwind":
        uLupwind = xp.roll(uR, 1, axis=axis)
        return np.asarray(uLupwind), np.asarray(uR)

    if mode == "hybrid":
        uLhybrid = xp.where(~shockFlagL, (uL + xp.roll(uR, 1, axis=axis)) / 2, uL)
        uRhybrid = xp.where(~shockFlagR, (uR + xp.roll(uL, -1, axis=axis)) / 2, uR)
        return np.asarray(uLhybrid), np.asarray(uRhybrid)

    raise ValueError(f"Invalid mode: {mode}. Options: 'inner', 'upwind', and 'hybrid'.")


def TENO7Mx(
    u: ArrayLike, mode: str = "hybrid", CT: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """TENO7-M reconstruction in x direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["x"])
    return TENO7M(u, axis=0, mode=mode, CT=CT)

def TENO7My(
    u: ArrayLike, mode: str = "hybrid", CT: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """TENO7-M reconstruction in y direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["y"])
    return TENO7M(u, axis=1, mode=mode, CT=CT)

def TENO7Mz(
    u: ArrayLike, mode: str = "hybrid", CT: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """TENO7-M reconstruction in z direction."""
    if hasJAX:
        import jax
        u = jax.device_put(u, sharding["z"])
    return TENO7M(u, axis=2, mode=mode, CT=CT)
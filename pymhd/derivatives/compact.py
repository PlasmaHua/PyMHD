# PyMHD: Python for Magnetohydrodynamic Turbulence.
# Copyright (c) 2026 Yuyang Hua (华宇阳)
# License: MIT

"""
pymhd/derivatives/compact.py
----------------------------

Implements the TCS-M scheme:
    - TCS-M: Targeted Compact Scheme with Multi-stencil Discontinuity Detectors (MSDD)
    - Ref: Lele 1992 JCP, Jiang 2000 IJCFD
    - Currently only supports JAX implementation
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab

from jax import Array
from jax.typing import ArrayLike

from functools import partial

def MSDD7(
    u: ArrayLike, axis: int, CT: float = 0.01
) -> list[Array]:
    """Multi-stencil Discontinuity Detector (7-point stencils) for 3D arrays

    Parameters
    ----------
    u    : ArrayLike, 3D array
    axis : int, axis direction (0, 1, 2 corresponds to x, y, z)
    CT   : float, smoothness threshold, default: 0.01

    Returns
    -------
    conditions: list[Array], conditions for different discontinuity configurations

    References
    ----------
    [1] Balsara & Shu, JCP 160, no. 2 (May 2000): 405-52

    """
    u = jnp.asarray(u)

    uL3, uL2, uL1, u0, uR1, uR2, uR3 = [jnp.roll(u, shift, axis=axis) for shift in (3, 2, 1, 0, -1, -2, -3)]

    # Smoothness indicators for 7-point stencils
    # Ref: Balsara & Shu, Journal of Computational Physics 160, no. 2 (May 2000): 405–52

    # β0
    beta0 = 1/240 * (
                uL3 * (  547 * uL3 - 3882  * uL2 + 4642 * uL1 - 1854 * u0) + \
                uL2 * ( 7043 * uL2 - 17246 * uL1 + 7042 * u0) + \
                uL1 * (11003 * uL1 - 9402  * u0) + \
                u0  * ( 2107 * u0)
            )

    # β1
    beta1 = 1/240 * (
                uL2 * (  267 * uL2 - 1642 * uL1 + 1602 * u0 - 494 * uR1) + \
                uL1 * ( 2843 * uL1 - 5966 * u0  + 1922 * uR1) + \
                u0  * ( 3443 * u0  - 2522 * uR1) + \
                uR1 * (  547 * uR1) \
            )

    # β2
    beta2 = 1/240 * (
                uL1 * (  547 * uL1 - 2522 * u0  + 1922 * uR1 - 494 * uR2) + \
                u0  * ( 3443 * u0  - 5966 * uR1 + 1602 * uR2) + \
                uR1 * ( 2843 * uR1 - 1642 * uR2) + \
                uR2 * (  267 * uR2) \
            )

    # β3
    beta3 = 1/240 * (
                u0  * ( 2107 * u0  -  9402 * uR1 + 7042 * uR2 - 1854 * uR3) + \
                uR1 * (11003 * uR1 - 17246 * uR2 + 4642 * uR3) + \
                uR2 * ( 7043 * uR2 -  3882 * uR3) + \
                uR3 * (  547 * uR3 ) \
            )

    epsilon = 1e-6
    q = 6

    gamma0 = 1 / (beta0 + epsilon)**q
    gamma1 = 1 / (beta1 + epsilon)**q
    gamma2 = 1 / (beta2 + epsilon)**q
    gamma3 = 1 / (beta3 + epsilon)**q

    chi0 = gamma0 / (gamma0 + gamma1 + gamma2 + gamma3)
    chi1 = gamma1 / (gamma0 + gamma1 + gamma2 + gamma3)
    chi2 = gamma2 / (gamma0 + gamma1 + gamma2 + gamma3)
    chi3 = gamma3 / (gamma0 + gamma1 + gamma2 + gamma3)

    # ===== Computing delta values =====

    # chi values from neighboring stencils
    chi0R1 = jnp.roll(chi0, -1, axis=axis)
    chi0R2 = jnp.roll(chi0, -2, axis=axis)
    chi0R3 = jnp.roll(chi0, -3, axis=axis)

    chi1L1 = jnp.roll(chi1,  1, axis=axis)

    chi1R1 = jnp.roll(chi1, -1, axis=axis)
    chi1R2 = jnp.roll(chi1, -2, axis=axis)

    chi2L1 = jnp.roll(chi2,  1, axis=axis)
    chi2L2 = jnp.roll(chi2,  2, axis=axis)

    chi2R1 = jnp.roll(chi2, -1, axis=axis)

    chi3L1 = jnp.roll(chi3,  1, axis=axis)
    chi3L2 = jnp.roll(chi3,  2, axis=axis)
    chi3L3 = jnp.roll(chi3,  3, axis=axis)

    # deltas (sharp cutoff function)
    # Multi-stencil Discontinuity Detector
    delta0 = jnp.where((chi0 < CT) & (chi1L1 < CT) & (chi2L2 < CT) & (chi3L3 < CT), 0, 1)
    delta1 = jnp.where((chi1 < CT) & (chi2L1 < CT) & (chi3L2 < CT) & (chi0R1 < CT), 0, 1)
    delta2 = jnp.where((chi2 < CT) & (chi3L1 < CT) & (chi1R1 < CT) & (chi0R2 < CT), 0, 1)
    delta3 = jnp.where((chi3 < CT) & (chi2R1 < CT) & (chi1R2 < CT) & (chi0R3 < CT), 0, 1)

    # ===== Possible discontinuity Configurations =====

    # uHalfOptimal: smooth stencils
    condition0 = (delta0 == 1) & (delta1 == 1) & (delta2 == 1) & (delta3 == 1)

    # uHalf1: a single discontinuity between uL3 and uL2
    condition1 = (delta0 == 0) & (delta1 == 1) & (delta2 == 1) & (delta3 == 1)

    # uHalf2: a single discontinuity between uL2 and uL1
    condition2 = (delta0 == 0) & (delta1 == 0) & (delta2 == 1) & (delta3 == 1)

    # uHalf3: a single discontinuity between uL1 and u0
    condition3 = (delta0 == 0) & (delta1 == 0) & (delta2 == 0) & (delta3 == 1)

    # uHalf4: a single discontinuity between u0 and uR1
    condition4 = (delta0 == 1) & (delta1 == 0) & (delta2 == 0) & (delta3 == 0)

    # uHalf5: a single discontinuity between uR1 and uR2
    condition5 = (delta0 == 1) & (delta1 == 1) & (delta2 == 0) & (delta3 == 0)

    # uHalf6: a single discontinuity between uR2 and uR3
    condition6 = (delta0 == 1) & (delta1 == 1) & (delta2 == 1) & (delta3 == 0)

    conditions = [
        condition0,
        condition1,
        condition2,
        condition3,
        condition4,
        condition5,
        condition6
    ]

    return conditions


def TENO7M(
    u: ArrayLike, axis: int, dx: float, CT: float = 0.01
) -> Array:
    """TENO7-M finite difference scheme

    Estimation of the derivative for initial guess of the BiCGSTAB algorithm.

    Parameters
    ----------
    u    : ArrayLike, 3D array
    axis : int, axis direction (0, 1, 2 corresponds to x, y, z)
    dx   : float, grid spacing along the axis
    CT   : float, smoothness threshold, default: 0.01

    Returns
    -------
    dudx : Array, estimated derivative (initial guess)

    """
    u = jnp.asarray(u)

    uL3, uL2, uL1, u0, uR1, uR2, uR3 = [jnp.roll(u, shift, axis=axis) for shift in (3, 2, 1, 0, -1, -2, -3)]

    conditions = MSDD7(u, axis, CT=CT)

    # smooth stencils
    a1 = 0.77088238051822552      # Original: 3/4   = 0.75
    a2 = -0.166705904414580469    # Original: -3/20 = -0.15
    a3 = 0.02084314277031176      # Original: 1/60  = 0.01666...
    dudx0 = (a1 * (uR1 - uL1) + a2 * (uR2 - uL2) + a3 * (uR3 - uL3)) / dx

    # a single discontinuity between uL3 and uL2
    # 1/12	−2/3	0	2/3	−1/12
    dudx1 = (2/3 * (uR1 - uL1) - 1/12 * (uR2 - uL2)) / dx

    # a single discontinuity between uL2 and uL1
    dudx2 = (uR1 - uL1) / (2 * dx)

    # a single discontinuity between uL1 and u0
    dudx3 = (uR1 - u0 ) / dx

    # a single discontinuity between u0 and uR1
    dudx4 = (u0  - uL1) / dx

    # a single discontinuity between uR1 and uR2
    dudx5 = (uR1 - uL1) / (2 * dx)

    # a single discontinuity between uR2 and uR3
    # 1/12	−2/3	0	2/3	−1/12
    dudx6 = (2/3 * (uR1 - uL1) - 1/12 * (uR2 - uL2)) / dx

    dudxs = [dudx0, dudx1, dudx2, dudx3, dudx4, dudx5, dudx6]

    dudx = jnp.select(conditions, dudxs, default=dudx0)

    return dudx


@partial(jax.jit, static_argnums=(1, 2, 3))
def TCS7M(
    u: ArrayLike, axis: int, CT: float = 0.01, L: float = 1.0
) -> Array:
    """Targeted Compact Scheme with MSDD (TCS-M)

    Parameters
    ----------
    u    : ArrayLike, 3D array
    axis : int, axis direction (0, 1, 2 corresponds to x, y, z)
    CT   : float, smoothness threshold, default: 0.01
    L    : float, domain length along the axis, default: 1.0

    Returns
    -------
    dudx : Array, derivative along the given axis

    References
    ----------
    [1] Lele, 1992, JCP, 10.1016/0021-9991(92)90324-R
    [2] Jiang, 2001, IJCFD, 10.1080/10618560108970024

    """
    u = jnp.asarray(u)
    dx = L / u.shape[axis]

    uL3, uL2, uL1, u0, uR1, uR2, uR3 = [jnp.roll(u, shift, axis=axis) for shift in (3, 2, 1, 0, -1, -2, -3)]

    conditions = MSDD7(u, axis, CT=CT)
    x0 = TENO7M(u, axis, dx, CT=CT)

    # construct linear system: AX = B

    # ===== right-hand side B =====
    # B for smooth stencils:
    # cL, bL, aL, c, aR, bR, cR represent c^- b^-, a^-, c^0, a^+, b^+, c^+, respectively

    # B0: smooth stencils
    # Lele 1992 (3.1.6)
    a0, b0, c0 = 1.3025166, 0.9935500, 0.03750245
    B0: Array = c0 / (6 * dx) * (uR3 - uL3) + b0 / (4 * dx) * (uR2 - uL2) + a0 / (2 * dx) * (uR1 - uL1)

    # B1: a single discontinuity between uL3 and uL2
    # Lele 1992 (2.1.12)
    a1, b1 = 40/27, 25/54
    B1: Array = b1 / (4 * dx) * (uR2 - uL2) + a1 / (2 * dx) * (uR1 - uL1)

    # B2: a single discontinuity between uL2 and uL1
    # Jiang 2001, Lele 1992 (theta = 3/10)
    bL2, aL2, c2, aR2, bR2 = 0, -0.75 * 44/49, -2.65 * 5/49, 0.75 * 44/49 + 1.4 * 5/49, 1.25 * 5/49
    B2: Array = (bL2 * uL2 + aL2 * uL1 + c2 * u0 + aR2 * uR1 + bR2 * uR2) / dx

    # B3: a single discontinuity between uL1 and u0
    # Jiang 2001, Lele 1992 (theta = 3/10)
    bL3, aL3, c3, aR3, bR3 = 0, 0, -2.65, 1.4, 1.25
    B3: Array = (bL3 * uL2 + aL3 * uL1 + c3 * u0 + aR3 * uR1 + bR3 * uR2) / dx

    # B4: a single discontinuity between u0 and uR1
    # Jiang 2001, Lele 1992 (theta = 3/10)
    bL4, aL4, c4, aR4, bR4 = -1.25, -1.4, 2.65, 0, 0
    B4: Array = (bL4 * uL2 + aL4 * uL1 + c4 * u0 + aR4 * uR1 + bR4 * uR2) / dx

    # B5: a single discontinuity between uR1 and uR2
    # Jiang 2001, Lele 1992 (theta = 3/10)
    bL5, aL5, c5, aR5, bR5 = -1.25 * 5/49, -1.4 * 5/49 + -0.75 * 44/49, 2.65 * 5/49, 0.75 * 44/49, 0
    B5: Array = (bL5 * uL2 + aL5 * uL1 + c5 * u0 + aR5 * uR1 + bR5 * uR2) / dx

    # B6: a single discontinuity between uR2 and uR3
    # Lele 1992 (2.1.12)
    a6, b6 = 40/27, 25/54
    B6: Array = b6 / (4 * dx) * (uR2 - uL2) + a6 / (2 * dx) * (uR1 - uL1)

    Bs = [B0, B1, B2, B3, B4, B5, B6]
    B = jnp.select(conditions, Bs, default=B0)

    # ===== matrix-vector product mapping =====
    # here, x represents the array of unknown derivatives, not the coordinates
    # betaL, alphaL, alphaR, betaR represent \beta^-, \alpha^-, \alpha^+, \beta^+, respectively
    def A(x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        xL2, xL1, x0, xR1, xR2 = [jnp.roll(x, shift, axis=axis) for shift in (2, 1, 0, -1, -2)]

        # A0: smooth stencils
        # Lele 1992 (3.1.6)
        alpha0, beta0 = 0.5771439, 0.0896406
        A0 = beta0 * xL2 + alpha0 * xL1 + x0 + alpha0 * xR1 + beta0 * xR2

        # A1: a single discontinuity between uL3 and uL2
        # Lele 1992 (2.1.12)
        alpha1, beta1 = 4/9, 1/36
        A1 = beta1 * xL2 + alpha1 * xL1 + x0 + alpha1 * xR1 + beta1 * xR2

        # A2: a single discontinuity between uL2 and uL1
        # Jiang 2001, Lele 1992 (theta = 3/10)
        betaL2, alphaL2, alphaR2, betaR2 = 0, 0.25 * 44/49, 0.25 * 44/49 + 2.6 * 5/49, 0.3 * 5/49
        A2 = betaL2 * xL2 + alphaL2 * xL1 + x0 + alphaR2 * xR1 + betaR2 * xR2

        # A3: a single discontinuity between uL1 and u0
        # Jiang 2001, Lele 1992 (theta = 3/10)
        betaL3, alphaL3, alphaR3, betaR3 = 0, 0, 2.6, 0.3
        A3 = betaL3 * xL2 + alphaL3 * xL1 + x0 + alphaR3 * xR1 + betaR3 * xR2

        # A4: a single discontinuity between u0 and uR1
        # Jiang 2001, Lele 1992 (theta = 3/10)
        betaL4, alphaL4, alphaR4, betaR4 = 0.3, 2.6, 0, 0
        A4 = betaL4 * xL2 + alphaL4 * xL1 + x0 + alphaR4 * xR1 + betaR4 * xR2

        # A5: a single discontinuity between uR1 and uR2
        # Jiang 2001, Lele 1992 (theta = 3/10)
        betaL5, alphaL5, alphaR5, betaR5 = 0.3 * 5/49, 2.6 * 5/49 + 0.25 * 44/49, 0.25 * 44/49, 0
        A5 = betaL5 * xL2 + alphaL5 * xL1 + x0 + alphaR5 * xR1 + betaR5 * xR2

        # A6: a single discontinuity between uR2 and uR3
        # Lele 1992 (2.1.12)
        alpha6, beta6 = 4/9, 1/36
        A6 = beta6 * xL2 + alpha6 * xL1 + x0 + alpha6 * xR1 + beta6 * xR2

        As = [A0, A1, A2, A3, A4, A5, A6]
        A = jnp.select(conditions, As, default=A0)

        return A

    dudx, _ = bicgstab(A, B, x0=x0, tol=1e-8, atol=0, maxiter=200)

    return dudx


# Direction wrappers

def TCS7Mx(
    u: ArrayLike, CT: float = 0.01, L: float = 1.0
) -> Array:
    """TCS7-M derivative in x direction"""
    return TCS7M(u, axis=0, CT=CT, L=L)

def TCS7My(
    u: ArrayLike, CT: float = 0.01, L: float = 1.0
) -> Array:
    """TCS7-M derivative in y direction"""
    return TCS7M(u, axis=1, CT=CT, L=L)

def TCS7Mz(
    u: ArrayLike, CT: float = 0.01, L: float = 1.0
) -> Array:
    """TCS7-M derivative in z direction"""
    return TCS7M(u, axis=2, CT=CT, L=L)

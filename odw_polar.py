#!/usr/bin/env python3
"""Closed-form wave-angle solutions of the equilibrium oblique-detonation polar.

Reference implementation of the explicit formulae derived in

    J. Li and C. Luo, "Exact explicit wave-angle solutions of the equilibrium
    oblique-detonation polar".

Model: calorically perfect gas, constant specific-heat ratio ``gamma``,
instantaneous and complete heat release, planar zero-thickness equilibrium
front attached to a straight wedge.

Non-dimensional heat release follows the convention of the paper,

    Q  = Q* / (R T1),        q' = 2 (gamma - 1) Q / gamma

with ``Q*`` the chemical energy release per unit mass, ``R`` the gas constant
and ``T1`` the upstream static temperature.

Everything here is closed form.  The wave angles come from a cubic solved by
Cardano's method, the detachment point from a second cubic, and the downstream
total-sonic point from a quadratic.  No iteration is used anywhere.

Angles are in radians throughout; use ``math.degrees`` at the call site.

The routines are the ones exercised by ``verification/validate_closed_form.py``
and ``verification/sonic_numerical_check.py``; see the README for how those
checks are run and what tolerances they meet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "PolarModel",
    "CJPoint",
    "DetachmentPoint",
    "SonicPoint",
    "real_cubic_roots",
]

# Root-clustering and admissibility-filter tolerances.  These are the values
# used by the verification scripts; they are loose enough to survive
# double-precision cancellation near the CJ endpoint and tight enough not to
# admit a spurious branch.
_ROOT_TOL = 2.0e-11
_FILTER_TOL = 2.0e-9


# ---------------------------------------------------------------------------
# Cubic solver
# ---------------------------------------------------------------------------

def _unique_sorted(values: Iterable[float], tol: float = 2.0e-10) -> list[float]:
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
    out: list[float] = []
    for value in ordered:
        if not out or abs(value - out[-1]) > tol * max(1.0, abs(value)):
            out.append(value)
    return out


def real_cubic_roots(
    a: float, b: float, c: float, d: float, tol: float = _ROOT_TOL
) -> list[float]:
    """Distinct real roots of ``a x^3 + b x^2 + c x + d = 0``.

    Implemented with the real Cardano / trigonometric formulae rather than a
    companion-matrix eigensolver, so that what is evaluated is the closed form
    itself.  Degenerates to a quadratic or a linear solve only when the leading
    coefficient vanishes to rounding.
    """
    scale = max(1.0, abs(a), abs(b), abs(c), abs(d))
    eps = tol * scale

    if abs(a) <= eps:
        if abs(b) <= eps:
            if abs(c) <= eps:
                return []
            return [-d / c]
        disc = c * c - 4.0 * b * d
        disc_tol = tol * max(1.0, c * c, abs(4.0 * b * d))
        if disc < -disc_tol:
            return []
        sqrt_disc = math.sqrt(max(0.0, disc))
        return _unique_sorted(
            [(-c + sqrt_disc) / (2.0 * b), (-c - sqrt_disc) / (2.0 * b)]
        )

    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (27.0 * a * a * d - 9.0 * a * b * c + 2.0 * b * b * b) / (27.0 * a * a * a)
    shift = b / (3.0 * a)
    disc = (0.5 * q) ** 2 + (p / 3.0) ** 3
    disc_tol = tol * max(1.0, (0.5 * q) ** 2, abs((p / 3.0) ** 3))

    if disc > disc_tol:
        # One real root.
        sqrt_disc = math.sqrt(disc)
        z = _cbrt(-0.5 * q + sqrt_disc) + _cbrt(-0.5 * q - sqrt_disc)
        return [z - shift]

    if disc >= -disc_tol:
        # Repeated roots.
        if abs(q) <= tol * max(1.0, abs(p), abs(q)):
            return [-shift]
        u = _cbrt(-0.5 * q)
        return _unique_sorted([2.0 * u - shift, -u - shift])

    # Three distinct real roots: trigonometric form, no complex arithmetic.
    if p >= 0.0:
        raise ArithmeticError(
            "casus irreducibilis reached with p >= 0; coefficients are ill-scaled"
        )
    cos_arg = (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
    cos_arg = min(1.0, max(-1.0, cos_arg))
    phi = math.acos(cos_arg) / 3.0
    amplitude = 2.0 * math.sqrt(-p / 3.0)
    return _unique_sorted(
        [amplitude * math.cos(phi - 2.0 * math.pi * k / 3.0) - shift for k in range(3)]
    )


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CJPoint:
    """Chapman-Jouguet endpoint of the attached polar."""

    M_cj: float       # CJ Mach number of the mixture
    u_cj: float       # M_cj^2, the CJ bound on M^2 sin^2(beta)
    beta: float       # CJ wave angle [rad]
    theta: float      # CJ deflection angle [rad]
    r_cj: float       # reciprocal density ratio rho1/rho2 at the CJ point


@dataclass(frozen=True)
class DetachmentPoint:
    """Detachment point: the double root of the wave-angle cubic."""

    theta_max: float  # maximum attached deflection [rad]
    beta: float       # wave angle at detachment [rad]
    y: float          # double root y = tan(theta) tan(beta)
    s: float          # tan^2(theta_max)


@dataclass(frozen=True)
class SonicPoint:
    """Downstream total-sonic point, M2 = 1, on the high-compression branch."""

    theta: float      # deflection angle [rad]
    beta: float       # wave angle [rad]
    u: float          # M^2 sin^2(beta)
    r: float          # reciprocal density ratio rho1/rho2


@dataclass(frozen=True)
class WaveAngleRoot:
    """One algebraic root of the wave-angle cubic, with its admissibility flags."""

    t: float              # tan(beta)
    beta: float           # [rad]
    u: float              # M^2 sin^2(beta)
    r: float              # reciprocal density ratio
    sigma: float          # branch quantity 1 + gamma u - (gamma+1) u r
    attached: bool        # t > tan(theta)
    above_cj: bool        # u >= u_CJ
    high_compression: bool  # sigma >= 0
    admissible: bool      # all three of the above


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class PolarModel:
    """Equilibrium oblique-detonation polar for one upstream state.

    Parameters
    ----------
    M : float
        Upstream Mach number.
    gamma : float
        Specific-heat ratio, constant on both sides of the front (> 1).
    Q : float
        Non-dimensional heat release ``Q* / (R T1)``.

    Examples
    --------
    >>> import math
    >>> polar = PolarModel(M=7.0, gamma=1.3, Q=10.0)
    >>> round(math.degrees(polar.cj.theta), 4)
    11.0055
    >>> round(math.degrees(polar.detachment().theta_max), 4)
    40.3829
    >>> round(math.degrees(polar.weak_wave_angle(math.radians(30.0))), 6)
    45.777017
    """

    def __init__(self, M: float, gamma: float, Q: float) -> None:
        if gamma <= 1.0:
            raise ValueError(f"gamma must exceed 1, got {gamma!r}")
        if Q < 0.0:
            raise ValueError(f"Q must be non-negative, got {Q!r}")
        if M <= 0.0:
            raise ValueError(f"M must be positive, got {M!r}")
        self.M = float(M)
        self.gamma = float(gamma)
        self.Q = float(Q)
        self.cj = self._cj_point()

    # -- derived scalars ---------------------------------------------------

    @property
    def m(self) -> float:
        """M^2."""
        return self.M * self.M

    @property
    def q_prime(self) -> float:
        """q' = 2 (gamma - 1) Q / gamma."""
        return 2.0 * (self.gamma - 1.0) * self.Q / self.gamma

    def _abc(self) -> tuple[float, float, float]:
        """The grouped coefficients A, B, C of the detachment cubic."""
        return (
            self.q_prime + (self.gamma - 1.0) * self.m + 2.0,
            self.q_prime - (self.m - 1.0),
            (self.gamma + 1.0) * self.m + 2.0,
        )

    # -- CJ endpoint -------------------------------------------------------

    def _cj_point(self) -> CJPoint:
        gamma = self.gamma
        A_cj = 1.0 + (gamma * gamma - 1.0) * self.Q / gamma
        if A_cj < 1.0:
            raise ValueError("no real CJ state for these parameters")
        u_cj = A_cj + math.sqrt(max(0.0, A_cj * A_cj - 1.0))
        M_cj = math.sqrt(u_cj)
        if self.M <= M_cj:
            raise ValueError(
                f"M = {self.M:g} does not exceed M_CJ = {M_cj:.12g}; "
                "no non-degenerate attached branch exists"
            )
        beta_cj = math.asin(M_cj / self.M)
        r_cj = (1.0 + gamma * u_cj) / ((gamma + 1.0) * u_cj)
        theta_cj = beta_cj - math.atan(r_cj * math.tan(beta_cj))
        return CJPoint(M_cj, u_cj, beta_cj, theta_cj, r_cj)

    # -- forward problem ---------------------------------------------------

    def density_ratios(self, beta: float) -> tuple[float, float]:
        """Reciprocal density ratios ``(r_H, r_L)`` at wave angle ``beta``.

        ``r = rho1 / rho2``; ``r_H`` is the high-compression root.  Returns
        ``(nan, nan)`` below the CJ bound, where the roots are complex.
        """
        u = self.m * math.sin(beta) ** 2
        D = self.discriminant(u)
        if D < -_ROOT_TOL * max(1.0, (u - 1.0) ** 2):
            return math.nan, math.nan
        sqrt_D = math.sqrt(max(0.0, D))
        denom = (self.gamma + 1.0) * u
        return (
            (1.0 + self.gamma * u - sqrt_D) / denom,
            (1.0 + self.gamma * u + sqrt_D) / denom,
        )

    def discriminant(self, u: float) -> float:
        """``D = (u - 1)^2 - (gamma + 1) u q'``, the density-ratio discriminant."""
        return (u - 1.0) ** 2 - (self.gamma + 1.0) * u * self.q_prime

    def deflection(self, beta: float, high_compression: bool = True) -> float:
        """Deflection angle on either branch at wave angle ``beta`` [rad].

        This is the forward problem: it is already explicit, and is provided
        for plotting the polar and for independent checks of the inversion.
        """
        r_h, r_l = self.density_ratios(beta)
        r = r_h if high_compression else r_l
        if not math.isfinite(r) or r <= 0.0:
            return math.nan
        return beta - math.atan(r * math.tan(beta))

    # -- inverse problem ---------------------------------------------------

    def cubic_coefficients(self, theta: float) -> tuple[float, float, float, float]:
        """Coefficients ``(a3, a2, a1, a0)`` of the wave-angle cubic in ``t = tan(beta)``.

        The heat release enters as an additive shift of ``a3``, ``a2`` and
        ``a1``; setting ``q' = 0`` recovers the classical oblique-shock cubic.
        """
        tau = math.tan(theta)
        m, qp, gamma = self.m, self.q_prime, self.gamma
        return (
            tau * tau * (qp + (gamma - 1.0) * m + 2.0),
            2.0 * tau * (qp - (m - 1.0)),
            qp + tau * tau * ((gamma + 1.0) * m + 2.0),
            2.0 * tau,
        )

    def wave_angle_roots(self, theta: float) -> list[WaveAngleRoot]:
        """All real roots of the wave-angle cubic at deflection ``theta``, classified.

        Each root carries the three admissibility flags of the ordered filter:
        attachment ``t > tan(theta)``, the CJ bound ``u >= u_CJ``, and the sign
        of the branch quantity ``sigma = 1 + gamma u - (gamma + 1) u r``.
        A root on the high-compression branch has ``sigma >= 0``; one on the
        low-compression branch has ``sigma <= 0``.  The discriminant ``D``
        cannot make this distinction, since ``D = sigma^2`` on both branches.
        """
        tau = math.tan(theta)
        gamma, m = self.gamma, self.m
        out: list[WaveAngleRoot] = []
        for t in real_cubic_roots(*self.cubic_coefficients(theta)):
            u = m * t * t / (1.0 + t * t)
            r = (t - tau) / (t * (1.0 + t * tau))
            sigma = 1.0 + gamma * u - (gamma + 1.0) * u * r
            attached = t > tau - _FILTER_TOL * max(1.0, abs(tau))
            above_cj = u >= self.cj.u_cj - _FILTER_TOL * max(1.0, self.cj.u_cj)
            high = sigma >= -_FILTER_TOL * max(1.0, abs(sigma))
            out.append(
                WaveAngleRoot(
                    t=t,
                    beta=math.atan(t),
                    u=u,
                    r=r,
                    sigma=sigma,
                    attached=attached,
                    above_cj=above_cj,
                    high_compression=high,
                    admissible=attached and above_cj and high,
                )
            )
        return out

    def weak_wave_angle(self, theta: float) -> float:
        """Wave angle [rad] of the weak overdriven solution at deflection ``theta``.

        This is the smallest wave angle surviving the ordered filter.  The weak
        solution exists precisely on ``theta_CJ < theta < theta_max``.

        Raises
        ------
        ValueError
            If no admissible high-compression root exists, i.e. ``theta`` lies
            outside that interval.
        """
        # The CJ endpoint is where D = 0 and the two density-ratio roots
        # coalesce.  Taking it from the cubic there amplifies rounding through
        # the cancellation in D, so the exact CJ formula is used instead.
        if abs(theta - self.cj.theta) <= _ROOT_TOL * max(1.0, abs(self.cj.theta)):
            return self.cj.beta

        admissible = [root for root in self.wave_angle_roots(theta) if root.admissible]
        if not admissible:
            raise ValueError(
                f"no admissible high-compression root at theta = "
                f"{math.degrees(theta):.6g} deg; the deflection is outside "
                f"[{math.degrees(self.cj.theta):.6g}, "
                f"{math.degrees(self.detachment().theta_max):.6g}] deg"
            )
        return min(root.beta for root in admissible)

    def strong_wave_angle(self, theta: float) -> float:
        """Wave angle [rad] of the strong solution at deflection ``theta``.

        This is the largest admissible high-compression root.  It coincides
        with :meth:`weak_wave_angle` at detachment.
        """
        admissible = [root for root in self.wave_angle_roots(theta) if root.admissible]
        if not admissible:
            raise ValueError(
                f"no admissible high-compression root at theta = "
                f"{math.degrees(theta):.6g} deg"
            )
        return max(root.beta for root in admissible)

    # -- detachment --------------------------------------------------------

    def detachment(self) -> DetachmentPoint:
        """Detachment point from the double-root cubic.

        Solves ``A C y^3 + (B C + 3 A) y^2 + 4 B y + q' = 0`` for the double
        root ``y = tan(theta) tan(beta)``, recovers ``s = tan^2(theta)`` from
        ``s = -(3 A y^2 + 4 B y + q') / C``, and returns the admissible root.
        """
        A, B, C = self._abc()
        qp = self.q_prime
        gamma, m = self.gamma, self.m

        best: DetachmentPoint | None = None
        for y in real_cubic_roots(A * C, B * C + 3.0 * A, 4.0 * B, qp):
            if y <= 0.0:
                continue
            s = -(3.0 * A * y * y + 4.0 * B * y + qp) / C
            if s <= 0.0:
                continue
            tau = math.sqrt(s)
            t = y / tau
            if t <= tau:
                continue
            u = m * t * t / (1.0 + t * t)
            if u < self.cj.u_cj - _FILTER_TOL * max(1.0, self.cj.u_cj):
                continue
            r = (t - tau) / (t * (1.0 + t * tau))
            sigma = 1.0 + gamma * u - (gamma + 1.0) * u * r
            if sigma < -_FILTER_TOL * max(1.0, abs(sigma)):
                continue
            candidate = DetachmentPoint(math.atan(tau), math.atan(t), y, s)
            if best is None or candidate.s > best.s:
                best = candidate

        if best is None:
            raise RuntimeError("the detachment cubic has no admissible root")
        return best

    # -- downstream total-sonic locus -------------------------------------

    def sonic_coefficients(self) -> tuple[float, float, float]:
        """Coefficients ``(c2, c1, c0)`` of the sonic quadratic in ``u = M^2 sin^2(beta)``.

        With ``Bhat = q' - (m - 1)``,

            c2 = gamma (gamma q' + 2 Bhat)
            c1 = (gamma + 1) Bhat^2 + 2 Bhat + 2 gamma (m - 1)
            c0 = 2 (m - 1) - q'
        """
        gamma, m, qp = self.gamma, self.m, self.q_prime
        Bhat = qp - (m - 1.0)
        return (
            gamma * (gamma * qp + 2.0 * Bhat),
            (gamma + 1.0) * Bhat * Bhat + 2.0 * Bhat + 2.0 * gamma * (m - 1.0),
            2.0 * (m - 1.0) - qp,
        )

    def sonic_point(self) -> SonicPoint:
        """Downstream total-sonic point (``M2 = 1``) on the high-compression branch.

        Every attached high-compression polar carries exactly one such point,
        and it lies strictly between the CJ endpoint and detachment:
        ``theta_CJ < theta_s < theta_max``.  Only a square root is needed.
        """
        gamma, m, qp = self.gamma, self.m, self.q_prime
        c2, c1, c0 = self.sonic_coefficients()
        scale = max(abs(c2), abs(c1), abs(c0))

        if abs(c2) < 1.0e-14 * scale:
            roots = [-c0 / c1] if c1 != 0.0 else []
        else:
            disc = c1 * c1 - 4.0 * c2 * c0
            if disc < 0.0:
                raise RuntimeError("the sonic quadratic has no real root")
            sq = math.sqrt(disc)
            roots = [(-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2)]

        for u in roots:
            if not (self.cj.u_cj - 1.0e-9 <= u <= m * (1.0 + 1.0e-12)):
                continue
            r = (gamma * u + qp + 2.0 - m) / (1.0 + gamma * u)
            if r <= 0.0:
                continue
            if u >= m:  # degenerate normal-wave endpoint
                continue
            t = math.sqrt(u / (m - u))
            tau = t * (1.0 - r) / (1.0 + r * t * t)
            if tau < 0.0 or t <= tau:
                continue
            return SonicPoint(math.atan(tau), math.atan(t), u, r)

        raise RuntimeError("the sonic quadratic has no admissible root")

    def __repr__(self) -> str:
        return f"PolarModel(M={self.M!r}, gamma={self.gamma!r}, Q={self.Q!r})"


if __name__ == "__main__":
    import doctest

    failures, _ = doctest.testmod()
    raise SystemExit(1 if failures else 0)

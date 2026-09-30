"""Closed-form kernel for the locally linearised model of section 5.1.

At each iteration the mixture enthalpy at the current downstream composition
is linearised about the current temperature.  The resulting model is the
single-gamma jump problem with effective parameters (Gamma, Mcal, chi), where
Mcal = M_e^2 and chi = q'_e = 2 (Gamma - 1) Q_e / Gamma in the notation of the
paper.  In these variables the wave-angle cubic (2.12)-(2.13) reads

    tau^2 [chi + (Gamma-1) Mcal + 2] t^3 + 2 tau [chi - Mcal + 1] t^2
        + [chi + tau^2 ((Gamma+1) Mcal + 2)] t + 2 tau = 0,

with t = tan(beta) and tau = tan(theta).
"""

import numpy as np


def effective_params(a, d, K):
    """(a, d, K) -> (Gamma, Mcal, chi).

    ``a = cp2 / R2`` and ``d = (h2 - cp2 T2 - h1) / (R1 T1)`` describe the
    linearised enthalpy; ``K = u1^2 / (R1 T1)``.  Gamma is the effective ratio
    of (5.2), Mcal = M_e^2, and chi = q'_e.
    """
    Gamma = a / (a - 1.0)
    Mcal = K * (a - 1.0) / a
    chi = -2.0 * (a + d) / a
    return Gamma, Mcal, chi


def cardano(a3, a2, a1, a0, tol=1e-12):
    """Real roots of a3 t^3 + a2 t^2 + a1 t + a0 = 0 (Cardano / trigonometric form)."""
    if abs(a3) < tol * max(abs(a2), abs(a1), abs(a0), 1.0):      # degenerate: quadratic
        if abs(a2) < tol * max(abs(a1), abs(a0), 1.0):
            return [] if abs(a1) < 1e-300 else [-a0 / a1]
        disc = a1 * a1 - 4.0 * a2 * a0
        if disc < 0.0:
            return []
        sq = np.sqrt(disc)
        return sorted([(-a1 - sq) / (2 * a2), (-a1 + sq) / (2 * a2)])

    b, c, d = a2 / a3, a1 / a3, a0 / a3
    p = c - b * b / 3.0                              # depressed cubic, t = s - b/3
    q = 2.0 * b**3 / 27.0 - b * c / 3.0 + d
    shift = -b / 3.0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

    if disc > 0.0:                                   # one real root
        sq = np.sqrt(disc)
        s = np.cbrt(-q / 2.0 + sq) + np.cbrt(-q / 2.0 - sq)
        return [s + shift]
    if abs(p) < 1e-300:                              # p = q = 0
        return [shift]
    m = 2.0 * np.sqrt(-p / 3.0)                      # three real roots
    arg = 3.0 * q / (p * m)
    arg = min(1.0, max(-1.0, arg))
    phi = np.arccos(arg) / 3.0
    return sorted([m * np.cos(phi - 2.0 * np.pi * k / 3.0) + shift for k in range(3)])


def cubic_coeffs(Gamma, Mcal, chi, tau):
    """Coefficients of the wave-angle cubic, highest power first."""
    return (tau * tau * (chi + (Gamma - 1.0) * Mcal + 2.0),
            2.0 * tau * (chi - Mcal + 1.0),
            chi + tau * tau * ((Gamma + 1.0) * Mcal + 2.0),
            2.0 * tau)


def attached_roots(Gamma, Mcal, chi, theta):
    """All roots with t > tan(theta), each with its density ratio and sign quantity."""
    tau = np.tan(theta)
    out = []
    for t in cardano(*cubic_coeffs(Gamma, Mcal, chi, tau)):
        if not np.isfinite(t) or t <= tau:
            continue
        u = Mcal * t * t / (1.0 + t * t)
        r = (t - tau) / (t * (1.0 + t * tau))
        out.append(dict(beta=np.arctan(t), t=t, r=r, u=u))
    out.sort(key=lambda z: z["beta"])
    return out


def dtheta_dbeta(Gamma, Mcal, chi, beta, r):
    """Slope of the linearised polar at a root (beta, r).

    Positive on the weak segment of the high-compression branch and negative
    on the strong segment; it vanishes at detachment.
    """
    u = Mcal * np.sin(beta) ** 2
    dQdr = 2.0 * ((Gamma + 1.0) * u * r - (1.0 + Gamma * u))
    if abs(dQdr) < 1e-300:
        return None
    dQdu = ((Gamma + 1.0) * r - (Gamma - 1.0)) * (r - 1.0)
    drdb = -dQdu * (Mcal * np.sin(2.0 * beta)) / dQdr
    tb = np.tan(beta)
    denom = 1.0 + (r * tb) ** 2
    return float(1.0 - (drdb * tb + r / np.cos(beta) ** 2) / denom)


def classify_root(Gamma, Mcal, chi, beta, r):
    """Branch of one root: 'weak', 'strong' or 'low_compression'.

    The sign quantity sigma = 1 + Gamma u - (Gamma+1) u r of (2.17) separates
    the high- and low-compression branches; on the high-compression branch the
    slope d(theta)/d(beta) separates the weak and strong segments.
    """
    u = Mcal * np.sin(beta) ** 2
    sigma = 1.0 + Gamma * u - (Gamma + 1.0) * u * r
    if sigma < -1e-12:
        return "low_compression"
    sl = dtheta_dbeta(Gamma, Mcal, chi, beta, r)
    if sl is None:
        return "unknown"
    return "weak" if sl > 0.0 else "strong"


def detachment(Gamma, Mcal, chi):
    """Detachment point of the linearised model from the double-root cubic (2.28)-(2.29).

    Returns dict(theta, beta, r) or None.
    """
    A = chi + (Gamma - 1.0) * Mcal + 2.0
    B = chi - (Mcal - 1.0)
    C = (Gamma + 1.0) * Mcal + 2.0
    best = None
    for y in cardano(A * C, B * C + 3.0 * A, 4.0 * B, chi):
        if not (y > 0.0):
            continue
        s = y * y * (A * y + B)
        if not (s > 0.0):
            continue
        tau = np.sqrt(s)
        t = y / tau
        if t <= tau:
            continue
        th = np.arctan(tau)
        if best is None or th > best["theta"]:
            best = dict(theta=float(th), beta=float(np.arctan(t)),
                        r=float((t - tau) / (t * (1.0 + t * tau))))
    return best

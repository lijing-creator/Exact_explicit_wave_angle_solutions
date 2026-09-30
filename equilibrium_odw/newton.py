"""Fixed-composition Newton iteration for the wave angle.

This is the inner step of the reference solver: for a prescribed downstream
composition, the energy balance across the oblique front is solved for the
wave angle beta by Newton iteration, as in the two-step scheme of
Zhang et al. (2022).  It is an independent implementation written from the
published description, not the authors' code.
"""

import numpy as np


class NewtonResult(object):
    def __init__(self, ok, beta, T2, p2, u2, u2n, rho2, n_iter, reason=""):
        self.ok, self.beta, self.T2, self.p2 = ok, beta, T2, p2
        self.u2, self.u2n, self.rho2 = u2, u2n, rho2
        self.n_iter, self.reason = n_iter, reason


def oblique_newton(mix, X1, X2, T1, p1, u1, theta, beta0=None,
                   tol=1e-13, kmax=200, clamp=True):
    """Solve f(beta) = 0 for fixed downstream composition ``X2``.

    ``beta0`` defaults to max(theta, mu1) + 0.1 deg, with mu1 the upstream
    Mach angle, which selects the weak solution.
    """
    R1, R2 = mix.R(X1), mix.R(X2)
    h1 = mix.h(T1, X1)
    rho1 = p1 / (R1 * T1)

    if beta0 is None:
        g1 = mix.cp(T1, X1) / (mix.cp(T1, X1) - R1)
        a1 = np.sqrt(g1 * R1 * T1)
        s = a1 / u1
        if s >= 1.0:
            return NewtonResult(False, np.nan, np.nan, np.nan, np.nan, np.nan,
                                np.nan, 0, "subsonic upstream flow")
        beta0 = max(theta, np.arcsin(s)) + np.radians(0.1)

    lo, hi = theta + 1e-12, np.pi / 2 - 1e-12

    def parts(beta):
        u2n = u1 * np.cos(beta) * np.tan(beta - theta)           # tangential continuity
        T2 = (R1 * T1 / (u1 * np.sin(beta)) + u1 * np.sin(beta) - u2n) * u2n / R2
        return u2n, T2

    beta = float(beta0)
    reason = "max iter"
    for k in range(kmax):
        u2n, T2 = parts(beta)
        if not np.isfinite(T2) or T2 <= 0.0:
            return NewtonResult(False, beta, T2, np.nan, np.nan, u2n, np.nan,
                                k, "T2<=0")
        # energy residual; normal kinetic energies only
        f = mix.h(T2, X2) + 0.5 * u2n**2 - h1 - 0.5 * (u1 * np.sin(beta))**2

        cp2 = mix.cp(T2, X2)
        sb, cb = np.sin(beta), np.cos(beta)
        du2n_dbeta = u1 * (cb / np.cos(beta - theta)**2 - sb * np.tan(beta - theta))
        dT2_dbeta = u2n / R2 * (u1 * cb - (R1 * T1 / u1) * cb / sb**2)
        dT2_du2n = (R1 * T1 / (u1 * sb) + u1 * sb - 2.0 * u2n) / R2
        df_dbeta = -u1**2 * sb * cb
        fp = df_dbeta + u2n * du2n_dbeta + cp2 * (dT2_dbeta + dT2_du2n * du2n_dbeta)

        if not np.isfinite(fp) or abs(fp) < 1e-300:
            return NewtonResult(False, beta, T2, np.nan, np.nan, u2n, np.nan,
                                k, "f'=0")

        beta_new = beta - f / fp
        if clamp:
            beta_new = min(max(beta_new, lo), hi)
        elif not (lo < beta_new < hi):
            return NewtonResult(False, beta_new, T2, np.nan, np.nan, u2n,
                                np.nan, k + 1, "out of range")
        done = abs(beta_new - beta) < tol
        beta = beta_new
        if done:
            reason = "converged"
            break

    u2n, T2 = parts(beta)
    if not np.isfinite(T2) or T2 <= 0.0:
        return NewtonResult(False, beta, T2, np.nan, np.nan, u2n, np.nan,
                            k + 1, "T2<=0")
    rho2 = rho1 * u1 * np.sin(beta) / u2n
    p2 = rho2 * R2 * T2
    u2 = u2n / np.sin(beta - theta)
    return NewtonResult(reason == "converged", beta, T2, p2, u2, u2n, rho2,
                        k + 1, reason)

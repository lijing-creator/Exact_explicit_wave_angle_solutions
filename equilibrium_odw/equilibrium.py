"""Chemical-equilibrium composition at prescribed temperature and pressure.

Gibbs free-energy minimisation subject to element conservation, following the
NASA formulation of Gordon & McBride (1994).  For given (T, p) and element
moles b_j, the species moles x_i minimise

    G/(Ru T) = sum_i x_i [G_i0/(Ru T) + ln(x_i / sum x) + ln(p / p_ref)]

subject to sum_i a_ij x_i = b_j.  Each iteration solves the linear system for
the Lagrange multipliers and the total-moles ratio.  The update is applied to
ln x_i with a step limit, so that trace species cannot become negative; this
changes the path to the minimum, not the minimum itself.
"""

import numpy as np

from .thermo import P_REF


class EquilibriumSolver(object):
    """Equilibrium solver for one ``Mixture``.

    ``evaluations`` counts calls of :meth:`solve`, the number reported as
    chemical-equilibrium evaluations in the paper.  The defaults ``tol`` and
    ``kmax`` are the values used there.
    """

    def __init__(self, mixture, p_ref=P_REF, tol=1e-14, kmax=800, max_dln=2.0,
                 stat_tol=1e-6):
        self.mix = mixture
        self.p_ref = p_ref
        self.tol = tol
        self.kmax = kmax
        self.max_dln = max_dln
        self.stat_tol = stat_tol
        self.evaluations = 0           # calls of solve()
        self.inner_iters = 0           # accumulated minimisation iterations
        self.cold_fallbacks = 0        # warm starts rejected and redone cold

    def element_moles(self, X):
        """Element moles b_j of a composition given as mole fractions."""
        return self.mix.a.T.dot(np.asarray(X, float))

    def stationarity(self, T, p, n):
        """Residual of the stationarity condition of the Gibbs minimum.

        At the minimum, G_i0/(Ru T) + ln(x_i / sum x) + ln(p / p_ref) lies in the
        column space of the element matrix.  The unweighted residual is used,
        so that a trace species driven to a spurious zero is not hidden.
        """
        n = np.maximum(np.asarray(n, float), 1e-300)
        mu = self.mix.g0_over_RuT(T) + np.log(n / n.sum()) \
            + np.log(max(p, 1e-30) / self.p_ref)
        lam, *_ = np.linalg.lstsq(self.mix.a, mu, rcond=None)
        return float(np.max(np.abs(mu - self.mix.a.dot(lam))))

    def solve(self, T, p, b, y0=None):
        """Equilibrium at (T, p) for element moles ``b``.

        ``y0`` are species moles used as a warm start; a warm-started result
        that fails the stationarity check is discarded and recomputed from a
        cold start.  Returns (mole fractions, moles, iterations).
        """
        self.evaluations += 1
        if y0 is not None:
            X, n, k = self._solve_from(T, p, b, y0)
            if self.stationarity(T, p, n) < self.stat_tol:
                return X, n, k
            self.cold_fallbacks += 1
        return self._solve_from(T, p, b, None)

    def _solve_from(self, T, p, b, y0):
        mix, nsp = self.mix, self.mix.n
        nEl = len(mix.elements)
        a = mix.a                                        # (nsp, nEl)

        if y0 is None:                                   # CEA cold start
            y = np.full(nsp, 0.1 / nsp)
        else:
            y = np.maximum(np.asarray(y0, float), 1e-30)

        gRT = mix.g0_over_RuT(T)
        lnp = np.log(max(p, 1e-30) / self.p_ref)

        for k in range(self.kmax):
            ntot = y.sum()
            d = y * (gRT + np.log(y / ntot) + lnp)

            A = a.T.dot(y)
            B = (a.T * y).dot(a)
            D = a.T.dot(d)

            M = np.zeros((nEl + 1, nEl + 1))
            rhs = np.zeros(nEl + 1)
            M[0, :nEl] = A
            rhs[0] = d.sum()
            M[1:, :nEl] = B
            M[1:, nEl] = A
            rhs[1:] = b + D
            try:
                sol = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
            lam, u = sol[:nEl], sol[nEl]

            x = -d + y * (u + a.dot(lam))

            # limited step in ln x
            with np.errstate(divide="ignore", invalid="ignore"):
                dln = np.where(x > 0.0, np.log(np.maximum(x, 1e-300) / y), -5.0)
            big = np.max(np.abs(dln))
            omega = 1.0 if big <= self.max_dln else self.max_dln / big
            y_new = y * np.exp(omega * dln)
            y_new = np.maximum(y_new, 1e-300)

            err = np.max(np.abs(y_new - y)) / max(y_new.sum(), 1e-300)
            y = y_new
            self.inner_iters += 1
            if err < self.tol:
                return y / y.sum(), y, k + 1

        return y / y.sum(), y, self.kmax

#!/usr/bin/env python3
"""Independent check of the two-gamma reduction, section 4 of the paper.

The reference never uses the effective parameters (4.9).  For each wave angle
it solves the dimensional two-gamma jump conditions (4.1)-(4.3) directly: with
p1 = rho1 = 1 the energy balance is a quadratic in the density ratio r, whose
smaller root is the high-compression branch.  Critical points are then found
numerically on that polar:

  CJ point        Brent root of the discriminant of the r-quadratic in beta
  detachment      bounded maximisation of theta(beta)
  sonic point     Brent root of M2(beta) - 1 on the weak segment
  wave angles     Brent roots of theta(beta) - theta on each segment

and compared with ``odw_polar.two_gamma_polar``.  The grid includes
gamma1 < gamma2, gamma1 = gamma2 and small effective heat release.

Run:  python two_gamma_check.py      (exit code 0 on PASS, 1 on FAIL)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odw_polar import two_gamma_parameters, two_gamma_polar  # noqa: E402

DEG = 180.0 / math.pi
TOL_DEG = 1.0e-9


class DirectTwoGamma:
    """Two-gamma polar from the jump conditions, with p1 = rho1 = 1."""

    def __init__(self, M, gamma1, gamma2, Q):
        self.g1, self.g2, self.Q = gamma1, gamma2, Q
        self.U1 = M * math.sqrt(gamma1)          # upstream speed, since a1^2 = gamma1

    def _quadratic(self, beta):
        # gamma1/(gamma1-1) + a/2 + Q = k r [1 + a (1 - r)] + a r^2 / 2,  a = u_n1^2
        a = (self.U1 * math.sin(beta)) ** 2
        k = self.g2 / (self.g2 - 1.0)
        return a, (a / 2.0 - k * a, k * (1.0 + a), -(self.g1 / (self.g1 - 1.0) + a / 2.0 + self.Q))

    def discriminant(self, beta):
        _, (A2, A1, A0) = self._quadratic(beta)
        return A1 * A1 - 4.0 * A2 * A0

    def r_high(self, beta):
        a, (A2, A1, A0) = self._quadratic(beta)
        disc = max(A1 * A1 - 4.0 * A2 * A0, 0.0)
        return min((-A1 + s * math.sqrt(disc)) / (2.0 * A2) for s in (1.0, -1.0))

    def r_double(self, beta):
        # at the CJ point the two roots coincide; the root formula without the
        # square root avoids amplifying the residual of the discriminant
        _, (A2, A1, _) = self._quadratic(beta)
        return -A1 / (2.0 * A2)

    def theta(self, beta, r=None):
        r = self.r_high(beta) if r is None else r
        return beta - math.atan(r * math.tan(beta))

    def mach2(self, beta):
        r = self.r_high(beta)
        un1 = self.U1 * math.sin(beta)
        p2 = 1.0 + un1 * un1 * (1.0 - r)
        ut = self.U1 * math.cos(beta)
        return math.sqrt(((un1 * r) ** 2 + ut * ut) / (self.g2 * p2 * r))


CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    if not ok:
        print(f"[FAIL] {name}  {detail}")


def compare_case(M, g1, g2, Q):
    polar = two_gamma_polar(M, g1, g2, Q)
    ref = DirectTwoGamma(M, g1, g2, Q)
    tag = f"M={M:g}, gamma1={g1:g}, gamma2={g2:g}, Q={Q:g}"
    errs = {}

    # the discriminant turns positive at the CJ point; bracket it on a grid in beta
    grid = np.linspace(1.0e-3, math.pi / 2, 20001)
    neg = [i for i, b in enumerate(grid) if ref.discriminant(b) < 0.0]
    b_cj = brentq(ref.discriminant, grid[neg[-1]], grid[neg[-1] + 1], xtol=1e-15, rtol=1e-15)
    errs["beta_CJ"] = abs(b_cj - polar.cj.beta) * DEG
    errs["theta_CJ"] = abs(ref.theta(b_cj, ref.r_double(b_cj)) - polar.cj.theta) * DEG

    det = polar.detachment()
    opt = minimize_scalar(lambda b: -ref.theta(b), bounds=(b_cj, math.pi / 2 - 1e-9),
                          method="bounded", options={"xatol": 1e-12})
    errs["theta_max"] = abs(-opt.fun - det.theta_max) * DEG
    # beta_d is checked by evaluating the direct polar at the closed-form double root
    errs["theta(beta_d)"] = abs(ref.theta(det.beta) - det.theta_max) * DEG

    son = polar.sonic_point()
    b_s = brentq(lambda b: ref.mach2(b) - 1.0, b_cj + 1e-12, det.beta, xtol=1e-15, rtol=1e-15)
    errs["beta_s"] = abs(b_s - son.beta) * DEG
    errs["theta_s"] = abs(ref.theta(b_s) - son.theta) * DEG

    th = 0.5 * (polar.cj.theta + det.theta_max)
    f = lambda b: ref.theta(b) - th
    errs["weak"] = abs(brentq(f, b_cj, det.beta, xtol=1e-15, rtol=1e-15)
                       - polar.weak_wave_angle(th)) * DEG
    errs["strong"] = abs(brentq(f, det.beta, math.pi / 2 - 1e-12, xtol=1e-15, rtol=1e-15)
                         - polar.strong_wave_angle(th)) * DEG

    worst = max(errs, key=errs.get)
    check(f"closed form = direct two-gamma solution ({tag})", errs[worst] < TOL_DEG,
          f"largest error {errs[worst]:.2e} deg in {worst}")
    return errs


def main():
    print("Two-gamma reduction: closed form vs direct solution of (4.1)-(4.3)")
    worst = {}
    n = 0
    for g1, g2 in [(1.4, 1.2), (1.3, 1.15), (1.2, 1.35), (1.4, 1.25), (1.33, 1.33)]:
        for Q in [5.0, 10.0, 30.0]:
            for M in [6.0, 8.0, 10.0]:
                _, _, Qe = two_gamma_parameters(M, g1, g2, Q)
                try:
                    two_gamma_polar(M, g1, g2, Q)
                except ValueError:
                    continue                      # Q_e < 0 or M below the CJ Mach number
                for k, v in compare_case(M, g1, g2, Q).items():
                    worst[k] = max(worst.get(k, 0.0), v)
                n += 1
    for k, v in worst.items():
        print(f"    max |error| {k:>14s}: {v:.2e} deg")
    check("parameter cases compared", n >= 30, f"{n} cases")

    g, Me, Qe = two_gamma_parameters(7.0, 1.3, 1.3, 10.0)
    check("gamma1 = gamma2 recovers the single-gamma model",
          g == 1.3 and abs(Me - 7.0) < 1e-15 and abs(Qe - 10.0) < 1e-12)
    try:
        two_gamma_polar(8.0, 1.4, 1.2, 1.0)       # Q_e = 1 + 3.5 - 6 < 0
        rejected = False
    except ValueError:
        rejected = True
    check("Q_e < 0 is rejected", rejected)

    n_ok = sum(ok for _, ok, _ in CHECKS)
    verdict = "PASS" if n_ok == len(CHECKS) else "FAIL"
    print(f"{verdict}: {n_ok}/{len(CHECKS)} checks, {n} parameter cases, tolerance {TOL_DEG:g} deg")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

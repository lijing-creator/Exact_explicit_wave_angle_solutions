"""Equilibrium oblique detonations at a prescribed deflection angle, section 5 of the paper.

Both solvers use the two-step scheme of Zhang et al. (2022): the flow state is
computed at a fixed downstream composition, the equilibrium composition is
computed at the resulting temperature and pressure, and the composition is
relaxed,

    X2 <- (1 - lam) X2 + lam X2_eq,

until the flow state and composition converge.  They differ only in the flow
step.

``solve_closed_form``  linearises the mixture enthalpy about the current state
                       and takes the wave angle from the closed-form cubic of
                       the resulting single-gamma model (section 5.1);
``solve_newton``       solves the fixed-composition energy balance for the
                       wave angle by Newton iteration, starting each outer
                       iteration from the previous wave angle.

The Newton solver is an independent implementation of the published scheme,
used as the reference in the paper; it is not the authors' code.

Arguments common to both solvers
--------------------------------
mix, eq      ``thermo.Mixture`` and ``equilibrium.EquilibriumSolver``
X1           upstream mole fractions
T1, p1, u1   upstream temperature [K], pressure [Pa] and speed [m/s]
theta        deflection angle [rad]
branch       'weak' or 'strong'
lam, tol, kmax   composition relaxation factor, convergence tolerance and
                 maximum number of outer iterations
"""

from dataclasses import dataclass, field

import numpy as np

from .closed_form import attached_roots, classify_root, detachment, effective_params
from .equilibrium import EquilibriumSolver
from .newton import oblique_newton
from .thermo import Mixture, load_nasa9

LAM, TOL, KMAX = 0.4, 1e-10, 300
BETA0_STRONG = 0.97 * np.pi / 2       # initial wave angle of the Newton solver, strong branch

AIR_SPECIES = ['H2', 'H', 'O2', 'O', 'OH', 'HO2', 'H2O2', 'H2O', 'N', 'N2', 'NO']
FUEL_MOLES = {'H2': 0.42, 'CH4': 0.105, 'C2H4': 0.07}   # per 0.21 O2 + 0.79 N2 at phi = 1


@dataclass
class Solution:
    """Result of one solve."""

    ok: bool = False
    reason: str = ""
    branch: str = ""
    beta: float = np.nan          # wave angle [rad]
    T2: float = np.nan            # downstream temperature [K]
    p2: float = np.nan            # downstream pressure [Pa]
    u2: float = np.nan            # downstream speed [m/s]
    X2: np.ndarray = field(default=None, repr=False)   # downstream mole fractions
    iterations: int = 0           # outer iterations
    eq_evaluations: int = 0       # chemical-equilibrium evaluations
    inner: int = 0                # cubic solutions (closed form) or Newton steps
    relocated: bool = False       # first-iteration linearisation point moved (closed form)
    T_seed: float = np.nan        # temperature of the moved linearisation point


def fuel_air(fuel, thermo, phi=1.0):
    """Fuel-air mixture of section 5.2: returns (mixture, equilibrium solver, X1).

    ``fuel`` is 'H2', 'CH4' or 'C2H4'; ``thermo`` is the path of a NASA-9
    library or a dictionary returned by ``load_nasa9``.  Air is 0.21 O2 +
    0.79 N2 by volume.
    """
    db = load_nasa9(thermo) if isinstance(thermo, str) or hasattr(thermo, '__fspath__') else thermo
    names = list(AIR_SPECIES)
    if fuel != 'H2':
        names = [fuel, 'CO', 'CO2'] + names + ['CH3', 'HCO']
    missing = [n for n in names if n not in db]
    if missing:
        raise KeyError("species missing from the thermodynamic library: " + ", ".join(missing))
    mix = Mixture([db[n] for n in names])
    X = np.zeros(len(names))
    X[names.index(fuel)] = FUEL_MOLES[fuel] * phi
    X[names.index('O2')], X[names.index('N2')] = .21, .79
    return mix, EquilibriumSolver(mix), X / X.sum()


def frozen_sound_speed(mix, X1, T1):
    """Frozen upstream sound speed, used to convert a Mach number into ``u1``."""
    R1, cp1 = mix.R(X1), mix.cp(T1, X1)
    return float(np.sqrt(cp1 / (cp1 - R1) * R1 * T1))


def _convergence_error(p2, p2_old, T2, T2_old, Xeq, X2):
    """Largest change in composition, pressure and temperature."""
    e = max(abs(Xeq - X2))
    if p2_old is not None:
        e = max(e, abs(p2 - p2_old) / abs(p2), abs(T2 - T2_old) / abs(T2))
    else:
        e = 1.0
    return e


def _pick(Gamma, Mcal, chi, theta, branch):
    """Root of the requested branch with the smallest wave angle, or None."""
    hit = [z for z in attached_roots(Gamma, Mcal, chi, theta)
           if classify_root(Gamma, Mcal, chi, z["beta"], z["r"]) == branch]
    return hit[0] if hit else None


def _closed_form_iteration(mix, eq, X1, T1, p1, u1, theta, branch, lam, tol, kmax,
                           X2, T2, sol):
    b = eq.element_moles(X1)
    R1 = mix.R(X1)
    h1 = mix.h(T1, X1)
    v1 = R1 * T1 / p1
    rho1 = 1.0 / v1
    K = u1 * u1 / (R1 * T1)
    y0 = None
    p2_old = T2_old = None

    for k in range(kmax):
        # linearised enthalpy at the current state -> effective parameters
        R2 = mix.R(X2)
        cp2 = mix.cp(T2, X2)
        h2 = mix.h(T2, X2)
        Gamma, Mcal, chi = effective_params(cp2 / R2, (h2 - cp2 * T2 - h1) / (R1 * T1), K)

        sol.inner += 1
        pick = _pick(Gamma, Mcal, chi, theta, branch)
        if pick is None:
            sol.reason = "no %s root of the cubic (iteration %d)" % (branch, k)
            sol.iterations = k + 1
            return sol, k
        beta, r = pick["beta"], pick["r"]

        # wave angle -> density ratio -> pressure and temperature
        u1n = u1 * np.sin(beta)
        p2 = p1 + rho1 * u1n * u1n * (1.0 - r)
        T2s = p2 * (r * v1) / R2
        u2 = u1 * np.cos(beta) / np.cos(beta - theta)
        if not np.isfinite(T2s) or T2s <= 0.0:
            sol.reason = "non-positive temperature (iteration %d)" % k
            sol.iterations = k + 1
            return sol, None

        Xeq, n_eq, _ = eq.solve(T2s, p2, b, y0)
        y0 = n_eq
        err = _convergence_error(p2, p2_old, T2s, T2_old, Xeq, X2)
        p2_old, T2_old = p2, T2s
        sol.iterations = k + 1
        if err < tol:
            sol.ok, sol.reason = True, "converged"
            sol.beta, sol.T2, sol.p2, sol.u2 = beta, T2s, p2, u2
            sol.X2 = X2.copy()
            return sol, None
        X2 = (1.0 - lam) * X2 + lam * Xeq
        T2 = T2s
    sol.reason = "outer iteration did not converge"
    return sol, None


def solve_closed_form(mix, eq, X1, T1, p1, u1, theta, branch="weak",
                      lam=LAM, tol=TOL, kmax=KMAX):
    """Closed-form solver of section 5.1.

    The iteration starts from the frozen upstream state, T2 = T1 and X2 = X1,
    where the linearised model is the inert oblique shock of the upstream gas.
    Near detachment the requested deflection can exceed the maximum deflection
    of that model, so the cubic has no root in the first iteration.  The
    linearisation temperature is then moved to the temperature at the
    detachment point of that model, given in closed form by (2.28)-(2.29), and
    the iteration restarts at the requested deflection.  The composition stays
    frozen, and no equilibrium evaluation is made for the move.
    """
    X1 = np.asarray(X1, float)
    n0 = eq.evaluations
    sol, fail_k = _closed_form_iteration(mix, eq, X1, T1, p1, u1, theta, branch,
                                         lam, tol, kmax, X1.copy(), float(T1),
                                         Solution(branch=branch))
    if fail_k == 0:
        R1 = mix.R(X1)
        cp1 = mix.cp(T1, X1)
        rho1 = p1 / (R1 * T1)
        v1 = 1.0 / rho1
        frozen = effective_params(cp1 / R1, -cp1 * T1 / (R1 * T1), u1**2 / (R1 * T1))
        dt = detachment(*frozen)
        if dt is not None and theta > dt["theta"]:
            ps = p1 + rho1 * u1**2 * np.sin(dt["beta"])**2 * (1 - dt["r"])
            seed = ps * dt["r"] * v1 / R1
            retry = Solution(branch=branch, relocated=True, T_seed=float(seed), inner=sol.inner)
            sol, _ = _closed_form_iteration(mix, eq, X1, T1, p1, u1, theta, branch,
                                            lam, tol, kmax, X1.copy(), float(seed), retry)
    sol.eq_evaluations = eq.evaluations - n0
    return sol


def solve_newton(mix, eq, X1, T1, p1, u1, theta, branch="weak",
                 lam=LAM, tol=TOL, kmax=KMAX, beta0=None, warm_start=True):
    """Reference solver: fixed-composition Newton iteration for the wave angle.

    The branch is set by the initial wave angle: by default the weak branch
    starts from max(theta, mu1) + 0.1 deg and the strong branch from
    0.97 x 90 deg.  With ``warm_start`` (the setting used in the paper) every
    later outer iteration starts from the previous wave angle.
    """
    X1 = np.asarray(X1, float)
    sol = Solution(branch=branch)
    if beta0 is None and branch == "strong":
        beta0 = BETA0_STRONG
    b = eq.element_moles(X1)
    n0 = eq.evaluations
    X2 = X1.copy()
    y0 = None
    p2_old = T2_old = None
    bstart = beta0

    for k in range(kmax):
        res = oblique_newton(mix, X1, X2, T1, p1, u1, theta, beta0=bstart, clamp=True)
        sol.inner += res.n_iter
        sol.iterations = k + 1
        if not res.ok:
            sol.reason = "Newton iteration failed: " + res.reason
            break
        Xeq, n_eq, _ = eq.solve(res.T2, res.p2, b, y0)
        y0 = n_eq
        err = _convergence_error(res.p2, p2_old, res.T2, T2_old, Xeq, X2)
        p2_old, T2_old = res.p2, res.T2
        if err < tol:
            sol.ok, sol.reason = True, "converged"
            sol.beta, sol.T2, sol.p2, sol.u2 = res.beta, res.T2, res.p2, res.u2
            sol.X2 = X2.copy()
            break
        X2 = (1.0 - lam) * X2 + lam * Xeq
        if warm_start:
            bstart = res.beta
    else:
        sol.reason = "outer iteration did not converge"
    sol.eq_evaluations = eq.evaluations - n0
    return sol

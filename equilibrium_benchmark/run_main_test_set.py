#!/usr/bin/env python3
"""Main test set of section 5.2: both equilibrium solvers on 150 cases x 2 branches.

Stoichiometric hydrogen-, methane- and ethylene-air at T1 = 300 K and
p1 = 1 atm, M = 6-10, ten deflection angles per condition,
theta = theta_CJ + f (theta_max - theta_CJ) with f = 0.05, 0.15, ..., 0.95.
The prescribed angles and upstream speeds are listed in
``main_test_set_tasks.csv``; theta_CJ and theta_max there are those of the
reference equilibrium polar.

The script reports convergence, the agreement between the two solvers, and
the elapsed times and chemical-equilibrium evaluations of Table 2.  Each round
solves every case with both solvers, alternating which solver goes first;
round 0 is an untimed warm-up.  The Newton-based solver starts every outer
iteration from the previous wave angle.

Thermodynamic data are not distributed.  Pass the path of a NASA-9 library:

    python run_main_test_set.py path/to/thermo.inp [--rounds 7]

The paper used the tables of Combustion Toolbox 1.2.9; other data give
slightly different states and timings.
"""

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from equilibrium_odw import fuel_air, load_nasa9, solve_closed_form, solve_newton  # noqa: E402

T1, P1 = 300.0, 101325.0
METHODS = (("Newton-based", solve_newton), ("Closed-form", solve_closed_form))
MIXTURES = {"H2": "Hydrogen-air", "CH4": "Methane-air", "C2H4": "Ethylene-air"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("thermo", help="NASA-9 thermodynamic library")
    ap.add_argument("--rounds", type=int, default=7, help="timed rounds (default 7)")
    args = ap.parse_args()

    with open(HERE / "main_test_set_tasks.csv", encoding="utf-8") as fh:
        tasks = list(csv.DictReader(fh))
    db = load_nasa9(args.thermo)
    gases = {g: fuel_air(g, db) for g in MIXTURES}

    times = []                      # per timed round: {(method, gas): seconds}
    evals = {}                      # first timed round: {(method, gas): evaluations}
    states = {}                     # first timed round: {(id, method, branch): Solution}
    for rnd in range(1 + args.rounds):
        now = {}
        for t in tasks:
            gas, tid = t["gas"], int(t["id"])
            mix, eq, X1 = gases[gas]
            theta, u1 = math.radians(float(t["theta_deg"])), float(t["U"])
            s = (rnd + tid) % len(METHODS)
            for name, solve in METHODS[s:] + METHODS[:s]:
                for branch in ("weak", "strong"):
                    tic = time.perf_counter()
                    sol = solve(mix, eq, X1, T1, P1, u1, theta, branch=branch)
                    sec = time.perf_counter() - tic
                    if rnd == 0:
                        continue
                    now[name, gas] = now.get((name, gas), 0.0) + sec
                    if rnd == 1:
                        evals[name, gas] = evals.get((name, gas), 0) + sol.eq_evaluations
                        states[tid, name, branch] = sol
        if rnd > 0:
            times.append(now)
            print(f"round {rnd}: " + ", ".join(f"{m} {sum(v for (n, _), v in now.items() if n == m):.3f} s"
                                               for m, _ in METHODS), flush=True)

    # convergence and agreement between the two solvers
    print()
    for name, _ in METHODS:
        ok = sum(sol.ok for (_, n, _), sol in states.items() if n == name)
        print(f"{name}: {ok}/{len(tasks) * 2} branch states converged")
    rel = {"wave angle": 0.0, "temperature": 0.0, "pressure": 0.0}
    dX = 0.0
    for t in tasks:
        for branch in ("weak", "strong"):
            a, b = states[int(t["id"]), "Newton-based", branch], states[int(t["id"]), "Closed-form", branch]
            if not (a.ok and b.ok):
                continue
            rel["wave angle"] = max(rel["wave angle"], abs(b.beta - a.beta) / abs(a.beta))
            rel["temperature"] = max(rel["temperature"], abs(b.T2 - a.T2) / abs(a.T2))
            rel["pressure"] = max(rel["pressure"], abs(b.p2 - a.p2) / abs(a.p2))
            dX = max(dX, float(np.max(np.abs(b.X2 - a.X2))))
    for k, v in rel.items():
        print(f"max relative difference, {k:<12s} {v:.2e}")
    print(f"max absolute difference, mole fractions {dX:.2e}")

    # Table 2
    print()
    print(f"{'Mixture':<14s}{'Newton (s)':>12s}{'Closed (s)':>12s}{'Newton evals':>14s}"
          f"{'Closed evals':>14s}{'Speedup':>9s}")
    rows = [(MIXTURES[g], [g]) for g in MIXTURES] + [("All cases", list(MIXTURES))]
    for label, gs in rows:
        tn = [sum(r["Newton-based", g] for g in gs) for r in times]
        tc = [sum(r["Closed-form", g] for g in gs) for r in times]
        en = sum(evals["Newton-based", g] for g in gs)
        ec = sum(evals["Closed-form", g] for g in gs)
        speedup = statistics.median(a / b for a, b in zip(tn, tc))
        print(f"{label:<14s}{statistics.median(tn):>12.3f}{statistics.median(tc):>12.3f}"
              f"{en:>14d}{ec:>14d}{speedup:>9.2f}")
    print(f"\nTimes are medians over {args.rounds} rounds; speedups are medians of the "
          "per-round ratios.")


if __name__ == "__main__":
    main()

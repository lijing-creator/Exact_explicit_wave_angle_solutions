#!/usr/bin/env python3
"""Worked example for section 5: stoichiometric hydrogen-air with equilibrium products.

    python example_equilibrium.py path/to/thermo.inp

Thermodynamic data are not distributed; pass a NASA-9 library such as the
``thermo.inp`` of NASA CEA or ``thermo_CT.inp`` of Combustion Toolbox.
"""

import math
import sys

from equilibrium_odw import fuel_air, frozen_sound_speed, solve_closed_form, solve_newton

if len(sys.argv) != 2:
    raise SystemExit(__doc__)

T1, p1, M = 300.0, 101325.0, 8.0
theta = math.radians(30.0)
mix, eq, X1 = fuel_air("H2", sys.argv[1])
u1 = M * frozen_sound_speed(mix, X1, T1)

print(f"H2-air, phi = 1, T1 = {T1:g} K, p1 = {p1:g} Pa, M = {M:g}, theta = 30 deg\n")
print(f"{'solver':<14s}{'branch':<8s}{'beta (deg)':>12s}{'T2 (K)':>10s}{'p2 (MPa)':>10s}"
      f"{'X_H2O':>8s}{'outer':>7s}{'eq evals':>10s}")
for name, solve in (("closed-form", solve_closed_form), ("Newton-based", solve_newton)):
    for branch in ("weak", "strong"):
        s = solve(mix, eq, X1, T1, p1, u1, theta, branch=branch)
        if not s.ok:
            print(f"{name:<14s}{branch:<8s}  {s.reason}")
            continue
        x_h2o = s.X2[mix.names.index("H2O")]
        print(f"{name:<14s}{branch:<8s}{math.degrees(s.beta):>12.6f}{s.T2:>10.2f}{s.p2 / 1e6:>10.4f}"
              f"{x_h2o:>8.4f}{s.iterations:>7d}{s.eq_evaluations:>10d}")

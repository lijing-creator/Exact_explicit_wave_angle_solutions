"""Equilibrium oblique detonations with chemically equilibrated products (section 5).

The closed-form wave-angle cubic replaces the inner wave-angle iteration of
the two-step chemical-equilibrium scheme.  Thermodynamic data are not
distributed; pass the path of a NASA-9 library to ``fuel_air`` or
``load_nasa9``.
"""

from .equilibrium import EquilibriumSolver
from .solvers import Solution, frozen_sound_speed, fuel_air, solve_closed_form, solve_newton
from .thermo import Mixture, load_nasa9

__all__ = ["EquilibriumSolver", "Mixture", "Solution", "frozen_sound_speed", "fuel_air",
           "load_nasa9", "solve_closed_form", "solve_newton"]

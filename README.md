**English** | [简体中文](README.zh-CN.md)

# Exact explicit wave-angle solutions for equilibrium oblique detonations

Code accompanying

> J. Li and C. Luo, *Exact explicit wave-angle solutions for equilibrium
> oblique detonations* (submitted).

The equilibrium oblique-detonation polar links the wedge deflection angle
$\theta$, the wave angle $\beta$, the upstream Mach number $M$ and the heat
release $Q$. Going forward — prescribe $\beta$, get $\theta$ — is elementary.
Going backward — prescribe the wedge angle $\theta$ and ask for the wave
angles — has normally been done by reading a plotted polar or by iterating.

The paper shows that the inverse problem is exactly solvable in closed form,
and this repository is the reference implementation. It also contains the
chemical-equilibrium solvers of section 5, in which the closed-form solution
replaces the inner wave-angle iteration. **If you just want the
wave angle for a given wedge angle, that is one call.**

```python
import math
from odw_polar import PolarModel

polar = PolarModel(M=7.0, gamma=1.3, Q=10.0)      # Q = Q*/(R T1)
beta = polar.weak_wave_angle(math.radians(30.0))  # wedge angle in
print(math.degrees(beta))                         # -> 45.777017  (deg)
```

No iteration, no initial guess, no bracketing interval. Everything reduces to
a cubic (solved by Cardano) or a quadratic (solved by a square root).

The same closed-form solution covers the two-γ model, in which the
specific-heat ratio differs on the two sides of the front: an exact change of
parameters maps it onto the single-γ model (§4 of the paper). See
[Two-γ model](#two-γ-model) below.

## Model and conventions

Calorically perfect gas with the same gas constant $R$ and specific-heat ratio
$\gamma>1$ on both sides. Heat release is instantaneous and complete, so the
front is a zero-thickness equilibrium discontinuity attached to a straight
wedge.

The non-dimensional heat release follows the paper:

```
Q  = Q* / (R T1)                q' = 2 (gamma - 1) Q / gamma
```

with `Q*` the chemical energy release per unit mass and `T1` the upstream
static temperature. **Check this convention against your own** — several
non-dimensionalisations of the heat release are in circulation, and they are
not interchangeable.

All angles in the API are radians. Use `math.degrees` at the call site.

## What is implemented

`odw_polar.py` is a single self-contained module that needs **only the Python
standard library**. Each entry point corresponds to a numbered result in the
paper.

| Call | Returns | Paper |
|---|---|---|
| `PolarModel(M, gamma, Q)` | model for one upstream state | §2.1 |
| `.cj` | CJ endpoint: `M_cj`, `u_cj`, `beta`, `theta`, `r_cj` | (2.7), (2.8) |
| `.density_ratios(beta)` | both roots `(r_H, r_L)` at a wave angle | (2.6) |
| `.deflection(beta, high_compression)` | forward problem, either branch | (2.4) |
| `.cubic_coefficients(theta)` | `(a3, a2, a1, a0)` of the wave-angle cubic | (2.12), (2.13) |
| `.wave_angle_roots(theta)` | **all** real roots, each with its admissibility flags | (2.12), (2.16), (2.17) |
| `.weak_wave_angle(theta)` | weak overdriven solution | §2.3 |
| `.strong_wave_angle(theta)` | strong solution | §2.3 |
| `.detachment()` | `theta_max`, `beta_d` from the double-root cubic | (2.27)–(2.29) |
| `.sonic_coefficients()` | `(c2, c1, c0)` of the sonic quadratic | (3.5), (3.6) |
| `.sonic_point()` | downstream total-sonic point, `M2 = 1` | (3.7), (3.8) |
| `real_cubic_roots(a,b,c,d)` | real Cardano / trigonometric cubic solver | (2.18)–(2.23) |
| `two_gamma_parameters(M, gamma1, gamma2, Q)` | effective parameters `(gamma_e, M_e, Q_e)` of the two-γ model | (4.9) |
| `two_gamma_polar(M, gamma1, gamma2, Q)` | `PolarModel` of the equivalent single-γ problem | (4.10), §4.3 |

### The branch filter

Eliminating the density ratio to reach the cubic destroys the branch identity
of each root, so a root of the cubic does not by itself describe a physical
detonation. `wave_angle_roots` returns every real root together with the
ordered filter used in the paper:

1. `attached` — `tan(beta) > tan(theta)`;
2. `above_cj` — `u = M^2 sin^2(beta) >= u_CJ`;
3. `high_compression` — the sign of the branch quantity
   `sigma = 1 + gamma*u - (gamma+1)*u*r`, positive on the high-compression
   branch and negative on the low-compression branch (2.17).

The discriminant `D` cannot do step 3: squaring either branch relation gives
`D = sigma^2`, so `D >= 0` holds on both branches and carries no sign
information. If you reimplement this, that is the step to get right.

Among the roots that pass, the smaller wave angle is the weak overdriven
solution and the larger is the strong solution. They exist precisely on
`theta_CJ < theta < theta_max`; outside that interval `weak_wave_angle` raises
`ValueError` rather than returning something misleading.

### Two-γ model

The two-γ model uses `gamma1` upstream and `gamma2` downstream, with the
upstream Mach number `M = U1 / sqrt(gamma1 p1 / rho1)` and
`Q = Q* / (R1 T1)`. Moving the difference of the two enthalpy coefficients
into the heat release gives the single-γ problem with

```
gamma_e = gamma2
M_e     = M * sqrt(gamma1 / gamma2)
Q_e     = Q + gamma1/(gamma1 - 1) - gamma2/(gamma2 - 1)
```

The mapping is exact and preserves pressures, densities and velocities, so
every wave angle, deflection angle and critical point of the equivalent model
belongs to the two-γ model unchanged:

```python
import math
from odw_polar import two_gamma_polar

polar = two_gamma_polar(M=8.0, gamma1=1.4, gamma2=1.2, Q=20.0)
print(math.degrees(polar.detachment().theta_max))                # -> 47.316588
print(math.degrees(polar.weak_wave_angle(math.radians(30.0))))   # -> 41.112323
```

The downstream Mach number is also unchanged, because the downstream state and
`gamma2` are both preserved, so `sonic_point()` needs no conversion. Mach
numbers measured with the upstream sound speed are effective ones: `polar.M`
is `M_e` and `polar.cj.M_cj` is the effective CJ Mach number. Convert them
back with `M = M_e * sqrt(gamma2 / gamma1)`. The branch classification needs
`Q_e >= 0`; `two_gamma_polar` rejects `Q_e < 0`, for which the equivalent
problem has no CJ point.

## Chemical-equilibrium application

`equilibrium_odw/` implements section 5 of the paper: oblique detonations
whose products are in chemical equilibrium, with temperature-dependent species
properties. Both solvers use the two-step scheme of Zhang et al. (2022). The
flow state is computed at a fixed downstream composition, the equilibrium
composition is computed at the resulting temperature and pressure, and the
composition is relaxed until both converge. The solvers differ only in the
flow step.

| Call | Flow step |
|---|---|
| `solve_closed_form(...)` | linearises the mixture enthalpy at the current state and takes the wave angle from the closed-form cubic of the resulting single-γ model, (5.1)–(5.3) |
| `solve_newton(...)` | Newton iteration on the fixed-composition energy balance; each outer iteration starts from the previous wave angle |

`solve_newton` is the reference solver of the paper. It is an independent
implementation written from the published description of the scheme, not the
authors' code.

The closed-form iteration starts from the frozen upstream state, `T2 = T1`
and `X2 = X1`, where the linearised model is the inert oblique shock of the
upstream gas. Near detachment the prescribed deflection can exceed the maximum
deflection of that model, and the first cubic has no root. The linearisation
temperature is then moved to the temperature at that model's detachment
point, obtained in closed form from (2.28)–(2.29), and the iteration restarts
at the prescribed deflection. The composition stays frozen, and no equilibrium
evaluation is made for this move. In the main test set of section 5.2 it is
triggered for 8 of the 300 branch states.

```python
import math
from equilibrium_odw import fuel_air, frozen_sound_speed, solve_closed_form

mix, eq, X1 = fuel_air("H2", "path/to/thermo.inp")     # stoichiometric H2-air
u1 = 8.0 * frozen_sound_speed(mix, X1, 300.0)           # M = 8 at T1 = 300 K
sol = solve_closed_form(mix, eq, X1, 300.0, 101325.0, u1,
                        math.radians(30.0), branch="weak")
print(math.degrees(sol.beta), sol.T2, sol.p2)
```

`python example_equilibrium.py path/to/thermo.inp` runs both solvers on both
branches for this case.

### Thermodynamic data

No thermodynamic data are distributed with this repository. Pass the path of
a library in the NASA-9 format of McBride, Zehe & Gordon (NASA TP-2002-211556),
for example `thermo.inp` from NASA CEA or `databases/thermo_CT.inp` from
Combustion Toolbox. `fuel_air` needs H2, H, O2, O, OH, HO2, H2O2, H2O, N, N2
and NO, and for the hydrocarbon mixtures also the fuel, CO, CO2, CH3 and HCO.
The standard-state pressure is 1 bar.

The results in section 5.2 were computed with this code and the data of
Combustion Toolbox 1.2.9, evaluated through interpolated tables exported from
Combustion Toolbox. Other data, including the NASA-9 polynomials of the same
library, give slightly different states and timings. With the polynomials of
`thermo_CT.inp`, for example, the Newton-based reference stops at its inner
step limit in one weak-branch case near detachment (methane–air, `M = 7`,
`f = 0.95`), while the closed-form solver converges on all 300 branch states.

### Reproducing section 5.2

```bash
cd equilibrium_benchmark
python run_main_test_set.py path/to/thermo.inp       # a few minutes
```

`main_test_set_tasks.csv` lists the 150 cases of the main test set: mixture,
`M`, upstream speed `U`, `f` and the prescribed deflection. The script solves
every case on both branches with both solvers, reports convergence and the
agreement between the two solvers, and prints the timing table (Table 2) as
medians over seven rounds after one warm-up round.

## Repository layout

```
odw_polar.py            the closed-form implementation (stdlib only)
example.py              worked example: run `python example.py`
equilibrium_odw/        chemical-equilibrium solvers of section 5 (numpy)
example_equilibrium.py  worked example for section 5 (needs a NASA-9 library)
equilibrium_benchmark/  main test set of section 5.2
requirements.txt        dependencies for the verification and figure scripts
verification/           independent checks of every closed-form relation
figures/                scripts that generate the figures in the paper
```

## Verification

The closed-form relations are checked against independent numerics that never
use them: root finding and maximisation applied directly to the original polar
with its square root intact.

```bash
python -m pip install -r requirements.txt
cd verification
python validate_closed_form.py          # exit code 0 on PASS, 1 on FAIL
```

Over the 27 combinations of `M in {5, 7, 10}`, `gamma in {1.2, 1.3, 1.4}` and
`Q in {2, 5, 10}`:

| Quantity | Closed form | Independent reference | Agreement |
|---|---|---|---|
| wave angle | cubic (2.12) via Cardano | Brent root of the polar (2.3) | `1e-11` deg |
| detachment angle | cubic (2.28)–(2.29) | bounded maximisation of the polar | `1e-13` deg |
| sonic angle | quadratic (3.7)–(3.8) | Brent root of `M2 = 1` along the polar | `1e-13` deg |

`two_gamma_check.py` tests the two-γ mapping against a direct numerical
solution of the two-γ jump conditions (4.1)–(4.3) that never uses the
effective parameters. Over 43 cases, including `gamma1 < gamma2` and
`gamma1 = gamma2`, the CJ, detachment and sonic points and both wave angles
agree to within `1e-12` deg.

Pre-computed output is in `verification/results/`; `validation_summary.md`
there is the readable PASS/FAIL report.

### The individual scripts

| Script | What it checks |
|---|---|
| `validate_closed_form.py` | wave angles and the detachment point against Brent and direct maximisation; branch filter applied root by root; writes CSVs and a summary |
| `symbolic_factorization.py` | the exact factorisation `N(t) = (1+t^2) C3(t)`, the coefficients (2.13), and the `Q -> 0` degeneration to the classical oblique-shock cubic |
| `irreducibility_check.py` | Galois group of the wave-angle cubic; confirms the general case is genuinely irreducible, so the trigonometric form is needed |
| `sonic_symbolic_audit.py` | the reduction of `M2 = 1` to the quadratic (3.5): the `u^3` coefficient vanishes identically, and the elimination agrees with the resultant |
| `sonic_structure_audit.py` | existence and uniqueness of the sonic point, and the subsonic downstream state at detachment |
| `sonic_numerical_check.py` | the sonic quadratic against an independent Brent root of `M2 = 1`; branch classification and the ordering `theta_CJ < theta_s < theta_max` on a parameter sweep |
| `two_gamma_check.py` | the two-γ mapping (4.9)–(4.10) against a direct solution of the two-γ jump conditions |

Each script exits non-zero if any assertion fails.

## Figures

`figures/` contains the scripts behind the figures in the paper. Each writes
its output next to itself.

| Script | Figure |
|---|---|
| `oblique_detonation_geometry_clean.py` | 1 — geometry and notation |
| `polar_cubic_roots.py` | 2 — three real roots |
| `polar_cubic_double_root.py` | 3 — the double root at detachment |
| `polar_cubic_positive_discriminant.py` | 4 — one real root beyond detachment |
| `critical_structure_map.py` | 5 — solution domain in the `(M, theta)` plane |

The geometry sketch in figure 1 was finished by hand from the output of
`oblique_detonation_geometry_clean.py`; the script reproduces the layout and
the labelled angles, not the final typesetting.

## Reproducing the example

```bash
python example.py
```

For `M = 7`, `gamma = 1.3`, `Q = 10` this prints the CJ, sonic and detachment
points, the two wave angles at a 30-degree wedge, every algebraic root with its
filter flags, and a sweep along the weak branch. The width of the subsonic
part of the weak branch it reports (`0.002258` deg) is the inset value in
figure 5 of the paper.

## Citation

```bibtex
@article{li_luo_odw_polar,
  author  = {Li, Jing and Luo, Changtong},
  title   = {Exact explicit wave-angle solutions for equilibrium
             oblique detonations},
  note    = {submitted},
  url     = {https://github.com/lijing-creator/Exact_explicit_wave_angle_solutions}
}
```

The Chapman--Jouguet endpoint formulae (2.7)–(2.8) are not new; they agree
digit for digit with equations (28) and (32) of Pratt, Humphrey & Glenn,
*J. Propulsion and Power* **7** (1991) 837–845, and should be credited there.
The inverse cubic, the branch filter, the fixed-`Q` detachment cubic and the
sonic quadratic are the contributions of this paper.

## Scope

These relations describe the equilibrium end states and the structure of the
polar. They do not represent finite-thickness induction and reaction zones,
curved transition structures, or multidimensional and cellular instabilities.
Within the model, `theta_CJ` is the lower deflection limit of the weak
high-compression branch — not a threshold for the onset of a finite-rate
detonation.

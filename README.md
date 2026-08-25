**English** | [简体中文](README.zh-CN.md)

# Exact explicit wave-angle solutions of the equilibrium oblique-detonation polar

Code accompanying

> J. Li and C. Luo, *Exact explicit wave-angle solutions of the equilibrium
> oblique-detonation polar* (submitted).

The equilibrium oblique-detonation polar links the wedge deflection angle
$\theta$, the wave angle $\beta$, the upstream Mach number $M$ and the heat
release $Q$. Going forward — prescribe $\beta$, get $\theta$ — is elementary.
Going backward — prescribe the wedge angle $\theta$ and ask for the wave
angles — has normally been done by reading a plotted polar or by iterating.

The paper shows that the inverse problem is exactly solvable in closed form,
and this repository is the reference implementation. **If you just want the
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

## Repository layout

```
odw_polar.py            the closed-form implementation (stdlib only)
example.py              worked example: run `python example.py`
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

Pre-computed output is in `verification/results/`; `validation_summary.md`
there is the readable PASS/FAIL report.

### The individual scripts

| Script | What it checks |
|---|---|
| `validate_closed_form.py` | wave angles and the detachment point against Brent and direct maximisation; branch filter applied root by root; writes CSVs and a summary |
| `symbolic_factorization.py` | the exact factorisation `N(t) = (1+t^2) C3(t)`, the coefficients (2.13), and the `Q -> 0` degeneration to the classical oblique-shock cubic |
| `irreducibility_check.py` | Galois group of the wave-angle cubic; confirms the general case is genuinely irreducible, so the trigonometric form is needed |
| `sonic_symbolic_audit.py` | the reduction of `M2 = 1` to the quadratic (3.5): the `u^3` coefficient vanishes identically, and the elimination agrees with the resultant |
| `sonic_gap_asymptotics.py` | existence and uniqueness of the sonic point, subsonic detachment, and the high-Mach expansion giving `K(gamma)/M^4` |
| `sonic_numerical_check.py` | the sonic quadratic against an independent Brent root of `M2 = 1`; branch classification and the ordering `theta_CJ < theta_s < theta_max` on a parameter sweep |

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
| `global_limit_collapses.py` | 6 — high-Mach collapse |
| `sonic_detachment_gap_conditions.py` | 7 — strip width across parameters |

`limiting_structure.py` and `limiting_structure_geometric.py` are shared helpers
imported by the last two.

The geometry sketch in figure 1 was finished by hand from the output of
`oblique_detonation_geometry_clean.py`; the script reproduces the layout and
the labelled angles, not the final typesetting.

## Reproducing the example

```bash
python example.py
```

For `M = 7`, `gamma = 1.3`, `Q = 10` this prints the CJ, sonic and detachment
points, the two wave angles at a 30-degree wedge, every algebraic root with its
filter flags, and a sweep along the weak branch. The subsonic strip it reports
(`0.002258` deg) is the inset value in figure 5 of the paper.

## Citation

```bibtex
@article{li_luo_odw_polar,
  author  = {Li, Jing and Luo, Changtong},
  title   = {Exact explicit wave-angle solutions of the equilibrium
             oblique-detonation polar},
  note    = {submitted}
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

"""Detonation polar and inverse cubic for a positive Cardano discriminant."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np


GAMMA = 1.3
MACH = 7.0
Q = 10.0
THETA_STAR = math.radians(42.0)

RAD_TO_DEG = 180.0 / math.pi
M2 = MACH**2
Q_PRIME = 2.0 * (GAMMA - 1.0) * Q / GAMMA
TAU_DET = 0.8505536808071810698
T_DOUBLE = 2.4025454988939569475


def density_roots(beta):
    """Return high- and low-compression reciprocal density ratios."""
    beta = np.asarray(beta)
    u = M2 * np.sin(beta) ** 2
    disc = (u - 1.0) ** 2 - (GAMMA + 1.0) * u * Q_PRIME
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denominator = (GAMMA + 1.0) * u
    r_high = (1.0 + GAMMA * u - sqrt_disc) / denominator
    r_low = (1.0 + GAMMA * u + sqrt_disc) / denominator
    return r_high, r_low


def polar_theta(beta, branch="high"):
    """Equilibrium deflection angle for a prescribed wave angle."""
    r_high, r_low = density_roots(beta)
    r = r_high if branch == "high" else r_low
    return np.asarray(beta) - np.arctan(r * np.tan(beta))


def cubic_coefficients(theta):
    tau = math.tan(theta)
    a3 = tau**2 * (Q_PRIME + (GAMMA - 1.0) * M2 + 2.0)
    a2 = 2.0 * tau * (Q_PRIME - (M2 - 1.0))
    a1 = Q_PRIME + tau**2 * ((GAMMA + 1.0) * M2 + 2.0)
    a0 = 2.0 * tau
    return np.array([a3, a2, a1, a0], dtype=float)


# CJ endpoint, detachment state, and polar curves.
cal_a = 1.0 + (GAMMA**2 - 1.0) * Q / GAMMA
u_cj = cal_a + math.sqrt(cal_a**2 - 1.0)
beta_cj = math.asin(math.sqrt(u_cj) / MACH)
r_cj = (1.0 + GAMMA * u_cj) / ((GAMMA + 1.0) * u_cj)
theta_cj = beta_cj - math.atan(r_cj * math.tan(beta_cj))
theta_det = math.atan(TAU_DET)
beta_det = math.atan(T_DOUBLE)

beta = np.linspace(beta_cj, math.pi / 2.0, 2400)
theta_high = polar_theta(beta, "high")
theta_low = polar_theta(beta, "low")
i_det = int(np.argmax(theta_high))


# The fixed-angle inverse cubic has one real root and a complex-conjugate pair.
coeff = cubic_coefficients(THETA_STAR)
roots = np.roots(coeff)
real_root = float(roots[np.argmin(np.abs(roots.imag))].real)
complex_root = roots[np.argmax(roots.imag)]

p = (3.0 * coeff[0] * coeff[2] - coeff[1] ** 2) / (3.0 * coeff[0] ** 2)
q = (
    27.0 * coeff[0] ** 2 * coeff[3]
    - 9.0 * coeff[0] * coeff[1] * coeff[2]
    + 2.0 * coeff[1] ** 3
) / (27.0 * coeff[0] ** 3)
delta_c = (q / 2.0) ** 2 + (p / 3.0) ** 3
tau_star = math.tan(THETA_STAR)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

weak_color = "#0072B2"
strong_color = "#D55E00"
low_color = "#777777"
neutral = "#222222"

fig, (ax_polar, ax_cubic) = plt.subplots(
    1, 2, figsize=(7.15, 3.35), gridspec_kw={"width_ratios": [1.05, 1.0]}
)


# (a) The prescribed deflection lies beyond detachment.
ax_polar.plot(
    theta_high[: i_det + 1] * RAD_TO_DEG,
    beta[: i_det + 1] * RAD_TO_DEG,
    color=weak_color,
    label="high-compression: weak",
)
ax_polar.plot(
    theta_high[i_det:] * RAD_TO_DEG,
    beta[i_det:] * RAD_TO_DEG,
    color=strong_color,
    label="high-compression: strong",
)
ax_polar.plot(
    theta_low * RAD_TO_DEG,
    beta * RAD_TO_DEG,
    color=low_color,
    linestyle="--",
    linewidth=1.2,
    label="low-compression equilibrium branch",
)

theta_star_deg = THETA_STAR * RAD_TO_DEG
theta_det_deg = theta_det * RAD_TO_DEG
beta_det_deg = beta_det * RAD_TO_DEG
ax_polar.axvline(theta_star_deg, color=neutral, linestyle=":", linewidth=1.0)
ax_polar.plot(
    theta_cj * RAD_TO_DEG,
    beta_cj * RAD_TO_DEG,
    "D",
    ms=4.8,
    color=neutral,
    zorder=5,
)
ax_polar.plot(
    theta_det_deg,
    beta_det_deg,
    "P",
    ms=6.0,
    color=neutral,
    zorder=5,
)

ax_polar.annotate(
    "CJ",
    (theta_cj * RAD_TO_DEG, beta_cj * RAD_TO_DEG),
    xytext=(5, 5),
    textcoords="offset points",
    color=neutral,
)
ax_polar.annotate(
    "detachment",
    (theta_det_deg, beta_det_deg),
    xytext=(-44, 10),
    textcoords="offset points",
    ha="right",
    color=neutral,
    arrowprops={"arrowstyle": "-", "color": neutral, "lw": 0.7},
)
ax_polar.text(
    theta_star_deg - 0.65,
    51.0,
    rf"$\theta_*={theta_star_deg:.1f}^\circ$",
    rotation=90,
    va="center",
    ha="right",
    color=neutral,
)

ax_polar.set_xlim(-0.5, 44.0)
ax_polar.set_ylim(28.0, 91.5)
ax_polar.set_xlabel(r"flow deflection $\theta$ (deg)")
ax_polar.set_ylabel(r"wave angle $\beta$ (deg)")
ax_polar.set_xticks([0, 10, 20, 30, 40])
ax_polar.set_yticks([30, 40, 50, 60, 70, 80, 90])
ax_polar.grid(color="#D8D8D8", linewidth=0.45, alpha=0.8)
ax_polar.legend(loc="upper left", frameon=False, handlelength=2.4)
ax_polar.text(
    -0.14,
    1.03,
    "(a)",
    transform=ax_polar.transAxes,
    fontweight="bold",
    va="bottom",
)


# (b) Only the negative simple root crosses the real t axis.
t_plot = np.linspace(-0.35, 5.1, 2600)
scaled_cubic = np.polyval(coeff, t_plot) / (coeff[0] * (1.0 + t_plot**2))
ax_cubic.axvspan(tau_star, t_plot[-1], color=weak_color, alpha=0.055, lw=0)
ax_cubic.axhline(0.0, color=neutral, linewidth=0.8)
ax_cubic.axvline(tau_star, color=neutral, linestyle=":", linewidth=1.0)
ax_cubic.plot(t_plot, scaled_cubic, color=neutral, linewidth=1.45)
ax_cubic.plot(
    real_root,
    0.0,
    marker="x",
    ms=6.0,
    mew=1.3,
    color=low_color,
    zorder=5,
)

ax_cubic.annotate(
    "real root\n" + rf"$t_r={real_root:.3f}<0$",
    (real_root, 0.0),
    xytext=(29, -20),
    textcoords="offset points",
    ha="right",
    va="top",
    color=neutral,
)
ax_cubic.text(
    2.3,
    -0.52,
    "complex pair\n"
    + rf"$t={complex_root.real:.3f}\pm{complex_root.imag:.3f}\,\mathrm{{i}}$",
    ha="center",
    color=neutral,
)
ax_cubic.text(
    3.35,
    0.10,
    "no real-axis crossing",
    ha="center",
    color=neutral,
)
ax_cubic.text(
    tau_star + 0.08,
    -2.20,
    r"$t=\tau$",
    rotation=90,
    va="bottom",
    color=neutral,
)
ax_cubic.text(
    3.25,
    1.47,
    r"attached domain $t>\tau$",
    ha="center",
    color=neutral,
)

ax_cubic.set_xlim(-0.35, 5.1)
ax_cubic.set_ylim(-2.4, 1.75)
ax_cubic.set_xlabel(r"$t=\tan\beta$")
ax_cubic.set_ylabel(r"$C_3(t)/[a_3(1+t^2)]$")
ax_cubic.set_xticks([0, 1, 2, 3, 4, 5])
ax_cubic.set_yticks([-2, -1, 0, 1])
ax_cubic.grid(color="#D8D8D8", linewidth=0.45, alpha=0.8)
ax_cubic.text(
    -0.14,
    1.03,
    "(b)",
    transform=ax_cubic.transAxes,
    fontweight="bold",
    va="bottom",
)


fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.94, wspace=0.30)

output_stem = Path(__file__).with_suffix("")
fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
plt.close(fig)

print(f"theta_star = {theta_star_deg:.12f} deg")
print(f"theta_det = {theta_det_deg:.12f} deg")
print("roots =", roots)
print(f"Delta_C = {delta_c:.12f}")
print("saved", output_stem.with_suffix(".pdf"))
print("saved", output_stem.with_suffix(".png"))

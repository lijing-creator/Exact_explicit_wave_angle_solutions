"""Publication-style illustration of the detonation polar and cubic roots.

The figure uses the convention Q = Q*/(R T1) and the same coefficients as
JFM_latex_template/JFM_writing_template/main.tex.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


GAMMA = 1.3
MACH = 7.0
Q = 10.0
THETA_STAR = math.atan(0.5)

RAD_TO_DEG = 180.0 / math.pi
M2 = MACH**2
Q_PRIME = 2.0 * (GAMMA - 1.0) * Q / GAMMA


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


# CJ endpoint.
cal_a = 1.0 + (GAMMA**2 - 1.0) * Q / GAMMA
u_cj = cal_a + math.sqrt(cal_a**2 - 1.0)
beta_cj = math.asin(math.sqrt(u_cj) / MACH)
r_cj = (1.0 + GAMMA * u_cj) / ((GAMMA + 1.0) * u_cj)
theta_cj = beta_cj - math.atan(r_cj * math.tan(beta_cj))


# Detachment point on the high-compression branch.
detachment = minimize_scalar(
    lambda beta: -float(polar_theta(beta, "high")),
    bounds=(beta_cj, math.pi / 2.0),
    method="bounded",
    options={"xatol": 1.0e-14},
)
beta_det = float(detachment.x)
theta_det = -float(detachment.fun)


# Polar curves.
beta = np.linspace(beta_cj, math.pi / 2.0, 2400)
theta_high = polar_theta(beta, "high")
theta_low = polar_theta(beta, "low")
i_det = int(np.argmax(theta_high))


# Cubic roots at the illustrated deflection angle.
coeff = cubic_coefficients(THETA_STAR)
roots = np.roots(coeff)
real_roots = np.sort(roots[np.abs(roots.imag) < 1.0e-10].real)
if real_roots.size != 3:
    raise RuntimeError(f"Expected three real roots, obtained {roots}")
t_negative, t_weak, t_strong = real_roots
beta_negative, beta_weak, beta_strong = np.arctan(real_roots)
tau_star = math.tan(THETA_STAR)


# Plot styling.
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


# (a) Textbook orientation: theta horizontal, beta vertical.
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
ax_polar.axvline(theta_star_deg, color=neutral, linestyle=":", linewidth=1.0)
ax_polar.plot(theta_star_deg, beta_weak * RAD_TO_DEG, "o", ms=5.2,
              color=weak_color, zorder=5)
ax_polar.plot(theta_star_deg, beta_strong * RAD_TO_DEG, "s", ms=5.0,
              color=strong_color, zorder=5)
ax_polar.plot(theta_cj * RAD_TO_DEG, beta_cj * RAD_TO_DEG, "D", ms=4.8,
              color=neutral, zorder=5)
ax_polar.plot(theta_det * RAD_TO_DEG, beta_det * RAD_TO_DEG, "P", ms=6.0,
              color=neutral, zorder=5)

ax_polar.annotate(
    "weak root",
    (theta_star_deg, beta_weak * RAD_TO_DEG),
    xytext=(5, -12),
    textcoords="offset points",
    ha="left",
    color=neutral,
)
ax_polar.annotate(
    "strong root",
    (theta_star_deg, beta_strong * RAD_TO_DEG),
    xytext=(5, 4),
    textcoords="offset points",
    ha="left",
    color=neutral,
)
ax_polar.annotate(
    "CJ",
    (theta_cj * RAD_TO_DEG, beta_cj * RAD_TO_DEG),
    xytext=(5, 5),
    textcoords="offset points",
    color=neutral,
)
ax_polar.annotate(
    "detachment\n(double root)",
    (theta_det * RAD_TO_DEG, beta_det * RAD_TO_DEG),
    xytext=(-54, 8),
    textcoords="offset points",
    ha="right",
    color=neutral,
    arrowprops={"arrowstyle": "-", "color": neutral, "lw": 0.7},
)
ax_polar.text(
    theta_star_deg + 0.7,
    56.0,
    rf"$\theta_*={theta_star_deg:.3f}^\circ$",
    rotation=90,
    va="center",
    color=neutral,
)

ax_polar.set_xlim(-0.5, 43.0)
ax_polar.set_ylim(28.0, 91.5)
ax_polar.set_xlabel(r"flow deflection $\theta$ (deg)")
ax_polar.set_ylabel(r"wave angle $\beta$ (deg)")
ax_polar.set_xticks([0, 10, 20, 30, 40])
ax_polar.set_yticks([30, 40, 50, 60, 70, 80, 90])
ax_polar.grid(color="#D8D8D8", linewidth=0.45, alpha=0.8)
ax_polar.legend(loc="upper left", frameon=False, handlelength=2.4)
ax_polar.text(-0.14, 1.03, "(a)", transform=ax_polar.transAxes,
              fontweight="bold", va="bottom")


# (b) The cubic, divided by a positive factor to retain its roots while
# keeping the three crossings visible on one linear vertical scale.
t_plot = np.linspace(-0.35, 8.2, 2600)
scaled_cubic = np.polyval(coeff, t_plot) / (coeff[0] * (1.0 + t_plot**2))
ax_cubic.axvspan(tau_star, t_plot[-1], color=weak_color, alpha=0.055, lw=0)
ax_cubic.axhline(0.0, color=neutral, linewidth=0.8)
ax_cubic.axvline(tau_star, color=neutral, linestyle=":", linewidth=1.0)
ax_cubic.plot(t_plot, scaled_cubic, color=neutral, linewidth=1.45)

ax_cubic.plot(t_negative, 0.0, marker="x", ms=6.0, mew=1.3,
              color=low_color, zorder=5)
ax_cubic.plot(t_weak, 0.0, "o", ms=5.2, color=weak_color, zorder=5)
ax_cubic.plot(t_strong, 0.0, "s", ms=5.0, color=strong_color, zorder=5)

ax_cubic.annotate(
    rf"$t_-={t_negative:.3f}$" + "\n(algebraic only)",
    (t_negative, 0.0),
    xytext=(4, -28),
    textcoords="offset points",
    ha="left",
    color=neutral,
)
ax_cubic.annotate(
    rf"weak: $t={t_weak:.3f}$",
    (t_weak, 0.0),
    xytext=(8, 11),
    textcoords="offset points",
    ha="left",
    color=neutral,
)
ax_cubic.annotate(
    rf"strong: $t={t_strong:.3f}$",
    (t_strong, 0.0),
    xytext=(-5, 11),
    textcoords="offset points",
    ha="right",
    color=neutral,
)
ax_cubic.text(tau_star + 0.08, -2.65, r"$t=\tau$", rotation=90,
              va="bottom", color=neutral)
ax_cubic.text(4.45, 1.27, r"attached domain $t>\tau$", ha="center",
              color=neutral)

ax_cubic.set_xlim(-0.35, 8.2)
ax_cubic.set_ylim(-2.9, 1.55)
ax_cubic.set_xlabel(r"$t=\tan\beta$")
ax_cubic.set_ylabel(r"$C_3(t)/[a_3(1+t^2)]$")
ax_cubic.set_xticks([0, 1, 2, 4, 6, 8])
ax_cubic.set_yticks([-2, -1, 0, 1])
ax_cubic.grid(color="#D8D8D8", linewidth=0.45, alpha=0.8)
ax_cubic.text(-0.14, 1.03, "(b)", transform=ax_cubic.transAxes,
              fontweight="bold", va="bottom")


fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.94, wspace=0.30)

output_stem = Path(__file__).with_suffix("")
fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
plt.close(fig)

print(f"theta_CJ = {theta_cj * RAD_TO_DEG:.9f} deg")
print(f"beta_CJ = {beta_cj * RAD_TO_DEG:.9f} deg")
print(f"theta_det = {theta_det * RAD_TO_DEG:.9f} deg")
print(f"beta_det = {beta_det * RAD_TO_DEG:.9f} deg")
print("roots t =", real_roots)
print("roots beta (deg) =", np.arctan(real_roots) * RAD_TO_DEG)
print("saved", output_stem.with_suffix(".pdf"))
print("saved", output_stem.with_suffix(".png"))

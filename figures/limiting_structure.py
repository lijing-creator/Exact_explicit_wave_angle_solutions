"""Three-panel limiting structure of the fixed-Q equilibrium ODW polar.

Panels show the three results grouped in Section 3.2:
  (a) high-Mach sonic--detachment gap;
  (b) near-CJ collapse of the attached window;
  (c) local square-root merging of the weak and strong roots.

The implementation evaluates the closed-form relations directly and is
independent of the earlier composite morphology figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


RAD_TO_DEG = 180.0 / np.pi
BLUE = "#176b87"
ORANGE = "#b4532a"
DARK = "#2f3336"
GREY = "#8a8f94"


@dataclass(frozen=True)
class PolarModel:
    gamma: float
    heat_release: float

    @property
    def q_prime(self) -> float:
        return 2.0 * (self.gamma - 1.0) * self.heat_release / self.gamma

    @property
    def u_cj(self) -> float:
        cal_a = 1.0 + (self.gamma**2 - 1.0) * self.heat_release / self.gamma
        return cal_a + np.sqrt(cal_a**2 - 1.0)

    @property
    def m_cj(self) -> float:
        return np.sqrt(self.u_cj)

    @property
    def r_cj(self) -> float:
        return (1.0 + self.gamma * self.u_cj) / ((self.gamma + 1.0) * self.u_cj)

    def theta_cj(self, mach: float) -> float:
        beta = np.arcsin(np.sqrt(self.u_cj) / mach)
        return beta - np.arctan(self.r_cj * np.tan(beta))

    def coefficients(self, mach: float) -> tuple[float, float, float]:
        m = mach**2
        a = self.q_prime + (self.gamma - 1.0) * m + 2.0
        b = self.q_prime - (m - 1.0)
        c = (self.gamma + 1.0) * m + 2.0
        return a, b, c

    def geometry(self, mach: float, tau: float, t: float) -> tuple[float, float]:
        m = mach**2
        u = m * t**2 / (1.0 + t**2)
        r = (t - tau) / (t * (1.0 + t * tau))
        return u, r

    def detachment(self, mach: float) -> tuple[float, float, float, float, float]:
        """Return theta_max, beta_d, y_d, s_d and t_d."""
        a, b, c = self.coefficients(mach)
        roots = np.roots([a * c, b * c + 3.0 * a, 4.0 * b, self.q_prime])
        candidates: list[tuple[float, float, float, float]] = []
        for root in roots:
            if abs(root.imag) > 2.0e-6:
                continue
            y = float(root.real)
            if y <= 0.0:
                continue
            s = y**2 * (a * y + b)
            if s <= 0.0:
                continue
            tau = np.sqrt(s)
            t = y / tau
            if t <= tau:
                continue
            u, r = self.geometry(mach, tau, t)
            sigma = 1.0 + self.gamma * u - (self.gamma + 1.0) * u * r
            if u < self.u_cj * (1.0 - 5.0e-8) or sigma < -2.0e-6:
                continue
            candidates.append((s, y, tau, t))
        if not candidates:
            raise RuntimeError(f"No physical detachment root for M={mach:.12g}")
        s, y, tau, t = max(candidates, key=lambda item: item[0])
        return np.arctan(tau), np.arctan(t), y, s, t

    def sonic(self, mach: float) -> tuple[float, float, float]:
        """Return theta_s, beta_s and u_s."""
        m = mach**2
        b = self.q_prime - (m - 1.0)
        c2 = self.gamma * ((self.gamma + 2.0) * self.q_prime - 2.0 * (m - 1.0))
        c1 = (self.gamma + 1.0) * b**2 + 2.0 * b + 2.0 * self.gamma * (m - 1.0)
        c0 = 2.0 * (m - 1.0) - self.q_prime
        disc = c1**2 - 4.0 * c2 * c0
        if disc <= 0.0 or c2 >= 0.0:
            raise RuntimeError(f"Invalid sonic quadratic for M={mach:.12g}")
        u_s = (c1 + np.sqrt(disc)) / (-2.0 * c2)
        if not (self.u_cj < u_s < m):
            raise RuntimeError(f"Sonic root outside physical interval for M={mach:.12g}")
        r_star = (self.gamma * u_s + self.q_prime + 2.0 - m) / (1.0 + self.gamma * u_s)
        t_s = np.sqrt(u_s / (m - u_s))
        tau_s = t_s * (1.0 - r_star) / (1.0 + r_star * t_s**2)
        return np.arctan(tau_s), np.arctan(t_s), u_s

    def high_compression_roots(self, mach: float, theta: float) -> list[float]:
        """Return admissible high-compression beta roots at fixed theta."""
        m = mach**2
        tau = np.tan(theta)
        a, b, c = self.coefficients(mach)
        coeffs = [a * tau**2, 2.0 * b * tau, self.q_prime + c * tau**2, 2.0 * tau]
        roots = np.roots(coeffs)
        beta: list[float] = []
        for root in roots:
            if abs(root.imag) > 2.0e-6:
                continue
            t = float(root.real)
            if t <= tau:
                continue
            u, r = self.geometry(mach, tau, t)
            sigma = 1.0 + self.gamma * u - (self.gamma + 1.0) * u * r
            if u < self.u_cj * (1.0 - 5.0e-8) or sigma < -2.0e-6:
                continue
            beta.append(np.arctan(t))
        return sorted(beta)

    def fold_coefficient(self, mach: float) -> tuple[float, float, float]:
        """Return C_d [rad^1/2], theta_max and beta_d."""
        theta_max, beta_d, y, s, t = self.detachment(mach)
        a, b, c = self.coefficients(mach)
        tau = np.sqrt(s)
        c_d = 2.0 / (1.0 + t**2) * np.sqrt(
            (1.0 + s) * (c * y + 2.0) / (2.0 * tau * (3.0 * a * y + 2.0 * b))
        )
        return c_d, theta_max, beta_d


def direct_label(ax, x, y, text, color, *, dx=0, dy=0, ha="left", va="center") -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=7.7,
        ha=ha,
        va=va,
    )


def high_precision_gap(model: PolarModel, mach: float) -> float:
    """Return theta_max - theta_s in degrees without subtractive cancellation."""
    mp.mp.dps = 70
    gamma = mp.mpf(str(model.gamma))
    heat_release = mp.mpf(str(model.heat_release))
    mach_mp = mp.mpf(repr(float(mach)))
    m = mach_mp**2
    q_prime = 2 * (gamma - 1) * heat_release / gamma

    a = q_prime + (gamma - 1) * m + 2
    b = q_prime - (m - 1)
    c = (gamma + 1) * m + 2

    # The double-precision physical root is only used as an initial value;
    # Newton refinement and the angular subtraction are both high precision.
    y_initial = mp.mpf(repr(model.detachment(float(mach))[2]))

    def polynomial(y):
        return a * c * y**3 + (b * c + 3 * a) * y**2 + 4 * b * y + q_prime

    def derivative(y):
        return 3 * a * c * y**2 + 2 * (b * c + 3 * a) * y + 4 * b

    y = mp.findroot(
        polynomial,
        y_initial,
        df=derivative,
        tol=mp.mpf("1e-60"),
        verify=False,
    )
    s = y**2 * (a * y + b)
    theta_max = mp.atan(mp.sqrt(s))

    c2 = gamma * ((gamma + 2) * q_prime - 2 * (m - 1))
    c1 = (gamma + 1) * b**2 + 2 * b + 2 * gamma * (m - 1)
    c0 = 2 * (m - 1) - q_prime
    u_s = (c1 + mp.sqrt(c1**2 - 4 * c2 * c0)) / (-2 * c2)
    r_star = (gamma * u_s + q_prime + 2 - m) / (1 + gamma * u_s)
    t_s = mp.sqrt(u_s / (m - u_s))
    tau_s = t_s * (1 - r_star) / (1 + r_star * t_s**2)
    theta_s = mp.atan(tau_s)

    return float((theta_max - theta_s) * 180 / mp.pi)


def panel_high_mach(ax, model: PolarModel) -> float:
    mach = np.geomspace(5.0, 400.0, 135)
    exact = np.array([high_precision_gap(model, value) for value in mach])

    inert = PolarModel(model.gamma, 0.0)
    inert_gap = np.array([high_precision_gap(inert, value) for value in mach])

    k_deg = np.sqrt(model.gamma**2 - 1.0) / (2.0 * (model.gamma + 1.0) ** 2) * RAD_TO_DEG
    asymptotic = k_deg / mach**4

    ax.loglog(mach, exact, color=BLUE, lw=1.8)
    ax.loglog(mach, inert_gap, color=GREY, lw=1.25, ls=(0, (1.2, 2.0)))
    ax.loglog(mach, asymptotic, color=DARK, lw=1.25, ls=(0, (4.0, 2.0)))

    idx = -21
    direct_label(ax, mach[idx], exact[idx], r"exact, $Q=10$", BLUE, dx=-2, dy=8, ha="right")
    direct_label(ax, mach[-6], asymptotic[-6], r"$K(\gamma)M^{-4}$", DARK, dx=-3, dy=-9, ha="right")
    direct_label(ax, mach[35], inert_gap[35], r"inert, $Q=0$", GREY, dx=5, dy=-8)

    ax.set_title(r"(a) $M\rightarrow\infty$", fontsize=9.2, pad=4)
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$\theta_{\max}-\theta_s$ (deg)")
    ax.grid(True, which="both", color="#d0d0d0", lw=0.45, alpha=0.55)
    ax.text(
        0.05,
        0.08,
        rf"$K(\gamma)={k_deg:.3f}^\circ$",
        transform=ax.transAxes,
        fontsize=7.7,
        color=DARK,
    )

    slope = np.polyfit(np.log(mach[-35:]), np.log(exact[-35:]), 1)[0]
    return float(slope)


def panel_near_cj(ax, model: PolarModel) -> tuple[float, float]:
    eps = np.geomspace(1.0e-6, 3.0e-2, 75)
    window = []
    strip = []
    for value in eps:
        mach = model.m_cj * (1.0 + value)
        theta_max = model.detachment(mach)[0]
        window.append((theta_max - model.theta_cj(mach)) * RAD_TO_DEG)
        strip.append((theta_max - model.sonic(mach)[0]) * RAD_TO_DEG)
    window = np.asarray(window)
    strip = np.asarray(strip)

    delta = model.u_cj * ((1.0 + eps) ** 2 - 1.0)
    w3 = (
        (model.gamma + 1.0) ** 2
        * (model.u_cj + 1.0)
        * np.sqrt(model.u_cj)
        / (2.0 * (1.0 + model.gamma * model.u_cj) ** 3)
    )
    lam = 1.0 / (1.0 + model.u_cj) ** 2
    window_asym = w3 * delta**1.5 * RAD_TO_DEG
    strip_asym = lam * window_asym

    ax.loglog(eps, window, color=BLUE, lw=1.8)
    ax.loglog(eps, strip, color=ORANGE, lw=1.8)
    ax.loglog(eps, window_asym, color=DARK, lw=1.15, ls=(0, (4.0, 2.0)))
    ax.loglog(eps, strip_asym, color=DARK, lw=1.15, ls=(0, (1.2, 2.0)))

    direct_label(ax, eps[-8], window[-8], r"attached window", BLUE, dx=-2, dy=8, ha="right")
    direct_label(ax, eps[-8], strip[-8], r"subsonic strip", ORANGE, dx=-2, dy=-9, ha="right")
    ax.text(
        0.05,
        0.08,
        rf"$\lambda=(1+M_{{\rm CJ}}^2)^{{-2}}={lam:.5f}$",
        transform=ax.transAxes,
        fontsize=7.5,
        color=DARK,
    )

    ax.set_title(r"(b) $M\rightarrow M_{\mathrm{CJ}}^+$", fontsize=9.2, pad=4)
    ax.set_xlabel(r"$\epsilon=M/M_{\mathrm{CJ}}-1$")
    ax.set_ylabel("angular width (deg)")
    ax.grid(True, which="both", color="#d0d0d0", lw=0.45, alpha=0.55)

    slope_window = np.polyfit(np.log(eps[:25]), np.log(window[:25]), 1)[0]
    slope_strip = np.polyfit(np.log(eps[:25]), np.log(strip[:25]), 1)[0]
    return float(slope_window), float(slope_strip)


def panel_fold(ax, model: PolarModel, mach: float) -> tuple[float, float]:
    c_rad, theta_max, beta_d = model.fold_coefficient(mach)
    c_deg = c_rad * np.sqrt(RAD_TO_DEG)

    eps_deg = np.geomspace(1.0e-6, 1.2e-1, 90)
    weak = []
    strong = []
    for value in eps_deg:
        theta = theta_max - np.radians(value)
        roots = model.high_compression_roots(mach, theta)
        if len(roots) != 2:
            raise RuntimeError(f"Expected two high-compression roots at epsilon={value:.6g} deg")
        weak.append((roots[0] - beta_d) * RAD_TO_DEG)
        strong.append((roots[1] - beta_d) * RAD_TO_DEG)
    weak = np.asarray(weak)
    strong = np.asarray(strong)
    x = np.sqrt(eps_deg)
    leading = c_deg * x

    # Include the coalescence point explicitly.
    x_plot = np.insert(x, 0, 0.0)
    weak_plot = np.insert(weak, 0, 0.0)
    strong_plot = np.insert(strong, 0, 0.0)
    leading_plot = np.insert(leading, 0, 0.0)

    ax.plot(x_plot, weak_plot, color=BLUE, lw=1.8)
    ax.plot(x_plot, strong_plot, color=ORANGE, lw=1.8)
    ax.plot(x_plot, -leading_plot, color=DARK, lw=1.15, ls=(0, (4.0, 2.0)))
    ax.plot(x_plot, leading_plot, color=DARK, lw=1.15, ls=(0, (4.0, 2.0)))
    ax.axhline(0.0, color="#aaaaaa", lw=0.65, zorder=0)

    direct_label(ax, x[-8], weak[-8], "weak", BLUE, dx=-2, dy=-7, ha="right")
    direct_label(ax, x[-8], strong[-8], "strong", ORANGE, dx=-2, dy=7, ha="right")
    ax.text(
        0.05,
        0.91,
        rf"$M=7$, $C_d={c_rad:.4f}\ {{\rm rad}}^{{1/2}}$",
        transform=ax.transAxes,
        fontsize=7.5,
        color=DARK,
        va="top",
    )

    ax.set_title(r"(c) $\theta\rightarrow\theta_{\max}^-$", fontsize=9.2, pad=4)
    ax.set_xlabel(r"$(\theta_{\max}-\theta)^{1/2}$ (deg$^{1/2}$)")
    ax.set_ylabel(r"$\beta-\beta_d$ (deg)")
    ax.grid(True, color="#d0d0d0", lw=0.45, alpha=0.55)

    ratio_weak = (-weak[0]) / leading[0]
    ratio_strong = strong[0] / leading[0]
    return float(ratio_weak), float(ratio_strong)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.linewidth": 0.75,
            "savefig.transparent": False,
        }
    )

    model = PolarModel(1.3, 10.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.62), constrained_layout=True)
    slope_high = panel_high_mach(axes[0], model)
    slope_window, slope_strip = panel_near_cj(axes[1], model)
    ratio_weak, ratio_strong = panel_fold(axes[2], model, 7.0)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.7)

    output_dir = Path(__file__).resolve().parent
    stem = output_dir / "limiting_structure"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"high-M fitted slope: {slope_high:.8f} (target -4)")
    print(
        "near-CJ fitted slopes: "
        f"window={slope_window:.8f}, strip={slope_strip:.8f} (target 1.5)"
    )
    print(
        "fold leading-order ratios at smallest epsilon: "
        f"weak={ratio_weak:.8f}, strong={ratio_strong:.8f} (target 1)"
    )
    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.svg')}")
    print(f"saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

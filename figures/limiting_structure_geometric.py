"""Geometric views of the three limiting structures in Section 3.2.

The panels retain the finite-parameter curvature instead of linearizing
all three asymptotic laws.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from limiting_structure import PolarModel


RAD_TO_DEG = 180.0 / np.pi
BLUE = "#176b87"
ORANGE = "#b4532a"
DARK = "#2f3336"
GREY = "#8a8f94"


def critical_angles_mp(model: PolarModel, mach: float) -> tuple[float, float, float]:
    """Return theta_CJ, theta_s and theta_max in radians at high precision."""
    mp.mp.dps = 70
    gamma = mp.mpf(str(model.gamma))
    heat_release = mp.mpf(str(model.heat_release))
    mach_mp = mp.mpf(repr(float(mach)))
    m = mach_mp**2
    q_prime = 2 * (gamma - 1) * heat_release / gamma

    cal_a = 1 + (gamma**2 - 1) * heat_release / gamma
    u_cj = cal_a + mp.sqrt(cal_a**2 - 1)
    r_cj = (1 + gamma * u_cj) / ((gamma + 1) * u_cj)
    beta_cj = mp.asin(mp.sqrt(u_cj) / mach_mp)
    theta_cj = beta_cj - mp.atan(r_cj * mp.tan(beta_cj))

    a = q_prime + (gamma - 1) * m + 2
    b = q_prime - (m - 1)
    c = (gamma + 1) * m + 2
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

    return float(theta_cj), float(theta_s), float(theta_max)


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


def panel_high_mach(ax, model: PolarModel) -> None:
    inert = PolarModel(model.gamma, 0.0)
    k_deg = np.sqrt(model.gamma**2 - 1.0) / (2.0 * (model.gamma + 1.0) ** 2) * RAD_TO_DEG
    mach_reactive = np.geomspace(model.m_cj * 1.0003, 80.0, 145)
    mach_inert = np.geomspace(1.01, 80.0, 170)

    def scaled_gap(current_model, mach_values):
        values = []
        for mach in mach_values:
            _, theta_s, theta_max = critical_angles_mp(current_model, mach)
            values.append(mach**4 * (theta_max - theta_s) * RAD_TO_DEG / k_deg)
        return np.asarray(values)

    reactive = scaled_gap(model, mach_reactive)
    inert_values = scaled_gap(inert, mach_inert)

    ax.semilogx(mach_reactive, reactive, color=BLUE, lw=1.8)
    ax.semilogx(mach_inert, inert_values, color=GREY, lw=1.4, ls=(0, (2.0, 1.8)))
    ax.axhline(1.0, color=DARK, lw=1.05, ls=(0, (4.0, 2.0)))

    peak_reactive = int(np.argmax(reactive))
    peak_inert = int(np.argmax(inert_values))
    direct_label(
        ax,
        mach_reactive[peak_reactive],
        reactive[peak_reactive],
        r"$Q=10$",
        BLUE,
        dx=5,
        dy=7,
    )
    direct_label(
        ax,
        mach_inert[peak_inert],
        inert_values[peak_inert],
        r"$Q=0$",
        GREY,
        dx=-8,
        dy=10,
        ha="right",
    )
    direct_label(ax, 52.0, 1.0, r"high-$M$ limit", DARK, dy=-9, ha="center")

    ax.plot(
        model.m_cj,
        0.0,
        marker="o",
        ms=3.6,
        mfc="white",
        mec=BLUE,
        mew=1.0,
        clip_on=False,
    )
    ax.annotate(
        r"$M_{\rm CJ}$",
        (model.m_cj, 0.0),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=BLUE,
    )

    ax.set_title(r"(a) Finite-$M$ approach", fontsize=9.2, pad=4)
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$M^4(\theta_{\max}-\theta_s)/K(\gamma)$")
    ax.set_xlim(1.0, 85.0)
    ax.set_ylim(0.0, 1.34)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, which="both", color="#d0d0d0", lw=0.45, alpha=0.55)
    ax.text(0.72, 0.08, r"$\gamma=1.3$", transform=ax.transAxes, fontsize=7.6, color=DARK)


def panel_near_cj(ax, model: PolarModel) -> None:
    mu = np.geomspace(1.0e-5, 3.5e-1, 115)
    window = []
    strip = []
    for value in mu:
        mach = model.m_cj * (1.0 + value)
        theta_cj, theta_s, theta_max = critical_angles_mp(model, mach)
        window.append(theta_max - theta_cj)
        strip.append(theta_max - theta_s)
    window = np.asarray(window)
    strip = np.asarray(strip)

    delta = model.u_cj * ((1.0 + mu) ** 2 - 1.0)
    w3 = (
        (model.gamma + 1.0) ** 2
        * (model.u_cj + 1.0)
        * np.sqrt(model.u_cj)
        / (2.0 * (1.0 + model.gamma * model.u_cj) ** 3)
    )
    lam = 1.0 / (1.0 + model.u_cj) ** 2
    window_ratio = window / (w3 * delta**1.5)
    strip_ratio = strip / (lam * w3 * delta**1.5)

    ax.semilogx(mu, window_ratio, color=BLUE, lw=1.8)
    ax.semilogx(mu, strip_ratio, color=ORANGE, lw=1.8)
    ax.axhline(1.0, color=DARK, lw=1.05, ls=(0, (4.0, 2.0)))

    direct_label(
        ax,
        mu[-14],
        window_ratio[-14],
        "attached window",
        BLUE,
        dx=-2,
        dy=8,
        ha="right",
    )
    direct_label(
        ax,
        mu[-14],
        strip_ratio[-14],
        "subsonic strip",
        ORANGE,
        dx=-2,
        dy=-8,
        ha="right",
    )
    direct_label(ax, 2.5e-4, 1.0, "leading order", DARK, dy=7, ha="center")

    ax.set_title(r"(b) Range of the near-CJ laws", fontsize=9.2, pad=4)
    ax.set_xlabel(r"$\mu=M/M_{\rm CJ}-1$")
    ax.set_ylabel("exact width / leading-order width")
    ax.set_xlim(mu[0], mu[-1])
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, which="both", color="#d0d0d0", lw=0.45, alpha=0.55)
    ax.text(
        0.05,
        0.10,
        rf"$\lambda=(1+M_{{\rm CJ}}^2)^{{-2}}={lam:.5f}$",
        transform=ax.transAxes,
        fontsize=7.4,
        color=DARK,
    )


def panel_fold(ax, model: PolarModel, mach: float) -> None:
    c_rad, theta_max, beta_d = model.fold_coefficient(mach)
    epsilon_deg = np.linspace(1.0e-6, 1.5, 190)
    theta = theta_max - np.radians(epsilon_deg)

    weak = []
    strong = []
    for current_theta in theta:
        roots = model.high_compression_roots(mach, current_theta)
        if len(roots) != 2:
            raise RuntimeError("Expected two high-compression roots below detachment")
        weak.append(roots[0])
        strong.append(roots[1])
    weak = np.asarray(weak)
    strong = np.asarray(strong)

    leading_offset = c_rad * np.sqrt(np.radians(epsilon_deg))
    leading_weak = beta_d - leading_offset
    leading_strong = beta_d + leading_offset

    theta_deg = theta * RAD_TO_DEG
    weak_deg = weak * RAD_TO_DEG
    strong_deg = strong * RAD_TO_DEG
    leading_weak_deg = leading_weak * RAD_TO_DEG
    leading_strong_deg = leading_strong * RAD_TO_DEG

    ax.fill_between(theta_deg, weak_deg, strong_deg, color=BLUE, alpha=0.035, lw=0)
    ax.plot(theta_deg, weak_deg, color=BLUE, lw=1.8)
    ax.plot(theta_deg, strong_deg, color=ORANGE, lw=1.8)
    ax.plot(theta_deg, leading_weak_deg, color=DARK, lw=1.05, ls=(0, (4.0, 2.0)))
    ax.plot(theta_deg, leading_strong_deg, color=DARK, lw=1.05, ls=(0, (4.0, 2.0)))
    ax.plot(
        theta_max * RAD_TO_DEG,
        beta_d * RAD_TO_DEG,
        marker="D",
        ms=4.2,
        color=DARK,
        zorder=5,
    )

    label_index = 145
    direct_label(
        ax,
        theta_deg[label_index],
        weak_deg[label_index],
        "weak",
        BLUE,
        dx=-5,
        dy=-8,
        ha="right",
    )
    direct_label(
        ax,
        theta_deg[label_index],
        strong_deg[label_index],
        "strong",
        ORANGE,
        dx=-5,
        dy=-8,
        ha="right",
    )
    ax.annotate(
        r"$\beta_d$",
        (theta_max * RAD_TO_DEG, beta_d * RAD_TO_DEG),
        xytext=(-9, 8),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=DARK,
    )
    ax.text(
        0.06,
        0.50,
        rf"$M=7$, $C_d={c_rad:.4f}\ {{\rm rad}}^{{1/2}}$",
        transform=ax.transAxes,
        fontsize=7.4,
        color=DARK,
    )

    ax.set_title(r"(c) Fold in the original angles", fontsize=9.2, pad=4)
    ax.set_xlabel(r"flow deflection $\theta$ (deg)")
    ax.set_ylabel(r"wave angle $\beta$ (deg)")
    ax.set_xlim(theta_max * RAD_TO_DEG - 1.55, theta_max * RAD_TO_DEG + 0.06)
    ax.set_ylim(weak_deg[-1] - 0.7, strong_deg[-1] + 0.7)
    ax.grid(True, color="#d0d0d0", lw=0.45, alpha=0.55)


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
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.68), constrained_layout=True)
    panel_high_mach(axes[0], model)
    panel_near_cj(axes[1], model)
    panel_fold(axes[2], model, 7.0)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.7)

    output_dir = Path(__file__).resolve().parent
    stem = output_dir / "limiting_structure_geometric"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.svg')}")
    print(f"saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

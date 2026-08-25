"""Plot the fixed-Q solution structure of the equilibrium ODW polar.

This script is intentionally independent of the existing morphology-map
figure.  It evaluates the closed-form relations used in Sections 2 and 3 and
creates a global (M, theta) map together with a magnified view of the sonic
and detachment curves near M=7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


GAMMA = 1.3
Q = 10.0
M_ANCHOR = 7.0


def cj_state(gamma: float, heat_release: float) -> tuple[float, float, float]:
    """Return (u_CJ, M_CJ, r_CJ)."""
    cal_a = 1.0 + (gamma**2 - 1.0) * heat_release / gamma
    u_cj = cal_a + np.sqrt(cal_a**2 - 1.0)
    m_cj = np.sqrt(u_cj)
    r_cj = (1.0 + gamma * u_cj) / ((gamma + 1.0) * u_cj)
    return u_cj, m_cj, r_cj


U_CJ, M_CJ, R_CJ = cj_state(GAMMA, Q)
Q_PRIME = 2.0 * (GAMMA - 1.0) * Q / GAMMA


def theta_cj(mach: float) -> float:
    beta = np.arcsin(np.sqrt(U_CJ) / mach)
    return beta - np.arctan(R_CJ * np.tan(beta))


def tangent_geometry(mach: float, tau: float, t: float) -> tuple[float, float]:
    m = mach**2
    u = m * t**2 / (1.0 + t**2)
    r = (t - tau) / (t * (1.0 + t * tau))
    return u, r


def detachment_state(mach: float) -> tuple[float, float, float]:
    """Return (theta_max, beta_d, y_d), selecting the physical cubic root."""
    m = mach**2
    a = Q_PRIME + (GAMMA - 1.0) * m + 2.0
    b = Q_PRIME - (m - 1.0)
    c = (GAMMA + 1.0) * m + 2.0

    roots = np.roots([a * c, b * c + 3.0 * a, 4.0 * b, Q_PRIME])
    candidates: list[tuple[float, float, float, float]] = []
    for root in roots:
        if abs(root.imag) > 2.0e-7:
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
        u, r = tangent_geometry(mach, tau, t)
        sigma = 1.0 + GAMMA * u - (GAMMA + 1.0) * u * r
        if u < U_CJ * (1.0 - 2.0e-8) or sigma < -2.0e-7:
            continue
        candidates.append((s, y, tau, t))

    if not candidates:
        return np.nan, np.nan, np.nan

    s, y, tau, t = max(candidates, key=lambda item: item[0])
    return np.arctan(tau), np.arctan(t), y


def sonic_state(mach: float) -> tuple[float, float, float]:
    """Return (theta_s, beta_s, u_s) from the sonic quadratic."""
    m = mach**2
    b = Q_PRIME - (m - 1.0)
    c2 = GAMMA * ((GAMMA + 2.0) * Q_PRIME - 2.0 * (m - 1.0))
    c1 = (GAMMA + 1.0) * b**2 + 2.0 * b + 2.0 * GAMMA * (m - 1.0)
    c0 = 2.0 * (m - 1.0) - Q_PRIME
    disc = c1**2 - 4.0 * c2 * c0
    if disc <= 0.0 or c2 >= 0.0:
        return np.nan, np.nan, np.nan

    u_s = (c1 + np.sqrt(disc)) / (-2.0 * c2)
    if not (U_CJ < u_s < m):
        return np.nan, np.nan, np.nan

    r_star = (GAMMA * u_s + Q_PRIME + 2.0 - m) / (1.0 + GAMMA * u_s)
    t_s = np.sqrt(u_s / (m - u_s))
    tau_s = t_s * (1.0 - r_star) / (1.0 + r_star * t_s**2)
    if tau_s <= 0.0:
        return np.nan, np.nan, np.nan
    return np.arctan(tau_s), np.arctan(t_s), u_s


def compute_curves(mach_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cj = np.empty_like(mach_values)
    sonic = np.empty_like(mach_values)
    detach = np.empty_like(mach_values)
    for idx, mach in enumerate(mach_values):
        cj[idx] = theta_cj(mach)
        sonic[idx] = sonic_state(mach)[0]
        detach[idx] = detachment_state(mach)[0]
    return np.degrees(cj), np.degrees(sonic), np.degrees(detach)


def value_at(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.interp(x, xs, ys))


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "savefig.transparent": False,
        }
    )

    m_max = 14.0
    mach = np.linspace(M_CJ * (1.0 + 2.0e-5), m_max, 900)
    theta_cj_deg, theta_s_deg, theta_max_deg = compute_curves(mach)

    valid = np.isfinite(theta_s_deg) & np.isfinite(theta_max_deg)
    mach = mach[valid]
    theta_cj_deg = theta_cj_deg[valid]
    theta_s_deg = theta_s_deg[valid]
    theta_max_deg = theta_max_deg[valid]

    if not np.all((theta_cj_deg < theta_s_deg) & (theta_s_deg < theta_max_deg)):
        raise RuntimeError("Critical-curve ordering failed: expected theta_CJ < theta_s < theta_max.")

    # Add the common degenerate endpoint explicitly.
    mach = np.insert(mach, 0, M_CJ)
    theta_cj_deg = np.insert(theta_cj_deg, 0, 0.0)
    theta_s_deg = np.insert(theta_s_deg, 0, 0.0)
    theta_max_deg = np.insert(theta_max_deg, 0, 0.0)

    x_min = 0.91 * M_CJ
    y_max = max(54.0, float(np.nanmax(theta_max_deg)) + 4.0)

    fig, ax = plt.subplots(figsize=(7.15, 4.75), constrained_layout=True)

    # Restrained, colour-blind-friendly region fills.
    ax.axvspan(x_min, M_CJ, color="#dedede", alpha=0.72, zorder=0)
    ax.fill_between(mach, 0.0, theta_cj_deg, color="#dce3e7", alpha=0.88, zorder=0)
    ax.fill_between(mach, theta_cj_deg, theta_s_deg, color="#cfe8f3", alpha=0.78, zorder=0)
    ax.fill_between(mach, theta_s_deg, theta_max_deg, color="#f2b68f", alpha=0.82, zorder=1)
    ax.fill_between(mach, theta_max_deg, y_max, color="#f4f4f4", alpha=0.72, zorder=0)

    # Critical curves. Line style and direct labels carry meaning with or without colour.
    ax.plot(mach, theta_cj_deg, color="#4b5563", lw=1.65, ls=(0, (5, 2, 1.2, 2)), zorder=4)
    ax.plot(mach, theta_max_deg, color="#b4532a", lw=2.15, zorder=5)
    # Plot the dashed sonic curve last so both nearly coincident boundaries remain visible.
    ax.plot(mach, theta_s_deg, color="#176b87", lw=1.65, ls=(0, (4, 2.2)), zorder=6)

    ax.axvline(M_CJ, color="#666666", lw=0.9, ls=(0, (2, 2)), zorder=3)
    ax.plot(M_CJ, 0.0, marker="o", ms=5.0, color="#222222", zorder=7)
    ax.annotate(
        r"$(M_{\mathrm{CJ}},0)$",
        xy=(M_CJ, 0.0),
        xytext=(7, 9),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )

    # Anchor slice shared with the Section 2 examples.
    ax.axvline(M_ANCHOR, color="#7a7a7a", lw=0.75, ls=(0, (1.5, 2.5)), zorder=2)
    ax.text(M_ANCHOR + 0.10, 1.0, r"$M=7$", rotation=90, color="#555555", va="bottom")

    # Region labels list every attached equilibrium root in each band.
    ax.annotate(
        "no equilibrium detonation state\nfor $M<M_{\\mathrm{CJ}}$",
        xy=((x_min + M_CJ) / 2.0, 35.0),
        xytext=(3.90, 42.0),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.75},
        ha="left",
        va="center",
        color="#4b4b4b",
        fontsize=9.0,
    )
    ax.text(
        5.85,
        y_max - 4.8,
        "no attached equilibrium solution",
        ha="left",
        va="center",
        color="#4b4b4b",
        fontsize=9.4,
    )
    ax.text(
        9.1,
        24.0,
        "two high-compression roots\n"
        r"weak: $M_2>1$; strong: $M_2<1$",
        ha="center",
        va="center",
        color="#24333b",
        fontsize=9.4,
    )
    ax.text(
        10.8,
        4.2,
        r"strong high-compression: $M_2<1$" "\n"
        r"low-compression: $M_2>1$",
        ha="center",
        va="center",
        color="#39434a",
        fontsize=9.4,
    )

    # Direct curve labels with short leaders avoid a detached legend.
    x_label = 7.55
    y_cj = value_at(x_label, mach, theta_cj_deg)
    y_s = value_at(x_label, mach, theta_s_deg)
    y_d = value_at(x_label, mach, theta_max_deg)
    ax.annotate(
        r"$\theta_{\mathrm{CJ}}(M)$",
        xy=(x_label, y_cj),
        xytext=(13, 2),
        textcoords="offset points",
        color="#313943",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        r"$\theta_s(M)$",
        xy=(x_label, y_s),
        xytext=(12, -13),
        textcoords="offset points",
        color="#174f63",
        arrowprops={"arrowstyle": "-", "color": "#176b87", "lw": 0.8},
        ha="left",
        va="top",
    )
    ax.annotate(
        r"$\theta_{\max}(M)$",
        xy=(x_label, y_d),
        xytext=(12, 11),
        textcoords="offset points",
        color="#74351d",
        arrowprops={"arrowstyle": "-", "color": "#b4532a", "lw": 0.8},
        ha="left",
        va="bottom",
    )

    # Magnified view of the two curves near the M=7 anchor.
    zoom_mach = np.linspace(M_ANCHOR - 1.5e-3, M_ANCHOR + 1.5e-3, 240)
    _, zoom_s, zoom_max = compute_curves(zoom_mach)
    axins = ax.inset_axes([0.585, 0.60, 0.365, 0.315])
    axins.set_facecolor("white")
    axins.set_zorder(20)
    axins.fill_between(zoom_mach, zoom_s, zoom_max, color="#f2b68f", alpha=0.58)
    axins.plot(zoom_mach, zoom_max, color="#b4532a", lw=1.65)
    axins.plot(zoom_mach, zoom_s, color="#176b87", lw=1.45, ls=(0, (4, 2.2)))
    theta_s_anchor = np.degrees(sonic_state(M_ANCHOR)[0])
    theta_max_anchor = np.degrees(detachment_state(M_ANCHOR)[0])
    gap_anchor = theta_max_anchor - theta_s_anchor
    axins.axvline(M_ANCHOR, color="#777777", lw=0.65, ls=(0, (1.5, 2.0)))
    axins.plot(
        M_ANCHOR,
        theta_max_anchor,
        marker="o",
        ms=3.5,
        color="#b4532a",
        zorder=3,
    )
    axins.plot(
        M_ANCHOR,
        theta_s_anchor,
        marker="o",
        ms=3.5,
        color="#176b87",
        zorder=3,
    )
    bracket_x = M_ANCHOR + 7.0e-4
    bracket_s = np.interp(bracket_x, zoom_mach, zoom_s)
    bracket_max = np.interp(bracket_x, zoom_mach, zoom_max)
    axins.annotate(
        "",
        xy=(bracket_x, bracket_max),
        xytext=(bracket_x, bracket_s),
        arrowprops={"arrowstyle": "<->", "color": "#555555", "lw": 0.7},
    )
    gap_label = f"{gap_anchor:.4f}" + r"$^\circ$"
    axins.annotate(
        gap_label,
        xy=(bracket_x, 0.5 * (bracket_s + bracket_max)),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.5,
    )
    axins.annotate(
        r"$\theta_{\max}(M)$",
        xy=(zoom_mach[50], zoom_max[50]),
        xytext=(4, 4),
        textcoords="offset points",
        color="#74351d",
        fontsize=7.4,
        ha="left",
        va="bottom",
    )
    axins.annotate(
        r"$\theta_s(M)$",
        xy=(zoom_mach[50], zoom_s[50]),
        xytext=(4, -2),
        textcoords="offset points",
        color="#174f63",
        fontsize=7.4,
        ha="left",
        va="top",
    )
    y_pad = 8.0e-4
    axins.set_xlim(zoom_mach[0], zoom_mach[-1])
    axins.set_ylim(float(np.min(zoom_s)) - y_pad, float(np.max(zoom_max)) + y_pad)
    axins.set_xlabel(r"$M$", labelpad=0.5, fontsize=8.5)
    axins.set_ylabel(r"$\theta$ (deg)", labelpad=1.0, fontsize=8.5)
    axins.set_xticks([6.9985, 6.9995, 7.0000, 7.0005, 7.0015])
    axins.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axins.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axins.tick_params(axis="both", labelsize=7.5, length=2.5, pad=1.5)
    axins.grid(True, which="major", color="#d0d0d0", lw=0.45, alpha=0.65)
    axins.set_title(r"magnified near $M=7$", fontsize=8.5, pad=2.0)

    ax.set_xlim(x_min, m_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel(r"upstream Mach number, $M$")
    ax.set_ylabel(r"flow-deflection angle, $\theta$ (deg)")
    ax.set_xticks([M_CJ, 5.0, 7.0, 9.0, 11.0, 13.0])
    ax.set_xticklabels([r"$M_{\mathrm{CJ}}$", "5", "7", "9", "11", "13"])
    ax.set_yticks(np.arange(0.0, y_max + 0.1, 10.0))
    ax.grid(axis="y", color="#cfcfcf", lw=0.5, alpha=0.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir = Path(__file__).resolve().parent
    stem = output_dir / "critical_structure_map"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    anchor_cj = np.degrees(theta_cj(M_ANCHOR))
    anchor_s = np.degrees(sonic_state(M_ANCHOR)[0])
    anchor_max = np.degrees(detachment_state(M_ANCHOR)[0])
    print(f"gamma={GAMMA:.6g}, Q={Q:.6g}, M_CJ={M_CJ:.12f}")
    print(
        "M=7: "
        f"theta_CJ={anchor_cj:.9f} deg, "
        f"theta_s={anchor_s:.9f} deg, "
        f"theta_max={anchor_max:.9f} deg, "
        f"gap={anchor_max - anchor_s:.9f} deg"
    )
    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.svg')}")
    print(f"saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

"""High-Mach limits of the exact fixed-Q sonic and detachment loci.

Panel (a) normalises the sonic--detachment gap by its universal
high-Mach law K(gamma)/M^4. Panel (b) shows the sonic angle approaching
the common limit sin(theta_infinity)=1/gamma; together, the panels test
both parts of the high-Mach theorem.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from limiting_structure import PolarModel
from limiting_structure_geometric import critical_angles_mp


RAD_TO_DEG = 180.0 / np.pi
BLUE = "#176b87"
ORANGE = "#b4532a"
GREEN = "#4f7f52"
DARK = "#2f3336"
GRID = "#d0d0d0"


def direct_label(ax, x, y, text, color, *, dx=0, dy=0, ha="left", va="center") -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=8.0,
        ha=ha,
        va=va,
    )


def panel_high_mach(ax, gamma: float) -> None:
    q_values = [2.0, 10.0, 30.0]
    colors = [GREEN, BLUE, ORANGE]
    line_styles = [(0, (2.0, 1.6)), "solid", (0, (5.0, 2.0))]
    k_deg = np.sqrt(gamma**2 - 1.0) / (2.0 * (gamma + 1.0) ** 2) * RAD_TO_DEG

    for q_value, color, line_style in zip(q_values, colors, line_styles):
        model = PolarModel(gamma, q_value)
        mach = np.geomspace(model.m_cj * 1.0003, 100.0, 145)
        scaled_gap = []
        for current_mach in mach:
            _, theta_s, theta_max = critical_angles_mp(model, current_mach)
            scaled_gap.append(
                current_mach**4 * (theta_max - theta_s) * RAD_TO_DEG / k_deg
            )
        scaled_gap = np.asarray(scaled_gap)
        ax.semilogx(
            mach,
            scaled_gap,
            color=color,
            lw=1.8 if q_value == 10.0 else 1.5,
            ls=line_style,
        )

        peak = int(np.argmax(scaled_gap))
        offsets = {
            2.0: (-5, 9, "right"),
            10.0: (5, 8, "left"),
            30.0: (4, -10, "left"),
        }
        dx, dy, ha = offsets[q_value]
        direct_label(
            ax,
            mach[peak],
            scaled_gap[peak],
            rf"$Q={q_value:g}$",
            color,
            dx=dx,
            dy=dy,
            ha=ha,
        )
        ax.plot(
            model.m_cj,
            0.0,
            marker="o",
            ms=3.4,
            mfc="white",
            mec=color,
            mew=1.0,
            clip_on=False,
        )

    ax.axhline(1.0, color=DARK, lw=1.1, ls=(0, (4.0, 2.0)))
    direct_label(
        ax,
        67.0,
        1.0,
        r"$Q$-independent limit",
        DARK,
        dy=-9,
        ha="center",
    )
    ax.text(
        0.76,
        0.08,
        rf"$\gamma={gamma:g}$",
        transform=ax.transAxes,
        color=DARK,
        fontsize=7.8,
    )

    ax.set_title(r"(a) High-$M$ closure of the sonic strip", fontsize=9.4, pad=4)
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$M^4(\theta_{\max}-\theta_s)/K(\gamma)$")
    ax.set_xlim(1.5, 105.0)
    ax.set_ylim(0.0, 1.36)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, which="both", color=GRID, lw=0.45, alpha=0.58)


def panel_common_angle(ax, gamma: float) -> None:
    q_values = [2.0, 10.0, 30.0]
    colors = [GREEN, BLUE, ORANGE]
    line_styles = [(0, (2.0, 1.6)), "solid", (0, (5.0, 2.0))]
    label_levels = {2.0: 0.78, 10.0: 0.72, 30.0: 0.66}
    label_offsets = {
        2.0: (4, 7),
        10.0: (4, 0),
        30.0: (4, -7),
    }

    for q_value, color, line_style in zip(q_values, colors, line_styles):
        model = PolarModel(gamma, q_value)
        mach = np.geomspace(model.m_cj * 1.0003, 100.0, 145)
        normalized_sonic_angle = []
        for current_mach in mach:
            _, theta_s, _ = critical_angles_mp(model, current_mach)
            normalized_sonic_angle.append(gamma * np.sin(float(theta_s)))
        normalized_sonic_angle = np.asarray(normalized_sonic_angle)

        ax.semilogx(
            mach,
            normalized_sonic_angle,
            color=color,
            lw=1.8 if q_value == 10.0 else 1.5,
            ls=line_style,
        )

        label_index = int(
            np.argmin(np.abs(normalized_sonic_angle - label_levels[q_value]))
        )
        dx, dy = label_offsets[q_value]
        direct_label(
            ax,
            mach[label_index],
            normalized_sonic_angle[label_index],
            rf"$Q={q_value:g}$",
            color,
            dx=dx,
            dy=dy,
        )

        ax.plot(
            model.m_cj,
            0.0,
            marker="o",
            ms=3.4,
            mfc="white",
            mec=color,
            mew=1.0,
            clip_on=False,
        )

    ax.axhline(1.0, color=DARK, lw=1.1, ls=(0, (4.0, 2.0)))
    direct_label(
        ax,
        63.0,
        1.0,
        r"$\sin\theta_\infty=1/\gamma$",
        DARK,
        dy=-9,
        ha="center",
    )
    ax.text(
        0.76,
        0.08,
        rf"$\gamma={gamma:g}$",
        transform=ax.transAxes,
        color=DARK,
        fontsize=7.8,
    )

    ax.set_title(r"(b) Common high-$M$ angle", fontsize=9.4, pad=4)
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$\gamma\sin\theta_s$")
    ax.set_xlim(1.5, 105.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, which="both", color=GRID, lw=0.45, alpha=0.58)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.75,
            "savefig.transparent": False,
        }
    )

    gamma = 1.3
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), constrained_layout=True)
    panel_high_mach(axes[0], gamma)
    panel_common_angle(axes[1], gamma)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.0, width=0.7)

    output_dir = Path(__file__).resolve().parent
    stem = output_dir / "global_limit_collapses"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.svg')}")
    print(f"saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

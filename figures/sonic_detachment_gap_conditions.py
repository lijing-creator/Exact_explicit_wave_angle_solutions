"""Compare the sonic--detachment gap across several parameter conditions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
from scipy.optimize import minimize_scalar

from limiting_structure import PolarModel


M_MAX = 50.0
Q_CASES = (0.1, 1.0, 10.0, 50.0)
GAMMA_CASES = (1.1, 1.2, 1.3, 1.4, 1.67)

COLORS = ("#4b5563", "#176b87", "#3b7f5f", "#b4532a", "#7a5195")
LINESTYLES = (
    "-",
    (0, (5.0, 2.0)),
    (0, (3.0, 1.5, 1.0, 1.5)),
    (0, (1.2, 1.6)),
    (0, (7.0, 2.0, 1.2, 2.0)),
)
GRID = "#c9cdd0"


def gap_deg(model: PolarModel, mach: float) -> float:
    theta_max = model.detachment(mach)[0]
    theta_s = model.sonic(mach)[0]
    return float(np.degrees(theta_max - theta_s))


def curve_and_peak(
    model: PolarModel,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    delta_max = M_MAX / model.m_cj - 1.0
    delta = np.geomspace(2.0e-5, delta_max, 750)
    mach = model.m_cj * (1.0 + delta)
    gap = np.array([gap_deg(model, value) for value in mach])

    optimum = minimize_scalar(
        lambda value: -gap_deg(model, value),
        bounds=(model.m_cj * (1.0 + 1.0e-6), M_MAX),
        method="bounded",
        options={"xatol": 1.0e-11},
    )
    m_peak = float(optimum.x)
    return mach, gap, m_peak, gap_deg(model, m_peak)


def style_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, M_MAX)
    ax.set_ylim(2.0e-9, 2.0)
    ax.set_xticks([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.55, alpha=0.72)
    ax.grid(True, which="minor", axis="y", color=GRID, lw=0.35, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"upstream Mach number, $M$")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    peak_rows: list[tuple[float, float, float, float]] = []

    ax = axes[0]
    for index, heat_release in enumerate(Q_CASES):
        model = PolarModel(1.3, heat_release)
        mach, gap, m_peak, gap_peak = curve_and_peak(model)
        color = COLORS[index]
        line_style = LINESTYLES[index]
        ax.plot(
            mach,
            gap,
            color=color,
            lw=1.75,
            ls=line_style,
            label=rf"$Q={heat_release:g}$",
        )
        ax.plot(m_peak, gap_peak, marker="o", ms=4.0, color=color)
        peak_rows.append((model.gamma, model.heat_release, m_peak, gap_peak))

    ax.set_title(r"(a) fixed $\gamma=1.3$")
    ax.set_ylabel(r"$\theta_{\max}-\theta_s$ (deg)")
    ax.legend(frameon=False, loc="lower left")
    ax.annotate(
        r"common $K(\gamma)M^{-4}$ tail",
        xy=(40.0, gap_deg(PolarModel(1.3, 10.0), 40.0)),
        xytext=(-12, 34),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#343a40",
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.7},
    )

    ax = axes[1]
    for index, gamma in enumerate(GAMMA_CASES):
        model = PolarModel(gamma, 10.0)
        mach, gap, m_peak, gap_peak = curve_and_peak(model)
        color = COLORS[index]
        line_style = LINESTYLES[index]
        ax.plot(
            mach,
            gap,
            color=color,
            lw=1.75,
            ls=line_style,
            label=rf"$\gamma={gamma:g}$",
        )
        ax.plot(m_peak, gap_peak, marker="o", ms=4.0, color=color)
        peak_rows.append((model.gamma, model.heat_release, m_peak, gap_peak))

    ax.set_title(r"(b) fixed $Q=10$")
    ax.legend(frameon=False, loc="lower left")
    ax.text(
        0.97,
        0.97,
        "circles mark maxima",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#4b4b4b",
        fontsize=8.2,
    )

    for current_ax in axes:
        style_axis(current_ax)

    fig.suptitle("Sonic--detachment gap across parameter conditions", fontsize=12.0)

    stem = Path(__file__).with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    print("gamma,Q,M_CJ,M_peak,gap_peak_deg")
    for gamma, heat_release, m_peak, gap_peak in peak_rows:
        model = PolarModel(gamma, heat_release)
        print(
            f"{gamma:.6g},{heat_release:.6g},{model.m_cj:.9f},"
            f"{m_peak:.9f},{gap_peak:.12f}"
        )
    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.svg')}")
    print(f"saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

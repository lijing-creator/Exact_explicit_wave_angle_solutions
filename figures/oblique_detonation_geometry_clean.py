"""Text-free version of the oblique-detonation geometry schematic.

Same geometry, same coordinates and same colours as
``oblique_detonation_geometry.py``, with every text label and every
text-leader line removed. Intended for placing the labels by hand in
PowerPoint. The original script is left untouched.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Polygon


THETA = math.radians(14.0)
BETA = math.radians(52.0)

FLOW = "#0072B2"
WAVE = "#D55E00"
INK = "#222222"
MUTED = "#6F6F6F"
WEDGE = "#D9D9D9"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


fig, ax = plt.subplots(figsize=(7.15, 3.35))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0.0, 10.2)
ax.set_ylim(0.0, 5.15)
ax.set_aspect("equal", adjustable="box")
ax.axis("off")


apex = (3.65, 0.85)
x_right = 9.85
y_wedge = apex[1] + (x_right - apex[0]) * math.tan(THETA)
y_top = 5.0
x_wave_top = apex[0] + (y_top - apex[1]) / math.tan(BETA)


# Wedge body and upper surface.
wedge = Polygon(
    [apex, (x_right, y_wedge), (x_right, 0.12), (apex[0], 0.12)],
    closed=True,
    facecolor=WEDGE,
    edgecolor="none",
    zorder=1,
)
ax.add_patch(wedge)
ax.plot(
    [apex[0], x_right],
    [apex[1], y_wedge],
    color=INK,
    lw=2.0,
    solid_capstyle="round",
    zorder=5,
)


# Upstream-direction reference and detonation front.
ax.plot(
    [apex[0], 7.2],
    [apex[1], apex[1]],
    color=MUTED,
    lw=0.9,
    ls=(0, (4, 4)),
    zorder=2,
)
ax.plot(
    [apex[0], x_wave_top],
    [apex[1], y_top],
    color=WAVE,
    lw=2.6,
    solid_capstyle="round",
    zorder=6,
)


# Incoming and downstream velocity vectors.
v1 = FancyArrowPatch(
    (0.55, 1.68),
    (3.15, 1.68),
    arrowstyle="-|>",
    mutation_scale=14,
    lw=2.1,
    color=FLOW,
    zorder=8,
)
ax.add_patch(v1)

v2_start = (5.55, 1.78)
v2_length = 2.65
v2_end = (
    v2_start[0] + v2_length * math.cos(THETA),
    v2_start[1] + v2_length * math.sin(THETA),
)
v2 = FancyArrowPatch(
    v2_start,
    v2_end,
    arrowstyle="-|>",
    mutation_scale=14,
    lw=2.1,
    color=FLOW,
    zorder=8,
)
ax.add_patch(v2)


# Local unit normal (the leader line of the front annotation is dropped).
normal_origin = (
    apex[0] + 2.55 * math.cos(BETA),
    apex[1] + 2.55 * math.sin(BETA),
)
normal_length = 0.92
normal_end = (
    normal_origin[0] + normal_length * math.cos(BETA - math.pi / 2.0),
    normal_origin[1] + normal_length * math.sin(BETA - math.pi / 2.0),
)
normal = FancyArrowPatch(
    normal_origin,
    normal_end,
    arrowstyle="-|>",
    mutation_scale=11,
    lw=1.2,
    color=MUTED,
    linestyle=(0, (3, 2)),
    zorder=7,
)
ax.add_patch(normal)


def add_arc(radius, start, stop, color=INK):
    arc = Arc(
        apex,
        2.0 * radius,
        2.0 * radius,
        angle=0.0,
        theta1=math.degrees(start),
        theta2=math.degrees(stop),
        lw=1.15,
        color=color,
        zorder=9,
    )
    ax.add_patch(arc)


# Geometric angles at the attachment point: theta, beta, beta-theta.
add_arc(0.72, 0.0, THETA)
add_arc(1.05, 0.0, BETA)
add_arc(1.48, THETA, BETA, color=MUTED)


ax.plot(apex[0], apex[1], marker="o", ms=3.2, color=INK, zorder=10)


fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=0.98)
output_stem = Path(__file__).with_suffix("")
fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
fig.savefig(
    output_stem.parent / (output_stem.name + "_transparent.png"),
    dpi=350,
    bbox_inches="tight",
    transparent=True,
)
plt.close(fig)

for suffix in (".pdf", ".svg", ".png"):
    print("saved", output_stem.with_suffix(suffix))
print("saved", output_stem.parent / (output_stem.name + "_transparent.png"))

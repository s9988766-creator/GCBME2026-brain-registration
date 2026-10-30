"""Accuracy versus topology across the 4x2 grid.

One point per trained run. The x axis is the folding rate on a log scale, which
spans six orders of magnitude across the grid, so a linear axis would collapse
every diffeomorphic run onto zero. Filled markers are weakly supervised, hollow
markers unsupervised; marker shape encodes the architecture.

The figure makes one point: the runs separate into two vertical bands on the x
axis according to whether the deformation is produced by a group operation
(exp of a stationary velocity) or by additive accumulation, and that separation
is essentially independent of where they sit on the y axis.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

RUNS = [
    # label, file, marker, colour, supervised
    ("(A) direct displacement",      "vxm1_a2s_unsup",       "o", "#C44E52", False),
    ("(A) direct displacement",      "vxm1_a2s_softdice",    "o", "#C44E52", True),
    ("(B) U-Net + scal.-and-squar.", "vxmdiff_a2s_unsup",    "s", "#4C72B0", False),
    ("(B) U-Net + scal.-and-squar.", "vxmdiff_a2s_softdice", "s", "#4C72B0", True),
    ("(C) frozen VAE + scal.-sq.",   "v5_a2s_unsup",         "^", "#55A868", False),
    ("(C) frozen VAE + scal.-sq.",   "v5_a2s_softdice",      "^", "#55A868", True),
    ("(D) frozen VAE + Euler",       "v7_a2s_unsup",         "D", "#8172B2", False),
    ("(D) frozen VAE + Euler",       "v7_a2s_softdice",      "D", "#8172B2", True),
]
REFERENCE = [
    ("VoxelMorph-1 (released)", "voxelmorph1_test", "x", "#8C8C8C"),
    ("TransMorph (released)",   "transmorph_test",  "+", "#8C8C8C"),
]

FLOOR = 1e-6  # plotting floor: runs at exactly 0 folding are drawn here


def read(stem):
    rows = list(csv.DictReader(open(R / f"{stem}.csv", encoding="utf-8")))
    dice = np.array([float(r["dice_official_grouped17"]) for r in rows])
    jneg = np.array([float(r["jneg_percent"]) for r in rows])
    return dice.mean(), dice.std(ddof=1) / np.sqrt(len(dice)), max(jneg.mean(), FLOOR)


fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

seen = set()
for label, stem, marker, colour, sup in RUNS:
    d, se, j = read(stem + "_test")
    ax.errorbar(j, d, yerr=se, fmt=marker, ms=9, color=colour,
                mfc=colour if sup else "white", mew=1.6, ecolor=colour,
                capsize=2, zorder=3,
                label=label if label not in seen else None)
    seen.add(label)

for label, stem, marker, colour in REFERENCE:
    d, se, j = read(stem)
    ax.errorbar(j, d, yerr=se, fmt=marker, ms=9, color=colour, mew=1.6,
                ecolor=colour, capsize=2, zorder=2, label=label)

ax.set_xscale("log")
ax.set_xlabel("folding rate,  % of voxels with non-positive Jacobian  (log scale)")
ax.set_ylabel("Dice (official grouped-17)")
ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
ax.set_axisbelow(True)

# band separating group-structured integration from additive accumulation
# Band separating the two mechanisms. Its right edge sits at 1e-2 so that it
# contains every group-structured run, including (B) unsupervised at 1.9e-3,
# while still leaving two orders of magnitude before the additive cluster.
ax.axvspan(FLOOR / 2, 1e-2, color="#4C72B0", alpha=0.07, zorder=0)
ax.text(2.5e-6, 0.8355, r"group-structured integration,  $\phi=\exp(v)$",
        fontsize=8.5, color="#2F4F7F", ha="left", va="top")
ax.text(0.06, 0.8355, "additive accumulation", fontsize=8.5, color="#7A3033",
        ha="left", va="top")

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles, labels, fontsize=7.5, loc="center left",
                bbox_to_anchor=(0.015, 0.47), framealpha=0.95,
                handletextpad=0.6, borderpad=0.6)
leg.set_zorder(5)
ax.text(0.985, 0.025, "filled = weakly supervised     hollow = unsupervised",
        transform=ax.transAxes, fontsize=8, ha="right", color="#333333")

ax.set_ylim(0.695, 0.840)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"dice_vs_folding.{ext}", bbox_inches="tight")
print(f"wrote {OUT / 'dice_vs_folding.png'}")

print()
print(f"{'run':<34}{'Dice':>9}{'Jneg %':>13}")
for label, stem, *_ in RUNS:
    d, se, j = read(stem + "_test")
    print(f"{stem:<34}{d:>9.4f}{j:>13.6f}")

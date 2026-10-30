"""Paired statistics over the 2x2 architecture x supervision design.

Every model is evaluated on the same 115 test subjects, so the comparisons are
paired: a paired t-test and a Wilcoxon signed-rank test on the per-subject
grouped-17 Dice, plus Cohen's dz and a win count. Comparing group means with an
unpaired test would throw away the pairing and understate significance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


RUNS = {
    "Affine": "affine_test.csv",
    "VoxelMorph-1 official": "voxelmorph1_test.csv",
    "TransMorph official": "transmorph_test.csv",
    "VoxelMorph-1 unsup (ours)": "vxm1_a2s_unsup_test.csv",
    "VoxelMorph-1 +softDice": "vxm1_a2s_softdice_test.csv",
    "v5 unsup (ours)": "v5_a2s_unsup_test.csv",
    "v5 +softDice": "v5_a2s_softdice_test.csv",
    "VxM+VecInt unsup": "vxmdiff_a2s_unsup_test.csv",
    "VxM+VecInt +softDice": "vxmdiff_a2s_softdice_test.csv",
}

COMPARISONS = [
    ("VxM+VecInt +softDice", "v5 +softDice",
     "does DROPPING the frozen VAE help, holding integration fixed?  <-- attribution"),
    ("VoxelMorph-1 +softDice", "VxM+VecInt +softDice",
     "Dice cost of scaling-and-squaring integration (folding drops ~230,000x)"),
    ("VxM+VecInt unsup", "v5 unsup (ours)",
     "dropping the frozen VAE, unsupervised"),
    ("VxM+VecInt unsup", "VoxelMorph-1 unsup (ours)",
     "integration, unsupervised"),
    ("VoxelMorph-1 +softDice", "v5 +softDice",
     "architecture, WITH soft-Dice  <-- the original core question"),
    ("VoxelMorph-1 unsup (ours)", "v5 unsup (ours)",
     "architecture, WITHOUT soft-Dice"),
    ("VoxelMorph-1 +softDice", "VoxelMorph-1 unsup (ours)",
     "soft-Dice effect on VoxelMorph"),
    ("v5 +softDice", "v5 unsup (ours)",
     "soft-Dice effect on v5"),
    ("VoxelMorph-1 unsup (ours)", "VoxelMorph-1 official",
     "our 100-epoch loop vs the official 423-epoch checkpoint"),
    ("VoxelMorph-1 +softDice", "TransMorph official",
     "weakly supervised 274K vs unsupervised 46.8M (NOT a fair Dice comparison)"),
]


def read(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return ([r["subject"] for r in rows],
            np.asarray([float(r["dice_official_grouped17"]) for r in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    results = Path(args.results)

    data, subjects = {}, None
    for name, filename in RUNS.items():
        path = results / filename
        if not path.exists():
            print(f"  (missing {filename} -- skipping {name})")
            continue
        names, values = read(path)
        if subjects is None:
            subjects = names
        elif names != subjects:
            raise SystemExit(f"{filename} subject order differs -- pairing would be invalid")
        data[name] = values

    print("=" * 82)
    print("Test 115, official grouped-17 Dice")
    print(f"  {'model':<30}{'mean':>10}{'sd':>10}{'min':>9}{'max':>9}")
    for name, values in data.items():
        print(f"  {name:<30}{values.mean():>10.4f}{values.std(ddof=1):>10.4f}"
              f"{values.min():>9.4f}{values.max():>9.4f}")

    print("=" * 82)
    print("2x2  (rows = architecture, cols = supervision), Test 115 mean Dice")
    cells = [("VoxelMorph-1", "VoxelMorph-1 unsup (ours)", "VoxelMorph-1 +softDice"),
             ("VxM + VecInt", "VxM+VecInt unsup", "VxM+VecInt +softDice"),
             ("v5 (frozen VAE)", "v5 unsup (ours)", "v5 +softDice")]
    print(f"  {'':<18}{'beta=0':>12}{'beta=1':>12}{'delta':>10}")
    for label, unsup, sup in cells:
        if unsup in data and sup in data:
            a, b = data[unsup].mean(), data[sup].mean()
            print(f"  {label:<18}{a:>12.4f}{b:>12.4f}{b - a:>+10.4f}")
    if all(k in data for k in ("VoxelMorph-1 unsup (ours)", "v5 unsup (ours)",
                               "VoxelMorph-1 +softDice", "v5 +softDice")):
        d0 = data["v5 unsup (ours)"].mean() - data["VoxelMorph-1 unsup (ours)"].mean()
        d1 = data["v5 +softDice"].mean() - data["VoxelMorph-1 +softDice"].mean()
        print(f"  {'v5 - VoxelMorph':<18}{d0:>+12.4f}{d1:>+12.4f}")
        print(f"  interaction (does the VAE help more once supervised?): {d1 - d0:+.4f}")

    print("=" * 82)
    print("Paired comparisons (same 115 subjects)")
    for a_name, b_name, note in COMPARISONS:
        if a_name not in data or b_name not in data:
            continue
        a, b = data[a_name], data[b_name]
        diff = a - b
        t_stat, t_p = ttest_rel(a, b)
        try:
            _, w_p = wilcoxon(a, b)
        except ValueError:
            w_p = float("nan")
        dz = diff.mean() / diff.std(ddof=1)
        wins = int((diff > 0).sum())
        print(f"\n  {a_name}  vs  {b_name}")
        print(f"    {note}")
        print(f"    mean diff = {diff.mean():+.4f}   dz = {dz:+.3f}   wins {wins}/{len(diff)}")
        print(f"    paired t p = {t_p:.3e}    Wilcoxon p = {w_p:.3e}")

    print("=" * 82)
    print("Reminder: Dice is only comparable within the same supervision level.")


if __name__ == "__main__":
    main()

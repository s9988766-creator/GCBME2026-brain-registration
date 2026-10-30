"""Calibrate the locked evaluator against the official TransMorph artefacts.

Two independent checks, neither of which needs a GPU:

1. ``LABEL_NAMES`` re-derived from the official ``label_info.txt`` must match the
   hard-coded list, and must match the header row of the official result CSVs.
2. Re-aggregating the official per-subject per-label CSVs with our ``grouped17``
   must reproduce the numbers ``analysis.py`` prints (Affine ~0.386,
   VoxelMorph-1 ~0.729, TransMorph ~0.753).

If this script passes, any disagreement between our reproduction and the paper
is caused by inference, not by the metric.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import LABEL_NAMES, NUM_CLASSES, OUTSTRUCT, build_label_names, dice_p30_raw, grouped17


OFFICIAL_FILES = {
    "Affine": "affine.csv",
    "SyN (ANTs)": "ants_IXI.csv",
    "NiftyReg": "NiftyReg_IXI.csv",
    "deedsBCV": "deedsBCV_IXI.csv",
    "LDDMM": "lddmm_IXI.csv",
    "VoxelMorph-1": "Vxm_1_ncc_1_diffusion_1.csv",
    "VoxelMorph-2": "Vxm_2_ncc_1_diffusion_1.csv",
    "VoxelMorph-diff": "VxmDiff.csv",
    "CycleMorph": "CycleMorph.csv",
    "MIDIR": "MIDIR_ncc_1_diffusion_1.csv",
    "TransMorph": "TransMorph_ncc_1_diffusion_1.csv",
}


def read_official(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[1][1 : NUM_CLASSES + 1]
    dice, jdet = [], []
    for row in rows[2:]:
        if len(row) < NUM_CLASSES + 2 or row[1] == "":
            continue
        dice.append([float(v) for v in row[1 : NUM_CLASSES + 1]])
        jdet.append(float(row[NUM_CLASSES + 1]))
    return header, np.asarray(dice), np.asarray(jdet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transmorph-root", default=str(Path(__file__).resolve().parents[1] / "external" / "TransMorph"))
    args = parser.parse_args()
    root = Path(args.transmorph_root)

    print("=" * 78)
    print("Check 1: label names")
    label_info = root / "IXI" / "Baseline_registration_methods" / "VoxelMorph" / "label_info.txt"
    derived = build_label_names(label_info)
    assert derived == LABEL_NAMES, "LABEL_NAMES disagrees with official label_info.txt"
    print(f"  OK  46 class names re-derived from {label_info.name} match common.LABEL_NAMES")

    print("=" * 78)
    print("Check 2: grouped-17 aggregation vs official per-subject CSVs")
    print(f"  {'method':<18}{'grouped17':>12}{'+/-':>10}{'rawP30':>10}{'Jneg%':>10}{'n':>6}")
    results_dir = root / "IXI" / "Results"
    for name, filename in OFFICIAL_FILES.items():
        path = results_dir / filename
        if not path.exists():
            print(f"  {name:<18}  (missing {filename})")
            continue
        header, dice, jdet = read_official(path)
        assert header == LABEL_NAMES, f"{filename} header differs from LABEL_NAMES"
        main_scores = np.asarray([grouped17(row).mean() for row in dice])
        raw_scores = np.asarray([dice_p30_raw(row) for row in dice])
        print(
            f"  {name:<18}{main_scores.mean():>12.4f}{main_scores.std():>10.4f}"
            f"{raw_scores.mean():>10.4f}{jdet.mean() * 100:>10.4f}{len(dice):>6d}"
        )

    print("=" * 78)
    print("Expected from the TransMorph paper: Affine 0.386, VoxelMorph-1 0.729,")
    print("VoxelMorph-2 0.732, VoxelMorph-diff 0.580, CycleMorph 0.737,")
    print("MIDIR 0.742, TransMorph 0.753.")
    print(f"Groups: {len(OUTSTRUCT)} anatomical structures, unweighted mean.")


if __name__ == "__main__":
    main()

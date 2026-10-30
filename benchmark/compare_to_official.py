"""Per-subject comparison of our reproduced Test results against the official CSVs.

A matching mean can hide compensating per-subject errors, so this compares the
115 subject-level grouped-17 scores one by one.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import LABEL_NAMES, NUM_CLASSES, grouped17


PAIRS = [
    ("Affine", "affine_test.csv", "affine.csv"),
    ("VoxelMorph-1", "voxelmorph1_test.csv", "Vxm_1_ncc_1_diffusion_1.csv"),
    ("TransMorph", "transmorph_test.csv", "TransMorph_ncc_1_diffusion_1.csv"),
]


def read_official(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    dice, jdet = [], []
    for row in rows[2:]:
        if len(row) < NUM_CLASSES + 2 or row[1] == "":
            continue
        dice.append([float(v) for v in row[1 : NUM_CLASSES + 1]])
        jdet.append(float(row[NUM_CLASSES + 1]))
    return np.asarray([grouped17(d).mean() for d in dice]), np.asarray(jdet) * 100.0


def read_ours(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dice = np.asarray([float(r["dice_official_grouped17"]) for r in rows])
    jneg = np.asarray([float(r["jneg_percent"]) for r in rows]) if "jneg_percent" in rows[0] else np.zeros(len(rows))
    return dice, jneg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--official", default=str(Path(__file__).resolve().parents[1] / "external" / "TransMorph" / "IXI" / "Results"))
    args = parser.parse_args()

    ours_dir, official_dir = Path(args.results), Path(args.official)
    print(f"{'method':<16}{'ours':>9}{'official':>10}{'d(mean)':>10}"
          f"{'max|d|':>9}{'corr':>8}{'Jneg ours':>11}{'Jneg off':>10}")
    for name, ours_file, official_file in PAIRS:
        ours_path, official_path = ours_dir / ours_file, official_dir / official_file
        if not ours_path.exists():
            print(f"{name:<16}  (not run yet: {ours_file})")
            continue
        ours, ours_jneg = read_ours(ours_path)
        official, official_jneg = read_official(official_path)
        if len(ours) != len(official):
            print(f"{name:<16}  (n mismatch {len(ours)} vs {len(official)})")
            continue
        delta = ours - official
        corr = float(np.corrcoef(ours, official)[0, 1])
        print(
            f"{name:<16}{ours.mean():>9.4f}{official.mean():>10.4f}{delta.mean():>+10.4f}"
            f"{np.abs(delta).max():>9.4f}{corr:>8.4f}{ours_jneg.mean():>11.4f}{official_jneg.mean():>10.4f}"
        )

    print()
    print("Affine is deterministic, so its delta should be ~0 to numerical precision.")
    print("The learned models re-run inference, so tiny per-subject deltas are expected")
    print("from GPU nondeterminism, but the mean and the correlation should be tight.")


if __name__ == "__main__":
    main()

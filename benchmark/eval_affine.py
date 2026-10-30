"""Affine-only baseline under the locked atlas-to-subject protocol.

"Affine" means the dataset's existing affine preprocessing with no deformable
warp, i.e. Dice between the atlas segmentation and the subject segmentation.
This is the row the TransMorph paper reports as 0.386.

The Test run additionally cross-checks every per-subject, per-label value against
the official ``TransMorph/IXI/Results/affine.csv``, which is the strongest
available proof that our Dice implementation matches theirs voxel for voxel.
"""

from __future__ import annotations

import os
import argparse
import csv
import glob
from pathlib import Path

import numpy as np

from common import (
    LABEL_NAMES,
    NUM_CLASSES,
    dice_all_labels,
    load_pkl,
    normalize_seg,
    report,
    summarize_dice,
    write_rows,
)


def load_official_affine(path: Path) -> np.ndarray:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    values = [
        [float(v) for v in row[1 : NUM_CLASSES + 1]]
        for row in rows[2:]
        if len(row) >= NUM_CLASSES + 2 and row[1] != ""
    ]
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
    parser.add_argument("--split", choices=("Val", "Test"), default="Test")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--official-csv",
        default=str(Path(__file__).resolve().parents[1] / "external" / "TransMorph" / "IXI" / "Results" / "affine.csv"),
        help="Official per-subject CSV used to cross-check the Test split.",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    _, atlas_seg = load_pkl(root / "atlas.pkl")
    atlas_seg = normalize_seg(atlas_seg)

    rows, dice_matrix = [], []
    for path in sorted(glob.glob(str(root / args.split / "*.pkl"))):
        _, subject_seg = load_pkl(path)
        subject_seg = normalize_seg(subject_seg)
        row = {"subject": Path(path).stem}
        row.update(summarize_dice(atlas_seg, subject_seg))
        rows.append(row)
        dice_matrix.append(dice_all_labels(atlas_seg, subject_seg))

    output = args.output or str(Path(__file__).parent / "results" / f"affine_{args.split.lower()}.csv")
    write_rows(output, rows)
    report(f"Affine [{args.split}]", rows)
    print(f"  wrote {output}")

    official_path = Path(args.official_csv)
    if args.split == "Test" and official_path.exists():
        official = load_official_affine(official_path)
        ours = np.asarray(dice_matrix)
        print(f"  cross-check vs {official_path.name}: shapes {ours.shape} vs {official.shape}")
        if ours.shape == official.shape:
            aligned = np.abs(np.sort(ours, axis=0) - np.sort(official, axis=0)).max()
            in_order = np.abs(ours - official).max()
            print(f"    max |diff| subject-aligned  = {in_order:.3e}")
            print(f"    max |diff| order-invariant  = {aligned:.3e}")
            worst = int(np.unravel_index(np.abs(np.sort(ours, 0) - np.sort(official, 0)).argmax(), ours.shape)[1])
            print(f"    worst label (order-invariant) = {worst} {LABEL_NAMES[worst]}")


if __name__ == "__main__":
    main()

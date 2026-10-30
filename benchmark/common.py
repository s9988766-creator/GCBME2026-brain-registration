"""Shared definitions for the atlas-to-subject IXI benchmark.

Everything here is a faithful port of the official TransMorph IXI evaluation
protocol (https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration),
so that reproduced numbers are directly comparable to the published table.

Protocol summary
----------------
* Direction: moving = atlas, fixed = subject (``IXIBrainInferDataset`` yields
  ``x = atlas``, ``y = subject`` and the model is fed ``cat(x, y)``).
* Labels: FreeSurfer IDs are remapped to 0..45 by ``trans.Seg_norm``.
* Label transfer: 46-class one-hot, *trilinear* warp, argmax
  (``infer.py`` loops ``reg_model`` with ``mode='bilinear'`` over 46 channels).
* Per-label Dice: ``2 * inter / (union + 1e-5)``, computed on the full volume.
* Primary metric: ``analysis.py`` averages left/right structures into 17
  anatomical groups and then takes an unweighted mean over the 17 groups.
* Folding metric: ``jacobian_determinant_vxm`` followed by
  ``sum(detJ <= 0) / prod(shape)``.
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SHAPE = (160, 192, 224)
NUM_CLASSES = 46

# trans.Seg_norm / utils.process_label seg_table: FreeSurfer ID -> class 0..45.
SEG_TABLE = np.asarray([
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52,
    53, 54, 58, 60, 62, 63, 72, 77, 80, 85, 251, 252, 253, 254, 255,
], dtype=np.int16)

# utils.dice_val_VOI: the 30 raw classes used for the "VOI" Dice printed during
# training. Kept as a secondary diagnostic only -- it is NOT the published metric.
P30 = np.asarray([
    1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36,
], dtype=np.int16)

# Class 0..45 names, verified byte-identical to the header row of the official
# per-subject CSVs in TransMorph/IXI/Results/.
LABEL_NAMES = [
    "Unknown", "Left-Cerebral-White-Matter", "Left-Cerebral-Cortex",
    "Left-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper*", "Left-Caudate",
    "Left-Putamen", "Left-Pallidum", "3rd-Ventricle", "4th-Ventricle",
    "Brain-Stem", "Left-Hippocampus", "Left-Amygdala", "CSF",
    "Left-Accumbens-area", "Left-VentralDC", "Left-vessel", "Left-choroid-plexus",
    "Right-Cerebral-White-Matter", "Right-Cerebral-Cortex",
    "Right-Lateral-Ventricle", "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter", "Right-Cerebellum-Cortex",
    "Right-Thalamus-Proper*", "Right-Caudate", "Right-Putamen", "Right-Pallidum",
    "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "Right-VentralDC", "Right-vessel", "Right-choroid-plexus", "5th-Ventricle",
    "WM-hypointensities", "non-WM-hypointensities", "Optic-Chiasm",
    "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior",
    "CC_Anterior",
]

# analysis.py `outstruct`, in the published order.
OUTSTRUCT = [
    "Brain-Stem", "Thalamus", "Cerebellum-Cortex", "Cerebral-White-Matter",
    "Cerebellum-White-Matter", "Putamen", "VentralDC", "Pallidum", "Caudate",
    "Lateral-Ventricle", "Hippocampus", "3rd-Ventricle", "4th-Ventricle",
    "Amygdala", "Cerebral-Cortex", "CSF", "choroid-plexus",
]

# analysis.py resolves each group by substring match against the CSV header,
# i.e. `if stct in item`. Reproduced here so the mapping is derived, not typed.
GROUP_INDICES = {
    name: tuple(i for i, label in enumerate(LABEL_NAMES) if name in label)
    for name in OUTSTRUCT
}


def build_label_names(label_info_path: str | Path) -> list[str]:
    """Re-derive LABEL_NAMES from the official label_info.txt (validation helper)."""
    lines = Path(label_info_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    names: dict[int, str] = {}
    for class_index, seg_label in enumerate(SEG_TABLE.tolist()):
        for line in lines:
            fields = re.sub(" +", " ", line).split(" ")
            try:
                value = int(fields[0])
            except (ValueError, IndexError):
                continue
            if value == seg_label:
                names[class_index] = fields[1]
    return [names[i] for i in range(NUM_CLASSES)]


def load_pkl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        image, seg = pickle.load(handle)
    return np.asarray(image, np.float32), np.asarray(seg)


def normalize_seg(seg: np.ndarray) -> np.ndarray:
    """Exact TransMorph IXI ``trans.Seg_norm`` mapping from FreeSurfer IDs to 0..45."""
    lut = np.zeros(256, dtype=np.int16)
    lut[SEG_TABLE] = np.arange(len(SEG_TABLE), dtype=np.int16)
    return lut[np.asarray(seg, dtype=np.uint8)]


def dice_all_labels(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-class Dice for classes 0..45, matching ``utils.dice_val_substruct``."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    pred_counts = np.bincount(pred.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    target_counts = np.bincount(target.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    agree = pred == target
    inter = np.bincount(pred[agree].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return (2.0 * inter) / (pred_counts + target_counts + 1e-5)


def grouped17(dice46: np.ndarray) -> np.ndarray:
    """Average left/right into the 17 anatomical groups used by ``analysis.py``."""
    return np.asarray(
        [float(np.mean([dice46[i] for i in GROUP_INDICES[name]])) for name in OUTSTRUCT],
        dtype=np.float64,
    )


def dice_p30_raw(dice46: np.ndarray) -> float:
    """Unweighted mean over the 30 raw ``dice_val_VOI`` classes (diagnostic only)."""
    return float(np.mean([dice46[i] for i in P30]))


def summarize_dice(pred: np.ndarray, target: np.ndarray) -> dict:
    """Return the locked metric bundle for one subject."""
    dice46 = dice_all_labels(pred, target)
    groups = grouped17(dice46)
    row = {
        "dice_official_grouped17": float(groups.mean()),
        "dice_p30_raw": dice_p30_raw(dice46),
    }
    row.update({f"group_{name}": float(value) for name, value in zip(OUTSTRUCT, groups)})
    row.update({f"label_{i}_{LABEL_NAMES[i]}": float(dice46[i]) for i in range(NUM_CLASSES)})
    return row


def warp_seg_one_hot(seg: torch.Tensor, flow: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Official label transfer: 46-class one-hot, trilinear warp, argmax.

    ``seg`` is (B,1,H,W,D) integer, ``flow`` is (B,3,H,W,D) displacement in voxels.
    """
    shape = flow.shape[2:]
    vectors = [torch.arange(0, size, device=flow.device, dtype=torch.float32) for size in shape]
    grid = torch.stack(torch.meshgrid(*vectors, indexing="ij"), dim=0)[None]
    locations = grid + flow
    for axis, size in enumerate(shape):
        locations[:, axis] = 2.0 * (locations[:, axis] / (size - 1) - 0.5)
    locations = locations.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

    labels = seg.long().squeeze(1)
    best_value = None
    best_index = None
    for start in range(0, NUM_CLASSES, chunk):
        stop = min(start + chunk, NUM_CLASSES)
        block = torch.stack([(labels == i).float() for i in range(start, stop)], dim=1)
        warped = F.grid_sample(block, locations, mode="bilinear", align_corners=True)
        value, index = warped.max(dim=1, keepdim=True)
        index = index + start
        if best_value is None:
            best_value, best_index = value, index
        else:
            take = value > best_value
            best_value = torch.where(take, value, best_value)
            best_index = torch.where(take, index, best_index)
    return best_index


def jacobian_determinant_vxm(disp: np.ndarray) -> np.ndarray:
    """Exact port of ``utils.jacobian_determinant_vxm`` (disp is (3,H,W,D))."""
    disp = np.asarray(disp, dtype=np.float64).transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in volshape], indexing="ij"), axis=-1)
    jac = np.gradient(disp + grid, axis=(0, 1, 2))
    dx, dy, dz = jac
    det0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    det1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    det2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
    return det0 - det1 + det2


def jacobian_stats(flow: np.ndarray) -> tuple[float, float, float]:
    """Return (official non-positive fraction, Jneg %, log-Jacobian SD)."""
    det = jacobian_determinant_vxm(flow)
    fraction = float(np.sum(det <= 0) / det.size)
    logstd = float(np.std(np.log(np.clip(det, 1e-6, None))))
    return fraction, fraction * 100.0, logstd


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(tag: str, rows: list[dict]) -> None:
    """Print the locked headline summary for a finished run."""
    main = np.asarray([row["dice_official_grouped17"] for row in rows])
    raw = np.asarray([row["dice_p30_raw"] for row in rows])
    print(f"{tag}: n={len(rows)}")
    print(f"  Dice (official grouped-17) = {main.mean():.6f} +/- {main.std():.6f}")
    print(f"  Dice (raw P30, diagnostic) = {raw.mean():.6f} +/- {raw.std():.6f}")
    if "jneg_percent" in rows[0]:
        jneg = np.asarray([row["jneg_percent"] for row in rows])
        logstd = np.asarray([row["logj_std"] for row in rows])
        print(f"  Jneg%                      = {jneg.mean():.6f} +/- {jneg.std():.6f}")
        print(f"  log-Jacobian SD            = {logstd.mean():.6f} +/- {logstd.std():.6f}")

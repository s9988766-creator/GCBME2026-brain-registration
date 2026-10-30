"""v5 inverse diagnostic: score the EXISTING subject->atlas weights in both directions.

v5 (`LatentRegUNet`) predicts a *static* velocity field `v` and integrates it with
scaling-and-squaring, so the exact inverse transform is available for free:

    phi     = VecInt( v)     displacement on the atlas grid, pulls from subject space
    phi_inv = VecInt(-v)     displacement on the subject grid, pulls from atlas space

This is not an approximation -- it is the standard diffeomorphic inverse (the same
trick VoxelMorph's bidirectional mode uses), accurate to the same order as the
forward integration.

What this measures, and what it does NOT
----------------------------------------
The network was still *optimized* for subject->atlas (its soft-Dice target was the
atlas segmentation, identical for every training sample). Re-expressing its
transform in the other direction therefore isolates the pure geometry/resampling
component of the direction change, while holding the learned deformation fixed.
It does NOT tell us what v5 would score if retrained atlas->subject -- that
requires actual retraining, because the supervision target changes character.

Three numbers are reported, all under the locked official grouped-17 protocol:

  1. subject->atlas (as trained) -- old direction, new metric. The gap to the
     legacy 0.766 is the *protocol* effect (nearest-neighbour + P33 -> official).
  2. atlas->subject via phi_inv  -- new direction, new metric. The gap to (1) is
     the *direction* effect at fixed transform quality.
  3. Affine, already known: 0.3858 Test / 0.3891 Val.

Writes only into benchmark_atlas_to_subject/results/. It never touches
outputs/**/eval/metrics.csv.
"""

from __future__ import annotations

import os
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    NUM_CLASSES,
    LABEL_NAMES,
    OUTSTRUCT,
    GROUP_INDICES,
    jacobian_stats,
    load_pkl,
    normalize_seg,
    report,
    summarize_dice,
    warp_seg_one_hot,
    write_rows,
)
from reg_common import SpatialVAE, LatentRegUNet  # noqa: E402


SHAPE = (160, 192, 224)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
    parser.add_argument("--lfm-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--split", choices=("Val", "Test"), default="Test")
    parser.add_argument("--run", default="ixi_register_unet_l1.0_b1.0", help="v5 run directory.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    here = Path(__file__).parent
    lfm = Path(args.lfm_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_ckpt = lfm / "outputs" / "ixi_vae" / "vae_ixi_best.pth"
    reg_ckpt = lfm / "outputs" / args.run / "register_ixi_best.pth"
    print(f"VAE  : {vae_ckpt}")
    print(f"v5   : {reg_ckpt}")

    vae = SpatialVAE().to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device)["model"])
    checkpoint = torch.load(reg_ckpt, map_location=device)
    int_steps = checkpoint.get("int_steps", 7)
    model = LatentRegUNet(vae, full_shape=SHAPE, int_steps=int_steps).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"  int_steps={int_steps}  epoch={checkpoint.get('epoch')}")
    print(f"  trainable params = {sum(p.numel() for p in model.dec.parameters()):,} (velocity decoder)")

    root = Path(args.data_root)
    atlas_img, atlas_seg_raw = load_pkl(root / "atlas.pkl")
    atlas_seg = normalize_seg(atlas_seg_raw)
    fixed = torch.from_numpy(atlas_img)[None, None].to(device)
    atlas_seg_t = torch.from_numpy(atlas_seg.astype(np.int16))[None, None].to(device)

    present_atlas = set(np.unique(atlas_seg).tolist())
    missing = [i for i in range(NUM_CLASSES) if i not in present_atlas]
    print(f"  classes absent from the atlas segmentation: {[LABEL_NAMES[i] for i in missing]}")
    affected = [g for g in OUTSTRUCT if any(i in missing for i in GROUP_INDICES[g])]
    print(f"  grouped-17 groups touched by those: {affected or 'none'}")

    paths = sorted(glob.glob(str(root / args.split / "*.pkl")))
    if args.limit:
        paths = paths[: args.limit]
    print(f"Split: {args.split}  n={len(paths)}")

    forward_rows, inverse_rows = [], []
    with torch.no_grad():
        for index, path in enumerate(paths):
            subject_img, subject_seg_raw = load_pkl(path)
            subject_seg = normalize_seg(subject_seg_raw)
            moving = torch.from_numpy(subject_img)[None, None].to(device)
            subject_seg_t = torch.from_numpy(subject_seg.astype(np.int16))[None, None].to(device)

            # As trained: moving = subject, fixed = atlas.
            _, phi, velocity = model(moving, fixed)
            phi_inv = model.vecint(-velocity)

            # Validity of the inverse: composing the two must return the identity.
            # (phi_inv o phi)(x) = phi(x) + phi_inv(x + phi(x)), which must be ~0.
            residual = phi + model.stn(phi_inv, phi)
            residual_mag = residual.norm(dim=1)
            phi_mag = phi.norm(dim=1)
            consistency = {
                "inv_residual_mean_vox": float(residual_mag.mean()),
                "inv_residual_p99_vox": float(residual_mag.flatten().quantile(0.99)),
                "disp_mean_vox": float(phi_mag.mean()),
                "disp_p99_vox": float(phi_mag.flatten().quantile(0.99)),
                "disp_max_vox": float(phi_mag.max()),
            }

            # (1) subject -> atlas: warp the subject labels onto the atlas.
            pred_fwd = warp_seg_one_hot(subject_seg_t, phi.float()).cpu().numpy()[0, 0]
            row_fwd = {"subject": Path(path).stem}
            row_fwd.update(summarize_dice(pred_fwd, atlas_seg))
            _, jneg_f, logj_f = jacobian_stats(phi.detach().cpu().numpy()[0])
            row_fwd["jneg_percent"], row_fwd["logj_std"] = jneg_f, logj_f
            row_fwd.update(consistency)
            forward_rows.append(row_fwd)

            # (2) atlas -> subject: warp the atlas labels onto the subject via exp(-v).
            pred_inv = warp_seg_one_hot(atlas_seg_t, phi_inv.float()).cpu().numpy()[0, 0]
            row_inv = {"subject": Path(path).stem}
            row_inv.update(summarize_dice(pred_inv, subject_seg))
            _, jneg_i, logj_i = jacobian_stats(phi_inv.detach().cpu().numpy()[0])
            row_inv["jneg_percent"], row_inv["logj_std"] = jneg_i, logj_i
            inverse_rows.append(row_inv)

            print(
                f"  [{index + 1:3d}/{len(paths)}] {Path(path).stem:<28} "
                f"s->a={row_fwd['dice_official_grouped17']:.4f} "
                f"a->s={row_inv['dice_official_grouped17']:.4f} "
                f"Jneg {jneg_f:.4f}/{jneg_i:.4f} "
                f"|disp|={consistency['disp_mean_vox']:.2f}vox "
                f"invres={consistency['inv_residual_mean_vox']:.4f}vox",
                flush=True,
            )

    suffix = f"_first{args.limit}" if args.limit else ""
    split = args.split.lower()
    forward_out = here / "results" / f"v5_subject_to_atlas_{split}{suffix}.csv"
    inverse_out = here / "results" / f"v5_inverse_atlas_to_subject_{split}{suffix}.csv"
    write_rows(forward_out, forward_rows)
    write_rows(inverse_out, inverse_rows)

    print()
    report(f"v5 subject->atlas (as trained) [{args.split}]", forward_rows)
    print()
    report(f"v5 atlas->subject via exp(-v)  [{args.split}]", inverse_rows)
    print()
    fwd = np.mean([r["dice_official_grouped17"] for r in forward_rows])
    inv = np.mean([r["dice_official_grouped17"] for r in inverse_rows])
    print(f"Direction effect at fixed transform: {inv - fwd:+.4f} grouped-17 Dice")
    print()
    print("Inverse validity and deformation amplitude:")
    for key, label in [
        ("inv_residual_mean_vox", "mean |phi_inv o phi|  (0 = exact inverse)"),
        ("inv_residual_p99_vox", "p99  |phi_inv o phi|"),
        ("disp_mean_vox", "mean |phi|            (deformation amplitude)"),
        ("disp_p99_vox", "p99  |phi|"),
        ("disp_max_vox", "max  |phi|"),
    ]:
        values = np.asarray([r[key] for r in forward_rows])
        print(f"  {label:<42} = {values.mean():.4f} voxels")
    ratio = np.mean([r["inv_residual_mean_vox"] for r in forward_rows]) / max(
        np.mean([r["disp_mean_vox"] for r in forward_rows]), 1e-9
    )
    print(f"  residual / displacement                    = {ratio:.4%}")
    print("  If the residual is a tiny fraction of the displacement, exp(-v) is a")
    print("  valid inverse and the direction gap above is real, not an artifact.")
    print(f"  wrote {forward_out.name}, {inverse_out.name}")


if __name__ == "__main__":
    main()

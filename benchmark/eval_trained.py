"""Evaluate our own trained checkpoints under the locked atlas->subject protocol.

Same evaluator as `eval_official.py` -- the one calibrated to reproduce the
official per-subject CSVs -- applied to checkpoints produced by
`train_softdice.py`.

    python eval_trained.py --run vxm1_a2s_softdice --split Test
"""

from __future__ import annotations

import os
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from common import jacobian_stats, load_pkl, normalize_seg, report, summarize_dice, warp_seg_one_hot, write_rows
from train_softdice import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="directory name under runs/")
    parser.add_argument("--data-root", default=os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
    parser.add_argument("--split", choices=("Val", "Test"), default="Test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    here = Path(__file__).parent
    device = torch.device("cuda")
    ckpt_path = here / "runs" / args.run / "best.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = checkpoint["arch"]
    print(f"run   : {args.run}")
    print(f"ckpt  : {ckpt_path}")
    print(f"  arch={arch} lambda={checkpoint['lam']} beta={checkpoint['beta']} "
          f"selected at epoch {checkpoint['epoch']} (Val Dice {checkpoint['val_dice']:.6f})")
    if arch == "v7":
        print(f"  n_steps={checkpoint.get('n_steps')} (inference uses noise_std=0)")

    model, forward = build_model(arch, device,
                                 n_steps=checkpoint.get('n_steps', 10),
                                 noise_std=checkpoint.get('noise_std', 0.05))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  params: trainable={trainable:,} total={total:,}")

    root = Path(args.data_root)
    atlas_image, atlas_seg = load_pkl(root / "atlas.pkl")
    atlas_seg = normalize_seg(atlas_seg)
    moving = torch.from_numpy(atlas_image)[None, None].to(device)
    atlas_seg_t = torch.from_numpy(atlas_seg.astype(np.int16))[None, None].to(device)

    paths = sorted(glob.glob(str(root / args.split / "*.pkl")))
    if args.limit:
        paths = paths[: args.limit]
    print(f"split : {args.split}  n={len(paths)}")

    rows = []
    with torch.no_grad():
        for index, path in enumerate(paths):
            subject_image, subject_seg = load_pkl(path)
            subject_seg = normalize_seg(subject_seg)
            fixed = torch.from_numpy(subject_image)[None, None].to(device)

            torch.cuda.synchronize()
            start = time.perf_counter()
            _, flow = forward(moving, fixed)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            pred = warp_seg_one_hot(atlas_seg_t, flow.float()).cpu().numpy()[0, 0]
            _, jneg, logj = jacobian_stats(flow.detach().cpu().numpy()[0])

            row = {"subject": Path(path).stem}
            row.update(summarize_dice(pred, subject_seg))
            row["jneg_percent"], row["logj_std"], row["infer_ms"] = jneg, logj, elapsed_ms
            rows.append(row)
            print(f"  [{index + 1:3d}/{len(paths)}] {Path(path).stem:<28} "
                  f"grouped17={row['dice_official_grouped17']:.4f} Jneg%={jneg:.4f} {elapsed_ms:.1f}ms",
                  flush=True)

    suffix = f"_first{args.limit}" if args.limit else ""
    output = args.output or str(here / "results" / f"{args.run}_{args.split.lower()}{suffix}.csv")
    write_rows(output, rows)
    report(f"{args.run} [{args.split}]", rows)
    print(f"  mean inference = {np.mean([r['infer_ms'] for r in rows]):.1f} ms")
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()

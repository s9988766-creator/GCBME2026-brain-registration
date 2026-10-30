"""Compare deformation amplitude across methods.

Low folding is only meaningful if the field is actually deforming. This reports
mean/p99/max displacement magnitude alongside Jneg for the official baselines, so
v5's 0.001% folding can be judged against how far it actually moves voxels.
"""

from __future__ import annotations

import os
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from common import jacobian_stats, load_pkl
from eval_official import BUILDERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
    parser.add_argument("--split", default="Val")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    here = Path(__file__).parent
    device = torch.device("cuda")
    root = Path(args.data_root)
    atlas_img, _ = load_pkl(root / "atlas.pkl")
    x = torch.from_numpy(atlas_img)[None, None].to(device)
    paths = sorted(glob.glob(str(root / args.split / "*.pkl")))[: args.limit]

    print(f"{'method':<16}{'|disp| mean':>13}{'p99':>9}{'max':>9}{'Jneg %':>10}{'logJ SD':>10}")
    for name, (builder, ckpt_name) in BUILDERS.items():
        model = builder((160, 192, 224))
        state = torch.load(here / "checkpoints" / "official" / ckpt_name,
                           map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

        stats = []
        with torch.no_grad():
            for path in paths:
                subject_img, _ = load_pkl(path)
                y = torch.from_numpy(subject_img)[None, None].to(device)
                _, flow = model(torch.cat((x, y), dim=1))
                magnitude = flow.norm(dim=1)
                _, jneg, logj = jacobian_stats(flow.detach().cpu().numpy()[0])
                stats.append((
                    float(magnitude.mean()),
                    float(magnitude.flatten().quantile(0.99)),
                    float(magnitude.max()),
                    jneg,
                    logj,
                ))
        mean_stats = np.asarray(stats).mean(axis=0)
        print(f"{name:<16}{mean_stats[0]:>13.2f}{mean_stats[1]:>9.2f}{mean_stats[2]:>9.2f}"
              f"{mean_stats[3]:>10.4f}{mean_stats[4]:>10.4f}")
        del model
        torch.cuda.empty_cache()

    print()
    print(f"(n={len(paths)} {args.split} subjects, displacement in voxels)")


if __name__ == "__main__":
    main()

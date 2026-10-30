"""Does the frozen VAE buy anything the Dice table cannot see?

Measures, on an otherwise idle GPU, with warmup and repeats:
  * inference latency per subject,
  * peak inference VRAM,
  * batch throughput for processing a whole cohort,
  * and one advantage that is structural rather than incidental --

In atlas-based registration the MOVING image is the same atlas for every subject.
v5 encodes moving and fixed separately with a frozen encoder, so the atlas
features are a constant that can be computed once and reused for the entire
cohort. VoxelMorph concatenates (atlas, subject) at the input and runs one joint
U-Net, so nothing can be hoisted out of the loop. That asymmetry is a property of
the frozen-two-stream design, not of the VAE weights.
"""

from __future__ import annotations

import os
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from common import load_pkl
from train_softdice import build_model

SHAPE = (160, 192, 224)


def timed(fn, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cohort", type=int, default=115, help="cohort size for the throughput estimate")
    args = parser.parse_args()

    here = Path(__file__).parent
    device = torch.device("cuda")
    root = Path(os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
    atlas, _ = load_pkl(root / "atlas.pkl")
    subject, _ = load_pkl(sorted(glob.glob(str(root / "Val" / "*.pkl")))[0])
    x = torch.from_numpy(atlas)[None, None].to(device)
    y = torch.from_numpy(subject)[None, None].to(device)

    # All four architectures of the 4x2 grid must be timed the SAME way. v7 was
    # missing here, so the paper's latency column took its two cells from the
    # eval logs instead -- which time a whole evaluation pass, not a warmed-up
    # forward, and gave the same architecture 83.8 ms in one row and 199.1 ms in
    # another. Inference does not depend on supervision; one number per row.
    runs = [
        ("VoxelMorph-1 +softDice", "vxm1_a2s_softdice"),
        ("VxM+VecInt +softDice", "vxmdiff_a2s_softdice"),
        ("v5 (frozen VAE) +softDice", "v5_a2s_softdice"),
        ("v7 (frozen VAE, Euler) +softDice", "v7_a2s_softdice"),
    ]

    print(f"{'model':<28}{'latency ms':>12}{'peak VRAM MB':>15}{'cohort ' + str(args.cohort) + ' s':>16}")
    latencies = {}
    with torch.no_grad():
        for label, run in runs:
            ck = torch.load(here / "runs" / run / "best.pth", map_location="cpu", weights_only=False)
            model, forward = build_model(ck["arch"], device)
            model.load_state_dict(ck["model"])
            model.eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ms = timed(lambda: forward(x, y), args.repeats)
            peak = torch.cuda.max_memory_allocated() / 1e6
            latencies[label] = ms
            print(f"{label:<28}{ms:>12.1f}{peak:>15.0f}{ms * args.cohort / 1000:>16.1f}")
            del model
            torch.cuda.empty_cache()

    # v5 only: hoist the constant atlas encoding out of the per-subject loop.
    print()
    print("v5 with the atlas features cached across the cohort")
    ck = torch.load(here / "runs" / "v5_a2s_softdice" / "best.pth", map_location="cpu", weights_only=False)
    model, _ = build_model("v5", device)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        mu_m, sk_m = model.vae.encoder.features(x)   # constant for every subject

        def cached():
            mu_f, sk_f = model.vae.encoder.features(y)
            v_half = model.dec(mu_m, mu_f, sk_m, sk_f)
            v = torch.nn.functional.interpolate(v_half, scale_factor=2, mode="trilinear",
                                                align_corners=True) * 2
            flow = model.vecint(v)
            return model.stn(x, flow), flow

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ms_cached = timed(cached, args.repeats)
        peak = torch.cuda.max_memory_allocated() / 1e6
    base = latencies["v5 (frozen VAE) +softDice"]
    print(f"{'v5, atlas encoded once':<28}{ms_cached:>12.1f}{peak:>15.0f}"
          f"{ms_cached * args.cohort / 1000:>16.1f}")
    print(f"  saves {base - ms_cached:.1f} ms/subject ({(base - ms_cached) / base * 100:.0f}%) "
          f"vs re-encoding the atlas every time")
    fastest_other = min(v for k, v in latencies.items() if not k.startswith("v5"))
    print(f"  vs the best non-VAE model: {fastest_other:.1f} -> {ms_cached:.1f} ms "
          f"({fastest_other / ms_cached:.2f}x)")

    print()
    print("Measured training cost from the completed 100-epoch runs (403 subjects/epoch):")
    print("  beta=1   VoxelMorph 126 s/ep | VxM+VecInt 154 s/ep | v5  95 s/ep")
    print("  beta=0   VoxelMorph  86 s/ep | VxM+VecInt 114 s/ep | v5  54 s/ep")


if __name__ == "__main__":
    main()

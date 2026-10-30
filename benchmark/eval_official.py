"""Reproduce the official VoxelMorph-1 / TransMorph IXI checkpoints.

The model definitions are imported directly from the vendored official repo at
``external/TransMorph`` -- nothing is reimplemented -- so a matching Dice is
evidence that our data pipeline and evaluator agree with the paper.

Usage
-----
    python eval_official.py --model voxelmorph1 --split Val --limit 1   # smoke test
    python eval_official.py --model voxelmorph1 --split Val
    python eval_official.py --model transmorph  --split Val

Notes
-----
* ``timm`` is not installed in the ``lfm`` env (it needs torchvision, which we do
  not want to pull into a working training environment). TransMorph only uses
  ``DropPath``/``trunc_normal_``/``to_3tuple`` from it, none of which affect a
  ``model.eval()`` forward pass with loaded weights, so a minimal shim is
  injected instead.
* The official ``SpatialTransformer.__init__`` calls ``.cuda()`` unconditionally,
  so a GPU is required.
"""

from __future__ import annotations

import os
import argparse
import glob
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from common import load_pkl, normalize_seg, jacobian_stats, report, summarize_dice, warp_seg_one_hot, write_rows


REPO = Path(__file__).resolve().parents[1] / "external" / "TransMorph"
VXM_DIR = REPO / "IXI" / "Baseline_registration_methods" / "VoxelMorph"
TM_DIR = REPO / "IXI" / "TransMorph"


def install_timm_shim() -> None:
    """Provide the three timm symbols TransMorph imports, without torchvision."""
    try:
        import timm.models.layers  # noqa: F401
        return
    except Exception:
        pass

    def drop_path(x, drop_prob: float = 0.0, training: bool = False):
        if drop_prob == 0.0 or not training:
            return x
        keep = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return torch.nn.init.trunc_normal_(tensor, mean, std, a, b)

    def to_3tuple(x):
        return tuple(x) if isinstance(x, (tuple, list)) else (x, x, x)

    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    layers.trunc_normal_ = trunc_normal_
    layers.to_3tuple = to_3tuple
    models = types.ModuleType("timm.models")
    models.layers = layers
    timm = types.ModuleType("timm")
    timm.models = models
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers
    print("  [shim] injected minimal timm.models.layers (eval-only, no torchvision)")


def build_voxelmorph1(shape):
    sys.path.insert(0, str(VXM_DIR))
    from models import VxmDense_1  # type: ignore

    return VxmDense_1(shape)


def build_transmorph(shape):
    install_timm_shim()
    # Both official trees ship a top-level `models`; VoxelMorph's is a plain module
    # and TransMorph's is a package, so a stale import shadows the other. Purge it
    # so the two can be built in one process.
    for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if p != str(VXM_DIR)]
    sys.path.insert(0, str(TM_DIR))
    from models.TransMorph import CONFIGS as CONFIGS_TM  # type: ignore
    import models.TransMorph as TransMorph  # type: ignore

    config = CONFIGS_TM["TransMorph"]
    assert tuple(config.img_size) == tuple(shape), f"config img_size {config.img_size} != {shape}"
    return TransMorph.TransMorph(config)


BUILDERS = {
    "voxelmorph1": (build_voxelmorph1, "voxelmorph1.zip"),
    "transmorph": (build_transmorph, "transmorph.zip"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--data-root", default=os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
    parser.add_argument("--split", choices=("Val", "Test"), default="Val")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test on the first N subjects.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    here = Path(__file__).parent
    shape = (160, 192, 224)
    device = torch.device("cuda")

    builder, ckpt_name = BUILDERS[args.model]
    ckpt_path = here / "checkpoints" / "official" / ckpt_name
    print(f"Model : {args.model}")
    print(f"Ckpt  : {ckpt_path}")

    model = builder(shape)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    print(f"  checkpoint epoch={checkpoint.get('epoch')} "
          f"best_mse={checkpoint.get('best_mse')} best_dsc={checkpoint.get('best_dsc')}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  WARNING missing={list(missing)} unexpected={list(unexpected)}")
    else:
        print("  state_dict loaded strictly (no missing / unexpected keys)")
    trainable = sum(p.numel() for p in model.parameters())
    print(f"  parameters (excl. STN grid buffer) = {trainable:,}")
    model.to(device).eval()

    root = Path(args.data_root)
    atlas_img, atlas_seg = load_pkl(root / "atlas.pkl")
    atlas_seg = normalize_seg(atlas_seg)
    x = torch.from_numpy(atlas_img)[None, None].to(device)
    x_seg = torch.from_numpy(atlas_seg.astype(np.int16))[None, None].to(device)

    paths = sorted(glob.glob(str(root / args.split / "*.pkl")))
    if args.limit:
        paths = paths[: args.limit]
    print(f"Split : {args.split}  n={len(paths)}")

    rows = []
    with torch.no_grad():
        for index, path in enumerate(paths):
            subject_img, subject_seg = load_pkl(path)
            subject_seg = normalize_seg(subject_seg)
            y = torch.from_numpy(subject_img)[None, None].to(device)

            torch.cuda.synchronize()
            start = time.perf_counter()
            _, flow = model(torch.cat((x, y), dim=1))
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            warped_seg = warp_seg_one_hot(x_seg, flow.float())
            pred = warped_seg.cpu().numpy()[0, 0]
            _, jneg_percent, logj_std = jacobian_stats(flow.detach().cpu().numpy()[0])

            row = {"subject": Path(path).stem}
            row.update(summarize_dice(pred, subject_seg))
            row["jneg_percent"] = jneg_percent
            row["logj_std"] = logj_std
            row["infer_ms"] = elapsed_ms
            rows.append(row)
            print(
                f"  [{index + 1:3d}/{len(paths)}] {Path(path).stem:<28} "
                f"grouped17={row['dice_official_grouped17']:.4f} "
                f"rawP30={row['dice_p30_raw']:.4f} "
                f"Jneg%={jneg_percent:.4f} {elapsed_ms:.1f}ms",
                flush=True,
            )

    suffix = f"_first{args.limit}" if args.limit else ""
    output = args.output or str(here / "results" / f"{args.model}_{args.split.lower()}{suffix}.csv")
    write_rows(output, rows)
    report(f"{args.model} [{args.split}]", rows)
    print(f"  mean inference             = {np.mean([r['infer_ms'] for r in rows]):.1f} ms")
    print(f"  parameters                 = {trainable:,}")
    print(f"  wrote {output}")


if __name__ == "__main__":
    main()

"""Train registration models atlas->subject with a shared, identical recipe.

The question this exists to answer:

    Is v5's high Dice caused by the frozen spatial VAE, or purely by the
    soft-Dice supervision?

To answer it, architecture must be the ONLY variable. Both arms therefore share
the direction, the loss, the optimizer, the schedule, the epoch budget, the
soft-Dice label set, and the model-selection rule. `--arch` is the only switch.

    --arch voxelmorph1        official VxmDense_1, direct displacement (274,387)
    --arch voxelmorph1_diff   same U-Net + VecInt(7) scaling-and-squaring (274,387)
    --arch v5                 frozen VAE + stationary velocity + VecInt(7) (678,787)
    --arch v7                 frozen VAE + time-varying velocity + Euler (685,699)

Completing the 2x2 (added 2026-09-08). The four architectures above cover only two
cells of the design the headline claim actually rests on, and they sit on two
different backbones:

                    | composition            | addition
    ----------------+------------------------+--------------------------
    stationary v(x) | voxelmorph1_diff (B)   | == voxelmorph1 (A) *
                    | v5 (C)                 |
    time-varying    |                        | v7 (D)
      v(x,t)        |                        |

    * a stationary field accumulated additively telescopes: N * (v/N) = v, so the
      cell is algebraically identical to using v directly as a displacement.

So (C) vs (D) moves along BOTH axes at once, and it does so on the frozen-VAE
backbone -- the one this same paper reports as a negative result. These two
architectures close the 2x2 on the plain U-Net instead, where (A) and (B) already
sit, so the algebra contrast can be read down a single column:

    --arch voxelmorph1_tv_comp  U-Net + time-varying velocity, composed  (278,707)
    --arch voxelmorph1_tv_sum   U-Net + time-varying velocity, added     (278,707)

With those, `tv_comp` vs `tv_sum` isolates composition-vs-addition at fixed
time-variance, and `voxelmorph1_diff` vs `tv_comp` isolates stationary-vs-
time-varying at fixed composition. The prediction under test: `tv_comp` should be
fold-free (it is LDDMM's algebra), which would show the guarantee comes from the
group operation and not from the field being stationary.

Both carry 278,707 parameters -- identical to each other, 4,320 more than the
2-channel models because the two layers that see the raw input are widened by 4
channels (current state x3, time). The extra channels are zero-initialised,
so the two time-varying arms differ from each other in exactly one line of
`VxmTimeVarying.forward`.

Loss (identical to the legacy v5 recipe, and to the official VoxelMorph recipe
apart from the added soft-Dice term):

    L = NCC(win 9) + lambda * Grad_l2(flow) + beta * (1 - softDice)

Both `reg_common.NCC/Grad` and the official `losses.NCC_vxm/Grad3d` implement the
same formulation; `reg_common`'s is used for both arms so the two arms are
bit-for-bit comparable. `reg_common.NCC` additionally clamps the local variance
at 0, a fix for a real numerical bug, so it is the safer of the two.

Direction: moving = atlas, fixed = subject. The soft-Dice target is therefore the
*subject* segmentation, which differs for every sample -- unlike the legacy
subject->atlas runs where the target was one constant atlas label map.

Model selection: best official grouped-17 Dice on the full Val 58, never on Test.
"""

from __future__ import annotations

import os
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import grouped17, dice_all_labels, load_pkl, normalize_seg, warp_seg_one_hot  # noqa: E402
from reg_common import (NCC, Grad, SpatialVAE, LatentRegUNet, LatentFlowReg,  # noqa: E402
                        SpatialTransformer, VecInt, one_hot_seg, soft_dice_loss)

REPO = Path(__file__).resolve().parents[1] / "external" / "TransMorph"
VXM_DIR = REPO / "IXI" / "Baseline_registration_methods" / "VoxelMorph"
SHAPE = (160, 192, 224)


class NCCFast:
    """`reg_common.NCC` with the 9^3 box filter applied separably.

    A uniform sum filter is the outer product of three 1-D sum filters, so this is
    the same function, not an approximation: verified identical loss (absdiff 0)
    and identical gradients (max absdiff 8e-12, i.e. float roundoff), while being
    ~16x faster (265 ms -> 16 ms per fwd+bwd at 160x192x224). Applied to both arms.
    """

    def __init__(self, win: int = 9):
        self.win = win

    def _boxsum(self, a: torch.Tensor) -> torch.Tensor:
        w, p = self.win, self.win // 2
        kernel = torch.ones(1, 1, w, 1, 1, device=a.device, dtype=a.dtype)
        a = torch.nn.functional.conv3d(a, kernel, padding=(p, 0, 0))
        a = torch.nn.functional.conv3d(a, kernel.view(1, 1, 1, w, 1), padding=(0, p, 0))
        return torch.nn.functional.conv3d(a, kernel.view(1, 1, 1, 1, w), padding=(0, 0, p))

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        Ii, Ji = y_true, y_pred
        wsz = float(self.win ** 3)
        I_sum, J_sum = self._boxsum(Ii), self._boxsum(Ji)
        I2_sum, J2_sum, IJ_sum = self._boxsum(Ii * Ii), self._boxsum(Ji * Ji), self._boxsum(Ii * Ji)
        u_I, u_J = I_sum / wsz, J_sum / wsz
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * wsz
        I_var = torch.clamp(I2_sum - 2 * u_I * I_sum + u_I * u_I * wsz, min=0.0)
        J_var = torch.clamp(J2_sum - 2 * u_J * J_sum + u_J * u_J * wsz, min=0.0)
        cc = torch.clamp(cross * cross / (I_var * J_var + 1e-5), max=1.0)
        return -torch.mean(cc)


def soft_dice_chunked(atlas_seg, subject_seg, label_values, flow, stn, chunk=6):
    """Identical to `reg_common.soft_dice_loss` on a 38-channel one-hot, but never
    materialises the full one-hot.

    A [1,38,160,192,224] float one-hot is 1.05 GB, and the step holds three of them
    (atlas, subject, warped-and-saved-for-backward). On this Windows/WDDM box the
    backward slows down monotonically with committed VRAM -- measured 756 ms at a
    small footprint versus 2700 ms with 10 GB of ballast -- so shrinking the
    footprint is worth more than shrinking the FLOPs.

    Labels are processed in chunks under `torch.utils.checkpoint`, so each chunk's
    warped volume is recomputed in backward instead of being retained. Peak drops
    from ~3 GB to ~0.3 GB for this term; the recompute costs one extra grid_sample.
    """
    from torch.utils.checkpoint import checkpoint

    inters, preds, targets = [], [], []
    for start in range(0, len(label_values), chunk):
        block = label_values[start : start + chunk]

        def run(fl, block=block):
            atlas_oh = torch.cat([(atlas_seg == v).float() for v in block], dim=1)
            subject_oh = torch.cat([(subject_seg == v).float() for v in block], dim=1)
            warped = stn(atlas_oh, fl)
            dims = (2, 3, 4)
            return (warped * subject_oh).sum(dims), warped.sum(dims), subject_oh.sum(dims)

        inter, pred, target = checkpoint(run, flow, use_reentrant=False)
        inters.append(inter)
        preds.append(pred)
        targets.append(target)

    inter = torch.cat(inters, dim=1)
    denom = torch.cat(preds, dim=1) + torch.cat(targets, dim=1)
    dice = (2 * inter + 1e-5) / (denom + 1e-5)
    return 1.0 - dice.mean()


class IXIPairs(Dataset):
    """Yields (subject image, subject seg). The atlas is a constant, held outside."""

    def __init__(self, directory: str):
        self.files = sorted(glob.glob(str(Path(directory) / "*.pkl")))
        assert self.files, f"no .pkl under {directory}"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image, seg = load_pkl(self.files[index])
        return torch.from_numpy(image)[None], torch.from_numpy(seg.astype(np.int16))[None]


def _widen_conv_input(conv, extra: int):
    """A copy of `conv` taking `extra` more input channels, appended at the end.

    The existing channels keep their weights and the new ones start at exactly
    zero, so the widened layer initially computes what the original did.
    """
    new = torch.nn.Conv3d(conv.in_channels + extra, conv.out_channels,
                          kernel_size=conv.kernel_size, stride=conv.stride,
                          padding=conv.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _widen_unet_input(unet, extra: int) -> None:
    """Let VoxelMorph's U-Net take `extra` further input channels.

    Two layers see the raw input, not one. `Unet.forward` seeds its skip list with
    the input itself (`x_enc = [x]`) and concatenates it back at full resolution
    before `extras`, which is why `extras[0]` is built with `prev_nf + 2` channels.
    Widening only the first conv leaves `extras[0]` expecting 34 channels while
    receiving 38.

    In both layers the raw input is the trailing block of the channel axis, so the
    added channels append cleanly at the end.
    """
    unet.downarm[0].main = _widen_conv_input(unet.downarm[0].main, extra)
    unet.extras[0].main = _widen_conv_input(unet.extras[0].main, extra)


class VxmTimeVarying(torch.nn.Module):
    """VoxelMorph-1's U-Net driven as a TIME-VARYING velocity field.

    At each of `n_steps` Euler steps the network sees the image pair, the current
    displacement, and the time, and predicts the velocity there. The two modes
    differ in the single line that folds that step into the running state:

        "sum"      u <- u + v*dt
                   Plain vector addition in displacement space. No group
                   structure, so nothing bounds the accumulated field. This is
                   v7's scheme with the frozen VAE swapped for the U-Net.

        "compose"  u <- u + (v*dt) o (Id + u)
                   Composition of the small map Id + v*dt onto Id + u. Same
                   algebra VecInt uses -- `vec + transformer(vec, vec)` is this
                   with the step equal to the state -- and the same algebra as
                   LDDMM. Each factor is a near-identity map, and composing
                   near-identity maps cannot fold.

    `transformer(a, b)` samples `a` at `grid + b`, i.e. a o (Id + b), which is why
    the composition line reads the way it does.
    """

    def __init__(self, base, mode: str, n_steps: int = 10, use_ckpt: bool = True):
        super().__init__()
        assert mode in ("sum", "compose"), mode
        self.unet_model = base.unet_model
        self.flow = base.flow
        self.transformer = base.transformer
        self.mode = mode
        self.n_steps = n_steps
        self.use_ckpt = use_ckpt
        # the pair is already 2 channels; add current displacement(3) + time(1)
        _widen_unet_input(self.unet_model, 4)

    def _velocity(self, moving, fixed, u, t_val):
        t = torch.full((u.shape[0], 1, 1, 1, 1), t_val, device=u.device, dtype=u.dtype)
        x = torch.cat((moving, fixed, u, t.expand(-1, -1, *SHAPE)), dim=1)
        return self.flow(self.unet_model(x))

    def forward(self, moving, fixed):
        u = torch.zeros(moving.shape[0], 3, *SHAPE,
                        device=moving.device, dtype=moving.dtype)
        dt = 1.0 / self.n_steps
        for k in range(self.n_steps):
            t_val = k * dt
            if self.use_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                v = checkpoint(lambda uu, tt=t_val: self._velocity(moving, fixed, uu, tt),
                               u, use_reentrant=False)
            else:
                v = self._velocity(moving, fixed, u, t_val)
            step = v * dt
            if self.mode == "sum":
                u = u + step
            else:
                u = u + self.transformer(step, u)
        return self.transformer(moving, u), u


def build_model(arch: str, device, n_steps: int = 10, noise_std: float = 0.05):
    if arch == "voxelmorph1":
        # the repo root has its own `models` package which would shadow VoxelMorph's
        # models.py; purge any cached binding and put the official dir first.
        for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
            del sys.modules[name]
        sys.path.insert(0, str(VXM_DIR))
        from models import VxmDense_1  # type: ignore

        model = VxmDense_1(SHAPE).to(device)
        count = sum(p.numel() for p in model.parameters())
        assert count == 274_387, f"expected the official VxmDense_1 (274,387 params), got {count:,}"

        def forward(moving, fixed):
            warped, flow = model(torch.cat((moving, fixed), dim=1))
            return warped, flow

        return model, forward

    if arch == "voxelmorph1_diff":
        # Ablation that separates the two things v5 changes at once. Keeps
        # VoxelMorph-1's U-Net and flow head EXACTLY as they are, and only swaps
        # the directly-predicted displacement for a static velocity field
        # integrated by scaling-and-squaring -- the same VecInt(7) v5 uses.
        # VecInt has no parameters, so the count stays 274,387 and integration is
        # the only variable versus `--arch voxelmorph1`.
        for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
            del sys.modules[name]
        sys.path.insert(0, str(VXM_DIR))
        from models import VxmDense_1  # type: ignore

        base = VxmDense_1(SHAPE)

        class VxmDiff(torch.nn.Module):
            def __init__(self, base, int_steps=7):
                super().__init__()
                self.unet_model = base.unet_model
                self.flow = base.flow
                self.transformer = base.transformer
                self.vecint = VecInt(SHAPE, nsteps=int_steps)

            def forward(self, x):
                source = x[:, 0:1]
                velocity = self.flow(self.unet_model(x))
                flow = self.vecint(velocity)
                return self.transformer(source, flow), flow

        model = VxmDiff(base).to(device)
        count = sum(p.numel() for p in model.parameters())
        assert count == 274_387, f"expected 274,387 params (VecInt adds none), got {count:,}"

        def forward(moving, fixed):
            warped, flow = model(torch.cat((moving, fixed), dim=1))
            return warped, flow

        return model, forward

    if arch == "v5":
        vae = SpatialVAE().to(device)
        vae_ckpt = Path(__file__).resolve().parents[1] / "outputs" / "ixi_vae" / "vae_ixi_best.pth"
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device)["model"])
        model = LatentRegUNet(vae, full_shape=SHAPE, int_steps=7).to(device)

        def forward(moving, fixed):
            warped, flow, _ = model(moving, fixed)
            return warped, flow

        return model, forward

    if arch == "v7":
        # Fourth architecture row: the frozen VAE is held fixed exactly as in v5,
        # and only the way the field is produced changes -- a TIME-VARYING velocity
        # conditioned on the current field, accumulated by N additive Euler steps,
        # instead of one stationary velocity integrated by scaling-and-squaring.
        # Euler accumulation has no group structure, so unlike VecInt it carries no
        # diffeomorphism guarantee; contrasting (C) with (D) isolates the FORM of
        # the integration while the encoder stays constant.
        vae = SpatialVAE().to(device)
        vae_ckpt = Path(__file__).resolve().parents[1] / "outputs" / "ixi_vae" / "vae_ixi_best.pth"
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device)["model"])
        model = LatentFlowReg(vae, full_shape=SHAPE, n_steps=n_steps, use_ckpt=True).to(device)

        def forward(moving, fixed):
            # noise only while training; inference is the deterministic point estimate
            ns = noise_std if model.training else 0.0
            warped, flow, _ = model(moving, fixed, noise_std=ns)
            return warped, flow

        return model, forward

    if arch in ("voxelmorph1_tv_comp", "voxelmorph1_tv_sum"):
        # The two missing cells of the 2x2, both on the plain U-Net so they line up
        # with `voxelmorph1` and `voxelmorph1_diff`. They share this branch and
        # differ only in `mode`, so composition-vs-addition is the sole variable.
        for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
            del sys.modules[name]
        sys.path.insert(0, str(VXM_DIR))
        from models import VxmDense_1  # type: ignore

        mode = "compose" if arch.endswith("_comp") else "sum"
        model = VxmTimeVarying(VxmDense_1(SHAPE), mode=mode, n_steps=n_steps).to(device)
        count = sum(p.numel() for p in model.parameters())
        # 274,387 + 4,320. Four channels are added to both layers that see the raw
        # input: downarm[0] (8*4*27 = 864) and extras[0] (32*4*27 = 3,456).
        assert count == 278_707, f"expected 278,707 params, got {count:,}"

        def forward(moving, fixed):
            return model(moving, fixed)

        return model, forward

    raise ValueError(arch)


@torch.no_grad()
def evaluate_val(forward, atlas_image, atlas_seg_t, val_files, device, limit=0):
    """Official grouped-17 Dice, atlas -> subject, on the Val split."""
    files = val_files[:limit] if limit else val_files
    scores = []
    for path in files:
        subject_image, subject_seg = load_pkl(path)
        subject_seg = normalize_seg(subject_seg)
        fixed = torch.from_numpy(subject_image)[None, None].to(device)
        _, flow = forward(atlas_image, fixed)
        pred = warp_seg_one_hot(atlas_seg_t, flow.float()).cpu().numpy()[0, 0]
        scores.append(grouped17(dice_all_labels(pred, subject_seg)).mean())
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True,
                        choices=("voxelmorph1", "voxelmorph1_diff", "v5", "v7",
                                 "voxelmorph1_tv_comp", "voxelmorph1_tv_sum"))
    parser.add_argument("--data-root", default=os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lam", type=float, default=1.0, help="smoothness weight")
    parser.add_argument("--beta", type=float, default=1.0, help="soft-Dice weight (0 = unsupervised control)")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--limit-train", type=int, default=0, help="smoke test")
    parser.add_argument("--limit-val", type=int, default=0, help="smoke test")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="print per-phase timings")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--dice-chunk", type=int, default=6, help="labels per checkpointed soft-Dice chunk")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<tag>/last.pth if present")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="v7 / voxelmorph1_tv_*: Euler integration steps")
    parser.add_argument("--noise-std", type=float, default=0.05, help="v7 only: training noise on the initial field")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    here = Path(__file__).parent
    tag = args.tag or f"{args.arch}_a2s_l{args.lam}_b{args.beta}"
    save_dir = here / "runs" / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    atlas_image_np, atlas_seg_raw = load_pkl(root / "atlas.pkl")
    atlas_image = torch.from_numpy(atlas_image_np)[None, None].to(device)
    atlas_seg_t = torch.from_numpy(normalize_seg(atlas_seg_raw).astype(np.int16))[None, None].to(device)

    # soft-Dice label set: the raw FreeSurfer labels present in the atlas, exactly
    # as the legacy v5 run used. Now it is the ATLAS one-hot that gets warped.
    labels = [int(v) for v in np.unique(atlas_seg_raw.astype(np.int32)) if v != 0]
    atlas_seg_raw_t = torch.from_numpy(atlas_seg_raw.astype(np.int16))[None, None].to(device)
    stn = SpatialTransformer(SHAPE).to(device) if args.beta > 0 else None

    model, forward = build_model(args.arch, device, n_steps=args.n_steps, noise_std=args.noise_std)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"arch={args.arch}  trainable params={trainable:,}")
    print(f"direction=atlas->subject  lambda={args.lam}  beta={args.beta}  epochs={args.epochs}")
    print(f"soft-Dice labels: {len(labels)}")
    print(f"save_dir={save_dir}")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    ncc, grad = NCCFast(9), Grad("l2")

    train_set = IXIPairs(root / "Train")
    if args.limit_train:
        train_set.files = train_set.files[: args.limit_train]
    loader_kwargs = dict(batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(train_set, **loader_kwargs)
    val_files = sorted(glob.glob(str(root / "Val" / "*.pkl")))
    print(f"train n={len(train_set)}  val n={len(val_files)}")

    best_dice, history, start_epoch = -1.0, [], 1

    # Crash recovery. `best.pth` stores weights only (that is the artefact the
    # evaluator consumes); `last.pth` additionally stores optimizer moments and
    # RNG state, so a resumed run continues the *same* trajectory rather than
    # restarting Adam from zero. Without the optimizer and RNG state a resume
    # would silently differ from the uninterrupted arms and break the comparison.
    last_path = save_dir / "last.pth"
    if args.resume and last_path.exists():
        # map_location="cpu": the RNG states are ByteTensors that torch.set_rng_state
        # requires on the CPU, and load_state_dict moves weights/moments to the
        # model's device anyway.
        snapshot = torch.load(last_path, map_location="cpu", weights_only=False)
        if snapshot["arch"] != args.arch or snapshot["beta"] != args.beta or snapshot["epochs"] != args.epochs:
            raise SystemExit(f"{last_path} was written by a different configuration -- refusing to resume")
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        torch.set_rng_state(snapshot["cpu_rng"])
        torch.cuda.set_rng_state(snapshot["cuda_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        best_dice = snapshot["best_dice"]
        history = snapshot["history"]
        start_epoch = snapshot["epoch"] + 1
        print(f"resumed from {last_path}: continuing at epoch {start_epoch}, best so far {best_dice:.6f}")
    elif args.resume:
        print(f"--resume given but {last_path} does not exist; starting from scratch")

    for epoch in range(start_epoch, args.epochs + 1):
        # Official poly LR decay.
        lr_now = round(args.lr * np.power(1 - (epoch - 1) / args.epochs, 0.9), 8)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        model.train()
        start = time.time()
        totals = np.zeros(4)
        phases = np.zeros(5)
        mark = time.perf_counter()
        for subject_image, subject_seg in loader:
            if args.profile:
                torch.cuda.synchronize(); phases[0] += time.perf_counter() - mark; mark = time.perf_counter()
            fixed = subject_image.to(device, non_blocking=True)
            warped, flow = forward(atlas_image, fixed)
            similarity = ncc.loss(fixed, warped)
            smooth = grad.loss(flow)
            loss = similarity + args.lam * smooth
            if args.profile:
                torch.cuda.synchronize(); phases[1] += time.perf_counter() - mark; mark = time.perf_counter()
            dice_term = torch.zeros((), device=device)
            if args.beta > 0:
                dice_term = soft_dice_chunked(atlas_seg_raw_t, subject_seg.to(device),
                                              labels, flow, stn, chunk=args.dice_chunk)
                loss = loss + args.beta * dice_term
            if args.profile:
                torch.cuda.synchronize(); phases[2] += time.perf_counter() - mark; mark = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if args.profile:
                torch.cuda.synchronize(); phases[3] += time.perf_counter() - mark; mark = time.perf_counter()
            totals += [loss.item(), similarity.item(), smooth.item(), float(dice_term)]
            if args.profile:
                torch.cuda.synchronize(); phases[4] += time.perf_counter() - mark; mark = time.perf_counter()
        totals /= len(loader)
        elapsed = time.time() - start
        line = (f"[ep {epoch:03d}] lr={lr_now:.2e} loss={totals[0]:.4f} NCC={-totals[1]:.4f} "
                f"smooth={totals[2]:.4f} softDiceLoss={totals[3]:.4f} ({elapsed:.0f}s)")
        if args.profile:
            per = phases / len(loader) * 1000
            line += (f"  |  dataload={per[0]:.0f} fwd+ncc={per[1]:.0f} softdice={per[2]:.0f} "
                     f"bwd+step={per[3]:.0f} item()={per[4]:.0f} ms/subj")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            val_dice = evaluate_val(forward, atlas_image, atlas_seg_t, val_files, device, args.limit_val)
            line += f"  valDice={val_dice:.4f}"
            history.append({"epoch": epoch, "val_dice": val_dice, "train_loss": totals[0]})
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice,
                     "arch": args.arch, "lam": args.lam, "beta": args.beta, "int_steps": 7,
                     "n_steps": args.n_steps, "noise_std": args.noise_std},
                    save_dir / "best.pth",
                )
                line += "  * saved best"

            # Resume point. Written atomically so a power loss mid-write cannot
            # leave a truncated file that would poison the next resume.
            tmp_path = save_dir / "last.pth.tmp"
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_dice": best_dice, "history": history,
                 "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "numpy_rng": np.random.get_state(),
                 "arch": args.arch, "lam": args.lam, "beta": args.beta, "epochs": args.epochs},
                tmp_path,
            )
            tmp_path.replace(save_dir / "last.pth")
        print(line, flush=True)

    print(f"\nbest Val grouped-17 Dice = {best_dice:.6f}")
    np.save(save_dir / "history.npy", np.array(history, dtype=object), allow_pickle=True)
    print(f"checkpoint: {save_dir / 'best.pth'}")


if __name__ == "__main__":
    main()

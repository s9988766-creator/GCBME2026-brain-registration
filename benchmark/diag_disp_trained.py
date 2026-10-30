import os
"""Deformation amplitude for our trained runs -- low folding only counts if the field moves."""
import argparse, glob
from pathlib import Path
import numpy as np, torch
from common import jacobian_stats, load_pkl
from train_softdice import build_model

p = argparse.ArgumentParser(); p.add_argument("--limit", type=int, default=8)
p.add_argument("--split", default="Val"); a = p.parse_args()
here = Path(__file__).parent; dev = torch.device("cuda")
root = Path(os.environ.get("IXI_DATA_ROOT", "data/IXI_data"))
atlas, _ = load_pkl(root / "atlas.pkl"); x = torch.from_numpy(atlas)[None, None].to(dev)
paths = sorted(glob.glob(str(root / a.split / "*.pkl")))[: a.limit]
print(f"{'run':<26}{'|disp| mean':>13}{'p99':>8}{'max':>8}{'Jneg %':>12}{'logJ SD':>9}")
for run in ["vxm1_a2s_unsup", "vxm1_a2s_softdice", "vxmdiff_a2s_unsup", "vxmdiff_a2s_softdice", "v5_a2s_unsup", "v5_a2s_softdice"]:
    ck = torch.load(here / "runs" / run / "best.pth", map_location="cpu", weights_only=False)
    model, fwd = build_model(ck["arch"], dev); model.load_state_dict(ck["model"]); model.eval()
    st = []
    with torch.no_grad():
        for path in paths:
            img, _ = load_pkl(path); y = torch.from_numpy(img)[None, None].to(dev)
            _, flow = fwd(x, y); mag = flow.norm(dim=1)
            _, jneg, logj = jacobian_stats(flow.detach().cpu().numpy()[0])
            st.append((float(mag.mean()), float(mag.flatten().quantile(0.99)), float(mag.max()), jneg, logj))
    m = np.asarray(st).mean(0)
    print(f"{run:<26}{m[0]:>13.2f}{m[1]:>8.2f}{m[2]:>8.2f}{m[3]:>12.6f}{m[4]:>9.3f}")
    del model; torch.cuda.empty_cache()
print(f"\n(n={len(paths)} {a.split} subjects, displacement in voxels)")

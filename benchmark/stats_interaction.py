"""The 2x2 interaction test the "orthogonal factors" claim actually needs.

Reporting that displacement beats velocity by 0.005 under one training scheme and
by 0.002 under the other is not a test of whether the two factors interact. The
quantity that is, is the difference of differences, computed per subject so the
pairing survives:

    interaction = (reflow - sampler)_velocity  -  (reflow - sampler)_displacement

A confidence interval that spans zero means the training effect is statistically
indistinguishable between the two spaces -- which is what "separable main effects"
would require. A CI clear of zero means the factors do interact and the paper must
say so.

Folding is handled on the log scale: it spans three orders of magnitude and is
bounded below by zero, so a difference of raw percentages is dominated entirely by
the displacement arms and says nothing about the velocity ones.

CIs are bootstrap over subjects (BCa is overkill here; the percentile interval on
20000 resamples is stable to the third decimal).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

HERE = Path(__file__).parent
RES = HERE / "results"
B = 20000
SEED = 0

FILES = {
    "vel+reflow":   "flowsvf_unet_std_persubject_euler_test.csv",
    "vel+sampler":  "flowsvf_unet_persubject_euler_test.csv",
    "disp+reflow":  "flowdisp_unet_std_persubject_euler_test.csv",
    "disp+sampler": "flowdisp_unet_persubject_euler_test.csv",
}
# Jneg is the PERCENTAGE of voxels with negative Jacobian, so over a
# 160x192x224 volume the smallest resolvable non-zero value is one voxel:
# 100 / 6881280 = 1.4532e-5 %. "Zero folding" therefore means "below the
# one-voxel detection limit", and that limit -- not an arbitrary epsilon -- is
# the right floor before taking a log. An earlier version used 1e-7, which
# inflated the folding interaction from +0.42 to +0.59 decades purely through
# the 10 zero-folding subjects in one arm.
LOG_FLOOR = 100.0 / (160 * 192 * 224)


def load(fname, steps=2):
    rows = [r for r in csv.DictReader(open(RES / fname)) if int(r["steps"]) == steps]
    rows.sort(key=lambda r: r["subject"])
    return ([r["subject"] for r in rows],
            np.array([float(r["dice"]) for r in rows]),
            np.array([float(r["jneg_percent"]) for r in rows]))


def boot_ci(x, rng, b=B):
    n = len(x)
    idx = rng.integers(0, n, size=(b, n))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def report(name, x, rng, unit=""):
    lo, hi = boot_ci(x, rng)
    t, pt = ttest_rel(x, np.zeros_like(x))
    try:
        _, pw = wilcoxon(x)
    except ValueError:
        pw = float("nan")
    dz = x.mean() / x.std(ddof=1)
    crosses = "spans zero" if lo <= 0 <= hi else "excludes zero"
    print(f"  {name:<46} {x.mean():>+10.5f}{unit}  "
          f"95% CI [{lo:>+9.5f}, {hi:>+9.5f}]  dz {dz:>+6.2f}  "
          f"p_t {pt:.1e}  p_w {pw:.1e}   {crosses}")


def main():
    rng = np.random.default_rng(SEED)
    data = {}
    subj_ref = None
    for k, f in FILES.items():
        s, d, j = load(f)
        if subj_ref is None:
            subj_ref = s
        assert s == subj_ref, f"subject lists differ for {k}"
        data[k] = (d, np.maximum(j, LOG_FLOOR))
    n = len(subj_ref)
    print(f"n = {n} test subjects, paired throughout\n")

    # ---- Dice -------------------------------------------------------------
    tr_vel = data["vel+reflow"][0] - data["vel+sampler"][0]
    tr_disp = data["disp+reflow"][0] - data["disp+sampler"][0]
    print("DICE")
    report("training effect within velocity", tr_vel, rng)
    report("training effect within displacement", tr_disp, rng)
    report("INTERACTION (training effect: velocity - displacement)", tr_vel - tr_disp, rng)
    print()

    sp_reflow = data["disp+reflow"][0] - data["vel+reflow"][0]
    sp_sampler = data["disp+sampler"][0] - data["vel+sampler"][0]
    report("space effect within reflow (disp - vel)", sp_reflow, rng)
    report("space effect within sampler (disp - vel)", sp_sampler, rng)
    print()

    # ---- folding, log10 scale --------------------------------------------
    lg = {k: np.log10(v[1]) for k, v in data.items()}
    print("FOLDING (log10 of Jneg %)")
    sp_reflow_j = lg["vel+reflow"] - lg["disp+reflow"]
    sp_sampler_j = lg["vel+sampler"] - lg["disp+sampler"]
    report("space effect within reflow (vel - disp), decades", sp_reflow_j, rng)
    report("space effect within sampler (vel - disp), decades", sp_sampler_j, rng)
    report("INTERACTION (space effect: reflow - sampler)", sp_reflow_j - sp_sampler_j, rng)
    print()
    tr_vel_j = lg["vel+reflow"] - lg["vel+sampler"]
    tr_disp_j = lg["disp+reflow"] - lg["disp+sampler"]
    report("training effect within velocity, decades", tr_vel_j, rng)
    report("training effect within displacement, decades", tr_disp_j, rng)
    print()

    print("Read: a training effect on Dice that is the same size in both spaces, and a")
    print("space effect on folding that is the same size under both training schemes,")
    print("is what a separable-main-effects claim needs. Any interaction CI that")
    print("excludes zero must be reported rather than described as orthogonality.")


if __name__ == "__main__":
    main()

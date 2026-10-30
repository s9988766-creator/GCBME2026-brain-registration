# -*- coding: utf-8 -*-
"""Re-derive Tables 2 and 3 of the GCBME ablation paper from the eval logs and
the per-subject CSVs, and check them against what the .docx actually prints.

Written after a review found the latency column mixing two measurement methods:
(A)(B)(C) came from bench_speed.py (warmup + repeats on an idle GPU) while (D)
came from the eval logs, which time a whole evaluation pass. That gave one
architecture two different latencies depending on supervision, which cannot be
right. Run this after touching any number in the paper.

    python verify_ablation_tables.py [path\\to\\paper.docx]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).parent
DEFAULT_DOC = HERE.parent / "[醫工年會] 論文v2.1_ablation.docx"

# paper row label -> run tag. The grid is 4 architectures x 2 supervision arms.
GRID = {
    ("(A) Direct displacement", "none"): "vxm1_a2s_unsup",
    ("(A) Direct displacement", "soft-Dice"): "vxm1_a2s_softdice",
    ("(B) + scaling-and-squaring", "none"): "vxmdiff_a2s_unsup",
    ("(B) + scaling-and-squaring", "soft-Dice"): "vxmdiff_a2s_softdice",
    ("(C) + frozen VAE encoder", "none"): "v5_a2s_unsup",
    ("(C) + frozen VAE encoder", "soft-Dice"): "v5_a2s_softdice",
    ("(D) frozen VAE, Euler sum", "none"): "v7_a2s_unsup",
    ("(D) frozen VAE, Euler sum", "soft-Dice"): "v7_a2s_softdice",
}

# one latency per architecture, all from the same warmed-up benchmark
LATENCY_LOG = HERE / "logs_bench_speed_4arch.txt"
BENCH_LABEL = {
    "(A) Direct displacement": "VoxelMorph-1 +softDice",
    "(B) + scaling-and-squaring": "VxM+VecInt +softDice",
    "(C) + frozen VAE encoder": "v5 (frozen VAE) +softDice",
    "(D) frozen VAE, Euler sum": "v7 (frozen VAE, Euler) +softDice",
}

CSV = HERE / "results"


def from_csv(tag):
    """Authoritative per-subject numbers. SD is the SAMPLE SD (ddof=1).

    The paper originally mixed conventions: six rows carried ddof=1 and the two
    (D) rows ddof=0, differing in the last printed digit. One convention, and
    it is the one the other six rows already used.
    """
    import pandas as pd

    d = pd.read_csv(CSV / f"{tag}_test.csv")
    dice = d["dice_official_grouped17"].values
    return {
        "Dice": (float(dice.mean()), float(dice.std(ddof=1))),
        "Jneg": (float(d["jneg_percent"].mean()),),
        "logJ": (float(d["logj_std"].mean()),),
    }


def bench_latencies():
    text = LATENCY_LOG.read_text(encoding="utf-8", errors="replace")
    out = {}
    for arch, label in BENCH_LABEL.items():
        m = re.search(re.escape(label) + r"\s+([\d.]+)\s", text)
        if not m:
            raise SystemExit(f"{label}: not in {LATENCY_LOG.name}")
        out[arch] = float(m.group(1))
    return out


def doc_table2(path):
    import docx
    d = docx.Document(str(path))
    rows = {}
    for t in d.tables:
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) == 7 and cells[1] in ("none", "soft-Dice"):
                rows[(cells[0], cells[1])] = cells
    return rows


def approx(printed, value, dp=None):
    if printed in ("-", "–", ""):
        return True
    dp = dp if dp is not None else (len(printed.split(".")[1])
                                    if "." in printed else 0)
    return f"{value:.{dp}f}" == printed


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC
    print(f"checking {path.name}\n")
    rows = doc_table2(path)
    lat = bench_latencies()
    bad = 0

    print("Table 2, every cell against the per-subject CSVs")
    for key, tag in GRID.items():
        if key not in rows:
            print(f"  MISSING row {key}")
            bad += 1
            continue
        cells = rows[key]
        log = from_csv(tag)
        printed_dice = cells[2].replace("±", "|").split("|")
        checks = [
            ("Dice", approx(printed_dice[0].strip(), log["Dice"][0])),
            ("SD", approx(printed_dice[1].strip(), log["Dice"][1])),
            ("Jneg", approx(cells[3], log["Jneg"][0])),
            ("logJ", approx(cells[4], log["logJ"][0])),
            ("latency", approx(cells[6], lat[key[0]])),
        ]
        wrong = [n for n, ok in checks if not ok]
        if wrong:
            bad += 1
            print(f"  WRONG {key[0]:28s} {key[1]:10s} -> {wrong}")
            print(f"        printed {cells[2]} | {cells[3]} | {cells[4]} | {cells[6]}")
            print(f"        logs    {log['Dice'][0]:.6f} +/- {log['Dice'][1]:.6f} | "
                  f"{log['Jneg'][0]:.6f} | {log['logJ'][0]:.6f} | {lat[key[0]]}")
        else:
            print(f"  ok    {key[0]:28s} {key[1]:10s}")

    print("\none latency per architecture (supervision cannot change inference)")
    for arch in BENCH_LABEL:
        vals = {rows[k][6] for k in rows if k[0] == arch}
        ok = len(vals) == 1
        bad += 0 if ok else 1
        print(f"  {'ok   ' if ok else 'SPLIT'} {arch:28s} {sorted(vals)}")

    print("\ncontrasts smaller than the seed spread (0.0022) must not be "
          "stated as measured effects")
    for a, b, label in [
        (("(B) + scaling-and-squaring", "soft-Dice"),
         ("(A) Direct displacement", "soft-Dice"), "integration, supervised"),
        (("(D) frozen VAE, Euler sum", "soft-Dice"),
         ("(C) + frozen VAE encoder", "soft-Dice"), "algebra, supervised"),
        (("(D) frozen VAE, Euler sum", "none"),
         ("(C) + frozen VAE encoder", "none"), "algebra, unsupervised"),
    ]:
        d = from_csv(GRID[a])["Dice"][0] - from_csv(GRID[b])["Dice"][0]
        print(f"  {label:26s} {d:+.4f}  "
              f"{'BELOW seed noise -- report as no detectable difference' if abs(d) < 0.0022 else 'above seed noise'}")

    print("\n" + ("TABLES VERIFIED" if not bad else f"{bad} PROBLEM(S)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

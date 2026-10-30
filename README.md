# A Matched-Protocol Ablation of Deep Brain MRI Registration

Code and per-subject results for **"A Matched-Protocol Ablation of Deep Brain MRI Registration: Supervision, Step Composition, and Deformation Regularity"**
Bo-Wei Lai, Ming-Chang Chiang, Po-Kai Yang. *2026 Annual Conference of the Taiwanese Society of Biomedical Engineering (GCBME).*

Ten models are trained under one recipe on the official TransMorph IXI atlas-to-subject protocol
(four architectures x two supervision levels, plus two U-Net models that differ only in whether each velocity step is composed or added) and scored on 115 held-out subjects.

| Finding | Number (paper) |
|---|---|
| Supervision raises VoxelMorph-1 Dice | +0.085, 115/115 subjects |
| Scaling-and-squaring vs direct displacement (weak supervision) | folding 1.080% -> 0.000004%, Dice -0.0015 |
| Compose vs add steps, same backbone (E vs F) | folding 0.0221% vs 1.078%, **48.8x** |
| Frozen VAE encoder (C) vs U-Net (B) | 20.0 vs 61.2 ms, but Dice -0.0132 |

## Layout

```
reg_common.py              models, losses, scaling-and-squaring, Jacobian
outputs/ixi_vae/           frozen spatial-VAE checkpoint (4 MB) used by models C/D
benchmark/
  common.py                locked protocol: 46 classes, grouped-17 Dice, trilinear warp, Jneg
  train_softdice.py        train any of the architectures (--arch, --beta, --lam, --seed)
  eval_trained.py          evaluate a trained run on the 115 test subjects
  eval_affine.py / eval_official.py / validate_protocol.py / compare_to_official.py
                           reproduce the published TransMorph numbers (Table 1)
  stats_2x2.py, stats_interaction.py   paired statistics (Table 3)
  verify_ablation_tables.py            re-derive Tables 2-3 from logs/CSVs and check a manuscript .docx
  make_figure.py           Figure 1 (accuracy vs folding)
  run_*.bat                the exact training queues used
  results/                 per-subject test CSVs for every run
  logs/                    training / evaluation logs
paper_figures/             figures as submitted
```

Model letters in the paper -> `--arch`: A `voxelmorph1`, B `voxelmorph1_diff`, E `voxelmorph1_tv_comp`, F `voxelmorph1_tv_sum`; C and D use the frozen VAE (see `train_softdice.py`).

## Reproducing

Statistics and figures need **no GPU and no data**, only the CSVs in `benchmark/results/`:

```
pip install -r requirements.txt
cd benchmark
python stats_2x2.py
python make_figure.py
```

Retraining needs the IXI data and the official TransMorph repo:

```
git clone https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration external/TransMorph
git -C external/TransMorph checkout 6357a1d     # commit the results were produced with
# TransMorph's preprocessed IXI release -> data/IXI_data   (or set IXI_DATA_ROOT)
cd benchmark
python train_softdice.py --arch voxelmorph1_diff --beta 1.0 --lam 1.0 --epochs 100 --seed 0 --tag my_run
```

Trained checkpoints (~30 GB) are not in the repo. IXI is CC BY-SA 3.0; do not commit the data.

## Notes

* Evaluation follows the official direction (atlas = moving) and grouped-17 Dice; `validate_protocol.py` reproduces all eleven published numbers.
* Weakly supervised models are trained on the structures they are scored on; their Dice is not comparable to unsupervised methods.
* Each configuration is a single training run; see the Limitations section of the paper.

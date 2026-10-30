@echo off
REM Close the 2x2 on the plain U-Net backbone.
REM
REM Why these two runs exist. Paper A's headline claim -- that topology is decided
REM by the algebra of the integration -- currently rests on (C) vs (D), which moves
REM along BOTH axes at once (stationary -> time-varying AND composition -> addition)
REM and does so on the frozen-VAE backbone that the same paper reports as a negative
REM result. These two runs put the missing cells on the plain U-Net, where (A) and
REM (B) already sit, so the algebra contrast can be read down a single column with
REM one variable moving:
REM
REM                     | composition                | addition
REM     ----------------+----------------------------+---------------------------
REM     stationary v(x) | voxelmorph1_diff  (B) done | == voxelmorph1 (A) done
REM     time-varying    | voxelmorph1_tv_comp  NEW   | voxelmorph1_tv_sum   NEW
REM
REM The prediction under test: tv_comp is fold-free, because composition is LDDMM's
REM algebra. If it holds, "the algebra, not the field type" stops being an assertion
REM backed by a citation and becomes a measurement.
REM
REM n-steps 5, both arms. Two reasons, neither of them cost:
REM   1. HANDOVER.md section 4 already measured N=5 vs N=10 as statistically
REM      indistinguishable (p = 0.43) on this pipeline.
REM   2. Sequential composition accumulates interpolation error -- measured 1e-6 at
REM      2 steps, 10% at 128. Five compositions is the same order as VecInt's seven,
REM      so tv_comp is not handicapped against the arm it is compared with. Ten
REM      steps would double its resampling error and dirty the contrast.
REM
REM Checkpointing stays on. Measured on this box: n-steps 5 costs 1.17 s/step at
REM 6.5 GB with it, and 26.50 s/step at 21.9 GB without -- the larger footprint
REM spills past the 4090's 24 GB and collapses backward throughput, the same effect
REM already documented in soft_dice_chunked's docstring.
REM
REM Everything else is the locked shared recipe: lambda 1, beta 1, 100 epochs,
REM Adam 1e-4 poly decay, batch 1, Val-58 grouped-17 model selection, seed 0 --
REM identical to the four architectures already in the paper. One seed per cell,
REM matching the paper's existing design; the effect being claimed is folding at
REM five orders of magnitude, which is nowhere near seed noise. Do NOT claim a Dice
REM difference from a single seed.
REM
REM About 17 h per run, ~34 h for both. Launch detached:
REM
REM   Start-Process -FilePath "cmd.exe" `
REM     -ArgumentList "/c","benchmark\run_queue_2x2.bat" `
REM     -WorkingDirectory "benchmark" -WindowStyle Hidden

cd /d %~dp0
if "%PY%"=="" set PY=python
set Q=logs_queue_2x2.txt

echo [%DATE% %TIME%] START 2x2 completion, two runs, ~34 h > %Q%

echo [%DATE% %TIME%] --- 1/2  voxelmorph1_tv_comp  (time-varying + composition) >> %Q%
%PY% train_softdice.py --arch voxelmorph1_tv_comp --n-steps 5 ^
    --lam 1.0 --beta 1.0 --epochs 100 --seed 0 --tag vxm_tv_comp_s0 >> %Q% 2>&1
echo [%DATE% %TIME%] --- 1/2 done, exit %ERRORLEVEL% >> %Q%

echo [%DATE% %TIME%] --- 2/2  voxelmorph1_tv_sum   (time-varying + addition) >> %Q%
%PY% train_softdice.py --arch voxelmorph1_tv_sum --n-steps 5 ^
    --lam 1.0 --beta 1.0 --epochs 100 --seed 0 --tag vxm_tv_sum_s0 >> %Q% 2>&1
echo [%DATE% %TIME%] --- 2/2 done, exit %ERRORLEVEL% >> %Q%

echo [%DATE% %TIME%] ALL DONE. Next: evaluate both on Test 115 under the locked >> %Q%
echo                 protocol and read the algebra contrast down the time-varying row. >> %Q%

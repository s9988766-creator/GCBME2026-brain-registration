@echo off
REM Diffeomorphic VoxelMorph: official VoxelMorph-1 U-Net + VecInt(7), no VAE.
REM Separates the TWO things v5 changes at once -- frozen VAE features vs
REM scaling-and-squaring integration. Params stay 274,387, so integration is the
REM only variable against `--arch voxelmorph1`.
REM
REM Completes a 3x2:  {voxelmorph1, voxelmorph1_diff, v5} x {beta=0, beta=1}
REM
REM Detached launch (Start-Process) -- must NOT be a child of the VS Code /
REM Claude Code process tree, which lives in a kill-on-close Job Object.
REM last.pth is written every --val-every epochs (model + optimizer + RNG) so a
REM power loss costs at most 5 epochs; recover with --resume. NOTE: a resumed run
REM is a different random realization (~1e-2 on smoke config) not a bit-identical
REM continuation, so paper runs should be uninterrupted where possible.
REM Progress: logs_train_vxmdiff_b1.txt / logs_train_vxmdiff_b0.txt
REM Queue state: logs_train_queue_vxmdiff.txt

cd /d %~dp0
if "%PY%"=="" set PY=python

echo [%DATE% %TIME%] QUEUE START > logs_train_queue_vxmdiff.txt

echo [%DATE% %TIME%] starting voxelmorph1_diff beta=1.0 >> logs_train_queue_vxmdiff.txt
"%PY%" -u train_softdice.py --arch voxelmorph1_diff --beta 1.0 --lam 1.0 --epochs 100 --val-every 5 --tag vxmdiff_a2s_softdice > logs_train_vxmdiff_b1.txt 2>&1
echo [%DATE% %TIME%] voxelmorph1_diff beta=1.0 finished exit=%ERRORLEVEL% >> logs_train_queue_vxmdiff.txt

echo [%DATE% %TIME%] starting voxelmorph1_diff beta=0.0 >> logs_train_queue_vxmdiff.txt
"%PY%" -u train_softdice.py --arch voxelmorph1_diff --beta 0.0 --lam 1.0 --epochs 100 --val-every 5 --tag vxmdiff_a2s_unsup > logs_train_vxmdiff_b0.txt 2>&1
echo [%DATE% %TIME%] voxelmorph1_diff beta=0.0 finished exit=%ERRORLEVEL% >> logs_train_queue_vxmdiff.txt

echo [%DATE% %TIME%] QUEUE DONE >> logs_train_queue_vxmdiff.txt

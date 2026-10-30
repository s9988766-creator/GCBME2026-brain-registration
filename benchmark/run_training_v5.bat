@echo off
REM v5 (frozen spatial VAE + LatentRegUNet) retrained atlas->subject.
REM Identical recipe to the VoxelMorph arms, so `--arch` is the only variable.
REM Completes the 2x2:  {voxelmorph1, v5} x {beta=0 unsupervised, beta=1 soft-Dice}
REM
REM Detached launch (Start-Process) -- must NOT be a child of the VS Code /
REM Claude Code process tree, which lives in a kill-on-close Job Object.
REM Progress: logs_train_v5_b1.txt / logs_train_v5_b0.txt
REM Queue state: logs_train_queue_v5.txt

cd /d %~dp0
if "%PY%"=="" set PY=python

echo [%DATE% %TIME%] QUEUE START > logs_train_queue_v5.txt

echo [%DATE% %TIME%] starting v5 beta=1.0 >> logs_train_queue_v5.txt
"%PY%" -u train_softdice.py --arch v5 --beta 1.0 --lam 1.0 --epochs 100 --val-every 5 --tag v5_a2s_softdice > logs_train_v5_b1.txt 2>&1
echo [%DATE% %TIME%] v5 beta=1.0 finished exit=%ERRORLEVEL% >> logs_train_queue_v5.txt

echo [%DATE% %TIME%] starting v5 beta=0.0 >> logs_train_queue_v5.txt
"%PY%" -u train_softdice.py --arch v5 --beta 0.0 --lam 1.0 --epochs 100 --val-every 5 --tag v5_a2s_unsup > logs_train_v5_b0.txt 2>&1
echo [%DATE% %TIME%] v5 beta=0.0 finished exit=%ERRORLEVEL% >> logs_train_queue_v5.txt

echo [%DATE% %TIME%] QUEUE DONE >> logs_train_queue_v5.txt

@echo off
REM v7: frozen VAE + TIME-VARYING velocity accumulated by N=10 additive Euler
REM steps. Completes the 4x2 grid as architecture row (D).
REM
REM The contrast that matters is (C) v5 vs (D) v7: the frozen VAE encoder is
REM identical in both, so the only difference is the FORM of the integration --
REM one stationary velocity through scaling-and-squaring (a group operation, so
REM diffeomorphic by construction) versus additive Euler accumulation (no group
REM structure, hence no guarantee).
REM
REM lambda is held at 1.0 like every other cell. The legacy v7 used lambda=5
REM because it folds at lambda=1; running it at its own tuned lambda would make
REM architecture no longer the only variable, so the matched-lambda number is the
REM controlled one. The legacy lambda sweep is reported separately as the
REM accuracy/folding trade-off curve.
REM
REM Detached launch (Start-Process) -- must NOT be a child of the VS Code /
REM Claude Code process tree, which lives in a kill-on-close Job Object.
REM Progress: logs_train_v7_b1.txt / logs_train_v7_b0.txt
REM Queue state: logs_train_queue_v7.txt

cd /d %~dp0
if "%PY%"=="" set PY=python

echo [%DATE% %TIME%] QUEUE START > logs_train_queue_v7.txt

echo [%DATE% %TIME%] starting v7 beta=1.0 N=10 >> logs_train_queue_v7.txt
"%PY%" -u train_softdice.py --arch v7 --n-steps 10 --beta 1.0 --lam 1.0 --epochs 100 --val-every 5 --tag v7_a2s_softdice > logs_train_v7_b1.txt 2>&1
echo [%DATE% %TIME%] v7 beta=1.0 finished exit=%ERRORLEVEL% >> logs_train_queue_v7.txt

echo [%DATE% %TIME%] starting v7 beta=0.0 N=10 >> logs_train_queue_v7.txt
"%PY%" -u train_softdice.py --arch v7 --n-steps 10 --beta 0.0 --lam 1.0 --epochs 100 --val-every 5 --tag v7_a2s_unsup > logs_train_v7_b0.txt 2>&1
echo [%DATE% %TIME%] v7 beta=0.0 finished exit=%ERRORLEVEL% >> logs_train_queue_v7.txt

echo [%DATE% %TIME%] QUEUE DONE >> logs_train_queue_v7.txt

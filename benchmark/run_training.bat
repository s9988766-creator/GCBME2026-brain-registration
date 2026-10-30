@echo off
REM Detached training queue for the atlas->subject fair benchmark.
REM Launched via Start-Process so it is NOT a child of the VS Code / Claude Code
REM process tree; closing the editor or dropping the SSH session must not kill it.
REM Progress: logs_train_vxm1_b1.txt / logs_train_vxm1_b0.txt
REM Queue state: logs_train_queue.txt

cd /d %~dp0
if "%PY%"=="" set PY=python

echo [%DATE% %TIME%] QUEUE START > logs_train_queue.txt

echo [%DATE% %TIME%] starting voxelmorph1 beta=1.0 >> logs_train_queue.txt
"%PY%" -u train_softdice.py --arch voxelmorph1 --beta 1.0 --lam 1.0 --epochs 100 --val-every 5 --tag vxm1_a2s_softdice > logs_train_vxm1_b1.txt 2>&1
echo [%DATE% %TIME%] voxelmorph1 beta=1.0 finished exit=%ERRORLEVEL% >> logs_train_queue.txt

echo [%DATE% %TIME%] starting voxelmorph1 beta=0.0 >> logs_train_queue.txt
"%PY%" -u train_softdice.py --arch voxelmorph1 --beta 0.0 --lam 1.0 --epochs 100 --val-every 5 --tag vxm1_a2s_unsup > logs_train_vxm1_b0.txt 2>&1
echo [%DATE% %TIME%] voxelmorph1 beta=0.0 finished exit=%ERRORLEVEL% >> logs_train_queue.txt

echo [%DATE% %TIME%] QUEUE DONE >> logs_train_queue.txt

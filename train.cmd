@echo off
@REM 
set /P msg=Commit message: 

@REM Activate the conda environment
set CONDAPATH=D:\Miniconda3
set ENVNAME=CrossyRoad
if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

@REM Run the training script
python .\train.py

@REM Deactivate the environment
call conda deactivate

@REM Git tracking
git add .
git commit -m "%msg%"
git push

echo.
pause
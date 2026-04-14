@echo off
setlocal

call "C:\Users\wocho\miniforge3\condabin\conda.bat" activate 4dgs
if errorlevel 1 (
  echo Failed to activate conda environment 4dgs.
  exit /b 1
)

python %*
exit /b %ERRORLEVEL%

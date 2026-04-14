@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "FOCUSGS_CONFIG=configs\dynerf\cook_spinach_focusgs_sanity.yaml"
set "FOCUSGS_EXTRA_ARGS="
if not "%~1"=="" (
  set "FOCUSGS_CONFIG=%~1"
  shift
)

:collect_args
if "%~1"=="" goto run
set "FOCUSGS_EXTRA_ARGS=%FOCUSGS_EXTRA_ARGS% %1"
shift
goto collect_args

:run
call "%SCRIPT_DIR%run_in_4dgs_env.cmd" train.py --config "%FOCUSGS_CONFIG%" %FOCUSGS_EXTRA_ARGS%
exit /b %ERRORLEVEL%

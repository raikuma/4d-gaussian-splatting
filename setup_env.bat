@echo off
setlocal

set "ENV_NAME=4dgs"
set "VS2019_DEV_CMD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"

call conda create -n %ENV_NAME% python=3.10 pip setuptools=69.5.1 wheel -y
call conda activate %ENV_NAME%
call conda install -y cuda -c nvidia/label/cuda-11.8.0

python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install setuptools==69.5.1 "numpy<2" plyfile==0.8.1 tqdm==4.66.1 torchmetrics==0.11.4 imagesize==1.4.1 kornia==0.6.12 omegaconf==2.3.0 tensorboard ninja scikit-image lpips opencv-python==4.8.1.78

if not exist "%CONDA_PREFIX%\etc\conda\activate.d" mkdir "%CONDA_PREFIX%\etc\conda\activate.d"
if not exist "%CONDA_PREFIX%\etc\conda\deactivate.d" mkdir "%CONDA_PREFIX%\etc\conda\deactivate.d"

> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo @echo off
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "__4DGS_OLD_DISTUTILS_USE_SDK=%%DISTUTILS_USE_SDK%%"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "__4DGS_OLD_MSSDK=%%MSSdk%%"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "__4DGS_OLD_CUDA_HOME=%%CUDA_HOME%%"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "__4DGS_OLD_CUDA_PATH=%%CUDA_PATH%%"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "__4DGS_OLD_TORCH_CUDA_ARCH_LIST=%%TORCH_CUDA_ARCH_LIST%%"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo call "%VS2019_DEV_CMD%" -arch=amd64 ^>nul
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "DISTUTILS_USE_SDK=1"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "MSSdk=1"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "CUDA_HOME=%%CONDA_PREFIX%%"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "CUDA_PATH=%%CONDA_PREFIX%%"
>> "%CONDA_PREFIX%\etc\conda\activate.d\4dgs_vs2019_cuda.bat" echo set "TORCH_CUDA_ARCH_LIST=9.0+PTX"

> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo @echo off
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo if defined __4DGS_OLD_DISTUTILS_USE_SDK ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set "DISTUTILS_USE_SDK=%%__4DGS_OLD_DISTUTILS_USE_SDK%%"
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^) else ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set DISTUTILS_USE_SDK=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^)
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo if defined __4DGS_OLD_MSSDK ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set "MSSdk=%%__4DGS_OLD_MSSDK%%"
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^) else ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set MSSdk=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^)
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo if defined __4DGS_OLD_CUDA_HOME ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set "CUDA_HOME=%%__4DGS_OLD_CUDA_HOME%%"
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^) else ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set CUDA_HOME=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^)
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo if defined __4DGS_OLD_CUDA_PATH ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set "CUDA_PATH=%%__4DGS_OLD_CUDA_PATH%%"
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^) else ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set CUDA_PATH=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^)
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo if defined __4DGS_OLD_TORCH_CUDA_ARCH_LIST ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set "TORCH_CUDA_ARCH_LIST=%%__4DGS_OLD_TORCH_CUDA_ARCH_LIST%%"
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^) else ^(
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo     set TORCH_CUDA_ARCH_LIST=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo ^)
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo set __4DGS_OLD_DISTUTILS_USE_SDK=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo set __4DGS_OLD_MSSDK=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo set __4DGS_OLD_CUDA_HOME=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo set __4DGS_OLD_CUDA_PATH=
>> "%CONDA_PREFIX%\etc\conda\deactivate.d\4dgs_vs2019_cuda.bat" echo set __4DGS_OLD_TORCH_CUDA_ARCH_LIST=

call "%VS2019_DEV_CMD%" -arch=amd64
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "CUDA_HOME=%CONDA_PREFIX%"
set "CUDA_PATH=%CONDA_PREFIX%"
set "TORCH_CUDA_ARCH_LIST=9.0+PTX"

python -m pip install --no-build-isolation .\simple-knn
python -m pip install --no-build-isolation .\pointops2

echo diff-gaussian-rasterization is JIT-compiled on first import.

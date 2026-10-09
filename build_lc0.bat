@echo off
rem Builds Lc0 from lc0-src with its CUDA backend, for this computer's GPU.
rem
rem Usage: build_lc0.bat [CC]
rem   CC is the GPU's CUDA compute capability without the dot, for example 89
rem   for an RTX 40xx. By default it is read from nvidia-smi.
rem
rem Needs: the CUDA toolkit (found through CUDA_PATH, or in its default
rem location), Visual Studio
rem 2022 or its Build Tools with the C++ workload, and meson and ninja
rem ("python -m pip install --user meson ninja").
rem
rem The result is lc0-src\build\lc0.exe.
rem
rem For Maia-3 opponent models it also builds with ONNX Runtime, when its GPU
rem package is unpacked at lc0\onnxruntime-win-x64-gpu_cuda13-1.31.0, and copies
rem its DLLs next to lc0.exe, along with cuDNN 9's DLLs from CUDNN_BIN (by
rem default the nvidia-cudnn-cu13 wheel in .venv), which ONNX Runtime needs.

setlocal

rem CUDA_PATH is set machine-wide by the CUDA installer, but terminals started
rem before the install (VS Code's included) do not have it. Read it from the
rem registry then, or take the newest toolkit in the default location.
if not defined CUDA_PATH (
    for /f "tokens=2,*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v CUDA_PATH 2^>nul') do set "CUDA_PATH=%%b"
)
if not defined CUDA_PATH (
    for /d %%d in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_PATH=%%d"
)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo Could not find the CUDA toolkit. Install it, or set CUDA_PATH to its folder.
    exit /b 1
)
echo Using the CUDA toolkit at %CUDA_PATH%

rem The Visual Studio C++ environment, unless it is already loaded
where /q cl
if errorlevel 1 (
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VSPATH=%%i"
)
if defined VSPATH call "%VSPATH%\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul
where /q cl
if errorlevel 1 (
    echo Could not find the Visual Studio C++ compiler.
    exit /b 1
)

rem meson and ninja installed with "pip install --user" live here
for /d %%d in ("%APPDATA%\Python\Python3*") do set "PATH=%%d\Scripts;%PATH%"
where /q meson
if errorlevel 1 (
    echo Could not find meson. Run "python -m pip install --user meson ninja".
    exit /b 1
)

set "CC_CUDA=%~1"
if "%CC_CUDA%"=="" (
    for /f "usebackq tokens=1,2 delims=." %%a in (`nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader`) do set "CC_CUDA=%%a%%b"
)
echo Building for CUDA compute capability %CC_CUDA%

set PATH=%CUDA_PATH%\bin;%PATH%
set CC=cl
set CXX=cl
set CC_LD=link
set CXX_LD=link

set "ORT_DIR=%~dp0lc0\onnxruntime-win-x64-gpu_cuda13-1.31.0"
if not defined CUDNN_BIN set "CUDNN_BIN=%~dp0.venv\Lib\site-packages\nvidia\cudnn\bin"
set ONNX_OPTIONS=-Donnx=false
if exist "%ORT_DIR%\include\onnxruntime_cxx_api.h" (
    set ONNX_OPTIONS=-Donnx=true -Donnx_libdir="%ORT_DIR%\lib" -Donnx_include="%ORT_DIR%\include"
) else (
    echo ONNX Runtime not found at %ORT_DIR%: building without Maia-3 support.
)

pushd "%~dp0lc0-src" || exit /b 1

if not exist build\build.ninja (
    meson setup build --backend ninja --buildtype release -Ddefault_library=static ^
        -Dplain_cuda=true -Dcudnn=false -Ddx=false -Dopencl=false -Dblas=false -Dgtest=false ^
        -Dcc_cuda=%CC_CUDA% ^
        -Dcudnn_include="%CUDA_PATH%\include" -Dcudnn_libdirs="%CUDA_PATH%\lib\x64" ^
        %ONNX_OPTIONS%
) else (
    meson configure build %ONNX_OPTIONS%
)
if errorlevel 1 (
    popd
    echo meson setup failed.
    exit /b 1
)

meson compile -C build
set "RESULT=%ERRORLEVEL%"

popd

if not "%RESULT%"=="0" (
    echo.
    echo Build failed.
    exit /b %RESULT%
)

if exist "%ORT_DIR%\lib\onnxruntime.dll" (
    copy /y "%ORT_DIR%\lib\onnxruntime*.dll" "%~dp0lc0-src\build\" >nul
    if exist "%CUDNN_BIN%\cudnn64_9.dll" (
        copy /y "%CUDNN_BIN%\cudnn*.dll" "%~dp0lc0-src\build\" >nul
    ) else (
        echo cuDNN 9 not found at %CUDNN_BIN%: Maia-3 opponent models will not load.
        echo Set CUDNN_BIN to the folder that holds cudnn64_9.dll.
    )
)

echo.
echo Lc0 built: %~dp0lc0-src\build\lc0.exe

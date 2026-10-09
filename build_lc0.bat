@echo off
rem Builds Lc0 from lc0-src with its CUDA backend, for this computer's GPU.
rem
rem Usage: build_lc0.bat [CC]
rem   CC is the GPU's CUDA compute capability without the dot, for example 89
rem   for an RTX 40xx. By default it is read from nvidia-smi.
rem
rem Needs: the CUDA toolkit (CUDA_PATH set by its installer), Visual Studio
rem 2022 or its Build Tools with the C++ workload, and meson and ninja
rem ("python -m pip install --user meson ninja").
rem
rem The result is lc0-src\build\lc0.exe.

setlocal

if not defined CUDA_PATH (
    echo CUDA_PATH is not set. Install the CUDA toolkit first.
    exit /b 1
)

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

pushd "%~dp0lc0-src" || exit /b 1

if not exist build\build.ninja (
    meson setup build --backend ninja --buildtype release -Ddefault_library=static ^
        -Dplain_cuda=true -Dcudnn=false -Ddx=false -Dopencl=false -Dblas=false -Dgtest=false ^
        -Dcc_cuda=%CC_CUDA% ^
        -Dcudnn_include="%CUDA_PATH%\include" -Dcudnn_libdirs="%CUDA_PATH%\lib\x64"
    if errorlevel 1 (
        popd
        echo meson setup failed.
        exit /b 1
    )
)

meson compile -C build
set "RESULT=%ERRORLEVEL%"

popd

if not "%RESULT%"=="0" (
    echo.
    echo Build failed.
    exit /b %RESULT%
)

echo.
echo Lc0 built: %~dp0lc0-src\build\lc0.exe

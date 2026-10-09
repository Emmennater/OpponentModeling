@echo off
rem Builds the engine (Stockfish 19 with opponent modeling) using MSYS2's MinGW toolchain.
rem
rem Usage: build.bat [ARCH]
rem   ARCH defaults to "native", which builds for this computer's CPU.
rem   Pass e.g. x86-64-avx2 to build for other machines. Run
rem   "mingw32-make help" in stockfish\src for the full list.
rem
rem MSYS2 is expected in C:\msys64. Set MSYS2_ROOT to use another location.

setlocal

if not defined MSYS2_ROOT set "MSYS2_ROOT=C:\msys64"

if not exist "%MSYS2_ROOT%\mingw64\bin\mingw32-make.exe" (
    echo Could not find mingw32-make.exe in "%MSYS2_ROOT%\mingw64\bin".
    echo Install MSYS2 with the mingw-w64-x86_64-gcc and mingw-w64-x86_64-make packages,
    echo or set MSYS2_ROOT to your MSYS2 folder.
    exit /b 1
)

rem MSYS2's own sh.exe must come first: the Makefile runs shell scripts, and
rem they fail under a shell whose path contains a space (such as Git's).
set "PATH=%MSYS2_ROOT%\mingw64\bin;%MSYS2_ROOT%\usr\bin;%PATH%"

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=native"

pushd "%~dp0stockfish\src" || exit /b 1

rem Always rebuild from scratch, so objects built with other settings are never mixed in.
mingw32-make objclean
mingw32-make -j %NUMBER_OF_PROCESSORS% build ARCH=%ARCH% COMP=mingw
set "RESULT=%ERRORLEVEL%"

popd

if not "%RESULT%"=="0" (
    echo.
    echo Build failed.
    exit /b %RESULT%
)

echo.
echo Engine built: %~dp0stockfish\src\stockfish.exe

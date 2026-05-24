@echo off
REM ============================================================
REM  MTA for Master — First-time install (Windows)
REM ============================================================
REM  This script downloads an isolated Python and installs
REM  everything MTA needs, inside the folder it sits in. It does
REM  not touch anything else on your computer.
REM  Duration: 5 to 10 minutes depending on your connection.
REM ============================================================

chcp 65001 > nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo   MTA — Installation
echo ============================================================
echo.
echo   What will happen:
echo     1. Download uv (Python manager, about 30 MB)
echo     2. Download isolated Python 3.12 (about 50 MB)
echo     3. Install scientific libraries (about 400 MB)
echo.
echo   EVERYTHING stays inside this folder. To uninstall later,
echo   simply delete this entire folder.
echo.
echo   Estimated duration: 5 to 10 minutes.
echo.
pause

REM --- Step 1: download uv if not already present ---
if not exist "uv\uv.exe" (
    echo.
    echo [1/3] Downloading uv...
    mkdir uv 2>nul
    powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile 'uv\uv.zip' -UseBasicParsing } catch { Write-Host 'Download error.'; exit 1 }"
    if errorlevel 1 (
        echo.
        echo ERROR: could not download uv.
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )
    powershell -NoProfile -Command "Expand-Archive -Path 'uv\uv.zip' -DestinationPath 'uv' -Force"
    del "uv\uv.zip"
    echo   uv downloaded.
) else (
    echo [1/3] uv already present, skipping.
)

REM --- Step 2: create the isolated Python environment ---
echo.
echo [2/3] Creating isolated Python environment...
"uv\uv.exe" venv .venv --python 3.12
if errorlevel 1 (
    echo ERROR while creating the Python environment.
    pause
    exit /b 1
)
echo   Environment created.

REM --- Step 3: install dependencies ---
echo.
echo [3/3] Installing scientific libraries...
echo   (This may take a few minutes, please be patient.)
"uv\uv.exe" pip install --python .venv\Scripts\python.exe -r code\requirements.txt
if errorlevel 1 (
    echo ERROR while installing libraries.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Installation complete!
echo ============================================================
echo.
echo   You can now close this window and start MTA by
echo   double-clicking on:
echo.
echo       start_MTA.bat       (Windows)
echo       start_MTA.command   (Mac)
echo       start_MTA.sh        (Linux)
echo.
pause

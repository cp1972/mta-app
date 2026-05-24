@echo off
REM ============================================================
REM  MTA for Master — Launcher (Windows)
REM ============================================================
REM  Lets the user pick between:
REM    1. The Streamlit web app (default)
REM    2. The MTA_v3.py interactive CLI
REM    3. Batch / scripting help
REM  All paths go through .venv\Scripts\python.exe so that
REM  bundled libraries (pandas, scikit-learn, etc.) are found.
REM ============================================================

chcp 65001 > nul
cd /d "%~dp0"

REM Verify installation
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: MTA is not installed.
    echo Please first double-click on:
    echo    install_first_run.bat
    echo.
    pause
    exit /b 1
)

:menu
cls
echo.
echo ============================================================
echo   MTA — Multi-Text Analyser
echo ============================================================
echo.
echo   Choose an interface:
echo.
echo     [1] Streamlit web app                   ^(default^)
echo         Recommended for beginners. Click-and-drag
echo         interface in your browser.
echo.
echo     [2] Command-line interactive menu
echo         Text-based menu, runs in this window.
echo         Same workflow as the original MTA.py.
echo.
echo     [3] Show batch / scripting usage
echo         For Stata, R, or shell pipelines.
echo.
echo     [Q] Quit
echo.

set "choice="
set /p "choice=Your choice [1]: "
if "%choice%"=="" set "choice=1"

if /i "%choice%"=="1" goto streamlit
if /i "%choice%"=="2" goto cli
if /i "%choice%"=="3" goto batch_help
if /i "%choice%"=="Q" goto end
echo.
echo   Invalid choice: %choice%
timeout /t 2 > nul
goto menu

:streamlit
echo.
echo ============================================================
echo   Starting Streamlit web app...
echo ============================================================
echo.
echo   Your browser will open automatically.
echo   If nothing happens, open your browser at:
echo       http://localhost:8501
echo.
echo   IMPORTANT: DO NOT CLOSE this black window while using MTA.
echo   Close it to stop MTA.
echo ============================================================
echo.
start "" /min cmd /c "timeout /t 4 /nobreak > nul & start http://localhost:8501"
.venv\Scripts\streamlit.exe run code\streamlit_app.py --server.headless true --browser.gatherUsageStats false
goto end

:cli
echo.
echo ============================================================
echo   Starting CLI interactive menu...
echo ============================================================
echo.
.venv\Scripts\python.exe code\MTA_v3.py
echo.
echo CLI session ended.
pause
goto menu

:batch_help
echo.
echo ============================================================
echo   MTA_v3.py — Batch / scripting usage
echo ============================================================
echo.
.venv\Scripts\python.exe code\MTA_v3.py --help
echo.
echo ============================================================
echo   Example for Stata / R / shell:
echo.
echo   .venv\Scripts\python.exe code\MTA_v3.py ^
echo       --corpus  C:\path\to\corpus ^
echo       --stopwords C:\path\to\stop.txt ^
echo       --action nmf --n-topics 5 --json
echo ============================================================
echo.
pause
goto menu

:end
echo.
echo Goodbye!
timeout /t 2 > nul
exit /b 0

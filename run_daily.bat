@echo off
cd /d "%~dp0"

rem ===== One-click daily run with automatic logging =====
rem Double-click  : show progress + write log + auto-open dashboard on success
rem "run_daily.bat silent" : silent mode for Windows Task Scheduler
rem Logs: logs\run_YYYYMMDD_HHMMSS.log (+ last_success.log / last_error.log)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set SILENT=%1

if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOGFILE=logs\run_%TS%.log

if not "%SILENT%"=="silent" echo ===== Fusion Stock System daily run ===== log: %LOGFILE%
if not "%SILENT%"=="silent" echo Running, please wait 1-3 min...

python main.py > "%LOGFILE%" 2>&1
set EXITCODE=%ERRORLEVEL%

if not "%SILENT%"=="silent" powershell -NoProfile -Command "Get-Content '%LOGFILE%' -Encoding UTF8 -Tail 60"

if %EXITCODE% NEQ 0 goto error

copy /y "%LOGFILE%" logs\last_success.log >nul
if "%SILENT%"=="silent" exit /b 0

echo.
echo SUCCESS - opening today dashboard...
for /f "delims=" %%f in ('dir /b /o-d output\dashboard_*.html 2^>nul') do (
    start "" "output\%%f"
    goto done
)
echo dashboard file not found in output\
pause
exit /b 0

:error
copy /y "%LOGFILE%" logs\last_error.log >nul
if "%SILENT%"=="silent" exit /b %EXITCODE%
echo.
echo ================ ERROR ================
echo exit code: %EXITCODE%    full log: %LOGFILE%
echo common causes:
echo   1. network or proxy down (akshare needs internet)
echo   2. deps missing: pip install akshare pandas numpy PyYAML jinja2
echo fix: drag logs\last_error.log into TRAE chat, AI will diagnose it.
pause
exit /b %EXITCODE%

:done
exit /b 0

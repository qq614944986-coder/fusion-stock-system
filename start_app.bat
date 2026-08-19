@echo off
cd /d "%~dp0"

rem ===== Start local web app (double-click me) =====
rem Opens http://127.0.0.1:8199 in your browser automatically.
rem Data is fetched ONLY when you click the refresh button in the page.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
rem Domestic market data sites bypass the proxy (direct connection is faster and more stable)
set NO_PROXY=localhost,127.0.0.1,eastmoney.com,sina.com.cn,xueqiu.com,10jqka.com.cn,csindex.com.cn

if not exist logs mkdir logs

rem Already running? Just open the page.
powershell -NoProfile -Command "try{ $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',8199); $c.Close(); exit 0 }catch{ exit 1 }" >nul 2>&1
if %ERRORLEVEL%==0 (
    start "" http://127.0.0.1:8199
    exit /b 0
)

echo Starting service, browser will open in a second...
python server.py >> logs\server.log 2>&1
pause

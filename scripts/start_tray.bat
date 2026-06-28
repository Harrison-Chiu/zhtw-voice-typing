@echo off
REM 有視窗版啟動 ASR Input tray —— 首次測試或排錯用，看得到啟動 log 與錯誤。
REM 日常無視窗啟動請用 start_tray.vbs（或桌面捷徑）。
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m asr_input.tray
echo.
echo === tray 已結束（上方為輸出 / 錯誤訊息）===
pause

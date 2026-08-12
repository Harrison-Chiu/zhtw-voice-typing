@echo off
REM Windowed launcher for ASR Input tray - first run / troubleshooting.
REM For silent daily launch use start_tray.vbs (or the desktop shortcut).
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m asr_input.tray --startup-mode manual
echo.
echo === tray exited (output / errors shown above) ===
pause

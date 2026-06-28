' Silent launcher for ASR Input tray - double-click to start, no console window.
'
' Uses python.exe (not pythonw.exe) with hidden window style (0):
' tray.py uses print(); under pythonw sys.stdout is None and print() crashes.
' python.exe + hidden window keeps a console buffer so print() works while the
' window stays hidden. For troubleshooting use start_tray.bat (visible window).

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoDir = fso.GetParentFolderName(scriptDir)
pyExe = repoDir & "\.venv\Scripts\python.exe"

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = repoDir
sh.Run """" & pyExe & """ -m asr_input.tray", 0, False

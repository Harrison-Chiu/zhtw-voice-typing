' Silent launcher for ASR Input tray - double-click to start, no console window.
' Keep this file ASCII-only: Windows Script Host may parse UTF-8 without a BOM
' as an ANSI code page and report misleading unterminated-string errors.
'
' Uses python.exe (not pythonw.exe) with hidden window style (0):
' tray.py uses print(); under pythonw sys.stdout is None and print() crashes.
' python.exe + hidden window keeps a console buffer so print() works while the
' window stays hidden. For troubleshooting use start_tray.bat (visible window).
'
' Named arguments (WSH style, e.g. wscript start_tray.vbs /autostart):
'   /autostart  launch in autostart mode - record path only, model loads on the
'               first recording. Used by the Windows startup-folder shortcut.
'               Without it the tray uses manual mode and preloads Whisper.
'   /check      exit 0 immediately; lets tests parse this file without running it.

Set fso = CreateObject("Scripting.FileSystemObject")
If WScript.Arguments.Named.Exists("check") Then WScript.Quit 0

startupMode = "manual"
If WScript.Arguments.Named.Exists("autostart") Then startupMode = "autostart"

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoDir = fso.GetParentFolderName(scriptDir)
pyExe = repoDir & "\.venv\Scripts\python.exe"
logDir = repoDir & "\data\logs"
logFile = logDir & "\launcher.log"

If Not fso.FileExists(pyExe) Then
    MsgBox "ASR Input Python was not found:" & vbCrLf & pyExe & vbCrLf & _
        "Complete setup first, or use the windowed shortcut for diagnostics.", _
        vbCritical, "ASR Input startup failed"
    WScript.Quit 2
End If

If Not fso.FolderExists(repoDir & "\data") Then
    fso.CreateFolder repoDir & "\data"
End If
If Not fso.FolderExists(logDir) Then
    fso.CreateFolder logDir
End If
If fso.FileExists(logFile) Then
    If fso.GetFile(logFile).Size > 5242880 Then
        oldLog = logDir & "\launcher.previous.log"
        If fso.FileExists(oldLog) Then fso.DeleteFile oldLog, True
        fso.MoveFile logFile, oldLog
    End If
End If

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = repoDir
command = "%ComSpec% /d /s /c """"" & pyExe & """ -m asr_input.tray --startup-mode " & _
    startupMode & " >> """ & logFile & """ 2>&1"""
exitCode = sh.Run(sh.ExpandEnvironmentStrings(command), 0, True)

If exitCode <> 0 Then
    MsgBox "ASR Input exited unexpectedly (exit " & exitCode & ")." & vbCrLf & _
        "Diagnostic log: " & logFile, vbCritical, "ASR Input failed"
End If

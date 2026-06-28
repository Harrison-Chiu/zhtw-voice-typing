' 隱藏視窗啟動 ASR Input tray —— 雙擊即用，不開終端視窗。
'
' 刻意用 python.exe（非 pythonw.exe）搭配「視窗樣式 0（隱藏）」：
' tray.py 大量用 print()，pythonw 下 sys.stdout 為 None 會讓 print 崩潰；
' python.exe + 隱藏視窗則保留 console buffer，print 正常吞掉、tray 照常跑。
' 排錯請改用同目錄的 start_tray.bat（有視窗、看得到輸出與錯誤）。

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoDir = fso.GetParentFolderName(scriptDir)
pyExe = repoDir & "\.venv\Scripts\python.exe"

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = repoDir
sh.Run """" & pyExe & """ -m asr_input.tray", 0, False

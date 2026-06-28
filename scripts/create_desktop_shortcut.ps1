# 在桌面建立「ASR Input」捷徑，指向 start_tray.vbs（雙擊即啟動、無終端視窗）。
#
# 為何要由你在自己的終端跑：Claude Code Desktop 是 MSIX 沙箱，由它代建的捷徑
# 會落在沙箱內、你的真實桌面看不到，故這支腳本交給你手動執行。
#
# 跑法（在專案根目錄）：
#     powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1

$repo = Split-Path -Parent $PSScriptRoot
$vbs = Join-Path $repo 'scripts\start_tray.vbs'
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop 'ASR Input.lnk'

if (-not (Test-Path $vbs)) {
    Write-Error "找不到 $vbs，請確認在專案根目錄執行。"
    exit 1
}

$sh = New-Object -ComObject WScript.Shell
$s = $sh.CreateShortcut($lnk)
$s.TargetPath = 'wscript.exe'                 # 用 wscript 顯式開 vbs，不依賴 .vbs 副檔名關聯
$s.Arguments = '"' + $vbs + '"'
$s.WorkingDirectory = $repo
$s.Description = 'ASR 語音輸入（System Tray）— 雙擊啟動'
$s.IconLocation = (Join-Path $repo '.venv\Scripts\python.exe') + ',0'  # 暫用 python 圖示
$s.Save()

Write-Host "已建立桌面捷徑：$lnk"
Write-Host "雙擊即可啟動 ASR Input（首次仍需 ~30s 載入模型，圖示轉綠後可用）。"

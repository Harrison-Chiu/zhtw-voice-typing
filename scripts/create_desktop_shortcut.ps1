# 在桌面建立 ASR Input 捷徑（兩個）：
#   「ASR Input」        → start_tray.vbs（無視窗、背景常駐，日常用；結束用系統匣右鍵）
#   「ASR Input (視窗)」  → start_tray.bat（有視窗、看得到 log；關視窗即結束，排錯用）
#
# 為何要由你在自己的終端跑：Claude Code Desktop 是 MSIX 沙箱，由它代建的捷徑
# 可能落在沙箱內、你的真實桌面看不到，故這支腳本交給你手動執行。
#
# 跑法（在專案根目錄）：
#     powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1

$repo = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$pyIcon = (Join-Path $repo '.venv\Scripts\python.exe') + ',0'   # 暫用 python 圖示
$vbs = Join-Path $repo 'scripts\start_tray.vbs'
$bat = Join-Path $repo 'scripts\start_tray.bat'

if (-not (Test-Path $vbs)) {
    Write-Error "找不到 $vbs，請確認在專案根目錄執行。"
    exit 1
}

$sh = New-Object -ComObject WScript.Shell

function New-AsrShortcut {
    param($Name, $Target, $Args, $Desc)
    $lnk = Join-Path $desktop $Name
    $s = $sh.CreateShortcut($lnk)
    $s.TargetPath = $Target
    if ($Args) { $s.Arguments = $Args }
    $s.WorkingDirectory = $repo
    $s.Description = $Desc
    $s.IconLocation = $pyIcon
    $s.Save()
    Write-Host "已建立：$lnk"
}

# 無視窗版：用 wscript 顯式開 vbs，不依賴 .vbs 副檔名關聯
New-AsrShortcut 'ASR Input.lnk' 'wscript.exe' ('"' + $vbs + '"') `
    'ASR 語音輸入（背景常駐、無視窗）— 結束請用系統匣圖示右鍵'

# 有視窗版：直接指向 bat，雙擊開 cmd 視窗看 log
New-AsrShortcut 'ASR Input (視窗).lnk' $bat $null `
    'ASR 語音輸入（有視窗、看得到 log）— 關視窗即結束'

Write-Host "完成。日常用「ASR Input」，想看 log 用「ASR Input (視窗)」。"
Write-Host "首次啟動仍需 ~30s 載入模型，系統匣圖示轉綠後可用。"

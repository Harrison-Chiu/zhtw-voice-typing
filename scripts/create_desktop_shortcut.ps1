# 建立 ASR Input 的啟動捷徑。
#
# 預設只建一個桌面捷徑「ASR Input」→ start_tray.vbs（無視窗、背景常駐，日常用；
# 結束用系統匣圖示右鍵）。排錯用的有視窗版本改成選配，避免桌面長期擺兩個圖示。
#
# 為何要由你在自己的終端跑：Claude Code Desktop 是 MSIX 沙箱，由它代建的捷徑
# 可能落在沙箱內、你的真實桌面看不到，故這支腳本交給你手動執行。
#
# 跑法（在專案根目錄）：
#     powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1
#
# 參數：
#     -WithDebugShortcut  另外建「ASR Input (視窗)」→ start_tray.bat，看得到 log
#     -Autostart          另外在「啟動」資料夾放一份，開機自動常駐。走 autostart
#                         模式：只載錄音與 CPU VAD，第一次錄音才載 Whisper，
#                         開機後不占顯存
#     -Remove             移除本腳本建立過的三個捷徑（桌面兩個 + 啟動資料夾一個）

[CmdletBinding()]
param(
    [switch]$WithDebugShortcut,
    [switch]$Autostart,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$startupDir = [Environment]::GetFolderPath('Startup')
$vbs = Join-Path $repo 'scripts\start_tray.vbs'
$bat = Join-Path $repo 'scripts\start_tray.bat'
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'

$desktopMain = Join-Path $desktop 'ASR Input.lnk'
$desktopDebug = Join-Path $desktop 'ASR Input (視窗).lnk'
$startupMain = Join-Path $startupDir 'ASR Input.lnk'

if ($Remove) {
    $removed = 0
    foreach ($lnk in @($desktopMain, $desktopDebug, $startupMain)) {
        if (Test-Path $lnk) {
            Remove-Item $lnk -Force
            Write-Host "已移除：$lnk"
            $removed++
        }
    }
    if ($removed -eq 0) { Write-Host '沒有找到本腳本建立過的捷徑。' }
    exit 0
}

if (-not (Test-Path $vbs)) {
    Write-Error "找不到 $vbs，請確認在專案根目錄執行。"
    exit 1
}

# 捷徑本身不依賴 .venv，但沒有 .venv 時雙擊只會跳錯誤對話框，提早講清楚比較好排錯。
if (-not (Test-Path $venvPython)) {
    Write-Warning "尚未建立虛擬環境（找不到 $venvPython）。捷徑會照建，但要先在專案根目錄跑 uv sync 才能啟動。"
}

$pyIcon = $venvPython + ',0'   # 暫用 python 圖示
$sh = New-Object -ComObject WScript.Shell

function New-AsrShortcut {
    param($Path, $Target, $Args, $Desc)
    $s = $sh.CreateShortcut($Path)
    $s.TargetPath = $Target
    if ($Args) { $s.Arguments = $Args }
    $s.WorkingDirectory = $repo
    $s.Description = $Desc
    $s.IconLocation = $pyIcon
    $s.Save()
    Write-Host "已建立：$Path"
}

# 日常用：無視窗版。用 wscript 顯式開 vbs，不依賴 .vbs 副檔名關聯
New-AsrShortcut $desktopMain 'wscript.exe' ('"' + $vbs + '"') `
    'ASR 語音輸入（背景常駐、無視窗）— 結束請用系統匣圖示右鍵'

if ($WithDebugShortcut) {
    # 排錯用：直接指向 bat，雙擊開 cmd 視窗看 log
    New-AsrShortcut $desktopDebug $bat $null `
        'ASR 語音輸入（有視窗、看得到 log）— 關視窗即結束'
}

if ($Autostart) {
    New-AsrShortcut $startupMain 'wscript.exe' ('"' + $vbs + '" /autostart') `
        'ASR 語音輸入（開機自動啟動、無視窗）— 第一次錄音才載入模型'
}

Write-Host ''
Write-Host '完成。啟動後系統匣圖示會先出現；CPU VAD 就緒即可錄音。'
if (-not $WithDebugShortcut) {
    Write-Host '要排錯（看得到 log）時，直接雙擊 scripts\start_tray.bat，或加 -WithDebugShortcut 重跑本腳本。'
}
if (-not $Autostart) {
    Write-Host '要開機自動啟動，加 -Autostart 重跑本腳本。'
}
Write-Host '搬移或改名專案資料夾後要重跑本腳本；只改程式內容不用。'

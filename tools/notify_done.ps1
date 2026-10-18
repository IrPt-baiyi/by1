# notify_done.ps1 -- 目标完成时发声提醒（Windows）。
#
# 用法：powershell -ExecutionPolicy Bypass -File tools/notify_done.ps1
#
# 三层保险：先 beep 一小段旋律（不依赖任何音频文件），再播系统提示音，
# 最后打印一行大字（ssh/无音频设备时至少终端里有字）。
$ErrorActionPreference = 'Continue'

Write-Host ''
Write-Host '  ============================================' -ForegroundColor Green
Write-Host '   by1 目标完成 -- 目标完成，回来看结果' -ForegroundColor Green
Write-Host '  ============================================' -ForegroundColor Green
Write-Host ''

# 1) 上行三音（C5-E5-G5），重复两次
$notes = @(@(523, 160), @(659, 160), @(784, 260))
foreach ($round in 1..2) {
    foreach ($n in $notes) {
        try { [console]::beep([int]$n[0], [int]$n[1]) } catch { }
    }
    Start-Sleep -Milliseconds 120
}

# 2) 系统提示音（有就用）
$wav = Join-Path $env:WINDIR 'Media\Alarm01.wav'
if (Test-Path $wav) {
    try { (New-Object Media.SoundPlayer $wav).PlaySync() } catch { }
}

Write-Host '  [完成] 目标已标 complete。' -ForegroundColor Green
Write-Host ''
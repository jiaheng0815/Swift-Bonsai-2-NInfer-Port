# MMLU 评测看门狗：跑完 -> 生成报告 -> 延时关机。
# 不依赖 WMI，也不打开结果文件（只读元数据），避免和评测进程抢文件锁。
# 防睡眠用 SetThreadExecutionState（进程退出即失效，不改系统电源设置）。
param(
    [string]$Deadline = '2026-10-10 04:15',
    [string]$WorkDir  = 'D:\ai-ninfer\eval\mmlu',
    [int]$StallMinutes = 10,
    [int]$GraceSeconds = 120,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Continue'
$log = Join-Path $WorkDir 'shutdown_watch.log'
$res = Join-Path $WorkDir 'full.jsonl'
$sum = Join-Path $WorkDir 'summary.json'

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

function Get-LineCount($path) {
    try {
        $fs = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        $sr = New-Object IO.StreamReader($fs)
        $n = 0
        while ($null -ne $sr.ReadLine()) { $n++ }
        $sr.Close(); $fs.Close()
        return $n
    } catch { return -1 }
}

Add-Type -Namespace W -Name N -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);
'@
Say "watchdog 启动 | 防睡眠开 | 按期收尾 $Deadline | 停顿判定 ${StallMinutes}min | PID $PID"
[void][W.N]::SetThreadExecutionState([uint32]0x80000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$stop = Get-Date $Deadline

try {
    $lastTick = Get-Date
    $lastCount = -1
    while ($true) {
        try {
            $size = (Get-Item -LiteralPath $res -ErrorAction Stop).Length
        } catch { $size = -1 }
        $count = Get-LineCount $res
        if ($count -ne $lastCount) {
            $lastCount = $count
            $lastTick = Get-Date
            Say "进度 $count 题 | 结果文件 $size 字节"
        }
        if (Test-Path -LiteralPath $sum) { Say "summary.json 出现 -> 评测完成（$count 题）"; break }
        if ($count -gt 0 -and ((Get-Date) - $lastTick).TotalMinutes -gt $StallMinutes) {
            Say "结果文件 ${StallMinutes} 分钟无增长 -> 视为结束或中断（$count 题）"; break
        }
        if ((Get-Date) -gt $stop) { Say "到达收尾时刻 $Deadline（$count 题），强制收尾"; break }
        Start-Sleep -Seconds $PollSeconds
    }

    Push-Location $WorkDir
    try {
        $py = (Get-Command python -ErrorAction SilentlyContinue).Source
        if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
        if ($py) {
            $out = & $py make_report.py 2>&1 | Out-String
            Say ("报告: " + ($out.Trim() -replace "\r?\n", ' / '))
        } else { Say "找不到 python，跳过报告" }
    } catch { Say "报告生成失败: $_" }
    Pop-Location

    # 先撤掉兜底定时，再排自己的延时关机，避免冲突
    & shutdown.exe /a 2>&1 | Out-Null
    Say "提交延时关机（$GraceSeconds 秒；取消： shutdown /a）"
    & shutdown.exe /s /t $GraceSeconds /c "MMLU 评测已完成，机器将在 2 分钟后关机（取消：shutdown /a）"
    $rc = $LASTEXITCODE
    Say "shutdown.exe 退出码 $rc"
} catch {
    Say "看门狗异常退出: $_"
} finally {
    [void][W.N]::SetThreadExecutionState([uint32]0x80000000)
}

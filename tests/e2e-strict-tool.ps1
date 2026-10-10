# 端到端：工具 strict:true 的参数约束
$ErrorActionPreference = 'Continue'
$uri = 'http://127.0.0.1:8080/v1/chat/completions'

$schema = @{
  type = 'object'
  properties = @{
    city  = @{ type = 'string'; enum = @('北京','上海','广州') }
    days  = @{ type = 'integer'; minimum = 1; maximum = 5 }
    units = @{ type = 'string'; enum = @('celsius','fahrenheit') }
  }
  required = @('city','days','units')
  additionalProperties = $false
}

$body = @{
  model = 'bonsai2-27b'; max_tokens = 700
  messages = @(@{ role='user'; content='查询北京未来三天的天气，用摄氏度。请调用工具。' })
  tools = @(@{ type='function'; function=@{
      name = 'get_weather'
      description = 'Get the weather forecast for a city'
      parameters = $schema
      strict = $true } })
  tool_choice = 'auto'
} | ConvertTo-Json -Depth 40 -Compress

$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $r = Invoke-RestMethod -Uri $uri -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
} catch {
    $m = if ($_.ErrorDetails) { $_.ErrorDetails.Message } else { $_.Exception.Message }
    Write-Host "HTTP 错误: $m"
    exit 1
}
$sw.Stop()
$msg = $r.choices[0].message
Write-Host ("=== {0:N1}s  finish={1}" -f $sw.Elapsed.TotalSeconds, $r.choices[0].finish_reason)
Write-Host "content  = [$($msg.content)]"
Write-Host "tool_calls = $($msg.tool_calls | ConvertTo-Json -Depth 20 -Compress)"

$pass = 0; $fail = 0
function C($ok, $label) {
    if ($ok) { $script:pass++; Write-Host "  [OK  ] $label" }
    else { $script:fail++; Write-Host "  [BAD ] $label" }
}

$calls = @($msg.tool_calls)
C ($calls.Count -eq 1) "恰好一次工具调用"
if ($calls.Count -ge 1) {
    C ($calls[0].function.name -eq 'get_weather') "工具名是 get_weather"
    $args = $null
    try { $args = $calls[0].function.arguments | ConvertFrom-Json } catch { }
    C ($null -ne $args) "arguments 是合法 JSON"
    if ($args) {
        C ($args.city -in @('北京','上海','广州')) "city 落在 enum 内：$($args.city)"
        C ($args.days -is [long] -or $args.days -is [int]) "days 是整数：$($args.days)"
        C ($args.days -ge 1 -and $args.days -le 5) "days 落在 1..5：$($args.days)"
        C ($args.units -in @('celsius','fahrenheit')) "units 落在 enum 内：$($args.units)"
        C (@($args.PSObject.Properties.Name).Count -eq 3) "没有多余参数"
    }
}
Write-Host ""
Write-Host ("##########  PASS={0}  FAIL={1}  ##########" -f $pass, $fail)
exit $(if ($fail -eq 0) { 0 } else { 1 })

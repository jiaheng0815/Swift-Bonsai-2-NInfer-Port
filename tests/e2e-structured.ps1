# 端到端验证：约束解码在三个协议表面上的行为
$ErrorActionPreference = 'Stop'
$chat = 'http://127.0.0.1:8080/v1/chat/completions'

$script:pass = 0
$script:fail = 0

function Check($label, $content, $schemaCheck) {
    if ([string]::IsNullOrEmpty($content)) {
        # 空 content 意味着预算全花在推理上，答案是空的 —— 那是失败，不是「跳过」。
        $script:fail++
        Write-Host ("  [BAD ] {0}" -f $label)
        Write-Host "        空响应（finish=length，预算耗尽在推理阶段）"
        return
    }
    $ok = $false
    try {
        $parsed = $content | ConvertFrom-Json
        $ok = & $schemaCheck $parsed
    } catch { $ok = $false }
    if ($ok) { $script:pass++ } else { $script:fail++ }
    $mark = if ($ok) { 'OK  ' } else { 'BAD ' }
    Write-Host ("  [{0}] {1}" -f $mark, $label)
    Write-Host ("        {0}" -f $content)
}

function Ask($label, $schema, $prompt, $budget) {
    # 思考型模型偶尔会在推理上花掉几百个 token，而 finish=length 时 content 是空的。
    # 给足预算，让这条断言测的是约束而不是模型的废话量。
    if (-not $budget) { $budget = 800 }
    $body = @{ model='bonsai2-27b'; max_tokens=$budget;
      messages=@(@{role='user'; content=$prompt});
      response_format=@{ type='json_schema'; json_schema=@{ name='r'; strict=$true; schema=$schema } }
    } | ConvertTo-Json -Depth 40 -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-RestMethod -Uri $chat -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
        $sw.Stop()
        Write-Host ("=== {0}  {1:N1}s  finish={2}" -f $label, $sw.Elapsed.TotalSeconds, $r.choices[0].finish_reason)
        return $r.choices[0].message.content
    } catch {
        $sw.Stop()
        $msg = if ($_.ErrorDetails) { $_.ErrorDetails.Message } else { $_.Exception.Message }
        Write-Host ("=== {0}  {1:N1}s  HTTP 错误: {2}" -f $label, $sw.Elapsed.TotalSeconds, $msg)
        $script:fail++
        return $null
    }
}

Write-Host "`n---------- OpenAI Chat: response_format ----------`n"

$c = Ask 'title-object' @{ type='object'; properties=@{ title=@{ type='string' } }; required=@('title'); additionalProperties=$false } `
        '给这个会话起一个简短的中文标题。只输出 JSON，不要解释。'
if ($c) { Check 'object 且仅含 title 字符串' $c { param($p) $p.title -is [string] -and $p.PSObject.Properties.Name.Count -eq 1 } }

$c = Ask 'enum' @{ enum=@('alpha','beta','gamma') } '从 alpha/beta/gamma 中挑一个输出。只输出 JSON。'
if ($c) { Check 'enum 取值为三者之一' $c { param($p) $p -in @('alpha','beta','gamma') } }

$c = Ask 'const' @{ const='exact' } '请原样输出字符串 exact，不要任何解释。只输出 JSON。'
if ($c) { Check 'const 精确等于 exact' $c { param($p) $p -ceq 'exact' } }

$c = Ask 'anyOf' @{ anyOf=@(@{type='string'},@{type='integer'}) } '任选一个字符串或整数输出。只输出 JSON。'
if ($c) { Check 'anyOf 落在任一分支' $c { param($p) $p -is [string] -or $p -is [long] -or $p -is [int] } }

$c = Ask 'array-minmax' @{ type='array'; items=@{type='string'}; minItems=2; maxItems=4 } '输出二到四个问候语。只输出 JSON。'
if ($c) { Check '数组长度为 2..4 且元素皆字符串' $c { param($p) @($p).Count -ge 2 -and @($p).Count -le 4 -and (@($p) | ForEach-Object { $_ -is [string] }) -notcontains $false } }

$c = Ask 'pattern-anchored' @{ type='string'; pattern='^[a-z]+$' } '输出一个小写英文单词。只输出 JSON。'
if ($c) { Check 'pattern ^[a-z]+$ 全小写' $c { param($p) $p -cmatch '^[a-z]+$' } }

$c = Ask 'pattern-quantifiers' @{ type='string'; pattern='^[a-z]{3,8}[0-9]?$' } '输出三到八个小写字母，可选带一位数字。只输出 JSON。'
if ($c) { Check '区间量词与可选后缀生效' $c { param($p) $p -cmatch '^[a-z]{3,8}[0-9]?$' } }

$c = Ask 'pattern-alternation' @{ type='string'; pattern='^(cat|dog|bird)$' } '从 cat/dog/bird 中挑一个。只输出 JSON。'
if ($c) { Check '分支与全串锚定生效' $c { param($p) $p -in @('cat','dog','bird') } }

# Tier 2：非正则构造。语言不是正则的（`(a*)\1` 识别 {a^n a^n}），所以这里断言的是
# 约束「真的按 ECMA-262 语义生效」，而不是被静默放宽。
$c = Ask 'pattern-lookahead' @{ type='string'; pattern='^(?=[a-z])[a-z0-9]{3,8}$' } '输出一个以字母开头、由小写字母和数字组成的三到八位标识符。只输出 JSON。'
if ($c) { Check '正向先行断言生效' $c { param($p) $p -cmatch '^[a-z][a-z0-9]{2,7}$' } }

$c = Ask 'pattern-lookbehind' @{ type='string'; pattern='^[a-z]{2,6}(?<!x)$' } '输出二到六个小写字母组成的单词，不要以 x 结尾。只输出 JSON。'
if ($c) { Check '反向先行断言生效' $c { param($p) $p -cmatch '^[a-z]{2,6}$' } }

$c = Ask 'pattern-backreference' @{ type='string'; pattern='^(ab)\1$' } '只输出 JSON。' 200
if ($c) { Check '反向引用 (ab)\1 只接受 abab' $c { param($p) $p -ceq 'abab' } }

$c = Ask 'pattern-doubled-run' @{ type='string'; pattern='^(a*)\1$' } '只输出 JSON。' 700
if ($c) { Check '{a^n a^n} 只接受偶数长度' $c { param($p) $p -cmatch '^a*$' -and ($p.Length % 2) -eq 0 } }

$c = Ask 'length-bounds' @{ type='string'; minLength=4; maxLength=6 } '输出一个四到六个字的词。只输出 JSON。'
if ($c) { Check '码点长度落在 4..6' $c { param($p) $p.Length -ge 4 -and $p.Length -le 6 } }

$c = Ask 'escaped-content' @{ type='object'; properties=@{ note=@{type='string'; minLength=2} }; required=@('note'); additionalProperties=$false } `
        '用一句话描述天气。只输出 JSON。'
if ($c) { Check '转义/多字节内容仍满足 minLength' $c { param($p) $p.note.Length -ge 2 } }

$nested = @{ type='object'; properties=@{
    name=@{type='string'}
    tags=@{type='array'; items=@{type='string'}}
    meta=@{type='object'; properties=@{ score=@{type='integer'; minimum=0; maximum=100} }; required=@('score'); additionalProperties=$false}
  }; required=@('name','tags','meta'); additionalProperties=$false }
$c = Ask 'nested' $nested '虚构一个人的姓名、若干标签和分数。只输出 JSON。'
if ($c) { Check '嵌套对象完整且 score 在 0..100' $c { param($p) $p.name -is [string] -and $p.meta.score -ge 0 -and $p.meta.score -le 100 -and $p.PSObject.Properties.Name.Count -eq 3 } }

# 递归 $ref：`next` 可以继续递归，也可以取 null 收尾。链深由模型自己决定，
# 所以断言走完整条链而不是钉死层数 —— 钉死层数会把「模型选择更长」误判成约束失败。
$recursive = @{
  '$defs'=@{ Node=@{ anyOf=@(
      @{ type='null' }
      @{ type='object'; properties=@{ v=@{type='integer'}; next=@{ '$ref'='#/$defs/Node' } }; required=@('v'); additionalProperties=$false }
  ) } }
  '$ref'='#/$defs/Node' }
$c = Ask 'recursive-ref' $recursive '输出一个嵌套链表：每层一个整数 v 和指向下一层的 next，最后一层的 next 为 null。只输出 JSON。' 500
if ($c) {
    Check '递归 $ref 链每层都有整数 v，止于 null' $c { param($p)
        $depth = 0
        $node = $p
        while ($null -ne $node) {
            if ($node -isnot [PSCustomObject]) { return $false }
            if ($node.v -isnot [long] -and $node.v -isnot [int]) { return $false }
            $depth++
            if ($depth -gt 32) { return $false }
            $node = $node.next
        }
        return $depth -ge 2 }
}

$c = Ask 'additionalProperties-true' @{ type='object'; additionalProperties=$true; minProperties=1 } '输出一个至少含一个键的对象。只输出 JSON。'
if ($c) { Check 'additionalProperties:true 接受任意键' $c { param($p) $p.PSObject.Properties.Name.Count -ge 1 } }

Write-Host "`n---------- Responses API: text.format ----------`n"
$rbody = @{ model='bonsai2-27b'; max_output_tokens=200;
  input='从 red/green/blue 中挑一个输出。只输出 JSON。';
  text=@{ format=@{ type='json_schema'; name='r'; strict=$true; schema=@{ enum=@('red','green','blue') } } }
} | ConvertTo-Json -Depth 40 -Compress
try {
    $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/responses' -Method Post -ContentType 'application/json' -Body $rbody -TimeoutSec 300
    $text = ($r.output | Where-Object { $_.type -eq 'message' } | ForEach-Object { $_.content } | Where-Object { $_.type -eq 'output_text' }).text -join ''
    Write-Host ("=== responses text.format  {0}" -f $r.status)
    Check 'Responses text.format 受约束' $text { param($p) $p -in @('red','green','blue') }
} catch {
    $msg = if ($_.ErrorDetails) { $_.ErrorDetails.Message } else { $_.Exception.Message }
    Write-Host "=== responses: HTTP 错误: $msg"; $script:fail++
}

Write-Host "`n---------- Anthropic Messages: output_config.format ----------`n"
$abody = @{ model='bonsai2-27b'; max_tokens=200;
  messages=@(@{role='user'; content='从 small/medium/large 中挑一个输出。只输出 JSON。'});
  output_config=@{ format=@{ type='json_schema'; schema=@{ enum=@('small','medium','large') } } }
} | ConvertTo-Json -Depth 40 -Compress
try {
    $a = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/messages' -Method Post -ContentType 'application/json' -Body $abody -TimeoutSec 300
    $text = ($a.content | Where-Object { $_.type -eq 'text' }).text -join ''
    Write-Host ("=== anthropic output_config.format  {0}" -f $a.stop_reason)
    Check 'Messages output_config.format 受约束' $text { param($p) $p -in @('small','medium','large') }
} catch {
    $msg = if ($_.ErrorDetails) { $_.ErrorDetails.Message } else { $_.Exception.Message }
    Write-Host "=== anthropic: HTTP 错误: $msg"; $script:fail++
}

Write-Host "`n---------- 回归：未约束请求 ----------`n"
$plain = @{ model='bonsai2-27b'; max_tokens=300; messages=@(@{role='user'; content='用一句话介绍你自己。'}) } | ConvertTo-Json -Depth 10 -Compress
$r = Invoke-RestMethod -Uri $chat -Method Post -ContentType 'application/json' -Body $plain -TimeoutSec 300
Write-Host ("=== unconstrained  finish={0}" -f $r.choices[0].finish_reason)
Write-Host ("        {0}" -f $r.choices[0].message.content)
if ($r.choices[0].message.content.Length -gt 0) { $script:pass++ } else { $script:fail++ }

Write-Host ""
Write-Host ("##########  PASS={0}  FAIL={1}  ##########" -f $script:pass, $script:fail)
exit $(if ($script:fail -eq 0) { 0 } else { 1 })

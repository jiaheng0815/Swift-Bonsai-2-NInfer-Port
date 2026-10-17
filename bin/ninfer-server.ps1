# llama.cpp-flavoured launcher for the NInfer server.
#
# NInfer's own flags are kept, and the llama.cpp spellings most people have in muscle
# memory are translated onto them:
#
#   -m <model>        the .ninfer artifact (a .gguf is accepted and swapped for the
#                     artifact built from it, because NInfer runs artifacts, not GGUFs)
#   -mm <mmproj>      accepted and checked; the Vision tower lives INSIDE the artifact,
#                     so nothing is loaded from this path
#   -c N              --max-context N
#   -ctk / -ctv T     --kv-dtype <T>   (q4_0 -> nvfp4, q8_0 -> int8, f16 -> bf16, ...)
#   --parallel N      --max-concurrency N
#   --prefill-chunk N --prefill-chunk N (default 8192 = measured ceiling on this 16 GB
#                     card; 9216 drops below the 400 MiB VRAM safety line, see
#                     docs/TUNING.md 2)
#   --host / --port   passed through
#   --no-mmproj       drop --vision
#
# Anything not listed is passed to ninfer-serve.exe unchanged.  --dry-run prints the
# resolved command instead of starting the engine.
#
# NOTE: keep this file UTF-8 **with BOM**. Windows PowerShell 5.1 decodes BOM-less
# UTF-8 as ANSI, the Chinese strings below then swallow their closing quote, and the
# whole script dies with "MissingArrayIndexExpression". Do not strip the BOM.

$ErrorActionPreference = 'Stop'

$Root     = 'D:\ai-ninfer'
$Artifact = Join-Path $Root 'artifacts\bonsai2-27b-pruned.ninfer'
$Exe      = Join-Path $Root 'build\b1\apps\ninfer-serve.exe'

$model = $null
$mmproj = $null
$ctx = '262144'
$dtype = $null
$kvK = $null
$kvV = $null
$parallel = '1'
$prefillChunk = '8192'
$hostAddr = '127.0.0.1'
$port = '8080'
$vision = $true
$dryRun = $false
$force = $false
$extra = New-Object System.Collections.Generic.List[string]
$notes = New-Object System.Collections.Generic.List[string]

function Convert-CacheType([string] $t) {
  switch ($t.ToLower()) {
    'q4_0'   { return 'nvfp4' }
    'q4_1'   { return 'nvfp4' }
    'iq4_nl' { return 'nvfp4' }
    'q5_0'   { return 'nvfp4' }
    'q5_1'   { return 'nvfp4' }
    'q8_0'   { return 'int8'  }
    'q8_1'   { return 'int8'  }
    'int8'   { return 'int8'  }
    'f16'    { return 'bf16'  }
    'fp16'   { return 'bf16'  }
    'bf16'   { return 'bf16'  }
    'fp8'    { return 'fp8'   }
    'nvfp4'  { return 'nvfp4' }
    'k8v4'   { return 'k8v4'  }
    'rk4v4'  { return 'rk4v4' }
    'rk4v4-e8' { return 'rk4v4-e8' }
    default  { return $null }
  }
}

$argv = @($args)
$i = 0
while ($i -lt $argv.Count) {
  $a = [string] $argv[$i]
  $next = if ($i + 1 -lt $argv.Count) { [string] $argv[$i + 1] } else { $null }

  if ($a -eq '-m' -or $a -eq '--model') {
    $model = $next; $i += 2
  } elseif ($a -eq '-mm' -or $a -eq '--mmproj') {
    $mmproj = $next; $i += 2
  } elseif ($a -eq '-c' -or $a -eq '--ctx-size' -or $a -eq '--max-context') {
    $ctx = $next; $i += 2
  } elseif ($a -eq '-ctk' -or $a -eq '--cache-type-k') {
    $kvK = $next; $i += 2
  } elseif ($a -eq '-ctv' -or $a -eq '--cache-type-v') {
    $kvV = $next; $i += 2
  } elseif ($a -eq '--parallel' -or $a -eq '-np') {
    $parallel = $next; $i += 2
  } elseif ($a -eq '--prefill-chunk') {
    $prefillChunk = $next; $i += 2
  } elseif ($a -eq '--host') {
    $hostAddr = $next; $i += 2
  } elseif ($a -eq '--port') {
    $port = $next; $i += 2
  } elseif ($a -eq '--no-mmproj' -or $a -eq '--no-vision') {
    $vision = $false; $i += 1
  } elseif ($a -eq '--dry-run') {
    $dryRun = $true; $i += 1
  } elseif ($a -eq '--force' -or $a -eq '-f') {
    $force = $true; $i += 1
  } elseif ($a -eq '--temp') {
    $extra.Add('--temperature'); if ($next) { $extra.Add($next) }; $i += 2
  } elseif ($a -eq '--reasoning-preserve') {
    $extra.Add('--preserve-thinking'); $i += 1
  } elseif ($a -eq '--reasoning-effort') {
    # The server has no such startup flag: effort is a per-request field, and the model's
    # own default is already XHigh.
    if ($next -and $next.ToLower() -ne 'xhigh') {
      $notes.Add("--reasoning-effort $next is per-request on this engine (server default is xhigh); drop it here and set reasoning_effort=$next in the request body")
    } else {
      $notes.Add('--reasoning-effort xhigh is already the engine default; dropped')
    }
    if ($next) { $i += 2 } else { $i += 1 }
  } elseif (@('-ngl', '--n-gpu-layers', '--gpu-layers', '-t', '--threads', '-fa',
              '--flash-attn', '--no-warmup', '--mlock', '--jinja',
              '--chat-template-file', '--webui-mcp-proxy', '--image-min-tokens',
              '--no-webui', '--mmap', '--numa', '-cb', '--cont-batching') -contains $a) {
    $notes.Add("ignored: $a" + $(if ($next) { " $next" } else { '' }))
    $i += 2
  } else {
    $extra.Add($a); $i += 1
  }
}

# --- resolve the model ------------------------------------------------------
if (-not $model) { $model = $Artifact }
if ($model -match '\.gguf$') {
  $notes.Add("model argument is a GGUF; NInfer runs .ninfer artifacts, so $Artifact is used instead")
  $model = $Artifact
}
if (-not (Test-Path -LiteralPath $model)) { throw "artifact not found: $model" }
if (-not (Test-Path -LiteralPath $Exe))    { throw "engine not built: $Exe (run build.cmd)" }

# --- one engine per GPU -----------------------------------------------------
# Two engines on a 16 GB card do not just fight over VRAM: the loser also fails to
# pin its host state ("cudaMallocHost failed: ... out of memory"), which reads like a
# RAM bug when it is really "you already have one running".
if (-not $force) {
  $rivals = @(Get-Process -ErrorAction SilentlyContinue |
              Where-Object { $_.ProcessName -match '^(ninfer-serve|ninfer|llama-server|llama_server)' })
  if ($rivals.Count -gt 0) {
    Write-Host '[ninfer-server] 已经有一个引擎在跑了，先停掉再启动：' -ForegroundColor Red
    foreach ($r in $rivals) {
      Write-Host ("                   PID {0,-7} {1}   RAM {2:N0} MB" -f $r.Id, $r.ProcessName, ($r.WorkingSet64 / 1MB))
    }
    Write-Host '[ninfer-server] 停止： taskkill /IM ninfer-serve.exe /F   (或双击 stop-server.cmd)' -ForegroundColor Yellow
    Write-Host '[ninfer-server] 停完用 nvidia-smi 确认显存回到 1 GB 左右基线。' -ForegroundColor Yellow
    Write-Host '[ninfer-server] 确实要同时跑两个，加 --force。' -ForegroundColor DarkYellow
    exit 1
  }
}

# --- resolve the KV dtype ---------------------------------------------------
$fromK = if ($kvK) { Convert-CacheType $kvK } else { $null }
$fromV = if ($kvV) { Convert-CacheType $kvV } else { $null }
if ($kvK -and -not $fromK) { $notes.Add("unknown -ctk '$kvK'; falling back to nvfp4") }
if ($kvV -and -not $fromV) { $notes.Add("unknown -ctv '$kvV'; falling back to nvfp4") }
$rank = @{ 'nvfp4' = 4; 'rk4v4' = 4; 'rk4v4-e8' = 4; 'fp8' = 8; 'int8' = 8; 'k8v4' = 12; 'bf16' = 16 }
if ($fromK -and $fromV -and $fromK -ne $fromV) {
  $picked = if ($rank[$fromK] -le $rank[$fromV]) { $fromK } else { $fromV }
  $notes.Add("NInfer has one KV dtype for both K and V; -ctk $kvK / -ctv $kvV -> $picked")
  $dtype = $picked
} elseif ($fromK) { $dtype = $fromK }
elseif ($fromV)   { $dtype = $fromV }
else              { $dtype = 'nvfp4' }

# --- the prefill chunk ------------------------------------------------------
# A VRAM knob, not a speed knob: 1024 -> 8192 all prefill at ~650 tok/s; what
# grows is the runtime reservation (~0.16 MiB per chunk token).  Measured on
# this 16 GB card (docs/TUNING.md 2): 7168 -> 674 MiB free, 8192 -> 527 MiB,
# 9216 -> 382 MiB (under the 400 MiB safety line), 16384 -> fails to start.
if ($prefillChunk -match '^\d+$') {
  $pcValue = [int] $prefillChunk
  if ($pcValue % 128 -ne 0) {
    $notes.Add("--prefill-chunk $prefillChunk is not a multiple of 128; the engine will refuse it")
  }
  if ($pcValue -gt 8192) {
    $notes.Add("--prefill-chunk $prefillChunk is above the measured ceiling for this 16 GB card (8192); the free-VRAM safety line will be crossed")
  }
} else {
  $notes.Add("--prefill-chunk '$prefillChunk' is not a number")
}

# --- the mmproj check -------------------------------------------------------
if ($mmproj) {
  if (-not (Test-Path -LiteralPath $mmproj)) {
    $notes.Add("mmproj not found (and not needed): $mmproj")
  } elseif ($mmproj -notmatch 'BF16') {
    $notes.Add("mmproj '$([IO.Path]::GetFileName($mmproj))' is not the BF16 tower; the artifact carries the BF16 tower already, so this path is ignored")
  } else {
    $notes.Add('mmproj accepted: the artifact already contains that same BF16 Vision tower')
  }
}

# --- build the command ------------------------------------------------------
$serveArgs = New-Object System.Collections.Generic.List[string]
$serveArgs.Add($model)
$serveArgs.Add('--host');   $serveArgs.Add($hostAddr)
$serveArgs.Add('--port');   $serveArgs.Add($port)
$serveArgs.Add('--model-id'); $serveArgs.Add('bonsai2-27b')
$serveArgs.Add('--max-context'); $serveArgs.Add($ctx)
# Explicit capacity: `auto` demands 1 GiB of headroom on top of the reservation
# and cannot start at 262144 (docs/TUNING.md 5).
$serveArgs.Add('--kv-capacity'); $serveArgs.Add($ctx)
$serveArgs.Add('--kv-dtype'); $serveArgs.Add($dtype)
$serveArgs.Add('--prefill-chunk'); $serveArgs.Add($prefillChunk)
$serveArgs.Add('--max-concurrency'); $serveArgs.Add($parallel)
$serveArgs.Add('--spec'); $serveArgs.Add('mtp')
$serveArgs.Add('--draft-tokens'); $serveArgs.Add('3')
$serveArgs.Add('--lm-head-draft')
$serveArgs.Add('--host-kv-mib'); $serveArgs.Add('1024')
if ($vision) { $serveArgs.Add('--vision') }
$serveArgs.Add('--cors')
foreach ($e in $extra) { $serveArgs.Add($e) }

foreach ($n in $notes) { Write-Host "[ninfer-server] $n" -ForegroundColor DarkYellow }
Write-Host "[ninfer-server] $Exe $($serveArgs -join ' ')" -ForegroundColor DarkGray

if ($dryRun) { exit 0 }
& $Exe @serveArgs
exit $LASTEXITCODE

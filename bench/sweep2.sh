#!/usr/bin/env bash
# 第二轮：chunk 细化 + 解码参数（draft-tokens / lm-head-draft）+ 制品 A/B
set -u
cd /d/ai-ninfer/logs/tune
BASE="--host-kv-mib 0 --host-state-slots 0 --device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0"

run () {
  local label="$1"; shift
  echo "=========== $label ==========="
  python bench.py --label "$label" --prefill-tokens 12288 "$@" > "$label.out" 2>&1 || true
  python analyze.py | grep -E "^$label|^label" || true
}

# C. chunk 细化（找出"够用且安全"的上限）
run "chunk-8192"  --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 8192 $BASE
run "chunk-10240" --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 10240 $BASE
run "chunk-12288" --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 12288 $BASE

# B. 解码参数（固定 chunk 8192）
run "draft-2" --spec mtp --draft-tokens 2 --lm-head-draft --vision --prefill-chunk 8192 $BASE
run "draft-4" --spec mtp --draft-tokens 4 --lm-head-draft --vision --prefill-chunk 8192 $BASE
run "draft-5" --spec mtp --draft-tokens 5 --lm-head-draft --vision --prefill-chunk 8192 $BASE
run "no-lmhead" --spec mtp --draft-tokens 3 --vision --prefill-chunk 8192 $BASE

# D. 无 vision（省内存换更大 chunk 的可行性）
run "novision-8192" --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 8192 $BASE
run "novision-16384" --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 16384 $BASE
echo "=========== DONE ==========="
python analyze.py

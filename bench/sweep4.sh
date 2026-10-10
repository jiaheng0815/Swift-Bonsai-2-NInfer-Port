#!/usr/bin/env bash
# 第四轮：确定性解码 A/B（greedy）+ chunk 归因 + 复现性 + 前缀缓存
set -u
cd /d/ai-ninfer/logs/tune
BASE="--host-kv-mib 0 --host-state-slots 0 --device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0"

run () {
  local label="$1"; shift
  echo "=========== $label ==========="
  python bench.py --label "$label" --prefill-tokens 4096 "$@" > "$label.out" 2>&1 || true
  python analyze.py | grep -E "^$label|^label" || true
  python - "$label" <<'PY'
import json,sys
sys.path.insert(0,'.')
from bench import parse_log
label=sys.argv[1]
info=parse_log(f"{label}.log")
for r in info['requests']:
    if r['prompt']<200 and r['output']>=100:
        acc = f"{r['mtp'][0]}/{r['mtp'][1]} ({r['mtp'][0]/r['mtp'][1]*100:.0f}%)" if r['mtp'] else '-'
        print(f"   解码: {r['output']} tok | {r['decode_tps']} tok/s | mtp {acc}")
PY
}

# 确定性（greedy）解码 A/B：K=1/2/3 × lm-head-draft 开/关
run "g-k1"     --greedy --spec mtp --draft-tokens 1 --lm-head-draft --vision --prefill-chunk 8192 $BASE
run "g-k2"     --greedy --spec mtp --draft-tokens 2 --lm-head-draft --vision --prefill-chunk 8192 $BASE
run "g-k3"     --greedy --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 8192 $BASE
run "g-k3-nolm" --greedy --spec mtp --draft-tokens 3 --vision --prefill-chunk 8192 $BASE
run "g-k2-nolm" --greedy --spec mtp --draft-tokens 2 --vision --prefill-chunk 8192 $BASE

# chunk 归因：大 chunk + 充足显存（KV 降到 131072，腾出 ~2 GiB）
run "nomem-12288" --greedy --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 12288 \
    --max-context 131072 --kv-capacity 131072 $BASE

# 复现性：chunk 8192 重跑一次
run "chunk-8192-r2" --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 8192 $BASE
echo "=========== DONE ==========="

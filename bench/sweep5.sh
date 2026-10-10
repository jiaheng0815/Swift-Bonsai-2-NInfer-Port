#!/usr/bin/env bash
# 第五轮：greedy 下 K=2/K=3 各跑 3 次（确定性，可比）+ 前缀缓存验证
set -u
cd /d/ai-ninfer/logs/tune
BASE="--host-kv-mib 0 --host-state-slots 0 --device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0"

dec () {  # label
  python - "$1" <<'PY'
import sys; sys.path.insert(0,'.')
from bench import parse_log
info=parse_log(f"{sys.argv[1]}.log")
for r in info['requests']:
    if r['prompt']<200 and r['output']>=100:
        acc = f"{r['mtp'][0]}/{r['mtp'][1]} ({r['mtp'][0]/r['mtp'][1]*100:.1f}%)" if r['mtp'] else '-'
        print(f"   {sys.argv[1]}: decode {r['decode_tps']} tok/s | mtp {acc}")
PY
}

for n in r2 r3; do
  for k in 2 3; do
    label="gk${k}-${n}"
    echo "=========== $label ==========="
    python bench.py --label "$label" --prefill-tokens 1024 --greedy --spec mtp --draft-tokens $k \
      --lm-head-draft --vision --prefill-chunk 8192 $BASE > "$label.out" 2>&1 || true
    dec "$label"
  done
done

echo "=========== 前缀缓存 + 最终配置 ==========="
python prefix_test.py --label prefix-final --greedy --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --prefill-chunk 8192 --kv-capacity 262144 --max-context 262144 --max-concurrency 1 \
  --kv-dtype nvfp4 $BASE
echo "=========== DONE ==========="
python analyze.py

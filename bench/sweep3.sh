#!/usr/bin/env bash
# 第三轮：状态槽成本 / kv-capacity auto 行为 / 原始制品 A/B / 长上下文
set -u
cd /d/ai-ninfer/logs/tune
BASE="--host-kv-mib 0 --host-state-slots 0 --device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0"

run () {
  local label="$1"; shift
  echo "=========== $label ==========="
  python bench.py --label "$label" --prefill-tokens 12288 "$@" > "$label.out" 2>&1 || true
  python analyze.py | grep -E "^$label|^label" || true
}

# 状态槽的显存成本（去掉 1 个 cached device state）
run "state0-8192" --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 8192 \
    --device-state-slots 0 $BASE

# kv-capacity auto 会不会把 262144 砍掉
run "kvauto-7168" --kv-capacity auto --spec mtp --draft-tokens 3 --lm-head-draft --vision \
    --prefill-chunk 7168 --host-kv-mib 0 --host-state-slots 0 --device-state-slots 1 \
    --max-private-continuations 1 --max-shared-prefixes 0

# 原始（未裁剪）制品的 MTP 接受率对比
run "orig-8192" --artifact "D:\\ai-ninfer\\artifacts\\bonsai2-27b.ninfer" \
    --spec mtp --draft-tokens 3 --lm-head-draft --vision --prefill-chunk 8192 $BASE

# 长上下文：100k 级 prefill + 深度解码（验证 262k 设置下真能用）
run "long-100k" --prefill-tokens 78000 --spec mtp --draft-tokens 3 --lm-head-draft --vision \
    --prefill-chunk 8192 $BASE
echo "=========== DONE ==========="
python analyze.py

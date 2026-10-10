#!/usr/bin/env bash
# 第二轮：固定最佳 chunk，扫描 draft-tokens / lm-head-draft / vision
set -u
cd /d/ai-ninfer/logs/tune
CHUNK="${1:-16384}"
BASE="--host-kv-mib 0 --host-state-slots 0 --device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0"

run () {  # label, extra args
  local label="$1"; shift
  echo "=========== $label ==========="
  python bench.py --label "$label" --prefill-tokens 12288 "$@" > "$label.out" 2>&1
  python - "$label" <<'PY'
import json,sys
label=sys.argv[1]
rows=[json.loads(l) for l in open('results.jsonl',encoding='utf-8')]
r=[x for x in rows if x['label']==label][-1]
reqs=r.get('requests') or []
pre=max(reqs,key=lambda x:x['prompt']) if reqs else {}
dec=[x for x in reqs if x['prompt']<100 and x['output']>50]
print(f"{label}: started={r['started']} gpu={r.get('gpu_after_load_mib')}/{r.get('gpu_total_mib')} "
      f"slack={r.get('slack_mib')} kv={r.get('kv_tokens')} "
      f"prefill={pre.get('prefill_tps')} tok/s (prompt {pre.get('prompt')}) "
      f"decode={dec[-1].get('decode_tps') if dec else None} mtp={dec[-1].get('mtp') if dec else None}")
PY
}

run "draft-2" --spec mtp --draft-tokens 2 --lm-head-draft --vision --prefill-chunk "$CHUNK" $BASE
run "draft-4" --spec mtp --draft-tokens 4 --lm-head-draft --vision --prefill-chunk "$CHUNK" $BASE
run "draft-5" --spec mtp --draft-tokens 5 --lm-head-draft --vision --prefill-chunk "$CHUNK" $BASE
run "no-lmhead-draft" --spec mtp --draft-tokens 3 --vision --prefill-chunk "$CHUNK" $BASE
run "novision" --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk "$CHUNK" $BASE
echo "=========== DONE ==========="

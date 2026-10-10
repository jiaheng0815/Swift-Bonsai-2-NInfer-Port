#!/usr/bin/env bash
# prefill-chunk 扫描：固定其余参数，只改 --prefill-chunk
set -u
cd /d/ai-ninfer/logs/tune
COMMON="--spec mtp --draft-tokens 3 --lm-head-draft --host-kv-mib 0 --host-state-slots 0 \
--device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0 --vision"
for C in 1024 2048 4096 7168 16384 32768; do
  echo "=========== chunk=$C ==========="
  python bench.py --label "chunk-$C" --prefill-tokens 16384 $COMMON --prefill-chunk $C > "chunk-$C.out" 2>&1
  python - "$C" <<'PY'
import json,sys
c=sys.argv[1]
rows=[json.loads(l) for l in open('results.jsonl',encoding='utf-8')]
r=[x for x in rows if x['label']==f'chunk-{c}'][-1]
reqs=r.get('requests') or []
pre=max(reqs,key=lambda x:x['prompt']) if reqs else {}
dec=[x for x in reqs if x['prompt']<100 and x['output']>50]
print(f"chunk={c} started={r['started']} gpu={r.get('gpu_after_load_mib')} slack={r.get('slack_mib')} "
      f"kv={r.get('kv_tokens')} prefill={pre.get('prefill_tps')} tok/s (prompt {pre.get('prompt')}) "
      f"decode={dec[-1].get('decode_tps') if dec else None} mtp={dec[-1].get('mtp') if dec else None}")
PY
done
echo "=========== DONE ==========="

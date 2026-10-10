#!/usr/bin/env bash
# ============================================================
#  Swift Bonsai 2 27B on NInfer — Linux 启动模板
#
#  ⚠️ 未实测：本仓的实测环境是 Windows 11 + RTX 5060 Ti（见 docs/BENCHMARKS.md）。
#  Linux 侧参考社区仓 mengyou6688/Bonsai-2-27B-PQ2_0-CRACK-NInfer（Ubuntu）。
#  引擎需按本机 GPU 架构编译（见 engine/README.md），并把 ninfer-serve 放进 PATH。
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARTIFACT="${ARTIFACT:-$ROOT/artifacts/swift-bonsai2-27b-pruned.ninfer}"
EXE="${EXE:-$ROOT/build/b1/apps/ninfer-serve}"
MODELID="${MODELID:-swift-bonsai2-27b}"
PORT="${PORT:-8080}"
CTX="${CTX:-262144}"          # 262144 = 模型原生上限（nvfp4 KV 下要 4 GiB）
KVTYPE="${KVTYPE:-nvfp4}"     # nvfp4 | fp8 | bf16 | int8 | k8v4
PREFILL="${PREFILL:-8192}"
DRAFT="${DRAFT:-3}"

[[ -f "$ARTIFACT" ]] || { echo "artifact not found: $ARTIFACT"; exit 1; }
[[ -x "$EXE" ]] || { echo "engine binary not found: $EXE"; exit 1; }

echo "endpoint: http://127.0.0.1:$PORT/v1/models   (model id: $MODELID)"

exec "$EXE" "$ARTIFACT" \
  --host 127.0.0.1 --port "$PORT" \
  --model-id "$MODELID" \
  --max-context "$CTX" --kv-capacity "$CTX" --kv-dtype "$KVTYPE" \
  --prefill-chunk "$PREFILL" \
  --max-concurrency 1 \
  --spec mtp --draft-tokens "$DRAFT" \
  --lm-head-draft \
  --device-state-slots 1 --host-state-slots 0 --host-kv-mib 0 \
  --max-private-continuations 1 --max-shared-prefixes 0 \
  --vision \
  --cors

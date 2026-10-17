"""打印制品 frontend 资源与 pruned 目录的 SHA256（前 16 位），用于比对。

用法：python frontend_hashes.py <artifact.ninfer> [--pruned-dir <dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(os.environ.get("NINFER_SRC") or _REPO_ROOT / "src" / "cranebw")))
from tools.artifact.container import Artifact  # noqa: E402

NAMES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/generation_config.json",
)


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--pruned-dir", type=Path, default=None)
    args = ap.parse_args()

    with Artifact.open(args.artifact) as a:
        print("identity:", a.identity)
        print("objects :", len(a.objects))
        for n in NAMES:
            print(f"  {h(bytes(a.payload(n)))}  {n}")

    if args.pruned_dir:
        print("--- pruned dir ---")
        for n in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "vocab_map.json"):
            p = args.pruned_dir / n
            if p.exists():
                print(f"  {h(p.read_bytes())}  {n}")
        plan_path = args.pruned_dir / "vocab_map.json"
        if plan_path.exists():
            d = json.loads(plan_path.read_text(encoding="utf-8"))
            print("  new_token_count:", d["new_token_count"],
                  "kept:", len(d["kept_vocab_ids"]),
                  "draft ids:", len(d["draft_new_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

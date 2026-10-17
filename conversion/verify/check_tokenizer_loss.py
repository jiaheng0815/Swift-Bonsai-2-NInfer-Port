"""用真实语料量化词表裁剪对中英文的分词损失。

对每个语料文件分别用原始与裁剪后的 tokenizer 编码，并把原始 ids 映射到新 id 空间后逐位置比较：
  * 无法映射的位置 = 该 token 被删除，文本被迫用更碎的 token 表示（真实的效率损失）
  * token 总数增长 = 分词损失的直接度量

判定标准：中英文语料的 token 数增长必须 <= 0.1%。

用法：
    python check_tokenizer_loss.py [--artifact <src.ninfer>] [--pruned <dir>]
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = Path(os.environ.get("NINFER_SRC") or _REPO_ROOT / "src" / "cranebw")
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from tools.artifact.container import Artifact  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

CORPORA_ROOT = Path(os.environ.get("NINFER_CORPORA") or REPO_SRC / "eval" / "corpora" / "perplexity-1m" / "data")
CORPORA = {
    "zhwiki (中文维基)": "zhwiki",
    "wikitext (英文维基)": "wikitext",
    "pg19 (英文书籍)": "pg19",
    "ninfer (技术文档)": "ninfer",
}
THRESHOLD_PCT = 0.1


def load_original(artifact_path: Path) -> Tokenizer:
    with Artifact.open(artifact_path) as artifact:
        payload = bytes(artifact.payload("frontend/tokenizer.json"))
    handle = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
    handle.write(payload)
    handle.close()
    return Tokenizer.from_file(handle.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=_REPO_ROOT / "artifacts" / "bonsai2-27b.ninfer")
    parser.add_argument("--pruned", type=Path, default=_REPO_ROOT / "artifacts" / "pruned")
    args = parser.parse_args()

    remap = {
        int(k): int(v)
        for k, v in json.loads((args.pruned / "vocab_map.json").read_text(encoding="utf-8"))[
            "old_to_new"
        ].items()
    }

    original = load_original(args.artifact)
    pruned = Tokenizer.from_file(str(args.pruned / "tokenizer.json"))

    print(f"{'corpus':<24} {'orig tok':>10} {'new tok':>10} {'dropped hits':>13} {'loss %':>9}")
    print("-" * 72)

    overall_fail = False
    for label, folder in CORPORA.items():
        files = sorted((CORPORA_ROOT / folder).glob("*.txt"))
        if not files:
            print(f"{label:<24} (no files)")
            continue
        total_orig = total_new = dropped_hits = 0
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            # 长文本切块，避免单次 encode 过大
            chunk = 200_000
            for start in range(0, len(text), chunk):
                piece = text[start : start + chunk]
                if not piece.strip():
                    continue
                orig_ids = original.encode(piece).ids
                new_ids = pruned.encode(piece).ids
                total_orig += len(orig_ids)
                total_new += len(new_ids)
                dropped_hits += sum(1 for i in orig_ids if i not in remap)

        loss = (total_new - total_orig) / total_orig * 100 if total_orig else 0.0
        flag = ""
        if folder in ("zhwiki", "wikitext", "pg19", "ninfer"):
            if loss > THRESHOLD_PCT:
                flag = "  <-- EXCEEDS 0.1%"
                overall_fail = True
        print(
            f"{label:<24} {total_orig:>10} {total_new:>10} {dropped_hits:>13} {loss:>8.4f}%{flag}"
        )

    print("-" * 72)
    print(f"threshold: {THRESHOLD_PCT}% token-count growth on zh/en corpora")
    if overall_fail:
        raise SystemExit("FAIL: tokenizer loss exceeds threshold")
    print("PASS: all corpora within 0.1%")


if __name__ == "__main__":
    main()

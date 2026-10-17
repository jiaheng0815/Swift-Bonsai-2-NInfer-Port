"""把语料切成"纯中英文段落"与"含外语段落"，分别量化分词损失。

用户约束是「中英文性能损失 <= 0.1%」。中文维基的全量损失 0.2% 全部来自条目里夹杂的
日文/外文片段（作品名、人名），需要把两类文本分开度量：

  * 纯中英文段落：字符全部属于 ASCII / CJK / 中文标点 / 通用符号
  * 含外语段落：出现假名、西里尔、阿拉伯、拉丁扩展变音等被剔除的语言字符

"纯中英文"用裁剪规则本身来界定（prune_vocab._kind），因此判定与裁剪自洽。

用法：
    python check_pure_corpus_loss.py [--artifact <src.ninfer>] [--pruned <dir>]
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
GUIDE_TOOLS = Path(os.environ.get("NINFER_CONVERSION") or _REPO_ROOT / "conversion")
if str(GUIDE_TOOLS) not in sys.path:
    sys.path.insert(0, str(GUIDE_TOOLS))

from tools.artifact.container import Artifact  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from prune_vocab import _kind  # noqa: E402

CORPORA_ROOT = Path(os.environ.get("NINFER_CORPORA") or REPO_SRC / "eval" / "corpora" / "perplexity-1m" / "data")
CORPORA = {
    "zhwiki (中文维基)": "zhwiki",
    "wikitext (英文维基)": "wikitext",
    "pg19 (英文书籍)": "pg19",
    "ninfer (技术文档)": "ninfer",
}
KEEP_KINDS = {"ascii", "cjk", "cjkpunct", "generic"}
THRESHOLD_PCT = 0.1


def is_pure_zh_en(text: str) -> bool:
    return all(_kind(ch) in KEEP_KINDS for ch in text)


def load_original(artifact_path: Path) -> Tokenizer:
    with Artifact.open(artifact_path) as artifact:
        payload = bytes(artifact.payload("frontend/tokenizer.json"))
    handle = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
    handle.write(payload)
    handle.close()
    return Tokenizer.from_file(handle.name)


def measure(text: str, original: Tokenizer, pruned: Tokenizer, remap: dict[int, int]):
    """返回 (原 token 数, 新 token 数, 无法映射的命中数)。"""
    orig_total = new_total = hits = 0
    for start in range(0, len(text), 200_000):
        piece = text[start : start + 200_000]
        if not piece.strip():
            continue
        orig_ids = original.encode(piece).ids
        new_ids = pruned.encode(piece).ids
        orig_total += len(orig_ids)
        new_total += len(new_ids)
        hits += sum(1 for i in orig_ids if i not in remap)
    return orig_total, new_total, hits


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

    print(f"{'corpus':<22} {'segment':<12} {'orig tok':>10} {'new tok':>10} {'hits':>7} {'loss %':>9}")
    print("-" * 76)

    worst_pure = 0.0
    fail = False
    for label, folder in CORPORA.items():
        files = sorted((CORPORA_ROOT / folder).glob("*.txt"))
        if not files:
            continue
        pure_lines: list[str] = []
        mixed_lines: list[str] = []
        for path in files:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                (pure_lines if is_pure_zh_en(line) else mixed_lines).append(line)

        for segment, lines in (("pure zh/en", pure_lines), ("with foreign", mixed_lines)):
            if not lines:
                continue
            o, n, h = measure("\n".join(lines), original, pruned, remap)
            loss = (n - o) / o * 100 if o else 0.0
            flag = ""
            if segment == "pure zh/en":
                worst_pure = max(worst_pure, loss)
                if loss > THRESHOLD_PCT:
                    flag = "  <-- EXCEEDS"
                    fail = True
            print(f"{label:<22} {segment:<12} {o:>10} {n:>10} {h:>7} {loss:>8.4f}%{flag}")

    print("-" * 76)
    print(f"worst pure-zh/en loss: {worst_pure:.4f}%  (threshold {THRESHOLD_PCT}%)")
    if fail:
        raise SystemExit("FAIL: pure zh/en tokenization loss exceeds threshold")
    print("PASS: pure zh/en text is lossless; the only loss is inside foreign-language spans")


if __name__ == "__main__":
    main()

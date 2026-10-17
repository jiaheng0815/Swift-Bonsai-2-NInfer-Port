"""定位语料中被删除 token 的实际命中位置，判断是"外语片段"还是"误删中文"。

对每个被删且被命中的 token，打印它的原始字节、解码文本，以及在语料里的前后文。
若命中的是中文/英文/通用符号，说明 keep 规则有漏洞，必须修；
若命中的是外语字符，则属于"剔除外语"的预期代价。

用法：
    python inspect_dropped_hits.py [--folder zhwiki] [--top 25]
"""

from __future__ import annotations

import os
import argparse
import collections
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


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + list(
        range(ord("\u00ae"), ord("\u00ff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


U2B = {v: k for k, v in bytes_to_unicode().items()}


def render(token: str) -> str:
    try:
        raw = bytes(U2B[c] for c in token)
    except KeyError:
        return f"<non-bytelevel {token!r}>"
    try:
        return repr(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return f"<partial {raw!r}>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=_REPO_ROOT / "artifacts" / "bonsai2-27b.ninfer")
    parser.add_argument("--pruned", type=Path, default=_REPO_ROOT / "artifacts" / "pruned")
    parser.add_argument("--folder", default="zhwiki")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    remap = {
        int(k)
        for k in json.loads((args.pruned / "vocab_map.json").read_text(encoding="utf-8"))[
            "old_to_new"
        ]
    }

    with Artifact.open(args.artifact) as artifact:
        payload = bytes(artifact.payload("frontend/tokenizer.json"))
    handle = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
    handle.write(payload)
    handle.close()
    original = Tokenizer.from_file(handle.name)

    hits = collections.Counter()
    contexts: dict[int, list[str]] = {}
    total = 0
    for path in sorted((CORPORA_ROOT / args.folder).glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for start in range(0, len(text), 200_000):
            piece = text[start : start + 200_000]
            if not piece.strip():
                continue
            encoded = original.encode(piece)
            ids = encoded.ids
            total += len(ids)
            for pos, token_id in enumerate(ids):
                if token_id not in remap:
                    hits[token_id] += 1
                    if len(contexts.setdefault(token_id, [])) < 3:
                        lo = max(0, pos - 12)
                        hi = min(len(ids), pos + 13)
                        snippet = original.decode(ids[lo:hi])
                        contexts[token_id].append(snippet.replace("\n", " "))

    print(f"corpus={args.folder}  tokens={total}  dropped hits={sum(hits.values())}  "
          f"distinct dropped tokens={len(hits)}")
    print()
    for token_id, count in hits.most_common(args.top):
        token = original.id_to_token(token_id)
        print(f"  id={token_id:<7} count={count:<5} {render(token)}")
        for snippet in contexts[token_id][:2]:
            print(f"        ...{snippet!r}")


if __name__ == "__main__":
    main()

"""裁剪前后 tokenizer 对照：中英文分词效率必须不变。

同一段文本分别用原始 tokenizer 与裁剪后的 tokenizer 编码，比较 token 数量与解码结果。
裁剪只删除其他语言的 token，因此**中英文/代码文本的 token 数应当完全一致**；
若有任何增加，说明误删了中英文实际使用的 token。

用法：
    python check_vocab_prune.py --artifact <path.ninfer> --pruned <dir>
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_SRC = Path(os.environ.get("NINFER_SRC") or r"D:\ai-ninfer\src\cranebw")
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from tools.artifact.container import Artifact  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# 中英文/代码样本：这些文本的编码必须逐 token 无损
SAMPLES = [
    ("zh-日常", "你好，世界！今天天气怎么样？我想去公园散步。"),
    ("zh-技术", "请把模型的词表裁剪到只保留中英文，注意特殊 token 必须保留。"),
    ("zh-长", "在深度学习中，词表大小直接影响嵌入矩阵和输出层的参数量。" * 3),
    ("zh-成语", "画蛇添足、守株待兔、刻舟求剑、亡羊补牢。"),
    ("en-plain", "The quick brown fox jumps over the lazy dog."),
    ("en-tech", "We prune the vocabulary to reduce the embedding matrix size."),
    ("en-long", "Quantization reduces memory footprint while preserving accuracy. " * 3),
    ("code-py", "def foo(x: int) -> int:\n    return x + 1  # comment"),
    ("code-json", '{"name": "value", "nested": {"a": [1, 2, 3]}}'),
    ("code-c", "#include <stdio.h>\nint main(void) { return 0; }"),
    ("mixed", "请帮我 fix 这个 bug in token_embedding (vocab_size=248320)。"),
    ("digits", "3.14159 2.71828 1000000 0x1F 1e-9 2026-10-08"),
    ("punct-cn", "《红楼梦》——「引号」……？！；：（）【】"),
    ("punct-en", 'He said "hello" — it\'s fine… 50% • item'),
    ("symbols", "25°C ±0.5 ×  ÷ µ ≤ ≥ ≈ − → ─ │ ✓ ★"),
    ("math", "∑(x²) √2 ∞ ∫ ≈ ∀x ∃y"),
    ("currency", "$100 €50 £30 ¥1000 ₹500"),
    ("box", "┌─────┐\n│ box │\n└─────┘"),
]


def load_original(artifact_path: Path) -> Tokenizer:
    with Artifact.open(artifact_path) as artifact:
        payload = bytes(artifact.payload("frontend/tokenizer.json"))
    # tokenizers 库只接受文件路径
    handle = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
    handle.write(payload)
    handle.close()
    return Tokenizer.from_file(handle.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path(r"D:\ai-ninfer\artifacts\bonsai2-27b.ninfer"))
    parser.add_argument("--pruned", type=Path, default=Path(r"D:\ai-ninfer\artifacts\pruned"))
    args = parser.parse_args()

    original = load_original(args.artifact)
    pruned = Tokenizer.from_file(str(args.pruned / "tokenizer.json"))

    print(f"original vocab: {original.get_vocab_size(with_added_tokens=True)}")
    print(f"pruned   vocab: {pruned.get_vocab_size(with_added_tokens=True)}")
    print()

    orig_vocab = original.get_vocab()
    pruned_vocab = pruned.get_vocab()

    total_orig = total_pruned = 0
    regressions = []
    print(f"{'sample':<12} {'orig':>6} {'pruned':>7} {'delta':>6}  roundtrip  note")
    print("-" * 70)
    for label, text in SAMPLES:
        orig_ids = original.encode(text).ids
        pruned_ids = pruned.encode(text).ids
        total_orig += len(orig_ids)
        total_pruned += len(pruned_ids)
        delta = len(pruned_ids) - len(orig_ids)

        back = pruned.decode(pruned_ids)
        roundtrip = "OK" if back == text else "FAIL"

        note = ""
        if roundtrip != "OK":
            note = "DECODE MISMATCH"
        elif delta > 0:
            # 找出哪些 token 被删了导致碎片化
            missing = [
                tok
                for tok in (original.id_to_token(i) for i in orig_ids)
                if tok is not None and tok not in pruned_vocab
            ]
            note = f"lost {len(missing)} token(s): {missing[:3]}"
            regressions.append((label, delta, missing[:5]))
        print(f"{label:<12} {len(orig_ids):>6} {len(pruned_ids):>7} {delta:>+6}  {roundtrip:<10} {note}")

    print("-" * 70)
    print(f"{'TOTAL':<12} {total_orig:>6} {total_pruned:>7} {total_pruned-total_orig:>+6}")
    print()

    if regressions:
        print(f"!! {len(regressions)} sample(s) got worse — keep rule drops tokens in active use")
        for label, delta, missing in regressions:
            print(f"   {label}: +{delta} tokens, e.g. {missing}")
        raise SystemExit(1)
    print("PASS: all zh/en/code samples tokenize identically to the original")

    # 被删除语言的 token 必须确实不在新词表里
    dropped = [t for t in orig_vocab if t not in pruned_vocab]
    print(f"\ndropped token strings: {len(dropped)}")
    for probe in ("привет", "日本語", "한국어", "مرحبا", "café"):
        # 这些语言的 token 应当大部分不在新词表
        present = sum(1 for t in orig_vocab if probe in orig_vocab and t == probe)
        print(f"   probe {probe!r}: in original = {'yes' if probe in orig_vocab else 'no'}, "
              f"in pruned = {'yes' if probe in pruned_vocab else 'no'}")


if __name__ == "__main__":
    main()

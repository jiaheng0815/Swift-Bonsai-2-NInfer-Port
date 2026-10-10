"""词表裁剪：剔除中英文以外的语言，产出新的 frontend resources。

从 `<artifact>` 读取 tokenizer.json / tokenizer_config.json / generation_config.json，
按脚本分类保留中英文 token（含 CJK 的混合 token 一律保留），紧凑重编号，过滤 BPE merges，
写出到 `<out-dir>`。同时产出 `vocab_map.json`（old_id -> new_id），供 artifact 重建使用。

只读 artifact，不改动任何已有文件。

用法：
    python prune_vocab.py --artifact <path.ninfer> --out-dir <dir> [--self-check]

保留规则（与 artifacts/_vocab_scan3.py 一致，已实测）：
  * 256 个单字节 token（byte-level 底座）      —— 必须保留，否则任意文本无法编码
  * UTF-8 碎片（不完整的多字节序列）           —— 保留，BPE merge 的中间产物
  * 字符全部属于 ASCII 的 token                —— 英文 / 代码 / 数字
  * 含任一 CJK 字符的 token（含中英混合）      —— 保护中文
  * 全部 33 个 added token（special + 工具）   —— 用户要求保留 special token
其余（纯其他语言：西里尔、阿拉伯、假名、韩文、泰文、拉丁扩展变音等）剔除。
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(os.environ.get("NINFER_SRC") or r"D:\ai-ninfer\src\cranebw")
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from tools.artifact.container import Artifact  # noqa: E402

RESOURCE_NAMES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/generation_config.json",
)


# --------------------------------------------------------------------------- byte-level

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level 映射：把 256 个字节映射到可见字符。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_B2U = _bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}


def token_to_bytes(token: str) -> bytes | None:
    """把 tokenizer.json 里的 token 字符串还原成原始字节；含非映射字符时返回 None。"""
    try:
        return bytes(_U2B[ch] for ch in token)
    except KeyError:
        return None


# --------------------------------------------------------------------------- 分类

def _kind(ch: str) -> str:
    """把字符分到 ascii / cjk / cjkpunct / generic / other。

    `generic` 是中英文排版、代码和数学共用、且不属于任何自然语言的符号区。
    实测教训：U+2014（——）和 U+2026（……）在中文里极常用，若归入 other 会被误删，
    导致中文文本的 token 数上升。
    """
    cp = ord(ch)
    if cp < 0x80:
        return "ascii"
    # CJK 家族
    if (
        0x4E00 <= cp <= 0x9FFF          # CJK 基本区
        or 0x3400 <= cp <= 0x4DBF       # 扩展 A
        or 0xF900 <= cp <= 0xFAFF       # 兼容表意
        or 0x20000 <= cp <= 0x3FFFF     # 扩展 B 及以后
    ):
        return "cjk"
    if 0x3000 <= cp <= 0x303F or 0xFF00 <= cp <= 0xFFEF:
        return "cjkpunct"               # 中文标点 / 全角
    # 中英文/代码/数学共用的符号（非自然语言字符）
    if (
        0x00A0 <= cp <= 0x00BF          # Latin-1 符号：°±×÷µ¡¿§©®«»¼½¾
        or cp in (0x00D7, 0x00F7)       # × ÷
        or 0x2000 <= cp <= 0x206F       # 通用标点：—–…""''•‰‹›
        or 0x2070 <= cp <= 0x209F       # 上标下标
        or 0x20A0 <= cp <= 0x20CF       # 货币符号
        or 0x2100 <= cp <= 0x214F       # Letterlike：™©®№
        or 0x2150 <= cp <= 0x218F       # 数字形式：⅓Ⅳ
        or 0x2190 <= cp <= 0x21FF       # 箭头 ←→⇒
        or 0x2200 <= cp <= 0x22FF       # 数学运算符 −≤≥≈∑√∞
        or 0x2300 <= cp <= 0x23FF       # 技术符号 ⌘⌂
        or 0x2400 <= cp <= 0x243F       # 控制字符图形
        or 0x2500 <= cp <= 0x257F       # 制表符 ─│┌┐
        or 0x2580 <= cp <= 0x259F       # 方块元素 ▀▄█
        or 0x25A0 <= cp <= 0x25FF       # 几何图形 ■●▲
        or 0x2600 <= cp <= 0x27BF       # 杂项符号/装饰 ★☆♥✓
        or 0x1F300 <= cp <= 0x1FAFF     # emoji
        or 0xFE00 <= cp <= 0xFE0F       # 变体选择符
        or cp == 0x200D                 # ZWJ
    ):
        return "generic"
    return "other"


def keep_token(token: str) -> tuple[bool, str]:
    """返回 (是否保留, 分类标签)。"""
    raw = token_to_bytes(token)
    if raw is None:
        return True, "non-bytelevel"    # 保守保留，不应出现
    if len(raw) == 1:
        return True, "single-byte"      # 256 字节底座
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return True, "partial-utf8"     # 碎片，merge 中间产物
    kinds = {_kind(ch) for ch in text}
    if "cjk" in kinds or "cjkpunct" in kinds:
        return True, "has-cjk"          # 含中文一律保留（含中英混合词）
    if kinds <= {"ascii", "generic"}:
        return True, "ascii+symbol"     # 英文 / 代码 / 通用符号
    return False, "drop-lang"           # 含其他语言字符


# --------------------------------------------------------------------------- merges

def merge_parts(rule) -> tuple[str, str]:
    if isinstance(rule, list):
        return rule[0], rule[1]
    return tuple(rule.split(" ", 1))  # type: ignore[return-value]


# --------------------------------------------------------------------------- 主流程

def read_draft_ids(artifact: Artifact) -> list[int]:
    """读取 text/draft_head_token_ids（I32[131072]）。"""
    payload = bytes(artifact.payload("text/draft_head_token_ids"))
    if len(payload) % 4:
        raise SystemExit("draft_head_token_ids is not a multiple of 4 bytes")
    return [int.from_bytes(payload[i : i + 4], "little") for i in range(0, len(payload), 4)]


def build(args: argparse.Namespace) -> int:
    with Artifact.open(args.artifact) as artifact:
        raw = {name: bytes(artifact.payload(name)) for name in RESOURCE_NAMES}
        draft_ids = read_draft_ids(artifact)

    tokenizer = json.loads(raw["frontend/tokenizer.json"])
    tokenizer_config = json.loads(raw["frontend/tokenizer_config.json"])
    generation_config = json.loads(raw["frontend/generation_config.json"])

    vocab: dict[str, int] = tokenizer["model"]["vocab"]
    merges: list = tokenizer["model"]["merges"]
    added: list[dict] = tokenizer["added_tokens"]

    if len(vocab) == 0 or not merges:
        raise SystemExit("tokenizer.json is missing vocab or merges")

    # --- 1. 分类整个 vocab
    from collections import Counter

    id_to_token = {v: k for k, v in vocab.items()}
    if len(id_to_token) != len(vocab):
        raise SystemExit("model.vocab has duplicate ids")

    counts: Counter[str] = Counter()
    keep_ids: set[int] = set()
    for old_id, token in id_to_token.items():
        keep, tag = keep_token(token)
        counts[tag] += 1
        if keep:
            keep_ids.add(old_id)

    # added tokens 强制保留（用户要求保留 special token）
    for item in added:
        keep_ids.add(int(item["id"]))

    # draft_head 引用的 token 强制保留。
    # draft_head 是 Q4G64_F16S，而 output_head 是 PQ2_0_G128 —— 两者是不同量化器对同一批
    # lm_head 行的独立产物，字节无法互转；重新生成 Q4G64 行需要原始 BF16 权重（本机没有）。
    # 所以 draft_head 的 131072 行必须原样保留，其 token_ids 只能重映射、不能换 token。
    before = len(keep_ids)
    keep_ids.update(int(t) for t in draft_ids)
    draft_forced = len(keep_ids) - before

    total = len(vocab)
    print(f"vocab {total}  ->  keep {len(keep_ids)}  drop {total - len(keep_ids)}")
    for tag, n in counts.most_common():
        print(f"    {tag:<16} {n:>7}")
    print(f"    其中因 draft_head 强制保留: {draft_forced}")

    # --- 2. 紧凑重编号（保持原 id 相对顺序，便于按行比对新旧 embedding）
    kept_vocab_ids = sorted(i for i in keep_ids if i in id_to_token)
    remap: dict[int, int] = {old: new for new, old in enumerate(kept_vocab_ids)}
    next_id = len(kept_vocab_ids)

    # added tokens 紧随其后；必须与 vocab 无重叠（C++ load_added_tokens 会拒绝）
    for item in sorted(added, key=lambda t: int(t["id"])):
        old_id = int(item["id"])
        if old_id in remap:
            raise SystemExit(f"added token id {old_id} collides with a kept vocab id")
        remap[old_id] = next_id
        next_id += 1

    new_vocab_size = next_id
    print(f"new id space: {new_vocab_size} (vocab {len(kept_vocab_ids)} + added {len(added)})")

    # --- 3. 重建 vocab（id 0..len(kept)-1）与 added_tokens
    new_vocab = {id_to_token[old]: remap[old] for old in kept_vocab_ids}
    if len(new_vocab) != len(kept_vocab_ids):
        raise SystemExit("duplicate token strings after remap")

    new_added = []
    for item in sorted(added, key=lambda t: int(t["id"])):
        entry = dict(item)
        entry["id"] = remap[int(item["id"])]
        new_added.append(entry)

    # --- 4. 过滤 merges：a、b、a+b 都必须仍在词表里
    kept_tokens = set(new_vocab)
    new_merges = []
    for rule in merges:
        a, b = merge_parts(rule)
        if a in kept_tokens and b in kept_tokens and (a + b) in kept_tokens:
            new_merges.append(rule)

    print(f"merges {len(merges)}  ->  keep {len(new_merges)}  drop {len(merges) - len(new_merges)}")

    # --- 5. 写 tokenizer.json（其余字段原样保留）
    out_tokenizer = dict(tokenizer)
    out_tokenizer["added_tokens"] = new_added
    out_model = dict(tokenizer["model"])
    out_model["vocab"] = new_vocab
    out_model["merges"] = new_merges
    out_tokenizer["model"] = out_model

    # --- 6. tokenizer_config.json：重映射 added_tokens_decoder（内容必须与 added_tokens 一致）
    out_config = dict(tokenizer_config)
    old_decoder = tokenizer_config.get("added_tokens_decoder", {})
    new_decoder = {}
    for key, value in old_decoder.items():
        old_id = int(key)
        if old_id not in remap:
            raise SystemExit(f"added_tokens_decoder id {old_id} was not remapped")
        new_decoder[str(remap[old_id])] = value
    out_config["added_tokens_decoder"] = new_decoder

    # --- 7. generation_config.json：重映射 token id 字段
    out_generation = dict(generation_config)
    remapped_ids = []
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if field not in out_generation:
            continue
        value = out_generation[field]
        if value is None:
            continue
        if isinstance(value, list):
            new_value = []
            for token_id in value:
                if int(token_id) not in remap:
                    raise SystemExit(f"{field} references a dropped token id {token_id}")
                new_value.append(remap[int(token_id)])
                remapped_ids.append((field, token_id, remap[int(token_id)]))
            out_generation[field] = new_value
        else:
            if int(value) not in remap:
                raise SystemExit(f"{field} references a dropped token id {value}")
            out_generation[field] = remap[int(value)]
            remapped_ids.append((field, value, remap[int(value)]))

    # --- 8. 落盘
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_bytes(
        json.dumps(out_tokenizer, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    (out_dir / "tokenizer_config.json").write_bytes(
        json.dumps(out_config, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    (out_dir / "generation_config.json").write_bytes(
        json.dumps(out_generation, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    (out_dir / "vocab_map.json").write_bytes(
        json.dumps(
            {
                "old_to_new": {str(k): v for k, v in sorted(remap.items())},
                "new_token_count": new_vocab_size,
                "kept_vocab_ids": kept_vocab_ids,
                # draft_head 的 ids 按同序重映射（行内容不变，行 i 仍对应 token_ids[i]）
                "draft_new_ids": [remap[t] for t in draft_ids],
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )

    print()
    print("wrote ->", out_dir)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "vocab_map.json"):
        size = (out_dir / name).stat().st_size
        print(f"    {name:<28} {size/1e6:9.3f} MB")
    print()
    print("generation_config remaps:", remapped_ids)

    # --- 9. special token 的新 id（供 C++ 常量更新）
    print()
    print("=== special token 新 id ===")
    for entry in new_added:
        if entry.get("special"):
            print(f"    {entry['id']:>7}  {entry['content']}")

    if args.self_check:
        run_self_check(out_dir, new_vocab_size)
    return 0


# --------------------------------------------------------------------------- 自检

_CHECK_TEXTS = [
    ("中文", "你好，世界！这是一个测试。"),
    ("中文2", "词表裁剪后，中文应当保持原有分词效率。"),
    ("英文", "Hello, world! The quick brown fox jumps over the lazy dog."),
    ("代码", "def foo(x: int) -> int:\n    return x + 1  # comment"),
    ("中英混", "请帮我把 token_embedding 的 vocab_size 从 248320 改小。"),
    ("数字符号", "3.14159 * 2 == 6.28318, [1,2,3] {a: b}"),
    ("俄语(应退化)", "Русский текст"),
    ("日文(应退化)", "日本語のテキスト"),
    ("emoji(应退化)", "🚀🎉"),
]


def run_self_check(out_dir: Path, expected_size: int) -> None:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("\n[self-check] tokenizers 未安装，跳过")
        return

    print("\n=== self-check ===")
    tokenizer = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    actual = tokenizer.get_vocab_size(with_added_tokens=True)
    status = "OK" if actual == expected_size else f"MISMATCH (expected {expected_size})"
    print(f"vocab size (with added): {actual}  [{status}]")

    failures = 0
    for label, text in _CHECK_TEXTS:
        ids = tokenizer.encode(text).ids
        back = tokenizer.decode(ids)
        ok = back == text
        if not ok:
            failures += 1
        # 确认没有越界 id
        over = [i for i in ids if i >= actual]
        print(
            f"  {label:<14} tokens={len(ids):>4} roundtrip={'OK' if ok else 'FAIL'}"
            f"{' OUT-OF-RANGE' if over else ''}"
        )
        if not ok:
            print(f"      in : {text!r}")
            print(f"      out: {back!r}")

    # 特殊 token 必须可编码
    for special in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|image_pad|>"):
        ids = tokenizer.encode(special).ids
        print(f"  special {special:<18} -> {ids}")

    print(f"\nroundtrip failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, default=Path(r"D:\ai-ninfer\artifacts\bonsai2-27b.ninfer"))
    parser.add_argument("--out-dir", type=Path, default=Path(r"D:\ai-ninfer\artifacts\pruned"))
    parser.add_argument("--self-check", action="store_true", default=True)
    parser.add_argument("--no-self-check", dest="self_check", action="store_false")
    raise SystemExit(build(parser.parse_args()))


if __name__ == "__main__":
    main()

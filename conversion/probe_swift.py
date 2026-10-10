"""比对 Swift-Bonsai-2-PQ2_0.gguf 与基线 Ternary-Bonsai-2-27B-PQ2_0.gguf 的结构一致性。

判据（Swift 是"纯微调"，除权重值外应与基线逐项相同）：
  1. tensor 名称列表（顺序 + 集合）完全一致
  2. 每个 tensor 的 quant type 一致
  3. tokenizer 元数据（tokens / merges / token_type / special ids）逐字节一致
  4. chat template 一致
  5. 其余非数值 kv 一致（记录差异，数值类如训练相关的允许不同）

用法：python probe_swift.py <swift.gguf> <base.gguf>
"""
from __future__ import annotations

import hashlib
import struct
import sys
from collections import Counter


def read_gguf_header(path):
    """返回 (kv, tensors)，只读头部（tensor 数据不读）。"""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", path
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def s():
        return f.read(u64()).decode("utf-8", "replace")

    def val(t):
        if t in (0, 1, 7):
            return f.read(1)[0]
        if t in (2, 3):
            return struct.unpack("<h", f.read(2))[0]
        if t == 4:
            return u32()
        if t == 5:
            return struct.unpack("<i", f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", f.read(4))[0]
        if t == 8:
            return s()
        if t in (10, 11):
            return struct.unpack("<q", f.read(8))[0]
        if t == 12:
            return struct.unpack("<d", f.read(8))[0]
        if t == 9:
            et, n = u32(), u64()
            if et == 8:
                return [s() for _ in range(n)]
            if et == 6:
                return list(struct.unpack("<%df" % n, f.read(4 * n)))
            if et in (0, 1, 7):
                return list(f.read(n))
            if et in (2, 3):
                return list(struct.unpack("<%dh" % n, f.read(2 * n)))
            if et == 4:
                return list(struct.unpack("<%dI" % n, f.read(4 * n)))
            if et == 5:
                return list(struct.unpack("<%di" % n, f.read(4 * n)))
            raise SystemExit("unsupported array elem type %d" % et)
        raise SystemExit("unsupported kv type %d" % t)

    kv = {}
    for _ in range(n_kv):
        k = s()
        kv[k] = val(u32())

    tensors = []
    for _ in range(n_tensors):
        name = s()
        nd = u32()
        ne = [u64() for _ in range(nd)]
        tt = u32()
        off = u64()
        tensors.append((name, ne, tt, off))
    f.close()
    return ver, kv, tensors


def h_str_list(items) -> str:
    h = hashlib.sha256()
    for it in items:
        b = it.encode("utf-8")
        h.update(len(b).to_bytes(4, "little"))
        h.update(b)
    return h.hexdigest()


def main() -> int:
    swift_path, base_path = sys.argv[1], sys.argv[2]
    sv, skv, st = read_gguf_header(swift_path)
    bv, bkv, bt = read_gguf_header(base_path)

    print(f"gguf version: swift={sv} base={bv}")
    print(f"tensors: swift={len(st)} base={len(bt)}")

    snames = [t[0] for t in st]
    bnames = [t[0] for t in bt]
    print(f"tensor 名称列表完全一致（含顺序）: {snames == bnames}")
    if snames != bnames:
        only_s = sorted(set(snames) - set(bnames))
        only_b = sorted(set(bnames) - set(snames))
        print(f"  仅 Swift 有: {only_s[:10]}")
        print(f"  仅 基线 有: {only_b[:10]}")

    stypes = Counter(t[2] for t in st)
    btypes = Counter(t[2] for t in bt)
    print(f"quant type 直方图一致: {stypes == btypes}")
    if stypes != btypes:
        print(f"  swift: {dict(stypes)}")
        print(f"  base : {dict(btypes)}")
    mismatch = [(s[0], s[2], b[2]) for s, b in zip(st, bt) if s[2] != b[2]]
    if mismatch:
        print(f"  逐 tensor 类型不同的数量: {len(mismatch)}，前 10: {mismatch[:10]}")

    shapes_same = all(s[1] == b[1] for s, b in zip(st, bt))
    print(f"shape 完全一致: {shapes_same}")

    # tokenizer 逐字节比对
    tok_keys = [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.scores",
    ]
    print("\n=== tokenizer ===")
    all_same = True
    for k in tok_keys:
        s_has = k in skv
        b_has = k in bkv
        if not (s_has and b_has):
            print(f"  {k}: 存在性 swift={s_has} base={b_has}")
            all_same = False
            continue
        s_val, b_val = skv[k], bkv[k]
        same = s_val == b_val
        s_hash = h_str_list(s_val) if s_val and isinstance(s_val[0], str) else hashlib.sha256(
            repr(s_val).encode()).hexdigest()
        b_hash = h_str_list(b_val) if b_val and isinstance(b_val[0], str) else hashlib.sha256(
            repr(b_val).encode()).hexdigest()
        print(f"  {k:<28} n={len(s_val):<7} 一致={same}")
        print(f"      swift sha256={s_hash[:32]}...")
        print(f"      base  sha256={b_hash[:32]}...")
        all_same &= same
    print(f"tokenizer 元数据逐字节一致: {all_same}")

    # chat template
    print("\n=== chat template ===")
    tmpl_keys = [k for k in skv if "chat_template" in k]
    for k in tmpl_keys:
        s_val = skv.get(k)
        b_val = bkv.get(k)
        print(f"  {k}: 一致={s_val == b_val} (len swift={len(s_val) if s_val else 0})")

    # 其余 kv 差异
    print("\n=== kv 键差异 ===")
    only_s = sorted(set(skv) - set(bkv))
    only_b = sorted(set(bkv) - set(skv))
    print(f"  仅 Swift 有: {only_s}")
    print(f"  仅 基线 有: {only_b}")
    diffs = []
    for k in sorted(set(skv) & set(bkv)):
        if k in tok_keys or "chat_template" in k:
            continue
        if skv[k] != bkv[k]:
            sv_ = str(skv[k])[:60]
            bv_ = str(bkv[k])[:60]
            diffs.append((k, sv_, bv_))
    print(f"  值不同的键 {len(diffs)} 个:")
    for k, s_, b_ in diffs[:40]:
        print(f"    {k}\n      swift: {s_}\n      base : {b_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

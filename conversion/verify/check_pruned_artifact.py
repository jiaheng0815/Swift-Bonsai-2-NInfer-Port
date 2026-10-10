"""裁剪后 artifact 的端到端校验。

逐项验证 rebuild_artifact.py 的产物相对原件的正确性：
  1. 结构：identity、对象数、名称集合、shape/format/layout
  2. text/token_embedding / text/output_head：新行 i 必须等于原行 kept_old_ids[i]（逐字节）
  3. text/draft_head：字节完全不变（Q4G64 的行无法重建，必须原样保留）
  4. text/draft_head_token_ids：每个值等于 remap[原值]，顺序不变
  5. frontend/{tokenizer,tokenizer_config,generation_config}.json：与 pruned 目录一致
  6. 其余对象：抽样逐字节一致

用法：
    python check_pruned_artifact.py [--src <old.ninfer>] [--dst <new.ninfer>] [--pruned <dir>]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

REPO_SRC = Path(r"D:\ai-ninfer\src\cranebw")
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from tools.artifact.container import Artifact, ResourceObject, TensorObject  # noqa: E402
from tools.artifact.layouts import row_split_geometry  # noqa: E402

FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"  FAIL  {message}")


def row_bytes_of(payload, geometry, row: int, plane: str) -> bytes:
    if plane == "base":
        begin = geometry.base_offset + row * geometry.base_row_bytes
        width = geometry.base_row_bytes
    else:
        begin = geometry.scale_offset + row * geometry.scale_row_bytes
        width = geometry.scale_row_bytes
    return bytes(payload[begin : begin + width])


def check(src_path: Path, dst_path: Path, pruned: Path) -> None:
    plan = json.loads((pruned / "vocab_map.json").read_text(encoding="utf-8"))
    remap = {int(k): int(v) for k, v in plan["old_to_new"].items()}
    kept_old_ids = sorted(remap, key=lambda old: remap[old])
    new_vocab = int(plan["new_token_count"])

    with Artifact.open(src_path) as src, Artifact.open(dst_path) as dst:
        print("=== 1. structure ===")
        if src.identity != dst.identity:
            fail(f"identity changed: {src.identity} -> {dst.identity}")
        if len(src.objects) != len(dst.objects):
            fail(f"object count {len(src.objects)} -> {len(dst.objects)}")
        src_names = [o.name for o in src.objects]
        dst_names = [o.name for o in dst.objects]
        if src_names != dst_names:
            missing = set(src_names) - set(dst_names)
            extra = set(dst_names) - set(src_names)
            fail(f"object names differ; missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
        else:
            print(f"  ok    {len(dst.objects)} objects, names and order identical")

        # 预期 shape 变化
        for name, expected in (
            ("text/token_embedding", (new_vocab, 5120)),
            ("text/output_head", (new_vocab, 5120)),
            ("text/draft_head", (131072, 5120)),
            ("text/draft_head_token_ids", (131072,)),
        ):
            obj = dst.find(name)
            if obj.shape != expected:
                fail(f"{name} shape {obj.shape}, expected {expected}")
            else:
                print(f"  ok    {name:<28} {list(obj.shape)}")

        print("\n=== 2. embedding / output_head row remap ===")
        import numpy as np

        index = np.asarray(kept_old_ids, dtype=np.int64)
        for name in ("text/token_embedding", "text/output_head"):
            src_obj, dst_obj = src.find(name), dst.find(name)
            src_payload, dst_payload = src.payload(src_obj), dst.payload(dst_obj)
            src_geom = row_split_geometry(src_obj.format, src_obj.shape)
            dst_geom = row_split_geometry(dst_obj.format, dst_obj.shape)

            for plane, row_bytes, src_off, dst_off in (
                ("base", src_geom.base_row_bytes, src_geom.base_offset, dst_geom.base_offset),
                ("scale", src_geom.scale_row_bytes, src_geom.scale_offset, dst_geom.scale_offset),
            ):
                src_plane = np.frombuffer(
                    bytes(src_payload[src_off : src_off + src_geom.n * row_bytes]), dtype=np.uint8
                ).reshape(src_geom.n, row_bytes)
                dst_plane = np.frombuffer(
                    bytes(dst_payload[dst_off : dst_off + dst_geom.n * row_bytes]), dtype=np.uint8
                ).reshape(dst_geom.n, row_bytes)
                # 全量：dst 的第 i 行必须逐字节等于 src 的第 kept_old_ids[i] 行
                mismatch = np.flatnonzero((src_plane[index] != dst_plane).any(axis=1))
                if mismatch.size:
                    first = int(mismatch[0])
                    fail(
                        f"{name} {plane}: {mismatch.size} rows differ, first new_row={first} "
                        f"(old_row={kept_old_ids[first]})"
                    )
                else:
                    print(f"  ok    {name:<28} {plane} plane: ALL {dst_geom.n} rows byte-exact")
                del src_plane, dst_plane
            del src_payload, dst_payload
            gc.collect()

        print("\n=== 3. draft_head is byte-identical ===")
        src_obj, dst_obj = src.find("text/draft_head"), dst.find("text/draft_head")
        src_payload, dst_payload = src.payload(src_obj), dst.payload(dst_obj)
        if src_obj.bytes != dst_obj.bytes:
            fail(f"draft_head size {src_obj.bytes} -> {dst_obj.bytes}")
        else:
            src_geom = row_split_geometry(src_obj.format, src_obj.shape)
            dst_geom = row_split_geometry(dst_obj.format, dst_obj.shape)
            rng = random.Random(7)
            rows = [0, 1, 131071] + [rng.randrange(131072) for _ in range(12)]
            bad = 0
            for row in rows:
                for plane in ("base", "scale"):
                    if row_bytes_of(src_payload, src_geom, row, plane) != row_bytes_of(
                        dst_payload, dst_geom, row, plane
                    ):
                        fail(f"draft_head row {row} {plane} plane changed")
                        bad += 1
                if bad:
                    break
            if not bad:
                print(f"  ok    draft_head {len(rows)} sampled rows unchanged (Q4G64 preserved)")
        del src_payload, dst_payload
        gc.collect()

        print("\n=== 4. draft_head_token_ids remap ===")
        src_ids_raw = bytes(src.payload("text/draft_head_token_ids"))
        dst_ids_raw = bytes(dst.payload("text/draft_head_token_ids"))
        src_ids = [int.from_bytes(src_ids_raw[i : i + 4], "little") for i in range(0, len(src_ids_raw), 4)]
        dst_ids = [int.from_bytes(dst_ids_raw[i : i + 4], "little") for i in range(0, len(dst_ids_raw), 4)]
        if len(src_ids) != len(dst_ids):
            fail(f"draft id count {len(src_ids)} -> {len(dst_ids)}")
        else:
            mismatches = [
                (i, s, d) for i, (s, d) in enumerate(zip(src_ids, dst_ids)) if remap.get(s) != d
            ]
            if mismatches:
                fail(f"{len(mismatches)} draft ids not remapped as expected, e.g. {mismatches[:3]}")
            elif len(set(dst_ids)) != len(dst_ids):
                fail("remapped draft ids are not unique (validate_draft_ids would reject)")
            elif max(dst_ids) >= new_vocab:
                fail(f"draft id {max(dst_ids)} >= token domain {new_vocab}")
            else:
                print(f"  ok    {len(dst_ids)} ids remapped, unique, max {max(dst_ids)} < {new_vocab}")

        print("\n=== 5. frontend resources ===")
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            want = (pruned / name).read_bytes()
            got = bytes(dst.payload(f"frontend/{name}"))
            if want == got:
                print(f"  ok    frontend/{name:<26} {len(got)} B matches pruned output")
            else:
                fail(f"frontend/{name} does not match {pruned / name}")
        # 未替换的 resource 必须与原文件一致。
        # 注意 dst.payload 必须按名字查找：跨 artifact 传对象描述符会用错 offset。
        untouched = 0
        for obj in src.objects:
            if not isinstance(obj, ResourceObject):
                continue
            if obj.name.startswith("frontend/") and (
                "tokenizer" in obj.name or "generation_config" in obj.name
            ):
                continue
            if bytes(src.payload(obj)) != bytes(dst.payload(obj.name)):
                fail(f"resource {obj.name} changed unexpectedly")
            else:
                untouched += 1
        print(f"  ok    {untouched} untouched resources byte-identical")

        print("\n=== 6. other tensors byte-identical (sampled) ===")
        rng = random.Random(31337)
        others = [
            o
            for o in src.objects
            if isinstance(o, TensorObject)
            and o.name not in ("text/token_embedding", "text/output_head")
            and o.name != "text/draft_head_token_ids"
        ]
        picked = rng.sample(others, min(40, len(others)))
        bad = 0
        for obj in picked:
            src_obj, dst_obj = src.objects[src_names.index(obj.name)], dst.find(obj.name)
            if src_obj.bytes != dst_obj.bytes:
                fail(f"{obj.name} size {src_obj.bytes} -> {dst_obj.bytes}")
                bad += 1
                continue
            # 抽样块比对
            total = src_obj.bytes
            spans = [(0, min(65536, total)), (max(0, total // 2 - 4096), min(total, total // 2 + 4096))]
            for begin, end in spans:
                if bytes(src.payload(src_obj)[begin:end]) != bytes(dst.payload(dst_obj)[begin:end]):
                    fail(f"{obj.name} differs in [{begin},{end})")
                    bad += 1
                    break
            if bad >= 3:
                break
        if not bad:
            print(f"  ok    {len(picked)} sampled tensors byte-identical")

    print()
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} check(s) FAILED")
    print("ALL CHECKS PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path(r"D:\ai-ninfer\artifacts\bonsai2-27b.ninfer"))
    parser.add_argument("--dst", type=Path, default=Path(r"D:\ai-ninfer\artifacts\bonsai2-27b-pruned.ninfer"))
    parser.add_argument("--pruned", type=Path, default=Path(r"D:\ai-ninfer\artifacts\pruned"))
    args = parser.parse_args()
    check(args.src, args.dst, args.pruned)


if __name__ == "__main__":
    main()

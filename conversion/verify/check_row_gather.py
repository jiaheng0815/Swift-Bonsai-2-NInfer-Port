"""验证 rebuild_artifact.rows_chunks 的行 gather 逻辑。

用原 artifact 的 row-split 张量做小样本测试：取 K 行按给定顺序重排，
检查生成 payload 的长度与几何一致、且解析回来的每一行都与源行逐字节相同。

这是写入 8.3 GB 之前的关键验证 —— 若 gather 有误，产出的模型会数值错乱但结构合法。

用法：
    python check_row_gather.py --artifact <src.ninfer>
"""

from __future__ import annotations

import os
import argparse
import gc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = Path(os.environ.get("NINFER_SRC") or _REPO_ROOT / "src" / "cranebw")
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
GUIDE_TOOLS = Path(os.environ.get("NINFER_CONVERSION") or _REPO_ROOT / "conversion")
if str(GUIDE_TOOLS) not in sys.path:
    sys.path.insert(0, str(GUIDE_TOOLS))

from tools.artifact.container import Artifact  # noqa: E402
from tools.artifact.layouts import row_split_geometry, split_row_planes  # noqa: E402
from rebuild_artifact import rows_chunks  # noqa: E402

SAMPLES = [
    "text/layers/0/gdn/output",
    "mtp/layer/attention/output",
]


def check_sample(artifact: Artifact, name: str, count: int) -> bool:
    """在一个函数作用域内完成检查，退出后所有 mmap 视图自动释放。"""
    obj = artifact.find(name)
    if obj.kind != "tensor":
        print(f"skip {name}: not a tensor")
        return True
    n, k = obj.shape
    source = artifact.payload(obj)
    geometry = row_split_geometry(obj.format, obj.shape)

    rows = [(count - 1 - i) % n for i in range(count)]
    rows[1] = rows[0]  # 重复一行，覆盖 gather 的重复风险

    target = row_split_geometry(obj.format, (count, k))
    blob = b"".join(bytes(chunk) for chunk in rows_chunks(source, geometry, rows, target, b"\x00"))

    ok = True
    detail = ""
    if len(blob) != target.payload_bytes:
        ok = False
        detail = f"payload {len(blob)} B, geometry expects {target.payload_bytes} B"
    else:
        planes = split_row_planes(blob, target)
        for index, source_row in enumerate(rows):
            # base 平面
            src_base = source[
                geometry.base_offset + source_row * geometry.base_row_bytes :
                geometry.base_offset + (source_row + 1) * geometry.base_row_bytes
            ]
            got_base = planes.base[index * target.base_row_bytes : (index + 1) * target.base_row_bytes]
            if bytes(src_base) != bytes(got_base):
                ok = False
                detail = f"base row {index} (source {source_row}) mismatch"
                break
            # scale 平面
            if target.scale_row_bytes:
                src_scale = source[
                    geometry.scale_offset + source_row * geometry.scale_row_bytes :
                    geometry.scale_offset + (source_row + 1) * geometry.scale_row_bytes
                ]
                got_scale = planes.scale[
                    index * target.scale_row_bytes : (index + 1) * target.scale_row_bytes
                ]
                if bytes(src_scale) != bytes(got_scale):
                    ok = False
                    detail = f"scale row {index} (source {source_row}) mismatch"
                    break
        # 行序不可被转置：第 probe 行必须来自 rows[probe]
        if ok:
            probe = 3
            src_head = bytes(
                source[geometry.base_offset + rows[probe] * geometry.base_row_bytes :
                       geometry.base_offset + rows[probe] * geometry.base_row_bytes + 16]
            )
            got_head = bytes(
                planes.base[probe * target.base_row_bytes : probe * target.base_row_bytes + 16]
            )
            if src_head != got_head:
                ok = False
                detail = f"row {probe} is not mapped to source row {rows[probe]}"

    print(f"{'OK' if ok else 'FAIL':<5} {name:<32} rows={count} "
          f"base_row={geometry.base_row_bytes}B scale_row={geometry.scale_row_bytes}B "
          f"high_row={geometry.high_row_bytes}B -> {len(blob)}B"
          + (f"   {detail}" if detail else ""))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=_REPO_ROOT / "artifacts" / "bonsai2-27b.ninfer")
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()

    failures = 0
    with Artifact.open(args.artifact) as artifact:
        for name in SAMPLES:
            if not check_sample(artifact, name, args.rows):
                failures += 1
            gc.collect()  # 丢弃上一样本的 mmap 导出视图，否则 close() 会 BufferError

    if failures:
        raise SystemExit(f"\n{failures} sample(s) FAILED")
    print("\nPASS: row gather is byte-exact and preserves row order")


if __name__ == "__main__":
    main()

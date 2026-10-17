"""按裁剪后的词表重建 .ninfer artifact。

读取原 artifact + `prune_vocab.py` 产出的 `vocab_map.json`，产出裁剪后的新 artifact：

  text/token_embedding        [248320,5120] -> [184888,5120]   按 old->new 映射重排行
  text/output_head            同上
  text/draft_head             [131072,5120] 保持行数不变，内容换成新词表下的高频子集
  text/draft_head_token_ids   [131072]      按新词表重算
  frontend/{tokenizer,tokenizer_config,generation_config}.json  换成裁剪后的版本
  其余对象                     原样搬运（零拷贝 memoryview）

行重排是纯字节级 gather：PQ2_0_G128 的每行 = base plane 1280 B + scale plane 80 B，
两向独立，**不解量化、不重新量化**，原始行的比特完全保留。

用法：
    python rebuild_artifact.py --artifact <src.ninfer> --pruned <dir> --out <dst.ninfer>
"""

from __future__ import annotations

import os
import argparse
import gc
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = Path(os.environ.get("NINFER_SRC") or _REPO_ROOT / "src" / "cranebw")
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from tools.artifact.container import (  # noqa: E402
    Artifact,
    ArtifactError,
    ArtifactIdentity,
    ResourceObject,
    ResourceSpec,
    TensorObject,
    TensorSpec,
    ArtifactWriter,
)
from tools.artifact.layouts import row_split_geometry  # noqa: E402

VOCAB_WIDTH = 5120

REPLACED_RESOURCES = {
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/generation_config.json",
}
# 按 old->new 映射重排行的张量（新行序 = old id 升序）
RESHAPED_TENSORS = ("text/token_embedding", "text/output_head")


def rows_chunks(source, old_geometry, rows, new_geometry, zero_row: bytes):
    """按新行序逐行产出 row-split 平面的分片。

    PQ2_0_G128 的物理布局是 [base plane][high plane][scale plane]，不是按行交错，
    所以必须先产出整个 base 平面，再产出整个 scale 平面。平面之间的对齐 padding 补零。

    每行复制成独立 bytes 而不是产出切片视图：memoryview 切片会导出 mmap 指针，
    只要有一个活着 Artifact.close() 就抛 BufferError（pack.py 里踩过同一个坑）。
    行大小在 1-6 KB，复制的开销远低于排错成本。
    """
    row_count = len(rows)

    # --- base plane
    for row in rows:
        begin = old_geometry.base_offset + row * old_geometry.base_row_bytes
        yield bytes(source[begin : begin + old_geometry.base_row_bytes])
    written = row_count * new_geometry.base_row_bytes

    # --- padding 到 high plane
    if new_geometry.high_offset > written:
        yield zero_row * (new_geometry.high_offset - written)

    # --- high plane（PQ2_0 为空）
    if new_geometry.high_row_bytes:
        for row in rows:
            begin = old_geometry.high_offset + row * old_geometry.high_row_bytes
            yield bytes(source[begin : begin + old_geometry.high_row_bytes])
    written = new_geometry.high_offset + row_count * new_geometry.high_row_bytes

    # --- padding 到 scale plane
    if new_geometry.scale_offset > written:
        yield zero_row * (new_geometry.scale_offset - written)

    # --- scale plane
    for row in rows:
        begin = old_geometry.scale_offset + row * old_geometry.scale_row_bytes
        yield bytes(source[begin : begin + old_geometry.scale_row_bytes])


class SourceGuard:
    """保证退出时既不掩盖原始异常，也不因 mmap 视图残留而抛 BufferError。

    Artifact.__exit__ 无条件调用 close()，一旦有 memoryview 导出指针存活就抛
    BufferError，把块内真正的异常盖掉（这个文件第一次跑就是这么丢掉真因的）。
    """

    def __init__(self, artifact: Artifact):
        self.artifact = artifact

    def __enter__(self) -> Artifact:
        return self.artifact

    def __exit__(self, exc_type, exc, tb) -> bool:
        for _ in range(5):
            gc.collect()
            try:
                self.artifact.close()
                return False
            except BufferError:
                continue
        if exc_type is None:
            raise SystemExit("failed to release mmap views before closing the source artifact")
        print("warning: source artifact could not be closed (mmap views still exported)")
        return False


def build(args: argparse.Namespace) -> int:
    pruned_dir = Path(args.pruned)
    override = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        override[f"frontend/{name}"] = (pruned_dir / name).read_bytes()

    plan = json.loads((pruned_dir / "vocab_map.json").read_text(encoding="utf-8"))
    remap = {int(k): int(v) for k, v in plan["old_to_new"].items()}
    new_vocab = int(plan["new_token_count"])
    kept_old_ids = sorted(remap, key=lambda old: remap[old])
    if len(kept_old_ids) != new_vocab:
        raise SystemExit("vocab_map is inconsistent")

    # --- draft_head：行内容与行序都不变，只把 ids 映射到新 id 空间。
    # 行 i 仍对应 token_ids[i]，token 身份不变，所以 draft_head 的字节可以整体原样搬运。
    draft_new_ids = [int(t) for t in plan["draft_new_ids"]]
    if len(set(draft_new_ids)) != len(draft_new_ids):
        raise SystemExit("remapped draft ids are not unique")
    if max(draft_new_ids) >= new_vocab:
        raise SystemExit("remapped draft id is outside the new token domain")
    draft_ids_blob = b"".join(int(t).to_bytes(4, "little") for t in draft_new_ids)

    print(f"new vocab      : {new_vocab}")
    print(f"draft ids      : {len(draft_new_ids)}  range {min(draft_new_ids)}..{max(draft_new_ids)}")

    with SourceGuard(Artifact.open(args.artifact)) as source:
        if source.identity.model_id != "qwen3.8-27b":
            raise SystemExit(f"unexpected model_id {source.identity.model_id!r}")

        by_name = {obj.name: obj for obj in source.objects}

        # --- 构造新的对象清单（顺序沿用原文件）
        specs = []
        for obj in source.objects:
            if obj.name in REPLACED_RESOURCES:
                specs.append(ResourceSpec(obj.name, obj.encoding, len(override[obj.name])))
            elif obj.name in RESHAPED_TENSORS:
                specs.append(TensorSpec(obj.name, (new_vocab, VOCAB_WIDTH), obj.format, obj.layout))
            elif isinstance(obj, TensorObject):
                # 含 text/draft_head 与 text/draft_head_token_ids：形状不变
                specs.append(TensorSpec(obj.name, obj.shape, obj.format, obj.layout))
            else:
                specs.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))

        out_path = Path(args.out)
        zero_row = b"\x00"

        # 预览各对象的处理方式与目标几何
        for spec in specs:
            if spec.name in REPLACED_RESOURCES:
                print(f"  replace  {spec.name:<34} {len(override[spec.name]):>12} B")
            elif spec.name in RESHAPED_TENSORS:
                geom = row_split_geometry(spec.format, spec.shape)
                print(f"  gather   {spec.name:<34} {list(spec.shape)} -> {geom.payload_bytes} B")
            elif spec.name == "text/draft_head_token_ids":
                print(f"  remap    {spec.name:<34} {list(spec.shape)} -> {len(draft_ids_blob)} B")
            elif spec.name == "text/draft_head":
                print(f"  verbatim {spec.name:<34} {list(spec.shape)} (字节不变)")

        if args.dry_run:
            print("\n[dry-run] no output written")
            return 0

        if out_path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {out_path}")

        with ArtifactWriter(
            out_path,
            ArtifactIdentity(source.identity.model_id, source.identity.weights_id),
            specs,
        ) as writer:
            for obj in source.objects:
                if obj.name in REPLACED_RESOURCES:
                    writer.write(obj.name, override[obj.name])
                elif obj.name in RESHAPED_TENSORS:
                    geometry = row_split_geometry(obj.format, obj.shape)
                    target = row_split_geometry(obj.format, (new_vocab, VOCAB_WIDTH))
                    writer.write(
                        obj.name,
                        rows_chunks(source.payload(obj), geometry, kept_old_ids, target, zero_row),
                    )
                elif obj.name == "text/draft_head_token_ids":
                    writer.write(obj.name, draft_ids_blob)
                else:
                    # 含 text/draft_head：Q4G64 的行无法从 PQ2_0 的 output_head 重建，原样搬运
                    writer.write(obj.name, source.payload(obj))
        # writer 已在此关闭；source 由 SourceGuard 收尾

    print(f"\nwrote -> {out_path}")
    print(f"size  : {out_path.stat().st_size/1e9:.3f} GB")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, default=_REPO_ROOT / "artifacts" / "bonsai2-27b.ninfer")
    parser.add_argument("--pruned", type=Path, default=_REPO_ROOT / "artifacts" / "pruned")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "artifacts" / "bonsai2-27b-pruned.ninfer")
    parser.add_argument("--dry-run", action="store_true", help="只打印处理计划，不写文件")
    raise SystemExit(build(parser.parse_args()))


if __name__ == "__main__":
    main()

# artifacts — 制品说明

本目录在 **发布时为空**（`.gitignore` 排除）：`.ninfer` 制品与 GGUF 权重都**不入库**
（体积 + 版权归原作者）。

> **下载**：**未裁剪**的 Swift 制品（`swift-bonsai2-27b.ninfer`，7.736 GiB）连同
> **词表剪切器**打包成 zip 分卷，放在
> [Release v1.0.0](https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port/releases/tag/v1.0.0)
> 的 Assets 里，包内含 `README.txt`（用法）与 `SHA256SUMS.txt`（校验）。
> 解压后按包内说明跑剪切器 → 得到引擎可直接加载的 208,030 域制品。

## 本机实际产物（本地路径）

| 文件 | 体积 | 说明 |
|---|---|---|
| `swift-bonsai2-27b.ninfer` | 7.736 GiB | Swift Bonsai 2 全量制品（1126 对象）|
| `swift-bonsai2-27b-pruned.ninfer` | 7.629 GiB | ★ **词表裁剪版**（208,030 域，直接可服务）|
| `bonsai2-27b-pruned.ninfer` | 7.629 GiB | 基线 Bonsai 2 的裁剪版（A/B 对比用）|
| `pruned/` | 12 MB | 裁剪词表资源（tokenizer + `vocab_map.json`）|
| `hadamard-meta.json` | 194 KB | `check_signs.py` 用的权威元数据 |

## 怎么得到它们

**从官方 GGUF 自己打**（推荐，全程可复现）：

```powershell
# 0. 下载权重（官方渠道；本仓提供 hf 镜像/代理并行下载器）
python conversion\parallel_dl.py --url <官方GGUF地址> --out E:\models\swift-bonsai2-27b.gguf `
  --size <字节数> --sha256 <官方SHA256> --conn 20 --proxy http://127.0.0.1:7897

# 1. 打包（魔搭指南的 pack.py，见 docs/CONVERSION_GUIDE.md）
cd guide && python tools\pack.py build D:\ai-ninfer\artifacts\swift-bonsai2-27b.ninfer `
  --gguf <...gguf> --template D:\ai-ninfer\models\template\model.ninfer

# 2. 四件套验证（必须全绿，判据见 docs/CONVERSION_GUIDE.md）
python tools\verify\check_payload_order.py ...
python tools\verify\check_signs.py ...
python tools\verify\check_row_order.py ...
python tools\verify\check_assembly.py ...

# 3. 词表裁剪 + 重建 + 审计
python conversion\prune_vocab.py     --artifact artifacts\swift-bonsai2-27b.ninfer --out-dir artifacts\pruned
python conversion\rebuild_artifact.py --artifact artifacts\swift-bonsai2-27b.ninfer --pruned artifacts\pruned `
  --out artifacts\swift-bonsai2-27b-pruned.ninfer
python conversion\verify\check_pruned_artifact.py --src artifacts\swift-bonsai2-27b.ninfer `
  --dst artifacts\swift-bonsai2-27b-pruned.ninfer --pruned artifacts\pruned
```

## 权重来源（官方）

| 模型 | 仓库 | 本机用到的文件 |
|---|---|---|
| Swift Bonsai 2（UkisAI）| `ukisai/Swift-Bonsai-2-GGUF` | `Swift-Bonsai-2-PQ2_0.gguf`（7,206,168,928 B，SHA256 `5912bb73…63d5`）|
| Ternary Bonsai 2（PrismML）| `prism-ml/Ternary-Bonsai-2-27B-gguf` | `…-PQ2_0.gguf`（基线）|
| 视觉塔 mmproj | PrismML | `…-mmproj-BF16.gguf` |

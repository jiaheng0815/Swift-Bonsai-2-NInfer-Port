# artifacts — 制品说明

本目录在 **发布时为空**（`.gitignore` 排除）：`.ninfer` 制品与 GGUF 权重都**不入库**
（体积 + 版权归原作者）。

> **下载**：全部发布物在两个平台提供，内容一致：
>
> - [Hugging Face · jiaheng0815/Swift-Bonsai-2-NInfer-Port](https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port)
> - [魔搭 ModelScope · jiaheng0815/Swift-Bonsai-2-NInfer-Port](https://www.modelscope.cn/models/jiaheng0815/Swift-Bonsai-2-NInfer-Port)
>
> | 文件 | 内容 | 体积 |
> |---|---|---|
> | `ninfer-swift-bonsai2-complete-v1.0.0-single.zip` | **完整包**：引擎 exe + DLL + 已裁剪模型 + 剪切器 + 启动脚本 | 6.78 GiB |
> | `swift-bonsai2-27b-unpruned-v1.0.0-single.zip` | **未裁剪制品** + 剪切器 + 自包含容器库 | 6.28 GiB |
> | `engine/ninfer-engine-ternary-bonsai-swift-port-v1.0.0.zip` | 完整 C++ 引擎源码 | 11 MB |
>
> 包内含 `README.txt`（用法）与 `SHA256SUMS.txt`（校验）。
> 源码镜像：[GitHub](https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port) ·
> [Gitee](https://gitee.com/fengjiaheng123/swift-bonsai-2-ninfer-port)。

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
$REPO = "D:\ai-ninfer"        # 本仓根目录

# 0. 下载权重（官方渠道；大文件走代理直连，见 conversion\parallel_dl.py）
python conversion\parallel_dl.py --url <官方GGUF地址> --out E:\models\swift-bonsai2-27b.gguf `
  --size <字节数> --sha256 <官方SHA256> --conn 20 --proxy http://127.0.0.1:7897

# 1. 打包（pack.py 来自魔搭指南，不在本 git 仓；未裁剪包内附 ninfer-tools 容器库）
$env:NINFER_SRC = "<path-to>\ninfer-tools"   # 用指南原仓时改为 "$REPO\src\cranebw"
python <path-to-guide>\tools\pack.py build "$REPO\artifacts\swift-bonsai2-27b.ninfer" `
  --gguf <...gguf> --template "$REPO\models\template\model.ninfer"

# 2. 四件套验证（脚本来自魔搭指南 tools\verify；本机参考副本 guide\ 不入库）
python <path-to-guide>\tools\verify\check_payload_order.py ...
python <path-to-guide>\tools\verify\check_signs.py ...
python <path-to-guide>\tools\verify\check_row_order.py ...
python <path-to-guide>\tools\verify\check_assembly.py ...

# 3. 词表裁剪 + 重建 + 审计
python conversion\prune_vocab.py     --artifact "$REPO\artifacts\swift-bonsai2-27b.ninfer" --out-dir "$REPO\artifacts\pruned"
python conversion\rebuild_artifact.py --artifact "$REPO\artifacts\swift-bonsai2-27b.ninfer" --pruned "$REPO\artifacts\pruned" `
  --out "$REPO\artifacts\swift-bonsai2-27b-pruned.ninfer"
python conversion\verify\check_pruned_artifact.py --src "$REPO\artifacts\swift-bonsai2-27b.ninfer" `
  --dst "$REPO\artifacts\swift-bonsai2-27b-pruned.ninfer" --pruned "$REPO\artifacts\pruned"
```

> 使用**未裁剪发布包**时，`NINFER_SRC` 指向包内 `ninfer-tools`；使用 **git 仓**时默认指向
> `src\cranebw`，无需设置。

## 权重来源（官方）

| 模型 | 仓库 | 本机用到的文件 |
|---|---|---|
| Swift Bonsai 2（UkisAI）| `ukisai/Swift-Bonsai-2-GGUF` | `Swift-Bonsai-2-PQ2_0.gguf`（7,206,168,928 B，SHA256 `5912bb73…63d5`）|
| Ternary Bonsai 2（PrismML）| `prism-ml/Ternary-Bonsai-2-27B-gguf` | `…-PQ2_0.gguf`（基线）|
| 视觉塔 mmproj | PrismML | `…-mmproj-BF16.gguf` |

> 下载策略：**走用户代理直连官方站**（`http://127.0.0.1:7897`），不要用 hf-mirror；
> `parallel_dl.py` 支持多连接 range、断点续传与 SHA256 校验。

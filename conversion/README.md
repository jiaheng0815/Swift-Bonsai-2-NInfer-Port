# conversion — 制品转换与验证工具

把官方 GGUF 打成 NInfer `.ninfer` 制品，并做**词表裁剪**与**逐字节验证**。
完整流程见 [docs/CONVERSION_GUIDE.md](../docs/CONVERSION_GUIDE.md)。

## 文件

| 文件 | 作用 |
|---|---|
| `prune_vocab.py` | **词表裁剪**：剔除中英文以外的语言，产出重映射词表 + `vocab_map.json` |
| `rebuild_artifact.py` | 按新词表**逐字节重建**制品（纯行 gather，零解量化） |
| `probe_swift.py` | Swift/基线 GGUF 结构比对（tensor 表 / tokenizer / chat template 逐字节） |
| `parallel_dl.py` | 并行 range 下载器（走代理、断点续传、SHA256 校验） |
| `frontend_hashes.py` | 打印制品 frontend 资源哈希（比对裁剪一致性用） |
| `speed_probe.py` | 镜像/代理带宽探针 |
| `verify/` | 验证脚本（见下） |

## verify/

| 脚本 | 判据 |
|---|---|
| `check_row_gather.py` | 行 gather 逻辑预演（写 8 GB 之前）|
| `check_vocab_prune.py` | 裁剪前后 tokenizer 对同一文本的分词数不变 |
| `check_tokenizer_loss.py` | 真实语料上的**原始**分词损失（zh/en）|
| `check_pure_corpus_loss.py` | 分离「纯中英文 / 含外语」段落的分词损失 |
| `check_pruned_artifact.py` | 重建制品的**逐字节**审计（全量行 + 抽样张量）|
| `check_pruned_serving.py` | 在线功能（中英/代码/结构化输出/已删语言）|
| `inspect_dropped_hits.py` | 定位语料里被删 token 的命中位置（判断是否误删）|
| `probe_ninfer_thinking.py` | 探测服务端接受的 thinking 开关 |

## 依赖

```bash
python -m pip install -r requirements.txt
```

脚本内的 `REPO_SRC`（引擎树）与制品路径默认指向本仓的 `src/cranebw` 与 `artifacts/`，
如目录不同请用命令行参数覆盖。

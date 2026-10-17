# SWIFT_DEPLOYMENT · Swift Bonsai 2 27B 移植到 NInfer

> **目标**：把 [UkisAI 的 Swift Bonsai 2](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)
> （Ternary Bonsai 2 27B 的推理效率微调版，官方称**思考 token 中位减少 39.8%**、得分持平或更高）
> 跑在 **NInfer** 引擎上，并套用本项目的**词表裁剪**。
>
> **公开状态**：NInfer 上已有更早的 Swift 移植（`fyb1214`、`knoopx`，2026 年 9 月底）；
> **带词表裁剪 + 逐字节验证的移植，本仓是第一个。**
>
> 实测机器：Windows 11 + RTX 5060 Ti 16 GB（`sm_120a`）。
>
> **发布渠道**：源码在 [GitHub](https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port) /
> [Gitee 镜像](https://gitee.com/fengjiaheng123/swift-bonsai-2-ninfer-port)；
> 完整包 / 未裁剪包在 [Hugging Face](https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port)
> 和 [魔搭](https://www.modelscope.cn/models/jiaheng0815/Swift-Bonsai-2-NInfer-Port)。
>
> 文档最后校对：2026年10月17日。

---

## 1. 为什么"只换权重"就能整条链复用

Swift 是**权重合并式微调**（HF 模型卡原文：*the Swift correction merged into the ternary weights;
no adapter file, patch, or extra flag is needed*），文件布局与基座完全一致：

| 项 | Swift-Bonsai-2-PQ2_0.gguf | 基线 Ternary-Bonsai-2-27B-PQ2_0.gguf |
|---|---|---|
| 字节数 | 7,206,168,928 | 7,206,168,928（**完全相同**）|
| SHA256 | `5912bb73…63d5`（HF 官方值，已校验）| — |
| 结构 | 见 §2 判据表 | 同左 |

所以移植策略是：**权重侧走与基线完全同一条流水线，只把 GGUF 换成 Swift 的**；
引擎、模板、裁剪域、启动参数全部复用。

## 2. 结构比对（移植前的第一道判据）

`conversion/probe_swift.py` 直读两个 GGUF 的头部逐项比对：

| 判据 | 结果 |
|---|---|
| tensor 列表（851 个，含顺序）| ✅ 完全一致 |
| quant type 直方图 / 每个 tensor 的 shape | ✅ 完全一致 |
| `tokenizer.ggml.tokens`（248,320）| ✅ sha256 `3fb92abc…` 两边相同 |
| `tokenizer.ggml.merges`（247,587）| ✅ sha256 `47c34e7a…` 两边相同 |
| `tokenizer.ggml.token_type` | ✅ 相同 |
| chat template（8,952 字符）| ✅ 逐字节一致 |
| 其余 kv 元数据（含 Hadamard 符号表）| ✅ **0 个差异** |

**结论**：Swift 与基线**只有权重值不同**。因此——

- 裁剪词表资源（208,030 域）可直接复用，`prune_vocab.py` 对 Swift 制品的输出与基线
  **逐字节相同**（4/4 文件 sha256 一致，见 §4）；
- 引擎无需任何新改动；
- Hadamard 符号表、special token id 全部不变。

## 3. 打包（与基线同一条命令）

打包器 `pack.py` **不在本 git 仓内**（它来自魔搭指南 `shensanshu/ninfer-ada-ternary`）：

1. 从魔搭指南原仓获取 `tools/pack.py` 与 `tools/verify/`（本机参考副本在 `guide/`，不入库）；
2. 若使用[Hugging Face](https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port) /
   [魔搭](https://www.modelscope.cn/models/jiaheng0815/Swift-Bonsai-2-NInfer-Port) 的
   **未裁剪包**，它提供自包含容器库 `ninfer-tools/`——把它指给 `NINFER_SRC`，
   打包器仍用第 1 步的 `pack.py`（包内 `README.txt` 有完整步骤）。

```powershell
# 例：Windows；%REPO% 是本仓根目录
set REPO=D:\ai-ninfer
set NINFER_SRC=%CD%\ninfer-tools
set PACK=path\to\pack.py

python %PACK% check --gguf <Swift.gguf> --template "%REPO%\models\template\model.ninfer"
python %PACK% build "%REPO%\artifacts\swift-bonsai2-27b.ninfer" `
  --gguf <Swift.gguf> --template "%REPO%\models\template\model.ninfer"
```

产物与基线逐字节同构：**8,306,927,628 B = 7.736 GiB、1126 对象、文本 6.696 GiB、
借用 1.040 GiB**（mtp 12 / vision 333 / frontend 6 / draft_head 2）。

**四件套验证**（判据与结果见 [CONVERSION_GUIDE.md](CONVERSION_GUIDE.md) §4）：

| 验证 | Swift 制品结果 |
|---|---|
| 载荷布局指纹 | ✅ scale 全正、无 NaN，`zero_share=0.3279`（理论 0.3278）|
| 符号表 vs 元数据 | ✅ `max\|diff\| = 0.0` |
| 行序规则（5 组）| ✅ 全部逐行成立，0 重复 0 缺失 |
| 全量装配审计 | ✅ **15/15**（含 embedding 248,320 行 0 顺序错位）|

## 4. 词表裁剪（Swift 版）

```powershell
# 在 git 仓内（默认用 src\cranebw 的容器库）；独立包内先 set NINFER_SRC=...\ninfer-tools
python conversion\prune_vocab.py     --artifact artifacts\swift-bonsai2-27b.ninfer --out-dir artifacts\pruned
python conversion\rebuild_artifact.py --artifact artifacts\swift-bonsai2-27b.ninfer --pruned artifacts\pruned `
  --out artifacts\swift-bonsai2-27b-pruned.ninfer
```

| 项 | 结果 |
|---|---|
| 裁剪资源 vs 基线 | ✅ **4/4 文件 sha256 逐字节相同**（tokenizer / tokenizer_config / generation_config / vocab_map）|
| 词表 | 248,044 → 207,997（删 40,047，全部是其它语言）+ 33 special = **208,030** |
| 制品 | 7.736 GiB → **7.629 GiB**（−110 MiB）|
| 重建审计 | ✅ **ALL CHECKS PASSED**（embedding/output_head 全 208,030 行两平面逐字节精确）|
| 分词损失 | ✅ **纯中英文 0.0000%**（原始 zhwiki 的 0.2% 全部在外语片段内部）|

方法细节见 [VOCAB_PRUNE.md](VOCAB_PRUNE.md)。

## 5. 上线与验收

启动（262k / nvfp4 / chunk 8192 / MTP K=3 / vision）：

```powershell
launch\launch_windows.bat
# engine ready | total 7.1s | weights 7.62 GiB
# capacity | KV 262,144 tokens, nvfp4, explicit | pages 4,096/4,096 | free 432.7 MiB
```

| 验收项 | 结果 |
|---|---|
| 在线上功能（中/英/代码/标点/已删语言/结构化输出）| ✅ **ALL SERVING CHECKS PASSED** |
| 结构化解码（JSON schema）| ✅ `{"city":"Shanghai","population":24870000}` |
| PPL 正确性 | ✅ **4.991217**（基线 4.989455；判据：个位数=正确）|
| 俄语（被裁语言）| ✅ 仍可应答、不崩（退化但可用）|
| 思考 token A/B | 见 [BENCHMARKS.md](BENCHMARKS.md) |

## 6. 复现清单

| 步 | 命令 | 期望 |
|---|---|---|
| 1 | 下载 Swift GGUF（官方；大文件走代理直连，见 `conversion/parallel_dl.py`）| SHA256 = `5912bb73…63d5` |
| 2 | `probe_swift.py` 结构比对 | 7 项判据全一致（§2）|
| 3 | `pack.py check` + `build` | `RESULT: OK`；7.736 GiB / 1126 对象 |
| 4 | 四件套验证 | 全绿（§3）|
| 5 | 裁剪 + 重建 + 审计 | ALL CHECKS PASSED（§4）|
| 6 | 起服务 + 在线验证 + PPL | ALL PASSED；PPL 4.99（§5）|

## 7. 诚实交代（已知边界）

- **`draft_head` 借用自模板**（基线 Qwen3.8-27B 的 NextN 头与高频短名单）——
  Swift 微调只改了主权重，草稿头无法从三元包重建，这与基线部署的取舍相同；
- **PTQ1_0（1-bit）变体未做**：HF 卡注明其是"较早的 Swift 版本"（PQ2_0 才是更新后的合并权重），
  本仓只移植 PQ2_0；引擎支持 PTQ1_0 几何，感兴趣者可照本仓流水线自行做；
- **Swift 卡自身声明**这是实验性研究发布（指令跟随 / 工具使用 / 开放式编码仍有波动），
  本仓的实测数据也按此口径理解；
- 长上下文 / MMLU 的完整测试只在**基线**制品上跑过（见 [BENCHMARKS.md](BENCHMARKS.md)），
  Swift 目前验证到 PPL + 在线功能 + 思考 token A/B。

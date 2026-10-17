# 01 · 编译与部署

> 目标：在 **Windows 11 + RTX 5060 Ti 16 GB（Blackwell `sm_120a`，36 SM）** 上
> 把 **NInfer** 引擎编出来，跑通 **Ternary Bonsai 2 27B**。
>
> 参照物：魔搭指南 **`shensanshu/ninfer-ada-ternary`**（目标卡 RTX 4080 SUPER / Ada `sm_89` / 32 GB）。
> 本机是 Blackwell + 16 GB，因此**引擎树、编译架构、SM 计数**三处做了适配，其余严格按指南走。
>
> 源码已在 [GitHub](https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port) 与
> [Gitee 镜像](https://gitee.com/fengjiaheng123/swift-bonsai-2-ninfer-port) 提供；
> 不想自己编译的话，用[完整包](https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port)
> （引擎 exe + DLL 已内置）。文档最后校对：2026年10月17日。

---

## 1. 环境

| 项 | 值 |
|---|---|
| 系统 | Windows 11 Pro for Workstations |
| GPU | RTX 5060 Ti 16 GB（Blackwell `sm_120a`，36 SM，可用 16,311 MiB）|
| 内存 | 32 GB（单个引擎提交量约 39–49 GB，**只够跑一个**）|
| CUDA | 13.3（VS Build Tools 2022）|
| Python | 3.10+（本机 3.12，numpy / tokenizers / huggingface_hub；见 `conversion/requirements.txt`）|

> **一张卡只跑一个引擎。** 两个实例不只是抢显存：先跑的会被 WDDM 挤到系统内存，
> 后启动的在 `pinning host state` 阶段报 `cudaMallocHost failed: ... out of memory`
> —— 这个报错看着像内存 bug，其实含义是「已经有一个引擎在跑了」。

## 2. 引擎树：为什么用 CraneBW 而不是指南基线

指南的基线是 `Ambolio/ninfer-4090-windows @ 6eb70a07`（v1.0.8 线），
**该仓已从 GitHub 消失（截至 2026年10月17日仍为 404）**。

改用 **`CraneBW/ninfer-ternary-bonsai-ada`**：它自述就是「Ambolio v1.0.8 Ada 线 + 魔搭三元补丁」
的派生，容器版本同为 v2，`patches/` 与指南逐文件对应。本机树在 `src/cranebw/`。

## 3. 三处适配（Blackwell 与 Ada 的差异）

| # | 问题 | 改动 |
|---|---|---|
| 1 | `src/core/device.h` 只有 `NINFER_SM89`(128 SM) / `NINFER_SM86`(82 SM) 分支，`120a` 直接 `#error` | 新增 `NINFER_SM120` 分支，`kTargetSmCount = 36`（5060 Ti = 36 SM）|
| 2 | 根 `CMakeLists.txt` 只对 `89` 定义架构宏，`120a` 不定义任何宏 | 新增 `MATCHES "^120"` → `NINFER_SM120=1` |
| 3 | 指南基线仓已 404 | 换 CraneBW 派生树（见 §2）|

**编译架构用 `120a`（上游参考架构）而不是 `120`**：warp-specialized TMA / NVFP4 / k8v4
那批内核只在 `120a` 下完整编译。好处实测验证：**`nvfp4` KV 能把池子开到 262,144 token**
（= 模型原生 256K 上限），这是指南 Ada 线做不到的（Ada 线只支持 bf16/int8/fp8）。

**编码注意**：VS 构建树是 UTF-8 工程，`engine\build_windows.bat` 里设了 `VSLANG=1033`；
`ninja` 增量编译改 `.h` 后必须让依赖它的 `.cpp` 重编（MSVC 头依赖扫描在该树里不可靠）。

## 4. 打包器的两处最小修正（不改算法）

`pack.py` 来自魔搭指南，**不在本 git 仓内**（本机参考副本 `guide/tools/pack.py` 不入库）。
在你的 pack.py 副本上做这两处修正：

- `NINFER_ROOT` 指到本机引擎树（`D:\ai-ninfer\src\cranebw`）；
- 模板已含 `text/hadamard_signs/widths` 时不再重复追加（否则 `plan_duplicate object name` 中止）。
  这两个对象**始终由 GGUF 元数据生成**，不会从模板借用。

## 5. 构建与产物

```powershell
engine\build_windows.bat   # 改了 C++ 源码后重编（vcvars64 + ninja -C build\b1）
```

产物在 `build\b1\apps\`，已加入用户级 PATH：

| exe | 用途 |
|---|---|
| `ninfer.exe` | 命令行单次问答 / 多轮（`--prompt` 或 `--messages <json>`）|
| `ninfer-serve.exe` | HTTP 服务（OpenAI + Anthropic 双协议）|
| `ninfer-perplexity.exe` | PPL 正确性验收 |

引擎侧的改动（相对 CraneBW 树）：

1. **Blackwell 适配**（上面 §3 的 1、2 两处）；
2. **裁剪词表域**：`kTokenDomain` 208030、`bindings.cpp` 的 embedding/output_head
   行数、33 个 special token 的新 id —— 见 [VOCAB_PRUNE.md](VOCAB_PRUNE.md)；
   三元代码本身**未改动一行**。

## 6. 制品来源与溯源闭环

制品是本地从**官方权重**打出来的（本仓不发权重）：

| 项 | 值 |
|---|---|
| 权重来源 | `prism-ml/Ternary-Bonsai-2-27B-gguf` 的 `PQ2_0`（1.58-bit 三元）与 `ukisai/Swift-Bonsai-2-GGUF`（Swift 版）；下载走代理直连官方站，见 `conversion/parallel_dl.py` |
| 底座 | Qwen3.8-27B（64 层：16 纯注意力 + 48 GDN）|
| 打包器 | `guide/tools/pack.py`（**逐字节搬运 2-bit 码字，绝不反量化再量化**）|
| 模板 | `wei231/ternary-bonsai-3080-sm86-ninfer` 的 `model.ninfer`（SHA256 已对上发布方校验和）|
| 借用对象 | `mtp/*` 12（MTP 投机头）、`frontend/*` 6、`text/draft_head*` 2、`vision/*` 333 |
| 产物 | 8,306,927,628 B = **7.736 GiB**；文本 6.696 GiB；1126 对象 |

**溯源闭环**：制品里的 `frontend/tokenizer.json` 与 ModelScope 上 `Qwen/Qwen3.8-27B`
的官方 `tokenizer.json` **逐字节相同**（SHA256 `0997f410…229b9f3`）。

打包流程与四件套验证的判据见 [CONVERSION_GUIDE.md](CONVERSION_GUIDE.md)；
验收数据（PPL / MMLU / 长上下文 / 视觉）见 [BENCHMARKS.md](BENCHMARKS.md)。

## 7. 排错

| 现象 | 处理 |
|---|---|
| 启动报显存不足 | 降 `-c`（如 131072 / 65536）；或去掉 `--vision` |
| 重新编译报 `LNK1104 无法打开 exe` | 先 `launch\stop_server.bat`（exe 被占用）|
| 控制台看日志中文乱码 | 编码问题不是模型问题；用 `chcp 65001` 或读日志时 `-Encoding UTF8` |

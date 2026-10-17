# Swift Bonsai 2 on NInfer

> **把 Swift Bonsai 2 27B 移植到 NInfer 引擎**，并附带本项目独有的
> **词表裁剪**（248,320 → 208,030，纯中英文零损失）与**逐字节验证**工具链。
>
> 实测环境：**Windows 11 · RTX 5060 Ti 16 GB（Blackwell `sm_120a`）**。
> 引擎、已裁剪模型、剪切器与启动脚本都已打包，**解压即用**；
> 源码与文档在本仓，发布包见 [Hugging Face](#下载) 与 [魔搭](#下载)。

**仓库 / 发布渠道**

| 渠道 | 地址 | 内容 |
|---|---|---|
| **GitHub（主仓）** | https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port | 源码、文档、引擎补丁 |
| **Gitee（国内镜像）** | https://gitee.com/fengjiaheng123/swift-bonsai-2-ninfer-port | 与 GitHub 同步的源码镜像 |
| **Hugging Face** | https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port | 完整包 / 未裁剪包 / 引擎源码 zip |
| **魔搭 ModelScope** | https://www.modelscope.cn/models/jiaheng0815/Swift-Bonsai-2-NInfer-Port | 同上（国内直连） |

> **定位**：本项目的重点是 **Swift-Bonsai-2**（基线 Ternary Bonsai 2 只用于 A/B 对照）。
> NInfer 上更早已有 [fyb1214](https://huggingface.co/fyb1214/Swift-Bonsai-2-27B-NInfer)
> 与 [knoopx](https://huggingface.co/knoopx/Swift-Bonsai-2-27B-NInfer) 的 Swift 移植；
> 本仓的首发点是**词表裁剪 + 逐字节验证 + 开箱即用完整包**——据我们所知，
> 公开移植中还没有第二家把 Swift-Bonsai-2 的 NInfer 制品裁剪到 208,030 并完成闭环验证。

---

## 性能亮点（全部本机实测）

| 指标 | 数值 |
|---|---|
| 上下文 | **262,144 token**（模型原生上限，`nvfp4` KV，16 KiB/token）|
| 预填充 | **654 tok/s**（20k 级）/ **484 tok/s**（131k）|
| 解码 | **64 tok/s**（散文）/ **82 tok/s**（131k 深度，MTP K=3）|
| 前缀缓存 | 命中后 TTFT **79 ms**（15k）/ **825 ms**（131k）|
| 长上下文召回 | ✅ **144,975 token** 大海捞针答对（基线制品）|
| PPL（正确性判据）| **Swift 4.9912** vs 基线 4.9895（同语料，个位数=正确）|
| 视觉 | ✅ 官方图表 / 场景图 / 双图 / 视频全部实测通过（基线制品；Swift 同构）|
| 词表裁剪 | 词表 **248,320 → 208,030**；纯中英文分词损失 **0.0000%**；制品 −110 MiB |
| 制品验证 | 四件套全绿（载荷布局 / 符号表 / 行序 / 全量装配 15/15）+ 裁剪审计全绿 |

## 硬件与软件要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，**Blackwell `sm_120a`**（RTX 50 系）为实测目标；其他架构需按 `engine/README.md` 改 SM 分支 |
| 显存 | **16 GB**（权重 7.62 GiB + KV 池 4 GiB + 运行时 ≈ 15.0 GiB）|
| 系统 | Windows 10/11 64-bit（完整包）；源码构建脚本为 Windows，`launch/launch_linux.sh` 为未实测模板 |
| 工具链 | 仅源码构建需要：VS Build Tools 2022 + **CUDA 13.3** + CMake + Ninja |
| Python | 3.10+（numpy、tokenizers；见 `conversion/requirements.txt`）|

## 下载

> 两个平台内容一致，国内用户优先走 **魔搭** 或 **Gitee**。

- **Hugging Face**：https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port
- **魔搭 ModelScope**：https://www.modelscope.cn/models/jiaheng0815/Swift-Bonsai-2-NInfer-Port

| 文件 | 内容 | 体积 |
|---|---|---|
| ★ **`ninfer-swift-bonsai2-complete-v1.0.0-single.zip`** | **完整包（单文件，免分卷）**：引擎 exe×3 + FFmpeg/VC 运行库 DLL + **已裁剪模型**（直接可服务）+ 词表剪切器 + 启动脚本。**解压 → 双击 `launch\launch_windows.bat` 即用**（CUDA 已静态链接，无需装 CUDA）| 6.85 GiB |
| `swift-bonsai2-27b-unpruned-v1.0.0-single.zip` | （可选）**未裁剪制品** + 剪切器 + 自包含容器库——只有想自己重跑裁剪 / 用未打补丁引擎时才需要 | 6.35 GiB |
| `engine/ninfer-engine-ternary-bonsai-swift-port-v1.0.0.zip` | **完整 C++ 引擎源码**（基树 `ad6cb46` + 本仓补丁）—— 同一份源码也在本仓 `src/cranebw/` | 11 MB |

### 镜像 / 代理说明

- 权重原始来源是 Hugging Face（`ukisai/Swift-Bonsai-2-GGUF`、`prism-ml/Ternary-Bonsai-2-27B-gguf`）。
- 本机下载大文件**走用户代理直连原站**（`http://127.0.0.1:7897`），不要用 hf-mirror；
  `conversion/parallel_dl.py` 支持代理 + 20 连接 + 断点续传 + SHA256 校验。
- 发布包本身已在国内平台（魔搭 / Gitee）镜像，通常不需要再走代理。

## 快速开始

### A. 用完整包（推荐，零构建）

```text
1) 从 Hugging Face 或魔搭下载 ninfer-swift-bonsai2-complete-v1.0.0-single.zip
2) 用 7-Zip 解压（单文件，无需分卷）
3) 双击 launch\launch_windows.bat
4) curl.exe http://127.0.0.1:8080/v1/models
```

客户端 Base URL `http://127.0.0.1:8080/v1`，`model` 填 **`swift-bonsai2-27b`**，
api_key 随便填（本机未开鉴权）。**思维链在 `reasoning_content` 字段**。

### B. 从源码构建

```powershell
git clone https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port   # 或 Gitee 镜像
cd Swift-Bonsai-2-NInfer-Port

# 1. 引擎源码已内置（src/cranebw，基树 ad6cb46 + 本仓补丁已套用），直接编译
engine\build_windows.bat          # 工具链不同则用 VCVARS / CUDA_PATH / CMAKE_BIN 覆盖

# 2. 生成制品（下载官方 GGUF -> pack -> 四件套 -> 词表裁剪）
#    打包器 pack.py 来自魔搭指南 shensanshu/ninfer-ada-ternary，不在本 git 仓内；
#    未裁剪发布包内附自包含容器库 ninfer-tools\tools。详见 docs/CONVERSION_GUIDE.md
#    产出：artifacts\swift-bonsai2-27b-pruned.ninfer

# 3. 起服务 + 调用
launch\launch_windows.bat
curl.exe http://127.0.0.1:8080/v1/models
```

> 如果你更喜欢从上游基树开始：`git clone https://github.com/CraneBW/ninfer-ternary-bonsai-ada src\cranebw`，
> `git checkout ad6cb46`，再 `git apply engine\patches\ninfer-bonsai2-port.patch`。

## 仓库结构

```
README.md  LICENSE  NOTICE  .gitignore
src/cranebw/    ★ 完整引擎源码（基树 ad6cb46 + 本仓补丁，1449 文件；可直接构建）
conversion/     制品转换与验证：词表裁剪 / 逐字节重建 / 全套验证脚本 / 下载器
engine/         引擎改动说明 + patch（相对基树的 diff）+ 构建脚本
launch/         启动脚本（Windows .bat / Linux .sh 模板）+ 聊天客户端 + PPL
bench/          性能与思考 token 基准（可复跑）
tests/          端到端验证（结构化输出 / strict 工具）
docs/           全部文档（下表）
artifacts/      制品说明（不发布权重与 .ninfer）
eval/           PPL 语料 + MMLU 5-shot 评测（完整报告）
```

> `guide/` 是本地参考的魔搭指南原仓，**不在本仓跟踪范围内**，也不随 git 分发。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/SWIFT_DEPLOYMENT.md](docs/SWIFT_DEPLOYMENT.md) | ★ **Swift Bonsai 2 移植全流程**（结构比对 → 打包 → 验证 → 上线）|
| [docs/CONVERSION_GUIDE.md](docs/CONVERSION_GUIDE.md) | 从官方 GGUF 到 `.ninfer` 的转换步骤详解 + 四件套判据 |
| [docs/VOCAB_PRUNE.md](docs/VOCAB_PRUNE.md) | 词表裁剪方法：规则、无损重建、验证、收益与边界 |
| [docs/ENGINE_BUILD.md](docs/ENGINE_BUILD.md) | 引擎编译与 Blackwell 适配 |
| [docs/USAGE.md](docs/USAGE.md) | 使用说明：API / 客户端 / Claude Code / 视觉 / 排错 |
| [docs/TUNING.md](docs/TUNING.md) | 262k 上下文最佳参数（30 次起服务实测）|
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | PPL / 长上下文 / 视觉 / MMLU / 思考 token A/B 实测 |
| [engine/README.md](engine/README.md) | 引擎改了哪四组、为什么 |

> 文档最后校对：**2026年10月17日**。命令与路径以本仓当前 `main` 为准。

## 引用与致谢

- **Swift Bonsai 2**（推理效率微调）— [UkisAI](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)，Apache-2.0
- **Ternary Bonsai 2 27B**（1.58-bit 三元权重）— [PrismML](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- **Qwen3.8-27B**（底座 / 分词器 / MTP 头）— Qwen
- **NInfer 引擎与三元移植** — [CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)（Ambolio v1.0.8 线）
- **打包器与映射规格** — 魔搭指南 [shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)
- 社区参考：[mengyou6688](https://www.modelscope.cn/models/mengyou6688/Bonsai-2-27B-PQ2_0-CRACK-NInfer)（Linux）、
  [yutils](https://github.com/yutils/ninfer-Bonsai-2-5070)（RTX 5070）、
  [fyb1214](https://huggingface.co/fyb1214/Swift-Bonsai-2-27B-NInfer)、
  [knoopx](https://huggingface.co/knoopx/Swift-Bonsai-2-27B-NInfer)

完整授权信息见 [NOTICE.md](NOTICE.md)；本仓代码与文档采用
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) **Apache License 2.0**。

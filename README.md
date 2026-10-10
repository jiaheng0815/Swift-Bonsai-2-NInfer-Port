# Swift Bonsai 2 on NInfer

> **把 Swift Bonsai 2 27B 移植到 NInfer 引擎** —— 社区里第一个把这款
> 「思考 token 少 39.8%」的三元微调模型跑在 NInfer 上的移植，附带本项目独家的
> **词表裁剪**（248,320 → 208,030，中英文零损失）与**逐字节验证**工具链。
>
> 实测环境：**Windows 11 · RTX 5060 Ti 16 GB（Blackwell `sm_120a`）**。
> 权重不随仓库分发；**未裁剪的 `.ninfer` 制品（配词表剪切器）随
> [Release v1.0.0](../../releases/tag/v1.0.0) 提供**，由官方 Apache-2.0 权重本地打包。

---

## 性能亮点（全部本机实测）

| 指标 | 数值 |
|---|---|
| 上下文 | **262,144 token**（模型原生上限，`nvfp4` KV，16 KiB/token）|
| 预填充 | **654 tok/s**（20k 级）/ **484 tok/s**（131k）|
| 解码 | **64 tok/s**（散文）/ **82 tok/s**（131k 深度，MTP K=3）|
| 前缀缓存 | 命中后 TTFT **79 ms**（15k）/ **825 ms**（131k）|
| 长上下文召回 | ✅ **144,975 token** 大海捞针答对 |
| PPL（正确性判据）| **Swift 4.9912** vs 基线 4.9895（同语料，个位数=正确）|
| 视觉 | ✅ 官方图表 / 场景图 / 双图 / 视频全部实测通过 |
| 词表裁剪 | 词表 **248,320 → 208,030**；中英文分词损失 **0.0000%**；制品 −110 MiB |
| 制品验证 | 四件套全绿（载荷布局 / 符号表 / 行序 / 全量装配 15/15）+ 裁剪审计全绿 |

## 硬件与软件要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，**Blackwell `sm_120a`**（RTX 50 系）为实测目标；其他架构需按 `engine/README.md` 改 SM 分支 |
| 显存 | **16 GB**（权重 7.62 GiB + KV 池 4 GiB + 运行时 ≈ 15.0 GiB）|
| 系统 | Windows 11（本仓脚本为 Windows；`launch/launch_linux.sh` 为未实测模板）|
| 工具链 | VS Build Tools 2022 + **CUDA 13.3** + CMake + Ninja |
| Python | 3.10+（numpy、tokenizers；见 `conversion/requirements.txt`）|

## 下载（Release assets）

| 包 | 内容 | 体积 |
|---|---|---|
| `swift-bonsai2-27b-unpruned-v1.0.0.zip.001..004` | **未裁剪制品**（7.736 GiB）+ **词表剪切器** + 包内说明/校验（分卷 zip，需全部下载后用 7-Zip 打开 `.001` 解压）| 6.5 GiB |
| `ninfer-engine-ternary-bonsai-swift-port-v1.0.0.zip` | **完整 C++ 引擎源码**（基树 `ad6cb46` + 本仓补丁，可直接构建）| 11 MB |

> 只要 tools + patch 的话，本仓 `engine/` 与 `conversion/` 已经够用；
> 上面两个包是给"不想自己 clone/pack"的场景准备的。

## 快速开始

```powershell
# 1. 拿基树 + 打补丁（引擎改动见 engine/README.md）
git clone https://github.com/CraneBW/ninfer-ternary-bonsai-ada src\cranebw
cd src\cranebw && git checkout ad6cb46 && git apply ..\..\engine\patches\ninfer-bonsai2-port.patch && cd ..\..

# 2. 编译引擎（改 build_windows.bat 顶部的工具链路径）
engine\build_windows.bat

# 3. 生成制品（下载官方 GGUF -> pack -> 四件套 -> 词表裁剪，见 docs/CONVERSION_GUIDE.md）
#    产出：artifacts\swift-bonsai2-27b-pruned.ninfer

# 4. 起服务 + 调用
launch\launch_windows.bat          # 双击也行
curl.exe http://127.0.0.1:8080/v1/models
```

客户端：Base URL `http://127.0.0.1:8080/v1`，`model` 填 **`swift-bonsai2-27b`**，
api_key 随便填（本机未开鉴权）。**思维链在 `reasoning_content` 字段**。

## 仓库结构

```
README.md  LICENSE  NOTICE  .gitignore
conversion/     制品转换与验证：词表裁剪 / 逐字节重建 / 全套验证脚本 / 下载器
engine/         引擎改动说明 + patch（相对 CraneBW 基树 ad6cb46）
launch/         启动脚本（Windows .bat / Linux .sh 模板）+ 聊天客户端 + PPL
bench/          性能与思考 token 基准（可复跑）
tests/          端到端验证（结构化输出 / strict 工具）
docs/           全部文档（下表）
artifacts/      制品说明（不发布权重与 .ninfer）
eval/           PPL 语料 + MMLU 5-shot 评测（完整报告）
```

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

## 引用与致谢

- **Swift Bonsai 2**（推理效率微调）— [UkisAI](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)，Apache-2.0
- **Ternary Bonsai 2 27B**（1.58-bit 三元权重）— [PrismML](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- **Qwen3.8-27B**（底座 / 分词器 / MTP 头）— Qwen
- **NInfer 引擎与三元移植** — [CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)（Ambolio v1.0.8 线）
- **打包器与映射规格** — 魔搭指南 `shensanshu/ninfer-ada-ternary`
- 社区参考：[mengyou6688](https://www.modelscope.cn/models/mengyou6688/Bonsai-2-27B-PQ2_0-CRACK-NInfer)（Linux）、[yutils](https://github.com/yutils/ninfer-Bonsai-2-5070)（RTX 5070）

完整授权信息见 [NOTICE.md](NOTICE.md)；本仓代码与文档采用
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) **Apache License 2.0**。

# NOTICE — 版权与归属

本仓是 **NInfer** 的派生作品，把 **Swift Bonsai 2 27B**（UkisAI 对
**PrismML Ternary Bonsai 2 27B** 的推理效率微调版，底座 **Qwen3.8-27B**）
移植到 NInfer 引擎，并给出**词表裁剪**与**逐字节验证**的完整工具链。

**本 git 仓不含任何模型权重或由权重派生的 `.ninfer` 制品**（`.gitignore` 已排除）。
可选的**开箱即用发布包**（含已裁剪制品）在以下渠道另行分发：

| 渠道 | 地址 | 内容 |
|---|---|---|
| Hugging Face | https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port | 完整包 / 未裁剪包 / 引擎源码 |
| 魔搭 ModelScope | https://www.modelscope.cn/models/jiaheng0815/Swift-Bonsai-2-NInfer-Port | 同上 |
| GitHub（主仓） | https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port | 源码与文档 |
| Gitee（镜像） | https://gitee.com/fengjiaheng123/swift-bonsai-2-ninfer-port | 源码与文档 |

> Swift Bonsai 2 与 Ternary Bonsai 2 均按 **Apache-2.0** 发布，允许再分发派生制品；
> 发布包内保留完整 `NOTICE.txt` 与许可文本。若上游许可发生变化，以官方模型卡为准。

## 上游组件

| 组件 | 作者 / 来源 | 许可 |
|---|---|---|
| NInfer 引擎（基树）| [CraneBW/ninfer-ternary-bonsai-ada](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)（Ambolio v1.0.8 Ada 线 + 三元补丁的派生）| Apache-2.0 |
| 上游引擎参考 | Ambolio/ninfer-4090-windows v1.0.8（原仓已下线）| Apache-2.0 |
| 打包器与映射规格 | 魔搭指南 [shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary) | Apache-2.0 |
| 本仓的引擎改动 | 见 [engine/](engine/)（patch，相对基树 `ad6cb46`）| Apache-2.0 |

## 模型权重（官方渠道获取，均不入库）

| 模型 | 作者 | 许可 |
|---|---|---|
| Swift Bonsai 2（PQ2_0）| [UkisAI](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF) | Apache-2.0 |
| Ternary Bonsai 2 27B（PQ2_0 / mmproj）| [PrismML](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | 见其模型卡 |
| Qwen3.8-27B（分词器 / MTP 头来源）| Qwen | 见其模型卡 |
| 制品模板（借 MTP 头 / 分词器 / 视觉塔）| wei231/ternary-bonsai-3080-sm86-ninfer | 见其仓声明 |

## 社区参考实现

| 仓 | 作者 | 平台 |
|---|---|---|
| mengyou6688/Bonsai-2-27B-PQ2_0-CRACK-NInfer | mengyou6688 | Ubuntu / Linux |
| yutils/ninfer-Bonsai-2-5070 | yutils | Windows / RTX 5070 |

## 本项目原创部分（Apache-2.0）

- **词表裁剪方法**：`conversion/prune_vocab.py`、`conversion/rebuild_artifact.py`
  与全部验证脚本（`conversion/verify/`）——248,320 → 208,030，逐字节重建；
- **引擎改动**：Blackwell（`sm_120a`）编译适配、裁剪词表域、结构化解码接线
  （`engine/patches/`）；
- 全部文档与实测数据（`docs/`）。

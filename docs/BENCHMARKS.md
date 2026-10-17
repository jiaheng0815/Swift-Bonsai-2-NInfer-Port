# BENCHMARKS · 实测数据汇总

> 机器：RTX 5060 Ti 16 GB（可用 16,311 MiB）+ 32 GB RAM，Windows 11。
> 引擎配置（除注明外）：`262144 / nvfp4 / prefill-chunk 8192 / MTP K=3 / vision`。
> **所有数字都是本机实测**，不是引用值；判据写在每节里。
> 最后校对：2026年10月17日（v1.0.0 发布包）。

---

## 1. 正确性：PPL（Swift vs 基线）

同一语料 `eval/ppl_sample.txt`（1,789 token）、同 `fp8` KV、同 stride：

| 制品 | mean NLL | **PPL** |
|---|---|---|
| **swift-bonsai2-27b-pruned** | 1.607680 | **4.991217** |
| bonsai2-27b-pruned（基线）| 1.607327 | **4.989455** |

**判据**：个位数~十几 = 数值管线正确；几百 = 错。两者都正确，且**几乎一致（差 0.04%）**——
Swift 的微调没有损伤语言建模能力。

```powershell
launch\run_ppl.bat [artifact]      # 默认跑 Swift
```

## 2. 思考 token A/B（Swift 的卖点，如实报告）

官方模型卡：**GPQA-Diamond 中位思考 token ↓39.8%**（20,066 → 16,245），得分 84.18% → 84.34%。
本机的 8 题本地 A/B（同 prompt、同参数 `temperature 1.0`、`reasoning_effort xhigh`）：

| prompt | Swift | 基线 |
|---|---|---|
| math1（平均速度）| 66 | 54 |
| math2（3/5 整除计数）| 130 | 166 |
| logic1（三个错标签盒子）| 168 | 89 |
| logic2（5 台机器 5 分钟）| 100 | 227 |
| code1（Python 一行反转词序）| 136 | 173 |
| zh1（三句话解释三重量化）| 779 | 827 |
| zh2（进出水管问题）| 135 | 113 |
| reason1（理发师悖论）| 301 | 131 |
| **均值 / 中位** | **227 / 136** | **222 / 148** |

**诚实结论**：在这组**短推理** prompt 上（每题思考 50–830 token），Swift 与基线**没有显著差异**。
官方 39.8% 是在 GPQA-Diamond 那种**长推理**（中位 2 万思考 token、输出上限 81,920）上测的——
本机没有该数据集，无法复现其规模；本表只说明：**在短任务上不应期待 Swift 的收益**。
（附带观察：zh1 上 Swift 对"三重量化"的解释更准确——基线答成了"分三组不同精度"。）

```powershell
python bench\swift_bench.py --label base  --model bonsai2-27b      # 先起基线服务
python bench\swift_bench.py --label swift --model swift-bonsai2-27b
```

答案正确性抽查（两者都对）：水管题 **4 小时 48 分**、理发师悖论"不存在"、78.5→84 取整、
代码 `print(' '.join(input().split()[::-1]))`。

## 3. 速度（基线制品实测，Swift 同构同参）

| 场景 | 数值 |
|---|---|
| 预填充（20.6k–27.6k prompt）| **654 tok/s** |
| 预填充（131,289 token）| **483.6 tok/s** |
| 解码（散文，greedy）| **64 tok/s** |
| 解码（131k 深度）| **82.2 tok/s**（浅层同任务 124.4）|
| 前缀缓存命中 | 15,259 前缀 **TTFT 79 ms**；131,282 前缀 **825 ms** |
| MTP 接受率 | K=3 ≈ 32%（greedy）/ 高至 75%（任务相关）|

调参取舍与全部 30 组数据见 [TUNING.md](TUNING.md)。

## 4. 长上下文召回

| 用例 | 结果 |
|---|---|
| **144,975 token** 大海捞针（nvfp4）| ✅ 答对 `QK-9024-TR`（prefill 448.6 tok/s，全程 5.4 分钟）|
| 21,038 token 大海捞针（fp8 KV，早期）| ✅ 答对 `ZQ-7731-KX` |
| 多轮记忆（4,271 token 处）| ✅ 答对；第二轮前缀缓存命中 62 token |

## 5. 视觉（`--vision`，BF16 视觉塔）

视觉塔来源已**逐字节证明**：用引擎自带量化器从官方 BF16 mmproj 与 Q8_0 mmproj 分别重编
同样的矩阵，再与制品载荷比对 —— 6/6 探针**从 BF16 源重编逐字节相同**，从 Q8_0 源差 0.7%–4.6%。

| 用例 | 真值 | 模型输出 |
|---|---|---|
| 官方图表（CLI 与 HTTP base64）| NIFER VISION 731；3 个红圆；蓝方块在左 | ✅ 全对 |
| 场景图 | 邮箱数字 24；太阳在右侧 | ✅ |
| 双图对比 | 左 2 红圆、右 3 个、新增黄色五角星 | ✅ |
| 视频（temporal_events.mp4）| 红色圆形移动；绿色 3；蓝方块仍在；END 9 | ✅ |

编码耗时 47–66 ms；含图 HTTP 请求 TTFT 946 ms、解码 94 tok/s；视频 prompt 1,231 token、
解码 88.9 tok/s。视觉上限 `min(--max-context, 32768)` merged token。

## 6. MMLU 5-shot（基线裁剪制品，全量）

- **14,042 / 14,042 题全量**（官方 dev 5-shot 协议，test 集）
- **micro 76.59%**（10,755/14,042）、**macro 77.71%**（57 科平均）
- 单题均耗时 1.27 s（中位 1.01）；平均 prompt 755 token；thinking 关闭、只输出答案字母

完整分科表：[eval/mmlu/MMLU报告.md](../eval/mmlu/MMLU报告.md)。
最强科：college_biology 95.1%；最弱：high_school_mathematics 39.6%。

> 这是**裁剪版**制品（词表 208,030）的成绩 —— 直接证明词表裁剪没有损伤能力。
> Swift 版的 MMLU 未跑（其定位是"同分省思考"，不是提分）。

## 7. Agent / Claude Code 链路（基线制品）

引擎裸层（HTTP）与 Claude Code 真实链路（`claude -p --output-format stream-json`）：

| 场景 | 结果 |
|---|---|
| 引擎裸层 agent 全循环（2 轮 + 工具）| **3.94 s**（工具执行本身 17 ms）|
| Claude Code 纯对话 | 15.3 s（TTFT 7.7 s，缓存命中 24k）|
| Claude Code Agent：列目录 / 读文件 / 写文件 | 20–25 s，工具全部真实执行 ✅ |
| 系统提示规模 | 27,815 token（首轮 TTFT 46.7 s → 缓存后 7.6 s）|

思考 token 占输出 40%–60% —— 这是该模型的工作方式；`MAX_THINKING_TOKENS=16384` 是合适档位。

## 8. 制品级验证数据

| 验证 | 基线 | Swift |
|---|---|---|
| 载荷布局 `zero_share` | 0.3278 | **0.3279** |
| 符号表 `max\|diff\|` | 0.0 | **0.0** |
| 行序规则 | 5/5 成立 | **5/5 成立** |
| 全量装配 | 15/15 | **15/15** |
| 裁剪重建审计 | ALL PASSED | **ALL PASSED** |
| 词表裁剪损失（纯中英文）| 0.0000% | **0.0000%** |

**发布包端到端（v1.0.0 完整包）**：解压 → 双击 `launch\launch_windows.bat` →
`engine ready 8.8 s` → `capacity | KV 262,144 tokens, nvfp4, explicit | free 435 MiB` →
`/v1/chat/completions` 正常返回。包内 CUDA 已静态链接，FFmpeg / VC++ 运行库随包提供。

## 9. 复现

| 数据 | 命令 |
|---|---|
| PPL | `launch\run_ppl.bat` |
| 思考 token A/B | `python bench\swift_bench.py --label <name> --model <model-id>` |
| chunk/draft/状态槽扫描 | `bench\sweep*.sh` + `bench\bench.py` + `bench\analyze.py` |
| 深度/前缀 | `bench\depth_test.py`、`bench\prefix_test.py` |
| 裁剪审计 | `conversion\verify\check_pruned_artifact.py` |
| 在线功能 | `conversion\verify\check_pruned_serving.py` |
| MMLU | `eval\mmlu\mmlu_eval.py`（用法见该目录）|

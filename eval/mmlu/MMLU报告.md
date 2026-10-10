# MMLU 5-shot 实测报告（Ternary Bonsai 2 27B / ninfer）

- 生成时间：2026年10月10日 03:50:04
- 状态：**全量完成** —— 14,042 / 14,042 题（100.0%）
- 引擎：`ninfer-serve` @127.0.0.1:8080，`bonsai2-27b-pruned.ninfer`，262144 上下文 / nvfp4 KV / prefill-chunk 8192 / MTP K=3
- 协议：MMLU **dev 集 5-shot**（每科 5 例，官方协议）+ **test 集全量**；`temperature 0`、thinking 关闭、只输出答案字母
- 注：引擎不支持 `logprobs`（服务端明确拒绝），故采用生成 + 字母解析而非似然比较；系统指令为「只答字母」，不加时该 chat 模型会写推导过程（实测掉到 35.1%，属协议不适配）

## 总成绩

| 指标 | 数值 |
|---|---|
| **micro（按题）** | **76.59%**（10,755/14,042）|
| **macro（57 科平均）** | **77.71%** |
| 随机基线 | 25% |
| 未解析（写满预算未给字母，按错计） | 110 题（0.8%）|
| 单题均耗时 | 1.27 s（中位 1.01，最长 5.2）|
| 平均 prompt | 755 token |
| 前缀缓存命中 | 0 / 14042 请求（每题都是新前缀，预期 0）|

## 分科成绩（按准确率）

| 科目 | 准确率 | 题数 |
|---|---|---|
| college_biology | 95.1% | 144 |
| high_school_government_and_politics | 94.8% | 193 |
| astronomy | 93.4% | 152 |
| high_school_microeconomics | 92.9% | 238 |
| marketing | 92.7% | 234 |
| high_school_psychology | 92.7% | 545 |
| high_school_biology | 91.9% | 310 |
| high_school_us_history | 90.2% | 204 |
| international_law | 90.1% | 121 |
| high_school_computer_science | 90.0% | 100 |
| high_school_world_history | 89.9% | 237 |
| high_school_european_history | 87.9% | 165 |
| world_religions | 87.1% | 171 |
| miscellaneous | 87.1% | 783 |
| sociology | 87.1% | 201 |
| jurisprudence | 87.0% | 108 |
| us_foreign_policy | 87.0% | 100 |
| prehistory | 86.4% | 324 |
| management | 86.4% | 103 |
| professional_medicine | 86.0% | 272 |
| conceptual_physics | 86.0% | 235 |
| logical_fallacies | 84.7% | 163 |
| high_school_geography | 84.3% | 198 |
| medical_genetics | 84.0% | 100 |
| professional_psychology | 83.7% | 612 |
| high_school_macroeconomics | 83.3% | 390 |
| human_sexuality | 83.2% | 131 |
| clinical_knowledge | 83.0% | 265 |
| philosophy | 82.6% | 311 |
| nutrition | 82.0% | 306 |
| computer_security | 82.0% | 100 |
| security_studies | 80.4% | 245 |
| human_aging | 79.8% | 223 |
| business_ethics | 79.0% | 100 |
| college_computer_science | 79.0% | 100 |
| college_medicine | 78.6% | 173 |
| moral_disputes | 78.0% | 346 |
| high_school_statistics | 76.4% | 216 |
| electrical_engineering | 75.9% | 145 |
| anatomy | 74.8% | 135 |
| high_school_chemistry | 72.9% | 203 |
| econometrics | 72.8% | 114 |
| elementary_mathematics | 72.5% | 378 |
| public_relations | 70.9% | 110 |
| high_school_physics | 69.5% | 151 |
| formal_logic | 64.3% | 126 |
| machine_learning | 64.3% | 112 |
| professional_accounting | 62.8% | 282 |
| professional_law | 61.7% | 1534 |
| college_physics | 53.9% | 102 |
| college_chemistry | 53.0% | 100 |
| college_mathematics | 53.0% | 100 |
| abstract_algebra | 52.0% | 100 |
| virology | 51.8% | 166 |
| moral_scenarios | 50.7% | 895 |
| global_facts | 47.0% | 100 |
| high_school_mathematics | 39.6% | 270 |

（共 57 科）

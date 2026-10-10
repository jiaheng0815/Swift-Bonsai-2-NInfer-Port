# 02 · 制品制作与验证

> 把官方 **PQ2_0 GGUF**（1.58-bit 三元）打成 NInfer 原生 `.ninfer` 制品，并做**四件套验证**。
> 打包器来自魔搭指南（`guide/tools/pack.py`），本仓只做两处最小修正（见 [ENGINE_BUILD.md](ENGINE_BUILD.md) §4）。
>
> **原则：逐字节搬运 2-bit 码字与 scale，绝不「解量化 → 再量化」。**

---

## 1. 制品是什么

`.ninfer` 是引擎的原生容器：**1126 个对象** = 文本权重（三元 PQ2_0）+ 借用件。

| 部分 | 对象数 | 来源 |
|---|---|---|
| `text/*` 三元权重 | 767 | **GGUF**（逐字节搬运，含符号表/宽度表由元数据生成）|
| `mtp/*`（投机头）| 12 | **模板借用**：三元包不带 MTP，借同族 Qwen3.8-27B 的 NextN 头 |
| `vision/*`（视觉塔）| 333 | **模板借用**：官方 mmproj（来源已逐字节证明）|
| `frontend/*` | 6 | **模板借用**：tokenizer 等（与官方逐字节相同）|
| `text/draft_head(+_token_ids)` | 2 | **模板借用**：高频短名单（同分词器 ⇒ 同一名单）|

> **为什么可以借**：模板与目标同族同布局；MTP 块不做 Hadamard 旋转（残差流在原始基），
> 借用时逐张量比对**余弦 ≥ 0.99966**、7 个 norm **逐字节相同**。

## 2. 前提：模板必须是 `groupwise-int`

`pack.py` 会**遍历模板自己的对象名表**逐个映射（表外的 `vision/*`/`mtp/*`/`frontend/*` 才走 borrow），
所以模板 schema 必须逐字一致。两种 schema 的判据：

| | **`groupwise-int`（要这个）** | `nvfp4`（用不了） |
|---|---|---|
| `identity.weights_id` | `groupwise-int` | `nvfp4` |
| `text/layers/3/` | `attention/query_key` ＋ `attention/gate_value` | 单个 `attention/query_key_gate_value` |
| GDN 投影 | `gdn/a_projection` ＋ `gdn/b_projection` | `gdn/a_b_projection`（融合）|

本机用的模板：`models/template/model.ninfer`（SHA256 已对上发布方校验和）。

## 3. 打包（两条命令）

```powershell
cd D:\ai-ninfer\guide

# ① 只读校验：几何 + 解码 + 往返证明（不写文件）
python tools\pack.py check `
  --gguf    E:\models\...\Ternary-Bonsai-2-27B-PQ2_0.gguf `
  --template D:\ai-ninfer\models\template\model.ninfer

# ② 全量打包
python tools\pack.py build D:\ai-ninfer\artifacts\bonsai2-27b.ninfer `
  --gguf    E:\models\...\Ternary-Bonsai-2-27B-PQ2_0.gguf `
  --template D:\ai-ninfer\models\template\model.ninfer
```

`check` 的判据（应全绿）：

```
byte round trip + decode equality : bytes_equal=True decode_equal=True   （每个探针张量）
scale 平面指纹                     : 全正 / 无 NaN / median 0.012–0.017 / 高字节种类 8–32
codes zero_share                  : 0.3278 ± 0.0002（PQ2_0 理论值）
RESULT: OK
```

打包耗时约 4 分钟（7.2 GB 读 + 7.74 GB 写）；产物与指南公布值必须一致：
**文本 6.696 GiB、对象 1126、总 7.736 GiB**。

## 4. 四件套验证（判据与结果）

```powershell
cd D:\ai-ninfer\guide

# ① 载荷布局：row-split 三平面 vs 交错块（分布指纹）
python conversion\verify\check_payload_order.py <artifact> text/layers/3/attention/query_key

# ② 符号表：artifact vs GGUF 元数据 逐值比对
python conversion\verify\check_signs.py <artifact> D:\ai-ninfer\artifacts\hadamard-meta.json

# ③ 行序：两条 INFERRED 规则的**逐行指纹**比对（抓过真 bug 的杀手判据）
python conversion\verify\check_row_order.py <artifact> <pq2.gguf>

# ④ 全量装配审计：每个对象、每一行的多重集 + 规则逐行比对
python conversion\verify\check_assembly.py <artifact> <pq2.gguf>
```

| # | 判据 | 基线制品 | **Swift 制品** |
|---|---|---|---|
| ① | A 平面 scale 全正、无 NaN、`zero_share ≈ 0.3278` | ✅ 0.3278 | ✅ **0.3279** |
| ② | `max\|diff\| = 0.0`，宽度前缀和 5120→0 / 6144→5120 / 17408→11264 | ✅ | ✅ **0.0** |
| ③ | `gdn_value_z` / `attn_q_per_head_interleave` 5/5 组逐行成立，0 重复 0 缺失 | ✅ | ✅ **全成立** |
| ④ | 15/15 装配规则；248,320 行 embedding 0 顺序错位 | ✅ | ✅ **15/15** |

> 第 ③ 项是**当年抓出 `gdn_value_z` 真 bug 的判据**：行被置换而不是变换，
> 所以字节级指纹是精确测试，不依赖旋转基。

## 5. 复现清单（从零到可服务）

| 步 | 命令 | 期望 |
|---|---|---|
| 1 | `pack.py check` | `RESULT: OK` |
| 2 | `pack.py build` | 7.736 GiB / 1126 对象 |
| 3 | 四件套验证 | 上表全绿 |
| 4 | 词表裁剪 + 重建（可选但推荐）| 见 [VOCAB_PRUNE.md](VOCAB_PRUNE.md) |
| 5 | 起服务 + 验收 | 见 [USAGE.md](USAGE.md) / [BENCHMARKS.md](BENCHMARKS.md) |

> **判据链的逻辑**：③④ 保证「每个字节搬对了」，① 保证「布局没搞反」，② 保证「旋转基的符号没搞错」。
> 这三类错误都会产出**结构合法但数值错乱**的模型 —— 只看 PPL 数值大小不一定能定位，
> 所以要在装配层先钉死。

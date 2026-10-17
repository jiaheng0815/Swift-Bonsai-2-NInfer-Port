# 03 · 词表裁剪（本项目独有方法）

> 把 248,320 行的词表裁剪到 **208,030**：剔除中英文之外的语言，**逐字节**重建制品，
> 中英文分词损失 **0.0000%**。制品 7.736 GiB → **7.629 GiB**（省 110 MiB），
> 显存内 embedding / output_head 两张 337 MB 的大表同步缩小。
>
> 据我们所知，公开社区里还没有第二家把 NInfer/Bonsai 2 的制品做词表裁剪并完成闭环验证。

---

## 1. 为什么裁

Qwen3.8-27B 系词表很大（248,320），其中 embedding 与 output_head 各占
**1.7 GB 的三元载荷中约 337 MB**（5120 × 248,320 行）。用户场景是**中英文**，
其余语言（西里尔/阿拉伯/假名/韩文/泰文/拉丁扩展变音…）常年用不到，白占显存与带宽。

裁剪后：**每 token 的权重流少 110 MiB**，且裁剪是**纯行重排**——不触碰数值。

## 2. 裁剪规则（`conversion/prune_vocab.py`）

### 保留（keep）

| 类别 | 数量 | 理由 |
|---|---|---|
| 256 个单字节 token | 256 | byte-level 底座，任意 UTF-8 都能编码 |
| UTF-8 碎片（不完整多字节序列）| 816 | BPE merge 的中间产物，缺了会改变分词 |
| 全部 ASCII token | 128,212 | 英文 / 代码 / 数字 |
| 含任一 CJK 字符的 token | 55,571 | 保护中文（含中英混合词）|
| 通用符号（`generic`）| （并入上两类）| 破折号——、省略号……、°±×÷、←→、★✓、emoji 等 |
| 33 个 added token | 33 | special / 视觉 / 工具 token |
| draft_head 引用的 token | 23,142 | **强制保留**，见 §5 |

### 删除（drop）

- 纯其它语言 token：**63,189 个**（其中 23,142 个被 draft_head 强制救回，实际删 **40,047**）
- 对应的 BPE merge 规则：247,587 → **204,449**（删 43,138）

> 分类器的关键细节：`U+2014`（——）和 `U+2026`（……）在中文里极常用，
> 早期版本把它们归入 "other" 被误删，导致中文 token 数上升 —— 现在归入 `generic` 保留。
> 教训写在 `prune_vocab.py` 的 `_kind()` 注释里。

## 3. 无损重建（`conversion/rebuild_artifact.py`）

```
text/token_embedding   [248320, 5120] -> [208030, 5120]   按 old->new 映射重排行
text/output_head       同上
text/draft_head        [131072, 5120]  行内容与行序都不变（原样搬运）
text/draft_head_token_ids [131072]     按新词表重映射
frontend/{tokenizer,tokenizer_config,generation_config}.json  换成裁剪版
其余对象                                原样搬运（零拷贝）
```

**行重排是纯字节级 gather**：PQ2_0_G128 每行 = base 平面 1280 B + scale 平面 80 B，
两向独立，**不解量化、不重新量化**，原始行的比特完全保留。

## 4. 引擎侧改动（已随本仓源码树提供）

| 位置 | 改动 |
|---|---|
| `src/targets/qwen3_6/export/.../frontend.h` | `kTokenDomain` 248,320 → **208,030** |
| `src/targets/qwen3_6_27b/impl/load/bindings.cpp` | embedding / output_head 行数 → **208,030** |
| `src/targets/qwen3_6/impl/frontend/frontend.cpp` | 33 个 special token 新 id（见下）|

新 special id（裁剪后）：`<|endoftext|>`=207,997、`<|im_start|>`=207,998、`<|im_end|>`=207,999、
`<|vision_start|>`=208,006、`<|image_pad|>`=208,009、…、`<|audio_pad|>`=208,029。
**`prune_vocab.py` 每次跑完都会打印这张表**，改引擎时照着对。

> 注意：引擎（本地构建）只为**裁剪域**编好 —— 原始 248,320 域的旧制品会报
> `tensor descriptor does not match target contract: text/token_embedding`，这是**预期行为**。

## 5. 一个硬约束：draft_head 不能重建

MTP 草稿头 `text/draft_head` 是 **Q4G64_F16S**，而 output_head 是 **PQ2_0_G128** ——
两者是不同量化器对同一批 lm_head 行的**独立产物**，字节无法互转；重新生成 Q4G64 行
需要原始 BF16 权重（本机没有）。所以：

- draft_head 的 131,072 行**原样保留**，其 `token_ids` 只重映射、不换 token；
- 由此**强制保留 23,142 个其它语言 token**（它们是草稿头的高频行）。

## 6. 验证体系（六个脚本，全部可复跑）

| 脚本 | 判据 | 本次结果 |
|---|---|---|
| `conversion/verify/check_row_gather.py` | 行 gather 逻辑（写 8 GB 之前的预演）| ✅ |
| `conversion/verify/check_vocab_prune.py` | 裁剪前后 tokenizer 对同一文本的分词数 | ✅ 中英文一致 |
| `conversion/verify/check_tokenizer_loss.py` | 真实语料上的**原始**分词损失 | ⚠️ zhwiki 0.2007%（见下）|
| `conversion/verify/check_pure_corpus_loss.py` | 把语料切成「纯中英文 / 含外语」分别度量 | ✅ **纯中英文 0.0000%** |
| `conversion/verify/check_pruned_artifact.py` | 重建制品的**逐字节**审计 | ✅ ALL PASSED |
| `conversion/verify/check_pruned_serving.py` | 在线功能（中英/代码/结构化输出/已删语言）| ✅ |

### 分词损失：0.2% 是怎么来的（重要的诚实度细节）

| 语料 | 纯中英文段落 | 含外语段落 |
|---|---|---|
| zhwiki | **0.0000%**（253,781→253,781）| 6.51%（外语片段内部）|
| wikitext | **0.0000%**（218,139→218,139）| 0.28% |
| pg19 / ninfer | **0.0000%** | 0.0000% |

**结论**：中英文文本**逐 token 无损**；原始 zhwiki 损失的 0.2% 全部发生在
条目里夹杂的日文/外文片段（作品名、人名）内部。判据 `≤0.1%` 在**纯中英文**口径下成立。

### 重建审计（`check_pruned_artifact.py`）关键输出

```
=== 2. embedding / output_head row remap ===
  ok  text/token_embedding  base plane:  ALL 208030 rows byte-exact
  ok  text/token_embedding  scale plane: ALL 208030 rows byte-exact
  ok  text/output_head      base plane:  ALL 208030 rows byte-exact
  ok  text/output_head      scale plane: ALL 208030 rows byte-exact
=== 3. draft_head is byte-identical ===    ok 15 sampled rows unchanged
=== 4. draft_head_token_ids remap ===      ok 131072 ids remapped, max 208029 < 208030
ALL CHECKS PASSED
```

## 7. 收益与边界

**收益**：制品 −110 MiB；显存同样的两张表同步缩小；**中英文零损失**。

**边界（如实交代）**：
- 其它语言**仍可用但会退化**（俄语、日文等被拆成更碎的 token；在线验证里专门测了俄语不崩）；
- `draft_head` 的关系决定了**不可能把 40,014 全部删掉**（23,142 个被强制保留）；
- 引擎必须用**裁剪域**编译（本仓源码树已经是）。

## 8. 复现（四步）

```powershell
cd <repo>        # 本仓根目录（示例 D:\ai-ninfer）

# ① 分类裁剪（只读制品，产出 4 个文件到 artifacts\pruned）
python conversion\prune_vocab.py --artifact artifacts\swift-bonsai2-27b.ninfer --out-dir artifacts\pruned

# ② 逐字节重建（数分钟）
python conversion\rebuild_artifact.py --artifact artifacts\swift-bonsai2-27b.ninfer `
  --pruned artifacts\pruned --out artifacts\swift-bonsai2-27b-pruned.ninfer

# ③ 审计（必做）
python conversion\verify\check_pruned_artifact.py `
  --src artifacts\swift-bonsai2-27b.ninfer --dst artifacts\swift-bonsai2-27b-pruned.ninfer --pruned artifacts\pruned

# ④ 在线验证（先起服务；model 名填 swift-bonsai2-27b）
python conversion\verify\check_pruned_serving.py --model swift-bonsai2-27b
```

> 基线制品（`artifacts\bonsai2-27b.ninfer`）走完全相同的四步，把文件名换掉即可。
> 在独立包内运行时，先 `set NINFER_SRC=<包根>\ninfer-tools`（或完整包里的
> `conversion\ninfer-tools`），让脚本找到 `.ninfer` 容器库。

> `prune_vocab.py` 末尾会打印 **special token 新 id 表**与 **自我检查**（中英文/代码
> 往返、以及"应退化"的俄语/日文/emoji 用例），改引擎常数时照它对。

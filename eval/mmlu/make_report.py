#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 full.jsonl 汇总成中文 Markdown 报告（关机前自动生成，也可随时手动跑）。"""
import collections
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "full.jsonl")
OUT = os.path.join(HERE, "MMLU报告.md")
TOTAL = 14042


def main():
    if not os.path.exists(SRC):
        print("没有结果文件:", SRC)
        return 1
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["subject"]][0] += int(bool(r["correct"]))
        per[r["subject"]][1] += 1
    subj = {s: (ok / n, n) for s, (ok, n) in per.items()}
    macro = sum(a for a, _ in subj.values()) / len(subj) if subj else 0.0
    micro = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    unparsed = sum(1 for r in rows if r["pred"] is None)
    secs = [r["sec"] for r in rows] or [0.0]
    ptok = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    cache = sum(1 for r in rows if r.get("cache_n"))
    done = len(rows) >= TOTAL

    L = []
    L.append("# MMLU 5-shot 实测报告（Ternary Bonsai 2 27B / ninfer）\n")
    L.append(f"- 生成时间：{time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    L.append(f"- 状态：**{'全量完成' if done else '未跑完（中断）'}** —— {len(rows):,} / {TOTAL:,} 题"
             f"（{len(rows)/TOTAL*100:.1f}%）")
    L.append(f"- 引擎：`ninfer-serve` @127.0.0.1:8080，`bonsai2-27b-pruned.ninfer`，"
             "262144 上下文 / nvfp4 KV / prefill-chunk 8192 / MTP K=3")
    L.append("- 协议：MMLU **dev 集 5-shot**（每科 5 例，官方协议）+ **test 集全量**；"
             "`temperature 0`、thinking 关闭、只输出答案字母")
    L.append("- 注：引擎不支持 `logprobs`（服务端明确拒绝），故采用生成 + 字母解析而非似然比较；"
             "系统指令为「只答字母」，不加时该 chat 模型会写推导过程（实测掉到 35.1%，属协议不适配）\n")
    L.append("## 总成绩\n")
    L.append("| 指标 | 数值 |")
    L.append("|---|---|")
    L.append(f"| **micro（按题）** | **{micro*100:.2f}%**（{sum(r['correct'] for r in rows):,}/{len(rows):,}）|")
    L.append(f"| **macro（57 科平均）** | **{macro*100:.2f}%** |")
    L.append(f"| 随机基线 | 25% |")
    L.append(f"| 未解析（写满预算未给字母，按错计） | {unparsed} 题（{unparsed/max(1,len(rows))*100:.1f}%）|")
    L.append(f"| 单题均耗时 | {st.mean(secs):.2f} s（中位 {st.median(secs):.2f}，最长 {max(secs):.1f}）|")
    if ptok:
        L.append(f"| 平均 prompt | {st.mean(ptok):.0f} token |")
    L.append(f"| 前缀缓存命中 | {cache} / {len(rows)} 请求（每题都是新前缀，预期 0）|\n")
    L.append("## 分科成绩（按准确率）\n")
    L.append("| 科目 | 准确率 | 题数 |")
    L.append("|---|---|---|")
    for s, (a, n) in sorted(subj.items(), key=lambda kv: -kv[1][0]):
        L.append(f"| {s} | {a*100:.1f}% | {n} |")
    L.append(f"\n（共 {len(subj)} 科）")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"报告已写入 {OUT} | micro {micro*100:.2f}% | macro {macro*100:.2f}% | {len(rows)} 题")
    return 0


if __name__ == "__main__":
    sys.exit(main())

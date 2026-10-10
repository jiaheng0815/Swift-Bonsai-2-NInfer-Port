#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMLU 提示写法对比：在 24 道题（含数学/理科重推导的科目）上比较哪种写法能让模型
在规定预算内把答案字母吐出来、并且准确率更高。"""
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict

import pandas as pd

HERE = r"D:\ai-ninfer\eval\mmlu"
BASE = "http://127.0.0.1:8080/v1/chat/completions"
LETTERS = "ABCD"

SYS_A = ("You are taking a multiple-choice exam. Answer with the single letter "
         "(A, B, C, or D) of the correct option, nothing else.")
SYS_B = ("You are taking a multiple-choice exam. Reply with ONLY the letter of the correct "
         "option. Do not write any explanation, working, derivation or LaTeX.")
TAIL_C = "\n\nRespond with only one character: A, B, C, or D."


def fmt_q(q, choices):
    out = [f"Question: {q}"]
    for i, c in enumerate(choices):
        out.append(f"{LETTERS[i]}. {c}")
    return "\n".join(out)


def build(shots, q, choices, tail=""):
    parts = [fmt_q(s["question"], s["choices"]) + " " + LETTERS[int(s["answer"])] for s in shots]
    return "\n\n".join(parts + [fmt_q(q, choices) + tail])


def ask(prompt, system, max_tokens):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    body = {"model": "bonsai2-27b", "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none"}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read().decode())
    return (j["choices"][0]["message"]["content"] or ""), time.time() - t0


def parse(t):
    """末尾优先：先找 'answer is X'，再找最后出现的独立字母。"""
    if not t:
        return None
    m = re.findall(r"answer\s*(?:is|would be|:)\s*\**\s*\(?\s*([A-Da-d])\b", t, re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"(?:^|\n)\s*\**\s*\(?([A-Da-d])[\).:]", t)
    if m:
        return m[-1].upper()
    m = re.match(r"^\s*[\*\s]*\(?([A-D])(?![A-Za-z])", t)
    if m:
        return m.group(1)
    return None


def main():
    dev = pd.read_parquet(f"{HERE}/dev.parquet")
    test = pd.read_parquet(f"{HERE}/test.parquet")
    # 12 个科目 × 2 题：数学/理科偏推导 + 文科
    subs = ["abstract_algebra", "college_mathematics", "college_physics", "conceptual_physics",
            "high_school_physics", "machine_learning", "college_chemistry", "econometrics",
            "astronomy", "world_religions", "philosophy", "high_school_geography"]
    variants = {
        "V1_系统指令+16": (SYS_A, "", 16),
        "V2_系统指令+256": (SYS_A, "", 256),
        "V3_强指令+64": (SYS_B, "", 64),
        "V4_尾指令+64": (SYS_A, TAIL_C, 64),
    }
    stats = defaultdict(lambda: {"ok": 0, "n": 0, "parsed16": 0, "sec": 0.0})
    for s in subs:
        shots = dev[dev.subject == s].to_dict("records")
        rows = test[test.subject == s].reset_index(drop=True).head(2)
        for _, row in rows.iterrows():
            gold = LETTERS[int(row["answer"])]
            for name, (sysmsg, tail, mt) in variants.items():
                p = build(shots, row["question"], list(row["choices"]), tail)
                try:
                    raw, dt = ask(p, sysmsg, mt)
                except Exception as exc:
                    raw, dt = f"<ERR {exc}>", 0.0
                pred = parse(raw)
                st = stats[name]
                st["n"] += 1
                st["sec"] += dt
                st["ok"] += int(pred == gold)
                if pred and len(raw) < 200:
                    st["parsed16"] += 1
                print(f"{name:16s} {s[:20]:22s} gold={gold} pred={pred} {dt:5.2f}s "
                      f"raw={raw[:60]!r}")
    print("\n=== 汇总 ===")
    for name, st in stats.items():
        print(f"{name:16s} 准确 {st['ok']}/{st['n']} = {st['ok']/st['n']*100:5.1f}% | "
              f"均耗时 {st['sec']/st['n']:.2f}s")


if __name__ == "__main__":
    main()

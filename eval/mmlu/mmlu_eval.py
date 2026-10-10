#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMLU 5-shot 标准评测：对已运行的 ninfer-serve 发请求（不自己起服务）。

协议：
  · 5-shot 示例 = MMLU dev 集（每科 5 题，官方协议）
  · 题目 = MMLU test 集（14,042 题 / 57 科）
  · temperature 0、thinking 关闭、答案只要字母
  · 结果逐题追加 results.jsonl（含 cache_n / 延迟），可断点续跑

用法：
    python mmlu_eval.py --pilot                     # 每科首题（57 题），冒烟
    python mmlu_eval.py                             # 全量
    python mmlu_eval.py --limit-per-subject 10      # 每科 10 题
"""
import argparse
import json
import os
import re
import time
import urllib.request
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "bonsai2-27b"
LETTERS = "ABCD"

RESULTS = os.path.join(HERE, "results.jsonl")
SUMMARY = os.path.join(HERE, "summary.json")
PROGRESS = os.path.join(HERE, "progress.log")
SYSTEM = ("You are taking a multiple-choice exam. Answer with the single letter "
          "(A, B, C, or D) of the correct option, nothing else.")


def fmt_question(q, choices):
    lines = [f"Question: {q}"]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_prompt(shots, question, choices):
    parts = [fmt_question(s["question"], s["choices"]) + " " + LETTERS[int(s["answer"])]
             for s in shots]
    parts.append(fmt_question(question, choices))
    return "\n\n".join(parts)


def parse_letter(text):
    """抽取答案字母：① 末尾的 'answer is X' ② 行首/独立字母 ③ 纯字母（容忍 markdown 星号）"""
    if not text:
        return None
    t = text.strip()
    m = re.findall(r"answer\s*(?:is|would be|:)\s*\**\s*\(?\s*([A-Da-d])\b", t, re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"(?:^|\n)\s*[\*\s]*\(?([A-Da-d])[\).:]", t)
    if m:
        return m[-1].upper()
    m = re.match(r"^\s*[\*\s]*\(?([A-D])(?![A-Za-z])", t)
    if m:
        return m.group(1)
    return None


def ask(prompt, max_tokens, timeout=600):
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none"}
    req = urllib.request.Request(BASE_URL, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    content = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return content, dt, body.get("usage") or {}, body.get("timings") or {}


def ask_retry(prompt, max_tokens, tries=5):
    last = None
    for k in range(tries):
        try:
            return ask(prompt, max_tokens)
        except Exception as exc:
            last = exc
            time.sleep(3 + 3 * k)
    raise last


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["subject"], int(r["index"])))
                except Exception:
                    pass
    return done


def summarize(path):
    per = defaultdict(lambda: [0, 0])
    n_ok = n_tot = cache = unparsed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            per[r["subject"]][0] += int(bool(r["correct"]))
            per[r["subject"]][1] += 1
            n_ok += int(bool(r["correct"]))
            n_tot += 1
            cache += int(bool(r.get("cache_n")))
            unparsed += int(r.get("pred") is None)
    subj_acc = {s: ok / n for s, (ok, n) in sorted(per.items()) if n}
    macro = sum(subj_acc.values()) / len(subj_acc) if subj_acc else 0.0
    summary = {"questions": n_tot, "correct": n_ok,
               "micro_accuracy": round(n_ok / max(1, n_tot), 4),
               "macro_accuracy": round(macro, 4), "subjects": len(subj_acc),
               "unparsed": unparsed, "prefix_cache_hits": cache,
               "per_subject": {s: round(a, 4) for s, a in subj_acc.items()}}
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n=== {n_tot} 题 | micro {n_ok/max(1,n_tot)*100:.2f}% | "
          f"macro({len(subj_acc)} 科) {macro*100:.2f}% | 未解析 {unparsed} | 命中缓存 {cache} ===")
    if subj_acc:
        print("最差 6 科：", [(s, round(a, 3)) for s, a in sorted(subj_acc.items(), key=lambda kv: kv[1])[:6]])
        print("最好 6 科：", [(s, round(a, 3)) for s, a in sorted(subj_acc.items(), key=lambda kv: -kv[1])[:6]])
    print(f"明细 {path} / 汇总 {SUMMARY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="每科只跑第一题")
    ap.add_argument("--limit-per-subject", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    test = pd.read_parquet(os.path.join(HERE, "test.parquet"))
    dev = pd.read_parquet(os.path.join(HERE, "dev.parquet"))
    shots_by_subject = {s: g.to_dict("records") for s, g in dev.groupby("subject")}

    done = load_done(args.out)
    plan = []
    for subject in sorted(test.subject.unique()):
        rows = test[test.subject == subject].reset_index(drop=True)
        n = 1 if args.pilot else (args.limit_per_subject or len(rows))
        for i in range(min(n, len(rows))):
            if (subject, i) not in done:
                plan.append((subject, i, rows.iloc[i]))

    total = len(plan)
    print(f"[mmlu] 待跑 {total} 题（已完成 {len(done)}）| 5-shot=dev | temp 0 | thinking off | "
          f"max_tokens {args.max_tokens}", flush=True)
    if not total:
        summarize(args.out)
        return

    correct, counts = defaultdict(int), defaultdict(int)
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                counts[r["subject"]] += 1
                correct[r["subject"]] += int(bool(r["correct"]))

    t_start = time.time()
    lat = []
    with open(args.out, "a", encoding="utf-8") as out:
        for n, (subject, idx, row) in enumerate(plan, 1):
            prompt = build_prompt(shots_by_subject[subject], row["question"], list(row["choices"]))
            gold = LETTERS[int(row["answer"])]
            try:
                content, dt, usage, timings = ask_retry(prompt, args.max_tokens)
                pred = parse_letter(content)
                ok = (pred == gold)
            except Exception as exc:
                content, dt, usage, timings, pred, ok = f"<ERROR {exc}>", 0.0, {}, {}, None, False
            counts[subject] += 1
            correct[subject] += int(ok)
            out.write(json.dumps({
                "subject": subject, "index": idx, "gold": gold, "pred": pred, "correct": bool(ok),
                "raw": content[:120], "sec": round(dt, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cache_n": timings.get("cache_n"),
                "predicted_n": timings.get("predicted_n"),
                "predicted_per_second": timings.get("predicted_per_second")},
                ensure_ascii=False) + "\n")
            out.flush()
            lat.append(dt)
            if n % 20 == 0 or n == total:
                el = time.time() - t_start
                eta = el / n * (total - n)
                acc = sum(correct.values()) / max(1, sum(counts.values()))
                msg = (f"[{n}/{total}] 累计正确率 {acc*100:.1f}% | 均延迟 {sum(lat)/len(lat):.2f}s | "
                       f"已用 {el/60:.1f}min | ETA {eta/60:.1f}min")
                print(msg, flush=True)
                with open(PROGRESS, "a", encoding="utf-8") as pf:
                    pf.write(msg + "\n")

    summarize(args.out)


if __name__ == "__main__":
    main()

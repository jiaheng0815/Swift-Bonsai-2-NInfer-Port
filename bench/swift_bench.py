#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swift Bonsai 2 vs 基线 Bonsai 2：思考 token 对比评测（引擎级）。

同一组 reasoning 型 prompt、同一套采样参数，分别打基线/ Swift 两个服务
（或同一服务的不同模型名），统计：
  * reasoning_tokens（引擎自报，权威）
  * completion_tokens / 总时长 / decode 速度
  * 答案文本（人工核对正确性）

用法：
    python swift_bench.py --label base  --base http://127.0.0.1:8080 --model bonsai2-27b
    python swift_bench.py --label swift --base http://127.0.0.1:8080 --model swift-bonsai2-27b
结果追加到 logs/swift_bench.jsonl，并打印汇总表。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "swift_bench.jsonl"

PROMPTS = [
    ("math1", "A train travels 120 km in 1.5 hours, then 90 km in 1 hour. What is the average speed for the whole journey? Give just the number."),
    ("math2", "How many integers between 1 and 1000 are divisible by neither 3 nor 5?"),
    ("logic1", "Three boxes are labelled 'Apples', 'Oranges', 'Mixed', but all labels are wrong. You may take one fruit from one box. Which box do you pick from and why? Answer briefly."),
    ("logic2", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer in one word."),
    ("code1", "Write a Python one-liner that reverses the words in a sentence. Output only the code."),
    ("zh1", "用三句话解释什么是三重量化，以及它为什么能省显存。"),
    ("zh2", "一个水池有两个进水管和一个出水管。甲管单独注满需 6 小时，乙管需 8 小时，出水管排空需 12 小时。三管齐开，多久注满？"),
    ("reason1", "Is it possible for a barber who shaves all and only those men who do not shave themselves to exist? Explain in 2 sentences."),
]

SYSTEM = "Answer in the user's language. Be precise."


def call(base: str, model: str, prompt: str, max_tokens: int, temperature: float, effort: str):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": effort,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    t0 = time.perf_counter()
    with opener.open(req, timeout=1800) as r:
        out = json.loads(r.read().decode("utf-8"))
    wall = time.perf_counter() - t0
    return out, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--only", default="", help="逗号分隔的 prompt key，仅跑这些")
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    rows = []
    for key, prompt in PROMPTS:
        if only and key not in only:
            continue
        try:
            out, wall = call(args.base, args.model, prompt, args.max_tokens, args.temperature, args.effort)
        except Exception as exc:  # noqa: BLE001
            print(f"[{key}] FAILED: {exc}")
            rows.append({"label": args.label, "key": key, "error": str(exc)})
            continue
        usage = out.get("usage", {}) or {}
        details = usage.get("completion_tokens_details", {}) or {}
        timings = out.get("timings", {}) or {}
        choice = (out.get("choices") or [{}])[0].get("message", {})
        row = {
            "label": args.label,
            "key": key,
            "model": args.model,
            "reasoning_tokens": details.get("reasoning_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "wall_s": round(wall, 2),
            "decode_tps": timings.get("predicted_per_second"),
            "answer": (choice.get("content") or "").strip()[:400],
            "thinking_chars": len(choice.get("reasoning_content") or ""),
        }
        rows.append(row)
        print(f"[{key}] reasoning={row['reasoning_tokens']} completion={row['completion_tokens']} "
              f"wall={row['wall_s']}s ans={row['answer'][:60]!r}")

    with RESULTS.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r.get("reasoning_tokens") is not None]
    if ok:
        print(f"\n=== {args.label}: n={len(ok)} "
              f"reasoning_tokens mean={statistics.mean(r['reasoning_tokens'] for r in ok):.0f} "
              f"median={statistics.median(r['reasoning_tokens'] for r in ok):.0f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

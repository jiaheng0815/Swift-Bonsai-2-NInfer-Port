#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 logs/tune/*.log 里的关键行汇总成表（不重新跑基准）。"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import parse_log  # noqa: E402

rows = []
for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.log"))):
    label = os.path.splitext(os.path.basename(path))[0]
    info = parse_log(path)
    led = info["ledger"] or ""
    cap = info["capacity"] or ""
    def grab(pattern, text, cast=float):
        m = re.search(pattern, text)
        if not m:
            return None
        return cast(m.group(1))

    def grab_mib(pattern, text):
        """'1.62 GiB' / '704.2 MiB' -> MiB"""
        m = re.search(pattern + r" ([\d.]+) (GiB|MiB)", text)
        if not m:
            return None
        v = float(m.group(1))
        return round(v * 1024.0) if m.group(2) == "GiB" else round(v)

    reqs = info["requests"]
    pre = max(reqs, key=lambda r: r["prompt"]) if reqs else None
    dec = [r for r in reqs if r["prompt"] < 200 and r["output"] >= 100]
    mi = lambda v: round(v / 1048576.0) if v else None
    kv = grab(r"KV ([\d,]+) tokens", cap, lambda s: int(s.replace(",", "")))
    rows.append({
        "label": label,
        "kv": kv,
        "free_after_weights_mib": grab_mib(r"after weights", led),
        "free_after_startup_mib": grab_mib(r"after startup", led),
        "slack_mib": grab_mib(r"slack", led),
        "graphs_mib": grab_mib(r"CUDA graphs", led),
        "prefill_tps": pre["prefill_tps"] if pre else None,
        "prefill_tokens": pre["prompt"] if pre else None,
        "ttft_s": round(pre["ttft_ms"] / 1000.0, 1) if pre else None,
        "decode_tps": dec[-1]["decode_tps"] if dec else None,
        "mtp_accept": (round(dec[-1]["mtp"][0] / dec[-1]["mtp"][1] * 100, 1)
                       if dec and dec[-1]["mtp"] else None),
        "started": "启动失败" if not (info["capacity"] or info["ledger"]) else "OK",
    })

hdr = f"{'label':20s} {'kv':>7s} {'startup_MiB':>11s} {'slack_MiB':>9s} {'graphs':>7s} {'prefill':>8s} {'ptok':>7s} {'ttft':>7s} {'decode':>7s} {'mtp%':>6s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    f = lambda v, n=2: ("-" if v is None else f"{v:.{n}f}")
    print(f"{r['label']:20s} {str(r['kv'] or '-'):>7s} {str(r['free_after_startup_mib'] or '-'):>11s} "
          f"{str(r['slack_mib'] or '-'):>9s} {str(r['graphs_mib'] or '-'):>7s} "
          f"{f(r['prefill_tps'],1):>8s} {str(r['prefill_tokens'] or '-'):>7s} {f(r['ttft_s'],1):>7s} "
          f"{f(r['decode_tps'],1):>7s} {f(r['mtp_accept'],1):>6s}")
if len(sys.argv) > 1 and sys.argv[1] == "--json":
    print(json.dumps(rows, ensure_ascii=False, indent=1))

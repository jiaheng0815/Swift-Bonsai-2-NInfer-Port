#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度上下文测试：131k 预填充 -> 同前缀追加一轮（缓存命中）-> 在 131k 深度上解码。

用同一份长前缀的第二次请求来量"深度解码"速度（KV 全量读取代价），
顺便验证 prefix cache 在 --device-state-slots/--max-private-continuations 下真的命中。
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import EXE, ART, LOGDIR, long_prompt, parse_log, wait_gpu_baseline  # noqa: E402

PORT = 8099


def post(messages, max_tokens, timeout=3600):
    payload = {"model": "bonsai2-27b", "messages": messages, "max_tokens": max_tokens,
               "reasoning_effort": "none"}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="depth")
    ap.add_argument("--system-tokens", type=int, default=78000)   # 实际约 x1.69
    ap.add_argument("--decode-tokens", type=int, default=160)
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]
    log_path = os.path.join(LOGDIR, f"{args.label}.log")
    argv = [EXE, ART, "--host", "127.0.0.1", "--port", str(PORT), "--model-id", "bonsai2-27b",
            "--log-level", "debug", "--log-stats-interval-ms", "0"] + extra
    print("[depth] " + " ".join(argv), flush=True)
    wait_gpu_baseline()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            while time.time() - t0 < 420:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(1.5)
            short = "Count slowly from 1 to 150, one number per line."
            print("① 浅层解码基准:", round(post([{"role": "user", "content": short}],
                                                args.decode_tokens)[1], 1), "s", flush=True)
            dossier = long_prompt(args.system_tokens)
            print("② 长前缀预填充 (约 131k token) ...", flush=True)
            b, wall = post([{"role": "user", "content": dossier}], 16)
            print(f"   完成 {wall:.0f}s  usage={b.get('usage')}", flush=True)
            print("③ 同前缀追加一轮 -> 131k 深度解码 ...", flush=True)
            b2, wall2 = post([{"role": "user", "content": dossier},
                              {"role": "assistant", "content": "ok"},
                              {"role": "user", "content": short}], args.decode_tokens)
            print(f"   完成 {wall2:.0f}s  usage={b2.get('usage')}", flush=True)
            print("④ 浅层解码基准（再来一次，确认没有漂移）:", flush=True)
            post([{"role": "user", "content": short + " "}], args.decode_tokens)
        finally:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:
                subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)], capture_output=True)
            time.sleep(3)
            wait_gpu_baseline()
    info = parse_log(log_path)
    print("\n请求明细：")
    for r in info["requests"]:
        acc = f"{r['mtp'][0]}/{r['mtp'][1]} ({r['mtp'][0]/r['mtp'][1]*100:.0f}%)" if r["mtp"] else "-"
        print(f"  req#{r['id']}: prompt {r['prompt']:>7,} | out {r['output']:>4} | "
              f"TTFT {r['ttft_ms']/1000:>7.2f}s | prefill {r['prefill_tps']:>6.1f} | "
              f"decode {r['decode_tps']:>5.1f} tok/s | mtp {acc}")
    print("\n缓存命中行：")
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if "req#" in line and "cache" in line:
            print("  " + line.strip()[line.find("req#"):][:150])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前缀缓存验证：同一段长 prompt 连发两次 + 带追加轮次，看 cache 命中与 TTFT。

用于验证 --device-state-slots/--max-private-continuations 这套 context cache 配置
在真实多轮（Claude Code 式）场景下是否生效。
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import EXE, ART, LOGDIR, gpu_used, long_prompt, parse_log, wait_gpu_baseline  # noqa: E402

PORT = 8099


def post(messages, max_tokens=16):
    payload = {"model": "bonsai2-27b", "messages": messages, "max_tokens": max_tokens,
               "reasoning_effort": "none"}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="prefix-cache")
    ap.add_argument("--serve-args", nargs=argparse.REMAINDER)
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]
    log_path = os.path.join(LOGDIR, f"{args.label}.log")
    argv = [EXE, ART, "--host", "127.0.0.1", "--port", str(PORT), "--model-id", "bonsai2-27b",
            "--log-level", "debug", "--log-stats-interval-ms", "0"] + extra
    print("[prefix] " + " ".join(argv), flush=True)
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
            system = long_prompt(9000)          # 模拟 Claude Code 的大系统提示
            turn1 = [{"role": "user", "content": system}]
            turn2 = turn1 + [{"role": "assistant", "content": "ok"},
                             {"role": "user", "content": "Now answer with the serial number only."}]
            print("A 首轮:", round(post(turn1), 2), "s", flush=True)
            print("B 同前缀重发:", round(post(turn1), 2), "s", flush=True)
            print("C 追加一轮:", round(post(turn2), 2), "s", flush=True)
        finally:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:
                subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)], capture_output=True)
            time.sleep(3)
            wait_gpu_baseline()
    info = parse_log(log_path)
    for r in info["requests"]:
        print(f"  req#{r['id']}: prompt {r['prompt']} tok | TTFT {r['ttft_ms']/1000:.2f}s | "
              f"prefill {r['prefill_tps']} tok/s | decode {r['decode_tps']} tok/s")
    print("原始行:")
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if "cache" in line and "req#" in line:
            print("   " + line.strip()[:180])


if __name__ == "__main__":
    main()

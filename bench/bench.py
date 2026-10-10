#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ninfer-serve 参数调优基准：起服务 -> 测 prefill/decode -> 杀进程 -> 汇总。

用法:
    python bench.py --label <名字> -- <ninfer-serve 的额外参数...>

输出: 一行 JSON（含内存账本、KV 容量、prefill/decode 吞吐），并追加到 results.jsonl。
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

EXE = r"D:\ai-ninfer\build\b1\apps\ninfer-serve.exe"
ART = r"D:\ai-ninfer\artifacts\bonsai2-27b-pruned.ninfer"
PORT = 8099
LOGDIR = r"D:\ai-ninfer\logs\tune"
RESULTS = os.path.join(LOGDIR, "results.jsonl")


def gpu_used():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()[0]
    used, total = out.split(",")
    return int(used), int(total)


def wait_gpu_baseline(target_mib=1600, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        used, _ = gpu_used()
        if used <= target_mib:
            return used
        time.sleep(2)
    return gpu_used()[0]


WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
         "rho sigma tau upsilon phi chi psi omega quantum vector matrix tensor gradient entropy "
         "kernel buffer tensor shard lattice orbital photon neutron boson hadron gluon lepton "
         "quasar pulsar nebula galaxy cluster filament baryon dark matter energy momentum").split()


def long_prompt(target_tokens):
    """生成一段唯一的英文长文本，约 target_tokens 个 token（>= 随机可保证前缀不命中缓存）。"""
    rnd = random.Random()
    head = "Nonce %08x. Read the following dossier and answer with its serial number.\n\n" % rnd.getrandbits(32)
    words = []
    while len(words) < int(target_tokens * 1.35):
        words.append(rnd.choice(WORDS))
    body = " ".join(words)
    tail = "\n\nEND OF DOSSIER. Serial number: SN-%06d. Reply with the serial number only." % rnd.randrange(1000000)
    return head + body + tail


def post(payload, timeout=1800):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0


REQ_DONE = re.compile(r"req#(\d+) done.*?prompt ([\d,]+) \| output ([\d,]+) \|.*?"
                      r"TTFT (?:(\d+)m )?([\d.]+)\s*(m?s).*?"
                      r"prefill ([\d.]+) tok/s \| decode ([\d.]+) tok/s(?: \| mtp accepted ([\d,]+)/([\d,]+))?")


def parse_log(path):
    """从运行日志里抽出 capacity / ledger 行和每个请求的吞吐行。"""
    info = {"capacity": None, "ledger": None, "cache": None, "media": None, "requests": [],
            "errors": []}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "capacity |" in line:
                info["capacity"] = line.strip()
            elif "memory ledger |" in line:
                info["ledger"] = line.strip()
            elif "context cache |" in line:
                info["cache"] = line.strip()
            elif "media |" in line:
                info["media"] = line.strip()
            m = REQ_DONE.search(line)
            if m:
                ttft = (float(m.group(4) or 0) * 60.0 + float(m.group(5))) * (1000.0 if m.group(6) == "s" else 1.0)
                plain = lambda s: int(s.replace(",", ""))
                info["requests"].append({
                    "id": int(m.group(1)), "prompt": plain(m.group(2)), "output": plain(m.group(3)),
                    "ttft_ms": ttft, "prefill_tps": float(m.group(7)),
                    "decode_tps": float(m.group(8)),
                    "mtp": (plain(m.group(9)), plain(m.group(10))) if m.group(9) else None})
            if " ERROR " in line or "error" in line.lower() and "status" not in line.lower():
                info["errors"].append(line.strip()[:300])
    return info


def parse_bytes(text):
    """把 '12.3 GiB' / '512 MiB' 归一成 MiB 浮点。"""
    if not text:
        return None
    m = re.search(r"([\d.]+)\s*(GiB|MiB|KiB|B)", text)
    if not m:
        return None
    v = float(m.group(1))
    return v * {"GiB": 1024.0, "MiB": 1.0, "KiB": 1 / 1024.0, "B": 1 / 1048576.0}[m.group(2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--artifact", default=ART)
    ap.add_argument("--prefill-tokens", type=int, default=16384)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--okv", default="--kv-dtype nvfp4 --max-context 262144 --max-concurrency 1")
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]

    os.makedirs(LOGDIR, exist_ok=True)
    log_path = os.path.join(LOGDIR, f"{args.label}.log")
    argv = [EXE, args.artifact, "--host", "127.0.0.1", "--port", str(PORT), "--model-id", "bonsai2-27b",
            "--log-level", "debug", "--log-stats-interval-ms", "0"] + args.okv.split() + extra
    print("[bench] " + " ".join(argv), flush=True)

    baseline = wait_gpu_baseline()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        result = {"label": args.label, "args": extra, "okv": args.okv, "gpu_baseline_mib": baseline,
                  "started": False}
        try:
            t0 = time.time()
            ready = False
            while time.time() - t0 < 420:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
                        if r.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(1.5)
            result["load_s"] = round(time.time() - t0, 1)
            result["started"] = ready
            if ready:
                used, total = gpu_used()
                result["gpu_after_load_mib"] = used
                result["gpu_total_mib"] = total
                # 0) 预热：先发一个短请求，避开引擎启动后的内部校准/首次图捕获
                try:
                    post({"model": "bonsai2-27b", "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 8, "reasoning_effort": "none"}, timeout=600)
                except Exception as exc:
                    result["warmup_error"] = str(exc)
                # 1) prefill：长而唯一的 prompt
                prompt = long_prompt(args.prefill_tokens)
                body, wall = post({"model": "bonsai2-27b",
                                   "messages": [{"role": "user", "content": prompt}],
                                   "max_tokens": 16, "reasoning_effort": "none"}, timeout=1800)
                result["prefill_wall_s"] = round(wall, 2)
                used2, _ = gpu_used()
                result["gpu_peak_prefill_mib"] = used2
                # 2) decode：短 prompt + 散文输出（比"数数"更接近真实接受率）
                body2, wall2 = post({"model": "bonsai2-27b",
                                     "messages": [{"role": "user", "content":
                                                   "Write a 200 word story about a lighthouse keeper who "
                                                   "finds a message in a bottle. Do not think, just write."}],
                                     "max_tokens": args.decode_tokens, "reasoning_effort": "none"},
                                    timeout=1800)
                result["decode_wall_s"] = round(wall2, 2)
                used3, _ = gpu_used()
                result["gpu_peak_mib"] = max(used2, used3)
        finally:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:
                subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)], capture_output=True)
            time.sleep(3)
            wait_gpu_baseline()

    info = parse_log(log_path)
    result["capacity"] = info["capacity"]
    result["ledger"] = info["ledger"]
    result["cache"] = info["cache"]
    result["requests"] = info["requests"]
    result["tail"] = tail_of(log_path)
    led = info["ledger"] or ""
    result["weights_free_mib"] = parse_bytes(re.search(r"after weights ([^|]+)", led).group(1)) if "after weights" in led else None
    result["after_startup_mib"] = parse_bytes(re.search(r"after startup ([^|]+)", led).group(1)) if "after startup" in led else None
    result["headroom_mib"] = parse_bytes(re.search(r"headroom ([^|]+)", led).group(1)) if "headroom" in led else None
    result["slack_mib"] = parse_bytes(re.search(r"slack ([^|]+)", led).group(1)) if "slack" in led else None
    result["graphs_mib"] = parse_bytes(re.search(r"CUDA graphs ([^|]+)", led).group(1)) if "CUDA graphs" in led else None
    cap = info["capacity"] or ""
    m = re.search(r"KV ([\d,]+) tokens", cap)
    result["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None

    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tail"}, ensure_ascii=False, indent=1))
    return 0


def tail_of(path, n=6):
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for line in lines[-n:]:
        out.append(line.rstrip()[:200])
    return out


if __name__ == "__main__":
    sys.exit(main())

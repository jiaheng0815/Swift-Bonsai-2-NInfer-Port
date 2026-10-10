"""并行 range 下载器（走代理，断点续传，SHA256 校验）。

hf_transfer 的 Rust 实现不认 HTTPS_PROXY，直连原站会卡死；本脚本用标准库
urllib + 显式 ProxyHandler，N 个连接各拉一段 range 写入同一文件的不同偏移。

用法：
    python parallel_dl.py --url <url> --out <path> --size <bytes> [--sha256 <hex>]
                          [--conn 6] [--chunk-mb 32] [--proxy http://127.0.0.1:7897]

进度落盘在 <out>.done（已完成 chunk 索引），中断后重跑自动续传。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def build_opener(proxy: str | None):
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def worker(opener, url: str, fh, q: queue.Queue, done: set, lock: threading.Lock,
           counter: dict, chunk_bytes: int, total_size: int, retries: int = 4) -> None:
    while True:
        try:
            idx = q.get_nowait()
        except queue.Empty:
            return
        start = idx * chunk_bytes
        # 尾块必须按文件真实末尾截断（服务端会把超出的 Range 夹到 EOF，
        # 按整块长度校验会把"正确返回"误判成 short read —— 第一次就是这么失败的）
        end = min(start + chunk_bytes, total_size) - 1
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "parallel-dl/1.0"},
                )
                with opener.open(req, timeout=120) as resp:
                    data = resp.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} != {end - start + 1}")
                with lock:
                    fh.seek(start)
                    fh.write(data)
                    fh.flush()
                    done.add(idx)
                    counter["bytes"] += len(data)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == retries - 1:
                    print(f"  chunk {idx} FAILED: {exc}", flush=True)
                    with lock:
                        counter["failed"].append(idx)
                else:
                    time.sleep(2 * (attempt + 1))
        q.task_done()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--sha256", default=None)
    ap.add_argument("--conn", type=int, default=6)
    ap.add_argument("--chunk-mb", type=int, default=32)
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or None)
    args = ap.parse_args()

    out = Path(args.out)
    done_path = Path(str(out) + ".done")
    chunk_bytes = args.chunk_mb * 1024 * 1024
    n_chunks = (args.size + chunk_bytes - 1) // chunk_bytes

    done: set[int] = set()
    if done_path.exists():
        done = set(json.loads(done_path.read_text()))

    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "r+b" if out.exists() else "w+b")
    fh.truncate(args.size)

    q: queue.Queue = queue.Queue()
    for idx in range(n_chunks):
        if idx not in done:
            q.put(idx)

    todo = q.qsize()
    print(f"chunks: {n_chunks} total, {todo} to fetch, conn={args.conn}, proxy={args.proxy}",
          flush=True)
    if todo == 0:
        print("all chunks already present", flush=True)
    opener = build_opener(args.proxy)
    lock = threading.Lock()
    counter = {"bytes": sum(1 for _ in done) * 0 , "failed": []}
    counter["bytes"] = 0
    base_done = len(done)

    threads = [
        threading.Thread(target=worker,
                         args=(opener, args.url, fh, q, done, lock, counter, chunk_bytes, args.size),
                         daemon=True)
        for _ in range(args.conn)
    ]
    t0 = time.time()
    for t in threads:
        t.start()

    # 进度线程
    stop = threading.Event()

    def progress() -> None:
        last = 0
        while not stop.wait(5):
            with lock:
                n = len(done)
                got = n * chunk_bytes
            if got - last > 200 * 1024 * 1024 or n == n_chunks:
                speed = got / max(time.time() - t0, 1e-9) / 1e6
                print(f"  {n}/{n_chunks} chunks  ~{min(got, args.size)/1e6:.0f} MB  "
                      f"{speed:.1f} MB/s", flush=True)
                last = got
            done_path.write_text(json.dumps(sorted(done)))

    pt = threading.Thread(target=progress, daemon=True)
    pt.start()
    for t in threads:
        t.join()
    stop.set()
    fh.close()
    done_path.write_text(json.dumps(sorted(done)))

    failed = counter["failed"]
    if failed:
        print(f"FAILED chunks: {len(failed)} — rerun to resume", flush=True)
        return 1

    print(f"downloaded {args.size/1e9:.3f} GB in {time.time()-t0:.0f}s "
          f"({args.size/(time.time()-t0)/1e6:.1f} MB/s)", flush=True)

    if args.sha256:
        h = hashlib.sha256()
        with open(out, "rb") as f:
            for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
        ok = digest == args.sha256.lower()
        print(f"SHA256 {digest}  {'OK' if ok else 'MISMATCH expected ' + args.sha256}", flush=True)
        if not ok:
            return 1
    done_path.unlink(missing_ok=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

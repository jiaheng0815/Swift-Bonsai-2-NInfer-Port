"""官方站带宽探针：N 个并行 range 请求各拉 M MB，测聚合吞吐。

默认直连 Hugging Face 官方站并读取系统代理（HTTPS_PROXY / HTTP_PROXY）。
例：
    python conversion/speed_probe.py --proxy http://127.0.0.1:7897 --conn 20
"""
import argparse
import threading
import time
import urllib.request

DEFAULT_URL = "https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF/resolve/main/Swift-Bonsai-2-PTQ1_0.gguf"
CHUNK = 24 * 1024 * 1024  # 每个连接 24 MB

results = {}


def worker(i: int, url: str, opener) -> None:
    start = i * CHUNK
    end = start + CHUNK - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "probe"})
    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=120) as r:
            data = r.read()
        dt = time.perf_counter() - t0
        results[i] = (len(data), dt, len(data) / dt / 1e6)
    except Exception as exc:  # noqa: BLE001
        results[i] = (-1, 0.0, 0.0)
        print(f"  conn{i}: FAILED {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help="要探测的文件 URL（默认官方 Swift GGUF）")
    ap.add_argument("--conn", type=int, default=4, help="并行连接数（默认 4）")
    ap.add_argument("--proxy", default=None, help="代理地址，如 http://127.0.0.1:7897；不传则读环境变量")
    args = ap.parse_args()

    if args.proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
        )
    else:
        opener = urllib.request.build_opener()  # 默认读取 HTTP(S)_PROXY

    threads = [threading.Thread(target=worker, args=(i, args.url, opener)) for i in range(args.conn)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total = sum(r[0] for r in results.values() if r[0] > 0)
    for i in sorted(results):
        n, dt, mbps = results[i]
        print(f"  conn{i}: {n/1e6:7.1f} MB in {dt:5.1f}s = {mbps:5.2f} MB/s")
    print(f"聚合: {total/1e6:.1f} MB in {wall:.1f}s = {total/wall/1e6:.2f} MB/s (N={args.conn})")


if __name__ == "__main__":
    main()

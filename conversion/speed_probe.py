"""hf-mirror 带宽探针：N 个并行 range 请求各拉 M MB，测聚合吞吐。"""
import threading
import time
import urllib.request

URL = "https://hf-mirror.com/ukisai/Swift-Bonsai-2-GGUF/resolve/main/Swift-Bonsai-2-PTQ1_0.gguf"
CHUNK = 24 * 1024 * 1024  # 每个连接 24 MB
N = 4

results = {}


def worker(i: int) -> None:
    start = i * CHUNK
    end = start + CHUNK - 1
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "probe"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=120) as r:
            data = r.read()
        dt = time.perf_counter() - t0
        results[i] = (len(data), dt, len(data) / dt / 1e6)
    except Exception as exc:  # noqa: BLE001
        results[i] = (-1, 0.0, 0.0)
        print(f"  conn{i}: FAILED {exc}")


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
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
print(f"聚合: {total/1e6:.1f} MB in {wall:.1f}s = {total/wall/1e6:.2f} MB/s (N={N})")

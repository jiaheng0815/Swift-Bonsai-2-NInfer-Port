# TUNING · 262k 上下文最佳参数（30 次起服务实测）

> 对象：`artifacts/swift-bonsai2-27b-pruned.ninfer`（7.63 GiB，裁剪词表制品，1119 张量）
> 机器：RTX 5060 Ti 16 GB（可用 16,311 MiB）+ 32 GB RAM
> 日期：2026年10月09日首测，2026年10月17日校对 · 原始数据在 `logs/tune/`（脚本见 `bench/`，可复跑）

---

## 一、结论：推荐命令

```powershell
build\b1\apps\ninfer-serve.exe artifacts\swift-bonsai2-27b-pruned.ninfer ^
  --host 127.0.0.1 --port 8080 --model-id swift-bonsai2-27b ^
  --max-context 262144 --kv-capacity 262144 --kv-dtype nvfp4 ^
  --prefill-chunk 8192 --max-concurrency 1 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --device-state-slots 1 --host-state-slots 0 --host-kv-mib 0 ^
  --max-private-continuations 1 --max-shared-prefixes 0 ^
  --vision --cors
```

（就是 `launch\launch_windows.bat` 里跑的东西。）

| 参数 | 取值 | 为什么是这个值 |
|---|---|---|
| `--max-context` | 262144 | 模型原生上限（`kNativeContext`）|
| `--kv-capacity` | **262144（显式）** | `auto` 在该上下文下**起不来**（见 §5）；省略等于 262144 |
| `--kv-dtype` | nvfp4 | 16 KiB/token，262k 只要 4.0 GiB；也是深度解码快的根本原因（§4）|
| `--prefill-chunk` | **8192** | 不掉速区间内的最大值；**再高只烧显存、不提速**（§2）|
| `--prefill-chunk` 保险值 | 7168 | 多 200 MiB 余量，速度一样（观察 `capacity \|` 行 free < 400 MiB 就退到这档）|
| `--max-concurrency` | 1 | 单用户；多并发按并发数分摊 KV |
| `--spec/--draft-tokens` | mtp / 3 | K=1/2/3 速度实测相同（§3）；**K=4/5 崩到 11 tok/s，绝对不要用** |
| `--lm-head-draft` | 开 | 显存预留与关掉完全相同（6.51 GiB），速度在噪声内 → 保留 |
| 状态槽三件套 | 最小可用缓存 | 换来的前缀复用极大：15k 前缀二次请求 TTFT **79 ms**、131k **825 ms**（§4）|
| `--vision` | 开 | 只占约 0.3 GiB 权重 |

启动后确认这一行没超：

```
capacity | KV 262,144 tokens, nvfp4, explicit | pages 4,096/4,096 | runtime 6.51 GiB | free 500~750 MiB
```

`free` ≥ 400 MiB 才算安全；低于这个数桌面（Chrome / Wallpaper Engine / 抖音）一涨显存
就会被 WDDM 换页，速度断崖。

## 二、prefill-chunk：它不是速度旋钮，是显存旋钮

同一份 262k/nvfp4/vision 配置，只改 `--prefill-chunk`（prompt 20.6k–27.6k token）：

| chunk | 运行时预留 | 启动后空闲显存 | prefill | 结论 |
|---|---|---|---|---|
| 1024（引擎默认）| 5.57 GiB | 1.44 GiB | 645.3 tok/s | |
| 2048 | 5.58 GiB | 1.45 GiB | 642.8 | 2k 以下几乎不要钱 |
| 4096 | 5.87 GiB | 1.17 GiB | 642.7 | |
| 7168 | 6.35 GiB | 715 MiB | 647.3 / 652.0 | 旧推荐值 |
| **8192** | **6.51 GiB** | **500–546 MiB** | **654.4 / 654.3** | **推荐：不发烫的最大值** |
| 9216 | 6.67 GiB | 382 MiB | — | 越线（< 400 MiB 安全线）|
| 10240 | 6.82 GiB | 197 MiB | **506.1** ⬇ | 掉速 23% |
| 12288 | 7.14 GiB | **0 B** ⬅ 榨干 | **517.9** ⬇ | 掉速 |
| 16384 | 需 7.78 GiB | — | **起不来** | `requested Engine runtime reservation requires 8350087936 bytes, but only 7724669440 bytes are available` |

- **1024 → 8192，速度 645 ↔ 654 tok/s（±1%，就是噪声）**：大 chunk 省下的 kernel 启动
  开销完全被 GEMM/反量化吞掉，所以「chunk 越高越快」在本机不成立。
- 显存成本线性：**约 160 KB / token**（0.156 MiB/token），从 chunk≈2048 起算。
- **掉速的根因是显存榨干，不是 chunk 本身**：把 KV 降到 131072（腾出 2.19 GiB）再跑
  chunk=12288 → **657.6 tok/s，满速**。

2026年10月11日复测（`--host-kv-mib 1024`，其余同）：7168 → free 674 MiB、8192 → 527–536 MiB、
9216 → 382 MiB（越线）。与上表一致：**8192 是不越线的最大值，往上没有余量**，
`bin\ninfer-server.ps1` 与 `launch\*.bat` 的默认值就取这一档，回退档 7168。

## 三、解码（MTP）：K=1/2/3 一样快，K=4/5 是陷阱

greedy（确定性，256 token 散文输出）：

| 配置 | decode | MTP 接受率 |
|---|---|---|
| K=1 + lm-head-draft | 64.1 tok/s | 62.8% |
| K=2 + lm-head-draft | 63.4 / 64.6 / 64.6 | 40.1% |
| K=3 + lm-head-draft | 64.8 / 64.7 / 64.4 | 32.3% |
| K=2 无 lm-head-draft | 68.9 | 44.6% |
| K=3 无 lm-head-draft | 57.6 | 24.0% |
| **K=4** | **11.5** ⬇⬇ | 21.1% |
| **K=5** | **11.7** ⬇⬇ | 18.8% |

- **K=1/2/3 吞吐相同（64–65 tok/s）**：K 越大接受率越低、但每步验证窗口也越大，两者抵消。
- **K=4/5 掉到 11 tok/s（5–6 倍损失）**：验证窗口（draft_window=K+1）越过引擎分档边界，
  走了一条极慢的 kernel 分解。**永远不要用 4/5。**
- 非 greedy（默认采样）会明显拉低接受率（24–43%），decode 在 57–66 tok/s 波动 —— 这是
  采样随机性决定的，不是参数能救的。

## 四、长上下文 / 深度解码（262k 的真实代价）

131k token 实测（`--prefill-chunk 8192`，nvfp4）：

| 场景 | 结果 |
|---|---|
| 131,289 token 预填充 | 271.6 s = **483.6 tok/s**（比短 prompt 的 650 慢 25%）|
| 前缀缓存命中后追加一轮 | 命中 131,282 / 100%，**TTFT 825 ms** |
| **131k 深度上的解码** | **82.2 tok/s**（浅层同任务 124.4 → 只掉 34%）|
| 15,259 token 前缀二次请求 | **TTFT 79 ms** |
| Claude Code 式 27.8k 系统提示 | 冷启动约 42 s，命中后 0.1–0.5 s |

**关键结论：262k 拉满之后解码并没有崩。** nvfp4 把 KV 压到 16 KiB/token（131k 深度 =
2.1 GiB），每步真正的大头仍是 7.6 GiB 权重流，所以深度解码只比浅层慢三分之一。

## 五、两个坑（实测确认）

1. **`--kv-capacity auto` 在 `--max-context 262144` 下起不来**：
   ```
   FATAL minimum Engine runtime reservation requires 6815734528 bytes in addition to
   1073741824 bytes of automatic headroom, but only 7724669440 bytes are available after weights
   ```
   auto 会额外要求 1024 MiB 余量、并把 262144 当作下限，等于「要 7.35 GiB 但只有 7.19 GiB」。
   改用显式 `--kv-capacity 262144`（或干脆省略）才是稳的。
2. **未裁剪制品加载不了**：引擎按裁剪域（208030）编译，248,320 域的旧制品报
   `tensor descriptor does not match target contract: text/token_embedding`。
   要做对比只能重编回旧版 —— 所以本仓从源头就统一走裁剪域。
3. `--device-state-slots 0` 无效（日志仍显示 `1 active + 1 cached device states`）。

## 六、复现方法

```
bench\bench.py        单配置基准：起服务 -> 预热 -> 长 prompt 预填充 -> 散文解码 -> 杀进程
bench\analyze.py      把 *.log 汇总成表
bench\sweep*.sh       chunk / draft / 状态槽 / kv-capacity 扫描
bench\depth_test.py   131k 深度往返（预填充 + 深度解码）
bench\prefix_test.py  前缀缓存专项
logs\tune\results.jsonl   全部结构化结果
```

例：

```powershell
python bench\bench.py --label my-test --prefill-tokens 12288 --spec mtp --draft-tokens 3 `
  --lm-head-draft --vision --prefill-chunk 8192 --host-kv-mib 0 --host-state-slots 0 `
  --device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0
```

## 七、一句话总结

**262,144 上下文 + nvfp4 + chunk 8192 + MTP K=3 + vision 的最小缓存配置**就是本机最优：
预填充 654 tok/s（短）/ 484 tok/s（131k），解码 64 tok/s（散文）/ 82 tok/s（131k 深度），
前缀命中后 TTFT 79–825 ms，启动后留 500+ MiB 显存余量。

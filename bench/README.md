# bench — 性能与行为基准（可复跑）

实测数据见 [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) 与 [docs/TUNING.md](../docs/TUNING.md)。

| 脚本 | 作用 |
|---|---|
| `swift_bench.py` | **Swift vs 基线：思考 token 对比**（引擎自报 `reasoning_tokens`）|
| `bench.py` | 起服务 → 预热 → 长 prompt 预填充 → 散文解码 → 杀进程，一行 JSON |
| `analyze.py` | 把 `*.log` 汇总成表 |
| `depth_test.py` | 131k 深度往返（预填充 + 深度解码）|
| `prefix_test.py` | 前缀缓存专项 |
| `agent_bench.py` | 引擎级 agent 循环（tool_use → tool_result → 最终答案）|
| `sweep*.sh` | 参数扫描（chunk / draft / 状态槽 / kv-capacity）|

## 用法示例

```powershell
# 思考 token A/B（先起对应服务）
python bench\swift_bench.py --label base  --model bonsai2-27b
python bench\swift_bench.py --label swift --model swift-bonsai2-27b

# 单配置基准（输出追加到 logs\tune\results.jsonl）
python bench\bench.py --label my-test --prefill-tokens 12288 --spec mtp --draft-tokens 3 `
  --lm-head-draft --vision --prefill-chunk 8192 --host-kv-mib 0 --host-state-slots 0 `
  --device-state-slots 1 --max-private-continuations 1 --max-shared-prefixes 0
```

> `bench.py` / `analyze.py` 的日志目录默认 `D:\ai-ninfer\logs\tune`（可用环境变量
> `NINFER_TUNE_LOGDIR` 覆盖）；`swift_bench.py` 的结果写在同目录的 `swift_bench.jsonl`。

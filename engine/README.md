# engine — NInfer 引擎的改动说明

> **完整引擎源码就在本仓库里： [`src/cranebw/`](../src/cranebw/)**（1449 文件 / 29.6 MB，
> 基树 `ad6cb46` + 本补丁已套用，可直接构建）—— 基树上游曾下线过一次（Ambolio 404），
> 所以这里做完整 vendoring。
>
> 本目录另存**改动记录**：`patches/ninfer-bonsai2-port.patch`（相对基树的完整 diff）、
> `stat.txt`（改动清单）与 Windows 构建脚本 `build_windows.bat`。
> [Hugging Face](https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port) 的
> `engine/` 目录也有同一份源码的 zip（含 `README-BUILD.txt`），供不做 git clone 的场景。

## 基树

| 项 | 值 |
|---|---|
| 基树 | [`CraneBW/ninfer-ternary-bonsai-ada`](https://github.com/CraneBW/ninfer-ternary-bonsai-ada)（Ambolio v1.0.8 Ada 线 + 魔搭三元补丁的派生）|
| 基准 commit | `ad6cb46`（patch 即相对它生成）|
| 许可 | Apache-2.0 |

```bash
git clone https://github.com/CraneBW/ninfer-ternary-bonsai-ada src/cranebw
cd src/cranebw && git checkout ad6cb46
git apply ../../engine/patches/ninfer-bonsai2-port.patch
```

## 改动分四组（31 文件，+875/−136）

### ① Blackwell / RTX 50 系编译适配（必改，否则编不过）

| 文件 | 改动 |
|---|---|
| `CMakeLists.txt` | `CMAKE_CUDA_ARCHITECTURES` 匹配 `^120` 时定义 `NINFER_SM120=1` |
| `src/core/device.h` | 新增 `NINFER_SM120` 分支：`kTargetSmCount = 36`（RTX 5060 Ti）|

编译用 `-DCMAKE_CUDA_ARCHITECTURES=120a`（参考架构）：warp-specialized TMA / NVFP4 /
k8v4 那批内核只在 `120a` 下完整编译 —— 这是本机能把 KV 池开到 262,144 token 的原因。

### ② 裁剪词表域（本项目的词表裁剪方法依赖它）

| 文件 | 改动 |
|---|---|
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h` | `kTokenDomain` → **208030** |
| `src/targets/qwen3_6_27b/impl/config.h` | `output_rows` → 208030（无 padding 尾巴）；`mask_token` → 208023（重编号）|
| `src/targets/qwen3_6_27b/impl/load/bindings.cpp` | embedding / output_head 绑定行数 → 208030 |
| `src/targets/qwen3_6/impl/frontend/frontend.cpp` | 33 个 special token 的新 id（见 docs/VOCAB_PRUNE.md §4）|

> **副作用（预期）**：引擎只认裁剪域制品；248,320 域的旧制品会报
> `tensor descriptor does not match target contract: text/token_embedding`。

### ③ 结构化解码（JSON schema 约束输出）

`src/serve/*`、`include/ninfer/ops/sampling.h`、`src/ops/kernel/sampling*.cuh`、
`src/runtime/*`、`src/targets/qwen3_6/impl/runtime/*`：把 `response_format` 的
`json_object` / `json_schema`（含 `$ref`、`anyOf`、递归引用等子集）编译成 token 级约束，
在采样前落 mask；`strict:true` 的工具参数同样受约束。约束请求按 token 逐位解码
（不投机）以保证许可前缀契约。

### ④ 采样与运行时（配套改动）

`speculative_round.cuh`、`program*`、`layouts*` 等为上面两组提供接线；改动清单见
`patches/stat.txt`。

## 验证

- 编译：`engine/build_windows.bat`（Windows）→ `build/b1/apps/{ninfer,ninfer-serve,ninfer-perplexity}.exe`
- 约束解码端到端：`tests/e2e-structured.ps1`、`tests/e2e-strict-tool.ps1`（需服务在跑）
- 数值正确性：PPL 判据见 `launch/run_ppl.bat`（个位数=正确）

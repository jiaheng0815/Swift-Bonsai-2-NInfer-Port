# USAGE · 使用说明

> 引擎：`CraneBW/ninfer-ternary-bonsai-ada`（本机按 `sm_120a` 重编，改动见 [engine/README.md](../engine/README.md)）
> 模型：Swift Bonsai 2 27B（`swift-bonsai2-27b`）/ 基线 Ternary Bonsai 2 27B（`bonsai2-27b`）
>
> 下载：[Hugging Face](https://huggingface.co/jiaheng0815/Swift-Bonsai-2-NInfer-Port) ·
> [魔搭 ModelScope](https://www.modelscope.cn/models/jiaheng0815/Swift-Bonsai-2-NInfer-Port)；
> 源码：[GitHub](https://github.com/jiaheng0815/Swift-Bonsai-2-NInfer-Port) ·
> [Gitee](https://gitee.com/fengjiaheng123/swift-bonsai-2-ninfer-port)。
> 文档最后校对：2026年10月17日。

> **两种目录布局**：git 仓里制品在 `artifacts\`、exe 在 `build\b1\apps\`；
> 完整包里制品在 `models\`、exe 在 `bin\`。下文命令以 git 仓为准，完整包用户把路径换成
> `models\swift-bonsai2-27b-pruned.ninfer` 与 `bin\ninfer-serve.exe` 即可
> （启动脚本已自动处理，不用改）。

---

## 1. 起服务

| 方式 | 命令 | 说明 |
|---|---|---|
| **双击（推荐）** | `launch\launch_windows.bat` | Swift 版，实测最优参数（完整包 / git 仓都有）|
| 双击 | `launch\launch_windows_base.bat` | 基线版（A/B 对比用；仅 git 仓，需自备基线制品）|
| 后台 | `launch\serve_detached.bat [model-id]` | 日志写文件，不开窗口（仅 git 仓）|
| 停止 | `launch\stop_server.bat` | 并确认显存回落 |
| Linux | `launch/launch_linux.sh` | ⚠️ 模板，未实测 |

**客户端 `model` 字段必须与服务端 `--model-id` 一致**（`swift-bonsai2-27b` 或 `bonsai2-27b`），
否则 `model_not_found`。本机未开鉴权，api_key 随便填。

> **一张卡只跑一个引擎。** 两个实例不只是抢显存：后启动的会在 `pinning host state` 阶段报
> `cudaMallocHost failed: ... out of memory`（页锁定内存拿不到）——这个报错的含义是
> 「已经有一个引擎在跑了」。先 `launch\stop_server.bat`。

### llama.cpp 风格启动器（可选）

`bin\ninfer-server.cmd`（可选；把 `bin\` 加入 PATH 后可在任意目录调用，见 §7）认
`-m / -mm / -c / -ctk / -ctv / --parallel / --prefill-chunk`，
自动翻译成 NInfer 参数；`--dry-run` 只打印最终命令：

```powershell
ninfer-server -m D:\ai-ninfer\artifacts\swift-bonsai2-27b-pruned.ninfer -c 262144 -ctk q4_0 -ctv q4_0 --parallel 1
```

`-ctk/-ctv` 映射：`q4_0/q5_0/q5_1/iq4_nl → nvfp4`、`q8_0/q8_1 → int8`、`f16/bf16 → bf16`、`fp8 → fp8`。

启动器已注入实测最优默认值（`--kv-capacity 262144`、`--prefill-chunk 8192`、MTP K=3 等），一般不用手动传；
`--prefill-chunk N` 可显式覆盖，传超过 8192 会提示越线（16 GB 卡实测上限，数据见 [TUNING §2](TUNING.md)）。

## 2. 调用（OpenAI / Anthropic 双协议）

Base URL：`http://127.0.0.1:8080/v1`（OpenAI），`http://127.0.0.1:8080`（Anthropic）。

```powershell
# 模型列表
curl.exe http://127.0.0.1:8080/v1/models

# 文本请求（body 写文件再 -d @文件，避免 Windows 引号地狱）
'{"model":"swift-bonsai2-27b","messages":[{"role":"user","content":"用三句话解释三重量化"}],"max_tokens":900}' > req.json
curl.exe -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '@req.json'
```

**思维链在 `reasoning_content` 字段**（流式下是同名 delta），`content` 只在收尾出答案；
只渲染 `content` 的客户端看起来会先空白一段，这是模型特性。**思考会吃 `max_tokens`**，
给太小会停在思考阶段、`content` 为空 —— 要长答案就调大到 1200+。

终端多轮聊天：`launch\chat.bat`（或 `python launch\chat.py swift-bonsai2-27b`），
命令：`/clear /think /stats /max N /temp F /exit`。

### 思考档位（每请求字段，不是启动参数）

```json
{"model":"swift-bonsai2-27b","reasoning_effort":"medium","messages":[...],"max_tokens":800}
```

这个模型的模板只接受 **`none / low / medium / xhigh`**（默认 `xhigh`，也是上限）；
`high` / `max` 会被 400 拒绝。

### 结构化解码（JSON schema）

`response_format` 支持 `text` / `json_object` / `json_schema`（含 `$ref`/`anyOf`/递归引用等子集），
约束在采样前编译成 token mask —— **输出按构造满足 schema**，不是事后校验：

```json
{"model":"swift-bonsai2-27b","messages":[{"role":"user","content":"Give me one Chinese city and its population."}],
 "max_tokens":400,"temperature":0,
 "response_format":{"type":"json_schema","json_schema":{"name":"city","schema":{"type":"object",
   "properties":{"city":{"type":"string"},"population":{"type":"integer"}},
   "required":["city","population"],"additionalProperties":false}}}}
```

### 图片 / 视频

源可以是 HTTP(S) URL 或 base64 data URI（服务端）；CLI 的 `--messages` 还接受本地路径。
视觉上限 `min(--max-context, 32768)` 个 merged token（约 131,072 原始 patch）。

```powershell
# 用引擎自带的官方测试图构造一次请求（base64 data URI）
python -c "import base64,json,pathlib;p=pathlib.Path(r'src\cranebw\examples\cli\media\visual_chart.png');print(json.dumps({'model':'swift-bonsai2-27b','max_tokens':400,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()}},{'type':'text','text':'图里有什么？'}]}]},ensure_ascii=False))" > req.json
curl.exe -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '@req.json'
```

视频同理，用 `"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}`。

## 3. 上下文能开多大 = KV 每 token 几个 bit

每 token 的 KV（16 个全注意力层 × 4 KV 头 × 256 dim × K+V = 32,768 元素）：

| `--kv-dtype` | 位/元素 | 每 token | 16 GB 卡（权重 7.62 GiB）实测 |
|---|---|---|---|
| `bf16` | 16 | 64 KiB | 262k 放不下 |
| `fp8` / `int8` | 8 | 32 KiB | 131,072 / 177,024 |
| `k8v4` | ≈12 | 24 KiB | 偏紧 |
| **`nvfp4`**（默认）| 4 | 16 KiB | **262,144 = 模型上限** |
| `rk4v4-e8` | 4 | 16 KiB | 本线独有（E8 格）|

对照 llama.cpp：`-ctk q4_0 -ctv q4_0` 是 4.5 bit（18 KiB/token），所以它能开 262k；
NInfer 的对应档位就是 `nvfp4`，实测同样开出 262,144。

**质量取舍**：4-bit KV 省一半显存但精度低于 fp8。本机实测 144,975 token 召回（nvfp4）
答对；要更保守就 `-c 131072 --kv-dtype int8`。

## 4. 参数对照表（llama.cpp → 本引擎）

| llama.cpp | 本引擎 |
|---|---|
| `-m <model>` | 第一个位置参数 `<artifact>`（给 GGUF 会自动换成 `.ninfer`）|
| `-mm <mmproj>` | 无（视觉塔在制品内，路径只做校验）|
| `-c N` | `--max-context N` |
| `-ub N`（概念近似）| `--prefill-chunk N`（预填充分块，显存旋钮；引擎默认 1024，本机实测上限 8192）|
| `-ctk` / `-ctv` | `--kv-dtype`（只有一个 KV 精度，K/V 必须一致）|
| `--parallel N` | `--max-concurrency N` |
| `--temp` | `--temperature` |
| `--reasoning-effort X` | 请求体 `reasoning_effort` |
| `--reasoning-preserve` | `--preserve-thinking` |
| `-ngl` / `-t` / `-fa` / `--chat-template-file` | 忽略（单卡全量加载；模板在制品 frontend）|
| `--webui-mcp-proxy` | 不支持 |

## 5. 排错

| 现象 | 处理 |
|---|---|
| 启动报显存不足 | 降 `--max-context`；或去掉 `--vision`；或把 `--prefill-chunk` 从 8192 降到 7168（判据：启动后 `capacity` 行的 `free` < 400 MiB 就该降）|
| `model_not_found` | `model` 必须与服务端 `--model-id` 一致 |
| 重新编译报 `LNK1104` | 先 `launch\stop_server.bat`（exe 被占用）|
| 中文 `ninfer.exe --prompt` 报 NFC 归一化失败 | 中文走 `--messages <utf8.json>` 或 HTTP |
| 控制台中文乱码 | 编码问题不是模型问题；`chcp 65001` |
| 答案为空但思维链正常 | `max_tokens` 太小，思考没结束 |
| 起不来 / 报显存被占 | `nvidia-smi` 看有无别的引擎；`launch\stop_server.bat` |
| `cudaMallocHost failed ... out of memory` | **不是内存 bug，是已经有一个引擎在跑** |

## 6. Claude Code 接入（已实测）

`~/.claude/settings.json` 里把 Anthropic 相关项指到本机服务：

```jsonc
"ANTHROPIC_BASE_URL":  "http://127.0.0.1:8080",
"ANTHROPIC_AUTH_TOKEN": "local-ninfer",       // 未开鉴权，随便填
"ANTHROPIC_MODEL":     "swift-bonsai2-27b",
"CLAUDE_CODE_EFFORT_LEVEL": "xhigh",          // 模型只认 none/low/medium/xhigh
"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",  // 本地只有 262144
"NO_PROXY": "127.0.0.1,localhost"             // 关键：否则被系统代理拦走
```

两个必须知道的点：

1. **`claude -p` 默认文本输出会报 400**（交互式不受影响）：Claude Code 非交互 + 文本输出时
   会发 `thinking.display="omitted"`，NInfer 明确拒绝。绕法：
   `claude -p "..." --output-format json --verbose`（或 `stream-json`）。
2. **首轮 TTFT 8–11 秒不是解码慢**，是 Claude Code 系统提示（27.8k token）的预填充；
   第二轮起前缀缓存命中 24k–108k token，TTFT 降到 0.1–0.5 s。

实测（基线模型）：`contextWindow 262144`、系统提示 27,815 token、首轮 TTFT 46.7 s→缓存后 7.6 s；
Agent 任务（4–8 轮、1–3 次工具调用）20–25 秒完成。

## 7. 文件都在哪

```
artifacts\            制品（.ninfer）与裁剪资源（git 仓）
build\b1\apps\        三个 exe：ninfer / ninfer-serve / ninfer-perplexity（git 仓）
bin\                  llama.cpp 风格启动器 ninfer-server（git 仓）
launch\               启动/停止/聊天/PPL 脚本
conversion\           词表裁剪 / 重建 / 验证工具
engine\               引擎改动说明与 patch
docs\                 全部文档
eval\                 PPL 语料与 MMLU 评测
logs\                 运行日志与调参数据（bench\ 脚本的默认输出目录）
src\cranebw\          引擎源码树（第三方 + engine\patches 应用后）
guide\                魔搭指南原仓（第三方，pack.py / tools\verify；不入库）

完整包布局：bin\（exe+DLL） models\（已裁剪制品） conversion\ launch\ eval\
```

**把 exe / 启动器加进 PATH（可选）**：在 PowerShell 里执行一次（用户级，永久）：

```powershell
[Environment]::SetEnvironmentVariable(
  'Path',
  [Environment]::GetEnvironmentVariable('Path','User') + ';D:\ai-ninfer\build\b1\apps;D:\ai-ninfer\bin',
  'User')
```

完整包用户把路径换成 `<解压目录>\bin` 即可。

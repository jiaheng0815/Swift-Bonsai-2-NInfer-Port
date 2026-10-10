# tests — 端到端验证（需服务在跑）

| 脚本 | 覆盖 |
|---|---|
| `e2e-structured.ps1` | 约束解码在三个协议表面（Chat / Responses / Messages）的行为 |
| `e2e-strict-tool.ps1` | 工具 `strict:true` 的参数约束 |

```powershell
# 先起服务（launch\launch_windows.bat），再：
pwsh -File tests\e2e-structured.ps1
pwsh -File tests\e2e-strict-tool.ps1
```

> 制品级与在线功能的验证脚本在 `conversion/verify/`（不需要引擎特性，只需要服务）。

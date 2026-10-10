"""Engine-level agent loop benchmark: tool_use -> tool_result -> final answer.

Mirrors what Claude Code does per turn, so the numbers are comparable to the
end-to-end Claude Code measurements.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8080/v1/messages"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": "local",
    "anthropic-version": "2023-06-01",
}
TOOLS = [{
    "name": "Bash",
    "description": "Run a shell command on the local machine.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The command to run"}},
        "required": ["command"],
    },
}]


def call(body: dict) -> tuple[dict, float]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(BASE, data=data, headers=HEADERS)
    t0 = time.perf_counter()
    with opener.open(req, timeout=900) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out, time.perf_counter() - t0


def summarize(tag: str, resp: dict, wall: float) -> None:
    u = resp.get("usage", {})
    kinds = [b.get("type") for b in resp.get("content", [])]
    print(f"{tag}: wall {wall:6.2f}s | in {u.get('input_tokens'):>6} "
          f"(cache_read {u.get('cache_read_input_tokens')}) | out {u.get('output_tokens'):>4} "
          f"| stop {resp.get('stop_reason'):<10} | blocks {kinds}")
    if "thinking_tokens" in (u.get("output_tokens_details") or {}):
        print(f"{'':>{len(tag)}}  thinking {u['output_tokens_details']['thinking_tokens']} tok"
              f" -> decode {(u.get('output_tokens') / wall):.1f} tok/s (incl. prefill+thinking)")


def main() -> int:
    task = ("用 Bash 工具执行命令 `dir /b D:\\ai-ninfer\\artifacts`，"
            "然后告诉我该目录下有哪些文件。")
    print("=== TURN 1 (user -> model: expect tool_use) ===")
    messages = [{"role": "user", "content": task}]
    r1, w1 = call({"model": "bonsai2-27b", "max_tokens": 1024, "temperature": 0,
                   "tools": TOOLS, "tool_choice": {"type": "auto"}, "messages": messages})
    summarize("turn1", r1, w1)

    tool_use = next((b for b in r1.get("content", []) if b.get("type") == "tool_use"), None)
    if not tool_use:
        print("!! model did not call the tool; stopping")
        return 1
    cmd = (tool_use.get("input") or {}).get("command", "")
    print(f"   -> tool_use {tool_use['name']}({cmd!r})")

    print("=== TOOL EXECUTION (the client side, like Claude Code does) ===")
    t0 = time.perf_counter()
    proc = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True)
    tool_ms = (time.perf_counter() - t0) * 1000
    tool_out = (proc.stdout or proc.stderr or "").strip()
    print(f"   tool wall {tool_ms:.0f} ms | output: {tool_out[:200]!r}")

    print("=== TURN 2 (tool_result -> model: expect final text) ===")
    messages += [
        {"role": "assistant", "content": r1["content"]},
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use["id"],
            "content": [{"type": "text", "text": tool_out or "(no output)"}],
        }]},
    ]
    r2, w2 = call({"model": "bonsai2-27b", "max_tokens": 1024, "temperature": 0,
                   "tools": TOOLS, "messages": messages})
    summarize("turn2", r2, w2)
    text = next((b.get("text") for b in r2.get("content", []) if b.get("type") == "text"), "")
    print(f"   final answer: {text.strip()[:200]!r}")
    print(f"\nAGENT LOOP TOTAL: {w1 + w2 + tool_ms / 1000:.2f}s "
          f"(turn1 {w1:.2f}s + tool {tool_ms/1000:.2f}s + turn2 {w2:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

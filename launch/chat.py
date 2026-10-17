#!/usr/bin/env python3
"""Terminal chat client for the local NInfer server (OpenAI-compatible).

Multi-turn, with prefix caching on the server side: each turn only pays for the new
suffix.  Reasoning comes back in `reasoning_content` -- this client prints it dim and
separately from the answer, because the model thinks before it answers and a client
that only renders `content` looks frozen for a while.

Commands: /clear  /think  /stats  /max N  /temp F  /exit
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8080/v1"
# 模型名可用命令行参数或环境变量覆盖（例如 bonsai2-27b），默认 Swift
MODEL = os.environ.get("NINFER_MODEL") or (sys.argv[1] if len(sys.argv) > 1 else "swift-bonsai2-27b")
DIM = "\033[2m"
OFF = "\033[0m"


def call(messages: list[dict], max_tokens: int, temperature: float | None) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach {BASE} ({e}). Start the server first: start-server.cmd")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    history: list[dict] = []
    max_tokens = 2048
    temperature: float | None = 1.0
    show_thinking = True

    print(f"NInfer chat -> {BASE} | model {MODEL}")
    print("commands: /clear /think /stats /max N /temp F /exit\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return 0
        if user == "/clear":
            history.clear()
            print("(history cleared)")
            continue
        if user == "/think":
            show_thinking = not show_thinking
            print(f"(thinking shown: {show_thinking})")
            continue
        if user.startswith("/max "):
            max_tokens = int(user.split()[1])
            print(f"(max_tokens={max_tokens})")
            continue
        if user.startswith("/temp "):
            temperature = float(user.split()[1])
            print(f"(temperature={temperature})")
            continue
        if user == "/stats":
            print(f"(turns={len(history)}, max_tokens={max_tokens}, temperature={temperature})")
            continue

        history.append({"role": "user", "content": user})
        data = call(history, max_tokens, temperature)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        thinking = msg.get("reasoning_content") or ""
        if show_thinking and thinking:
            print(f"{DIM}[thinking] {thinking.strip()}{OFF}")
        print(f"bot> {content.strip()}")
        t = data.get("timings") or {}
        usage = data.get("usage") or {}
        if t:
            print(
                f"     {usage.get('completion_tokens', t.get('predicted_n'))} tok | "
                f"decode {t.get('predicted_per_second', 0):.1f} t/s | "
                f"prefill {t.get('prompt_per_second', 0):.1f} t/s "
                f"({t.get('prompt_n', 0)} tok, cached {t.get('cache_n', 0)}) | "
                f"MTP {t.get('draft_n_accepted', 0)}/{t.get('draft_n', 0)}"
            )
        history.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    raise SystemExit(main())

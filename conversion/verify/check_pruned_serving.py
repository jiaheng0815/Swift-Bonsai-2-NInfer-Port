"""对裁剪后模型的在线功能验证。

直接打 http://127.0.0.1:8080 的 OpenAI 兼容端点，覆盖：
  1. 中文对话 / 英文对话 / 代码（必须流利、无乱码）
  2. 特殊 token 路径（chat template 走 <|im_start|>/<|im_end|>）
  3. 结构化输出（JSON schema 约束 → token trie + mask，是受 token_domain 影响的路径）
  4. 被删语言（俄语）必须仍能应答，不能崩

用法：
    python check_pruned_serving.py [--base http://127.0.0.1:8080]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

REPLACEMENT = "�"


def post(base: str, path: str, payload: dict, timeout: float = 180.0) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# 这是 thinking 模型：reasoning_content 会先消耗 token，预算必须够长，
# 否则 content 会因 finish_reason=length 而为空（不是模型坏了）。
TOKEN_BUDGET = 400


MODEL = "bonsai2-27b"   # main() 可用 --model 覆盖（如 swift-bonsai2-27b）


def chat(base: str, content: str, **extra) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": TOKEN_BUDGET,
        "temperature": 0.3,
        **extra,
    }
    result = post(base, "/v1/chat/completions", payload)
    return result["choices"][0]["message"]["content"]


CASES = [
    ("中文", "用一句话介绍你自己。"),
    ("中文-任务", "把下面这句话改写成更正式的书面语：这个方案我觉得不太行。"),
    ("英文", "Say hello in one short sentence."),
    ("代码", "写一个 Python 函数，返回两个数的和。只给代码。"),
    ("中英混", "解释一下什么是 tokenizer 的 vocab_size。"),
    ("标点符号", "请用中文标点写一句包含破折号和省略号的话：他说——算了……"),
    ("俄语-已删语言", "Привет! Как дела?"),
]


def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=MODEL, help="服务端 --model-id（默认 bonsai2-27b）")
    args = parser.parse_args()
    MODEL = args.model

    failures = 0
    for label, prompt in CASES:
        try:
            answer = chat(args.base, prompt)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            print(f"FAIL  {label:<18} HTTP {exc.code}: {body}")
            failures += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {label:<18} {type(exc).__name__}: {exc}")
            failures += 1
            continue

        clean = answer.strip()
        bad = REPLACEMENT in clean
        # 中英文用例里出现大段非中英文字符视为异常
        if not bad and label not in ("俄语-已删语言",):
            exotic = len(re.findall(r"[Ѐ-ӿ؀-ۿ가-힯]", clean))
            if exotic > max(3, len(clean) * 0.1):
                bad = True
        status = "FAIL" if bad else "OK"
        if bad:
            failures += 1
        preview = clean.replace("\n", " ")[:78]
        print(f"{status:<5} {label:<18} {preview}")

    print()
    print("=== structured output (JSON schema -> token trie + mask) ===")
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "city",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
                "required": ["city", "population"],
                "additionalProperties": False,
            },
        },
    }
    try:
        result = post(
            args.base,
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Give me one Chinese city and its population."}],
                "max_tokens": TOKEN_BUDGET,
                "temperature": 0.0,
                "response_format": schema,
            },
        )
        content = result["choices"][0]["message"]["content"]
        print(f"  raw: {content!r}")
        parsed = json.loads(content)
        print(f"  OK   parsed as JSON: city={parsed.get('city')!r} population={parsed.get('population')!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL structured output: {type(exc).__name__}: {exc}")
        failures += 1

    print()
    if failures:
        raise SystemExit(f"{failures} check(s) FAILED")
    print("ALL SERVING CHECKS PASSED")


if __name__ == "__main__":
    main()

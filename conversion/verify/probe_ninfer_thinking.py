"""探测 ninfer-serve 接受哪种 thinking 开关，以决定 Pi 的 thinkingFormat。"""
import json
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"


def probe(label, extra):
    payload = {
        "model": "bonsai2-27b",
        "messages": [{"role": "user", "content": "1+1=? 只回答数字"}],
        "max_tokens": 150,
        "temperature": 0.0,
        **extra,
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode())
        message = result["choices"][0]["message"]
        reasoning = message.get("reasoning_content") or ""
        content = (message.get("content") or "").replace("\n", " ")[:34]
        print(f"  {label:<40} reasoning={'YES (%d ch)' % len(reasoning) if reasoning else 'no ':>12}  content={content!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<40} ERROR {type(exc).__name__}: {str(exc)[:80]}")


print("probing ninfer-serve thinking switches")
probe("(nothing)", {})
probe("enable_thinking=False  (top level)", {"enable_thinking": False})
probe("enable_thinking=True   (top level)", {"enable_thinking": True})
probe("chat_template_kwargs.enable_thinking=False", {"chat_template_kwargs": {"enable_thinking": False}})
probe("thinking={type:enabled}  (deepseek style)", {"thinking": {"type": "enabled"}})

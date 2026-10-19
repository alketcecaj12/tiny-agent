"""A tiny fake OpenAI-compatible server for offline tests.

It answers `/chat/completions` and `/embeddings` with deterministic, rule-based
responses that exercise the same code paths as a real model:

* native tools in the request  -> first turn returns a `tool_calls` entry,
  the turn after a tool result returns a plain answer;
* prompt-based tools (`# Tools` in the system prompt) -> a ReAct-style
  THOUGHT/ACTION JSON blob, then `final_answer` after an OBSERVATION;
* otherwise -> an echo of the last user message.

Run standalone to try the examples without a model:

    python tests/fake_openai_server.py --port 8765
    TINYAGENT_BASE_URL=http://127.0.0.1:8765/v1 python examples/ch06_react.py
"""

import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _text(content) -> str:
    if isinstance(content, list):  # multimodal
        return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content or ""


def _dummy_args(properties: dict) -> dict:
    values = {"integer": 2, "number": 2.5, "boolean": True, "array": ["a"], "object": {}}
    return {name: values.get(spec.get("type"), "2") for name, spec in properties.items()}


def chat_reply(body: dict) -> dict:
    messages = body["messages"]
    last = messages[-1]
    tools = body.get("tools")
    system = _text(messages[0]["content"]) if messages[0]["role"] == "system" else ""
    message = {"role": "assistant", "content": None}

    if tools:
        if last["role"] == "tool":
            message["content"] = f"Final answer based on: {last['content']}"
        else:
            fn = tools[0]["function"]
            args = _dummy_args(fn["parameters"]["properties"])
            message["tool_calls"] = [
                {
                    "id": f"call_{len(messages)}",
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": json.dumps(args)},
                }
            ]
        if body.get("reasoning_effort") not in (None, "none"):
            message["reasoning"] = "Let me think about which tool to use."
    elif "# Tools" in system:
        content = _text(last["content"])
        if content.startswith("OBSERVATION:"):
            answer = content.removeprefix("OBSERVATION:").strip()
            message["content"] = (
                'THOUGHT: I have the result.\nACTION:\n'
                + json.dumps({"tool": "final_answer", "kwargs": f"The result is {answer}"})
            )
        else:
            name = re.search(r"`(\w+)`:", system).group(1)
            # Signature from "name(a: str, b: str)" or "path (str):" docstring lines
            signature = re.search(rf"{name}\(([^)]*)\)", system)
            if signature:
                params = [p.split(":")[0].strip() for p in signature.group(1).split(",")]
            else:
                params = re.findall(r"^\s+(\w+) \(\w+\):", system, re.MULTILINE)
            value = re.search(r"https?://\S+?(?=['\s]|$)", _text(messages[1]["content"]))
            kwargs = {p: value.group() if value else "3" for p in params if p}
            message["content"] = (
                f"THOUGHT: I should use {name}.\nACTION:\n"
                + json.dumps({"tool": name, "kwargs": kwargs})
            )
    else:
        message["content"] = f"Echo: {_text(last['content'])}"
        if body.get("reasoning_effort") not in (None, "none"):
            message["reasoning"] = "Thinking natively."

    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def embedding_reply(body: dict) -> dict:
    digest = hashlib.sha256(body["input"].encode()).digest()
    vector = [b / 255 for b in digest[:16]]
    # Make texts that share words more similar
    for word in set(re.findall(r"\w+", body["input"].lower())):
        vector[hash(word) % 16] += 1.0
    return {"data": [{"embedding": vector, "index": 0}], "model": body["model"]}


class Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
        if self.path == "/v1/chat/completions":
            payload, status = chat_reply(body), 200
        elif self.path == "/v1/embeddings":
            payload, status = embedding_reply(body), 200
        else:
            payload, status = {"error": "not found"}, 404
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def start(port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    port = parser.parse_args().port
    print(f"Fake OpenAI-compatible server on http://127.0.0.1:{port}/v1")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fake_openai_server  # noqa: E402

from tinyagent.llm import Response  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Never pick up a developer's real .env or TINYAGENT_* settings."""
    for key in list(os.environ):
        if key.startswith("TINYAGENT_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tinyagent.config.load_dotenv", lambda path=None: None)


@pytest.fixture
def server(monkeypatch):
    fake_openai_server.Handler.requests = []
    httpd = fake_openai_server.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    monkeypatch.setenv("TINYAGENT_BASE_URL", url)
    yield url, fake_openai_server.Handler.requests
    httpd.shutdown()


class FakeLLM:
    """Returns scripted responses and records the messages it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model = "fake"

    def generate(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        response = self.responses.pop(0)
        return response if isinstance(response, Response) else Response(content=response)


@pytest.fixture
def fake_llm():
    return FakeLLM

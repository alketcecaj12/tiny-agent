import pytest

from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM, EmbeddingModel
from tinyagent.toolbox import multiply
from tinyagent.tools import NativeTools


def test_generate_sends_openai_request(server):
    url, requests = server
    llm = LLM(model="m", api_key="secret", think=True, temperature=0.5)
    response = llm.generate([{"role": "user", "content": "hello"}])

    assert response.content == "Echo: hello"
    assert response.reasoning == "Thinking natively."
    assert response.metadata == {"model": "m", "prompt_tokens": 10, "completion_tokens": 5}
    sent = requests[-1]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["reasoning_effort"] == "medium"
    assert sent["body"]["temperature"] == 0.5
    assert sent["headers"]["Authorization"] == "Bearer secret"


def test_reasoning_effort_can_be_disabled(server, monkeypatch):
    _, requests = server
    LLM(think=False).generate([{"role": "user", "content": "x"}])
    assert requests[-1]["body"]["reasoning_effort"] == "none"
    monkeypatch.setenv("TINYAGENT_SEND_REASONING_EFFORT", "false")
    LLM(think=True).generate([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in requests[-1]["body"]


def test_env_defaults(server):
    url, _ = server
    llm = LLM()
    assert llm.base_url == url
    assert llm.model == "gemma4:e4b"
    assert llm.api_key == "no_key"


def test_native_agent_end_to_end_over_http(server):
    _, requests = server
    tools = NativeTools()
    tools.add_tool("multiply", multiply)
    agent = TinyAgent(llm=LLM(think=True), tools=tools)
    answer = agent.run("What is 2 times 2?")
    assert answer == "Final answer based on: 4.0"
    second = requests[-1]["body"]["messages"]
    assert second[-1] == {"role": "tool", "content": "4.0", "tool_call_id": "call_2"}


def test_embeddings(server):
    vector = EmbeddingModel().embed("Flamingos are pink")
    assert len(vector) == 16


def test_unreachable_endpoint_has_helpful_error():
    llm = LLM(base_url="http://127.0.0.1:9/v1")
    with pytest.raises(ConnectionError, match="TINYAGENT_BASE_URL"):
        llm.generate([{"role": "user", "content": "x"}])


def test_http_error_includes_body(server):
    url, _ = server
    llm = LLM(base_url=url.replace("/v1", "/nope"))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        llm.generate([{"role": "user", "content": "x"}])

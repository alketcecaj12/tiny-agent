"""The "brain" of the TinyAgent (Chapter 2 & 3).

Everything talks to an OpenAI-compatible `/chat/completions` endpoint using only
the Python standard library.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from tinyagent import config


@dataclass
class Response:
    """Structured response from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


def _post_json(url: str, body: dict, api_key: str | None, timeout: float) -> dict:
    """POST a JSON body and return the decoded JSON response."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"LLM endpoint {url} returned HTTP {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"Could not reach the LLM endpoint at {url} ({error.reason}). "
            "Is your server (e.g. `ollama serve`) running and is "
            "TINYAGENT_BASE_URL correct?"
        ) from error


class LLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        think: bool | None = False,
        temperature: float | None = None,
        timeout: float = 300,
    ):
        """Initialize the LLM with the given model.

        Arguments left as `None` fall back to the `TINYAGENT_*` environment
        variables (see `.env.example`).

        `think` toggles native reasoning: True sends `reasoning_effort="medium"`,
        False sends `"none"`, and None omits the parameter entirely. Set
        TINYAGENT_SEND_REASONING_EFFORT=false for providers that reject it.
        """
        self.model = model or config.model("native")
        self.base_url = (base_url or config.base_url()).rstrip("/")
        self.api_key = api_key or config.api_key()
        self.think = think
        self.temperature = temperature
        self.timeout = timeout
        if not config.env_bool("SEND_REASONING_EFFORT", True):
            self.think = None

    def generate(
        self, messages: list[dict], tools: list | None = None
    ) -> Response:
        """Generate a response from the LLM given a list of messages."""
        # Build the request body
        body = {
            "model": self.model,
            "messages": messages,
        }

        # Tools and Reasoning
        if tools:
            body["tools"] = tools
        if self.think is True:
            body["reasoning_effort"] = "medium"
        elif self.think is False:
            body["reasoning_effort"] = "none"
        if self.temperature is not None:
            body["temperature"] = self.temperature

        # POST to the OpenAI-compatible /chat/completions endpoint
        data = _post_json(
            f"{self.base_url}/chat/completions", body, self.api_key, self.timeout
        )

        # Extract message, tool_call, and metadata
        message = data["choices"][0]["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        usage = data.get("usage") or {}
        metadata = {
            "model": data.get("model", self.model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

        # Format as Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=reasoning,
            tool_call=tool_call,
            metadata=metadata,
        )


class EmbeddingModel:
    """Generate embeddings (Chapter 4, long-term memory)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120,
    ):
        """Initialize the embedding model with the given model."""
        self.model = model or config.model("embedding")
        self.base_url = (
            base_url or config.env("EMBEDDING_BASE_URL") or config.base_url()
        ).rstrip("/")
        self.api_key = (
            api_key or config.env("EMBEDDING_API_KEY") or config.api_key()
        )
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector."""
        # POST to the OpenAI-compatible /embeddings endpoint
        response = _post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": text},
            self.api_key,
            self.timeout,
        )

        # Extract and return the embedding
        return response["data"][0]["embedding"]

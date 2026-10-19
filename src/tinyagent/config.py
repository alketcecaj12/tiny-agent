"""Configuration loaded from environment variables (and an optional `.env` file).

The book uses an OpenAI-compatible endpoint for every LLM call. By default that
is a local Ollama server, but anything that speaks the OpenAI chat-completions
API works (llama.cpp, LM Studio, OpenRouter, OpenAI, ...).
"""

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Load `KEY=VALUE` lines from a `.env` file into `os.environ`.

    A tiny, dependency-free replacement for `python-dotenv`. Variables that are
    already set in the environment are never overwritten.
    """
    candidates = [Path(path)] if path else [Path.cwd() / ".env"]
    if not path:
        # Also look next to the project root when running from a checkout
        candidates.append(Path(__file__).resolve().parents[2] / ".env")

    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        return


def env(name: str, default: str | None = None) -> str | None:
    """Read a `TINYAGENT_*` setting, loading `.env` on first use."""
    load_dotenv()
    value = os.environ.get(f"TINYAGENT_{name}")
    return value if value not in (None, "") else default


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Defaults mirror the notebooks: Ollama on localhost with Gemma models
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "no_key"
DEFAULT_MODEL = "gemma4:e4b"  # native thinking + tool calling
DEFAULT_PROMPT_MODEL = "gemma3:12b"  # prompt-based reasoning/tools
DEFAULT_VISION_MODEL = "gemma4:e4b-it-qat"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"


def base_url() -> str:
    return env("BASE_URL", DEFAULT_BASE_URL)


def api_key() -> str:
    return env("API_KEY", DEFAULT_API_KEY)


def model(kind: str = "native") -> str:
    """Model name for a given use.

    Arguments:
        kind: One of "native" (thinking + tool calling), "prompt" (prompt-based
              chapters), "vision" (multimodal) or "embedding".
    """
    defaults = {
        "native": ("MODEL", DEFAULT_MODEL),
        "prompt": ("PROMPT_MODEL", DEFAULT_PROMPT_MODEL),
        "vision": ("VISION_MODEL", DEFAULT_VISION_MODEL),
        "embedding": ("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    }
    name, default = defaults[kind]
    return env(name, default)

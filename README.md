# TinyAgent

An AI agent built from scratch in plain Python: an LLM "brain", memory, tools,
skills, MCP, ReAct planning, evaluation, multi-agent collaboration, vision and
a coding agent you can run in your terminal.

This project is a plain-Python port of the Jupyter notebooks that accompany
[*An Illustrated Guide to AI Agents*](https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents)
by Maarten Grootendorst and Jay Alammar. Every notebook became a runnable
script in `examples/`, and the agent itself lives in the `tinyagent` package.

The core has **zero dependencies**: it talks to any OpenAI-compatible endpoint
using only the standard library. `rich` (prettier terminal) and `mcp`
(Model Context Protocol) are optional.

## Quickstart

1. **Run a local model.** The defaults match the book and use [Ollama](https://ollama.com):

   ```bash
   ollama pull gemma4:e4b        # native thinking + tool calling
   ollama pull gemma3:12b        # prompt-based chapters (3-6)
   ollama pull embeddinggemma    # long-term memory (chapter 4)
   ollama serve
   ```

   Any other OpenAI-compatible server works too (llama.cpp, LM Studio,
   OpenRouter, OpenAI...). See [Configuration](#configuration).

2. **Install** (Python 3.10+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -e ".[all]"          # or just `pip install -e .` for zero deps
   cp .env.example .env             # optional, edit to change provider/models
   ```

3. **Run the coding agent:**

   ```bash
   tinyagent                         # interactive, works in the current folder
   tinyagent -q "Create hello.py that prints the date and run it"
   tinyagent --help
   ```

   In the chat, `/help` lists commands (`/tools`, `/memory`, `/trajectory`,
   `/clear`, ...). `write_file` and `execute_python` ask for approval unless
   you pass `--yes`.

4. **Or walk through the book, chapter by chapter:**

   ```bash
   python examples/ch01_scaffold.py
   python examples/ch06_react.py
   ```

## Chapters

| Notebook | Script | What it shows |
|---|---|---|
| chapter01 | `examples/ch01_scaffold.py` | The agent skeleton (no LLM needed) |
| chapter02 | `examples/ch02_llm.py` | Raw HTTP call, `LLM` wrapper, `Response`, `Trajectory` |
| chapter03 | `examples/ch03_reasoning.py` | Few-shot, chain-of-thought, self-consistency, best-of-N, native reasoning (`--samples N`) |
| chapter04 | `examples/ch04_memory.py` | Agent without vs. with conversation memory |
| chapter04_short_term_memory | `examples/ch04_short_term_memory.py` | `TrimmingMemory`, `SummarizationMemory` |
| chapter04_long_term_memory | `examples/ch04_long_term_memory.py` | Embeddings, cosine similarity, `RAGMemory` |
| chapter05 | `examples/ch05_tools.py` | Prompt-based tool calling |
| chapter05_native_tool_calling | `examples/ch05_native_tool_calling.py` | JSON schemas from Python functions, native `tool_calls` |
| chapter05_skills | `examples/ch05_skills.py` | `SKILL.md` skills loaded progressively |
| chapter05_mcp | `examples/ch05_mcp.py` | Tools served by an MCP server (needs `.[mcp]`) |
| chapter06 | `examples/ch06_react.py` | ReAct loop with THOUGHT/ACTION/OBSERVATION |
| chapter06_native_react | `examples/ch06_native_react.py` | ReAct with native reasoning and tools |
| chapter07 | `examples/ch07_evaluation.py` | Benchmarks: exact match, programmatic checks, LLM-as-a-judge |
| chapter08 | `examples/ch08_multi_agent.py` | Orchestrator delegating to specialist agents |
| chapter09 | `examples/ch09_multimodal.py` | Image + text input (`[image] [question]` arguments) |
| chapter10 | `examples/ch10_coding_agent.py` | Minimal coding agent with human-in-the-loop |

## Project layout

```
tinyagent/
├── pyproject.toml
├── .env.example
├── src/tinyagent/
│   ├── config.py       # TINYAGENT_* settings and a tiny .env loader
│   ├── llm.py          # LLM, EmbeddingModel, Response          (ch 2-3)
│   ├── trajectory.py   # Step, Trajectory                       (ch 2)
│   ├── memory.py       # Memory, Trimming/Summarization/RAG/Multimodal (ch 4, 9)
│   ├── tools.py        # Tools, NativeTools, Skills, tool_to_schema (ch 5)
│   ├── mcp_server.py   # MCP server with a read_markdown tool    (ch 5)
│   ├── mcp_tools.py    # MCPTools client registry                (ch 5)
│   ├── planning.py     # ReAct, NativeReAct, PlanAndSolve, Reflexion (ch 6)
│   ├── evaluation.py   # Benchmark, Evaluator, scorers           (ch 7)
│   ├── toolbox.py      # Ready-made tools, incl. sandboxed code tools (ch 10)
│   ├── display.py      # Terminal output (plain ANSI or rich)    (ch 10)
│   ├── agent.py        # TinyAgent
│   ├── cli.py          # the `tinyagent` command
│   └── skills/file_analyzer/SKILL.md
├── examples/           # one script per notebook
└── tests/              # offline tests with a fake OpenAI-compatible server
```

## Use it as a library

```python
from tinyagent import LLM, Memory, NativeTools, NativeReAct, TinyAgent, Display
from tinyagent.toolbox import add, multiply

tools = NativeTools()
tools.add_tool("add", add)
tools.add_tool("multiply", multiply)

agent = TinyAgent(
    llm=LLM(think=True),           # reads TINYAGENT_* settings
    memory=Memory(),
    tools=tools,
    planner=NativeReAct(max_steps=10),
    display=Display(),             # omit for silent runs
)
print(agent.run("What is (4.6 + 6.685) x 4?"))
print(agent.trajectory.pretty())
```

## Configuration

All settings are environment variables (or lines in a `.env` file in the
folder you run from). Everything is optional.

| Variable | Default | Used for |
|---|---|---|
| `TINYAGENT_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `TINYAGENT_API_KEY` | `no_key` | Bearer token for the endpoint |
| `TINYAGENT_MODEL` | `gemma4:e4b` | Native thinking + tool calling (CLI, ch 5-10) |
| `TINYAGENT_PROMPT_MODEL` | `gemma3:12b` | Prompt-based chapters (3-6) |
| `TINYAGENT_VISION_MODEL` | `gemma4:e4b-it-qat` | Chapter 9 |
| `TINYAGENT_EMBEDDING_MODEL` | `embeddinggemma` | Chapter 4 long-term memory |
| `TINYAGENT_EMBEDDING_BASE_URL` / `_API_KEY` | main endpoint | Separate embedding provider |
| `TINYAGENT_JUDGE_MODEL` / `_BASE_URL` / `_API_KEY` | main LLM | Chapter 7 LLM-as-a-judge |
| `TINYAGENT_THINK` | `true` | Native reasoning in the CLI |
| `TINYAGENT_SEND_REASONING_EFFORT` | `true` | Set `false` if your provider rejects `reasoning_effort` |

For example, to use OpenRouter:

```bash
TINYAGENT_BASE_URL=https://openrouter.ai/api/v1
TINYAGENT_API_KEY=sk-or-...
TINYAGENT_MODEL=google/gemma-4-31b-it
TINYAGENT_PROMPT_MODEL=google/gemma-3-12b-it
```

## Tests

The tests run fully offline against a small fake OpenAI-compatible server,
including every example script:

```bash
pip install -e ".[dev]"
pytest
```

You can also start the fake server yourself to try the examples without a model:

```bash
python tests/fake_openai_server.py --port 8765
TINYAGENT_BASE_URL=http://127.0.0.1:8765/v1 python examples/ch06_react.py
```

## Differences from the notebooks

- Notebook-only helpers (`CodeAnnotator`, `DiffViewer`, `TrajectoryViewer`,
  Colab setup cells) are replaced by `Trajectory.pretty()` and printed output.
- The chapter-specific `TinyAgent` versions are folded into one final
  `TinyAgent`; `memory`, `tools`, `planner` and `display` are optional, so the
  early chapters still work with `TinyAgent(llm)`.
- `requests`, `httpx` and `yaml` were replaced by the standard library.
- Small robustness fixes: tool errors are returned to the agent as an
  observation instead of crashing, native tool results carry `tool_call_id`
  (required by OpenAI-style APIs), off-format ReAct answers are kept, the
  `/skills` CLI command works, and the code tools reject paths outside the
  workspace.

## Credits

The design, code and examples come from the book's companion repository by
Maarten Grootendorst and Jay Alammar
([HandsOnLLM/An-Illustrated-Guide-To-AI-Agents](https://github.com/HandsOnLLM/An-Illustrated-Guide-To-AI-Agents)),
whose package metadata declares the MIT license. Please keep this attribution
when you publish your copy.

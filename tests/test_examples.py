"""Run every example script against the fake server (no real model needed)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"

OFFLINE = {
    "ch01_scaffold.py": [],
    "ch02_llm.py": [],
    "ch03_reasoning.py": ["--samples", "2"],
    "ch04_memory.py": [],
    "ch04_short_term_memory.py": [],
    "ch04_long_term_memory.py": [],
    "ch05_tools.py": [],
    "ch05_native_tool_calling.py": [],
    "ch06_react.py": [],
    "ch06_native_react.py": [],
    "ch07_evaluation.py": [],
    "ch08_multi_agent.py": [],
}


def run(script, args, url, stdin=""):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TINYAGENT_")}
    env["TINYAGENT_BASE_URL"] = url
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True, text=True, timeout=120, env=env, input=stdin,
    )


@pytest.mark.parametrize("script", sorted(OFFLINE))
def test_example_runs(script, server):
    url, _ = server
    result = run(script, OFFLINE[script], url)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_coding_agent_example(server, tmp_path):
    url, _ = server
    result = run("ch10_coding_agent.py", [], url, stdin="List the files\nn\nexit\n")
    assert result.returncode == 0, result.stderr
    assert "ACTION" in result.stdout


def test_every_example_is_covered():
    scripts = {p.name for p in EXAMPLES.glob("ch*.py")}
    network = {"ch05_skills.py", "ch05_mcp.py", "ch09_multimodal.py", "ch10_coding_agent.py"}
    assert scripts == set(OFFLINE) | network

from tinyagent.llm import Response
from tinyagent.toolbox import build_code_tools, multiply
from tinyagent.tools import BUILTIN_SKILLS_DIR, NativeTools, Skills, Tools, tool_to_schema


def test_tool_to_schema():
    def f(a: str, b: int, c: float = 1.0):
        """Docs."""

    schema = tool_to_schema(f)
    params = schema["function"]["parameters"]
    assert schema["function"]["name"] == "f"
    assert schema["function"]["description"] == "Docs."
    assert params["properties"] == {
        "a": {"type": "string"},
        "b": {"type": "integer"},
        "c": {"type": "number"},
    }
    assert params["required"] == ["a", "b"]


def test_prompt_tools_parse_and_execute():
    tools = Tools()
    tools.add_tool("multiply", multiply, "Multiplies two numbers")
    assert "`multiply`: Multiplies two numbers" in tools.prompt

    response = tools.parse(Response(content='{"tool": "multiply", "kwargs": {"a": "3", "b": "2"} }'))
    assert response.tool_call == {"tool": "multiply", "kwargs": {"a": "3", "b": "2"}}
    assert tools.execute(response) == 6.0
    assert tools.observation(6.0) == ("user", "OBSERVATION: 6.0")


def test_parse_tolerates_trailing_comma_and_none_content():
    tools = Tools()
    response = tools.parse(Response(content='{"tool": "x", "kwargs": {"a": 1},}'))
    assert response.tool_call["tool"] == "x"
    assert tools.parse(Response(content=None)).tool_call is None


def test_execute_unknown_tool_and_errors():
    tools = Tools()
    tools.add_tool("multiply", multiply)
    assert "not found" in tools.execute(Response(tool_call={"tool": "nope"}))
    result = tools.execute(Response(tool_call={"tool": "multiply", "kwargs": {"a": "x", "b": "1"}}))
    assert result.startswith("Error while running 'multiply'")


def test_requires_approval(monkeypatch):
    tools = Tools(requires_approval=["multiply"])
    tools.add_tool("multiply", multiply)
    call = Response(tool_call={"tool": "multiply", "kwargs": {"a": "2", "b": "2"}})
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert "denied" in tools.execute(call)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert tools.execute(call) == 4.0


def test_final_answer_is_done():
    tools = Tools()
    response = Response(tool_call={"tool": "final_answer", "kwargs": "42"})
    assert tools.is_done(response) and response.content == "42"
    response = Response(tool_call={"tool": "final_answer", "kwargs": {"answer": "42"}})
    assert tools.is_done(response) and response.content == "42"
    assert not tools.is_done(Response(tool_call={"tool": "multiply"}))


def test_native_tools():
    tools = NativeTools()
    tools.add_tool("multiply", multiply)
    assert tools.prompt == ""
    assert tools.schemas[0]["function"]["name"] == "multiply"

    raw = {"id": "call_1", "function": {"name": "multiply", "arguments": '{"a": "2", "b": "5"}'}}
    response = tools.parse(Response(tool_call=raw))
    assert response.tool_call == {"id": "call_1", "tool": "multiply", "kwargs": {"a": "2", "b": "5"}}
    assert tools.execute(response) == 10.0
    assert tools.observation(10.0) == ("tool", "10.0")
    assert tools.is_done(Response(content="hi"))


def test_skills_load_builtin():
    skills = Skills()
    name = skills.add_skill(BUILTIN_SKILLS_DIR / "file_analyzer")
    assert name == "file_analyzer"
    assert skills.skills["file_analyzer"]["description"].startswith("Analyze files")
    assert skills.schemas[0]["function"]["parameters"]["properties"] == {}
    call = skills.parse(Response(tool_call={"function": {"name": "file_analyzer", "arguments": {}}}))
    assert "# File Analyzer Skill" in skills.execute(call)


def test_code_tools_are_sandboxed(tmp_path):
    tools = {name: func for name, (func, _) in build_code_tools(str(tmp_path)).items()}
    assert tools["write_file"]("a/b.txt", "hello\nworld") == "Written to 'a/b.txt'."
    assert tools["read_file"]("a/b.txt") == "hello\nworld"
    assert tools["read_lines"]("a/b.txt", "2", "2") == "2: world"
    assert tools["list_files"]("a") == "b.txt"
    assert tools["find_files"]("*.txt") == "a/b.txt"
    assert tools["search_file"]("wor", "a/b.txt") == "2: world"
    assert tools["execute_python"]("print(1 + 1)") == "2"
    try:
        tools["read_file"]("../outside.txt")
    except ValueError as error:
        assert "outside the workspace" in str(error)
    else:
        raise AssertionError("path traversal was not rejected")

import json

from tinyagent.agent import TinyAgent
from tinyagent.llm import Response
from tinyagent.planning import ReAct
from tinyagent.toolbox import add, multiply
from tinyagent.tools import NativeTools, Tools


def test_simple_agent_with_memory(fake_llm):
    llm = fake_llm(["Hi Maarten and Jay!", "You are Maarten and Jay."])
    agent = TinyAgent(llm=llm)
    agent.run("We are Maarten and Jay")
    assert agent.run("What are our names?") == "You are Maarten and Jay."
    roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[0]["tools"] == []  # no tools -> nothing sent


def test_prompt_based_react_loop(fake_llm):
    llm = fake_llm(
        [
            'THOUGHT: add first\nACTION:\n{"tool": "add", "kwargs": {"a": "4.6", "b": "6.685"}}',
            'THOUGHT: multiply\nACTION:\n{"tool": "multiply", "kwargs": {"a": "11.285", "b": "4"}}',
            'THOUGHT: done\nACTION:\n{"tool": "final_answer", "kwargs": "45.14"}',
        ]
    )
    tools = Tools()
    tools.add_tool("add", add, "add(a: str, b: str)")
    tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")
    agent = TinyAgent(llm=llm, tools=tools, planner=ReAct())

    assert agent.run("What is (4.6 + 6.685) x 4?") == "45.14"
    system = agent.memory.get_messages()[0]["content"]
    assert "# ReAct" in system and "# Tools" in system
    steps = agent.trajectory.runs[0]["steps"]
    assert [s.observation for s in steps[:2]] == ["OBSERVATION: 11.285", "OBSERVATION: 45.14"]
    assert steps[0].thought == "add first"
    assert steps[2].answer == "45.14"


def test_native_tool_loop_records_tool_call_id(fake_llm):
    call = {"id": "call_7", "type": "function",
            "function": {"name": "multiply", "arguments": json.dumps({"a": "5.1", "b": "7.3"})}}
    llm = fake_llm([Response(content=None, tool_call=call), Response(content="37.23")])
    tools = NativeTools()
    tools.add_tool("multiply", multiply)
    agent = TinyAgent(llm=llm, tools=tools)

    assert agent.run("What is 5.1 times 7.3?") == "37.23"
    messages = agent.memory.get_messages()
    assert messages[2]["tool_calls"] == [call]
    assert messages[3]["role"] == "tool" and messages[3]["tool_call_id"] == "call_7"
    assert llm.calls[0]["tools"][0]["function"]["name"] == "multiply"


def test_max_steps(fake_llm):
    call = {"id": "c", "function": {"name": "add", "arguments": '{"a": "1", "b": "1"}'}}
    llm = fake_llm([Response(tool_call=call)] * 3)
    tools = NativeTools()
    tools.add_tool("add", add)
    agent = TinyAgent(llm=llm, tools=tools, planner=ReAct(max_steps=3))
    agent.planner.parse = lambda r: r  # behave natively
    assert agent.run("loop") == "Max steps reached without completion."

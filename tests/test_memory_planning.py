from tinyagent.llm import Response
from tinyagent.memory import Memory, MultimodalMemory, RAGMemory, SummarizationMemory, TrimmingMemory
from tinyagent.planning import NativeReAct, PlanAndSolve, ReAct
from tinyagent.tools import Tools


def test_memory_tool_call_fields():
    memory = Memory()
    memory.add("assistant", None, tool_call={"id": "1"})
    memory.add("tool", "ok", tool_call_id="1")
    assert memory.get_messages() == [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "ok", "tool_call_id": "1"},
    ]


def test_trimming_memory_keeps_system_and_last_four():
    memory = TrimmingMemory()
    memory.add("system", "sys")
    for i in range(6):
        memory.add("user", f"u{i}")
    contents = [m["content"] for m in memory.get_messages()]
    assert contents == ["sys", "u2", "u3", "u4", "u5"]


def test_summarization_memory(fake_llm):
    llm = fake_llm(["Sarah asked about flamingos."])
    memory = SummarizationMemory(llm)
    memory.add("user", "Hi I'm Sarah")
    memory.add("assistant", "Hello Sarah")
    assert memory.get_messages() == [{"role": "system", "content": "Sarah asked about flamingos."}]
    assert "user: Hi I'm Sarah" in llm.calls[0]["messages"][0]["content"]


class FakeEmbedder:
    def embed(self, text):
        return [float("flamingo" in text.lower()), float("dolphin" in text.lower()), 0.1]


def test_rag_memory_retrieves_relevant_documents():
    docs = ["Dolphins are smart.", "Flamingos are pink.", "Cats sleep."]
    memory = RAGMemory(FakeEmbedder(), docs, k=1)
    memory.add("user", "Tell me about flamingos")
    content = memory.get_messages()[0]["content"]
    assert content.startswith("Context:\nFlamingos are pink.")
    assert content.endswith("Question: Tell me about flamingos")


def test_multimodal_memory():
    memory = MultimodalMemory(mime_type="image/jpeg")
    memory.add("user", "What is this?", image_data="QUJD")
    content = memory.get_messages()[0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert content[1] == {"type": "text", "text": "What is this?"}
    memory.add("user", "and this?", image_data="https://x/y.png")
    assert memory.get_messages()[1]["content"][0]["image_url"]["url"] == "https://x/y.png"


def test_react_parse():
    response = ReAct().parse(Response(content='THOUGHT: use add\nACTION:\n{"tool": "add"}'))
    assert response.reasoning == "use add"
    assert response.content == '{"tool": "add"}'
    # Off-format answers are kept rather than dropped
    assert ReAct().parse(Response(content="Just 4.")).content == "Just 4."
    assert NativeReAct().prompt == ""


def test_plan_and_solve_tools():
    tools = Tools()
    planner = PlanAndSolve(tools)
    assert {"update_plan", "mark_done"} <= set(tools.registry)
    planner.update_plan(["a", "b"])
    assert planner.mark_done(1) == "Plan:\n[x] 1. a\n[ ] 2. b"

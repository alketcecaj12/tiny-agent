"""The TinyAgent: LLM + Memory + Tools + Planning + Display (Chapters 1-10)."""

from dataclasses import replace

from tinyagent.display import SilentDisplay
from tinyagent.llm import LLM, Response
from tinyagent.memory import Memory
from tinyagent.planning import NativeReAct, ReAct
from tinyagent.tools import NativeTools, Tools
from tinyagent.trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory | None = None,
        tools: Tools | None = None,
        planner: ReAct | None = None,
        display=None,
        system_prompt: str = "You are a helpful assistant.\n\n",
    ):
        self.llm = llm
        self.memory = memory if memory is not None else Memory()
        self.tools = tools if tools is not None else NativeTools()
        self.planner = planner if planner is not None else NativeReAct()
        self.display = display if display is not None else SilentDisplay()

        self.trajectory = Trajectory()

        # Build system prompt with all components
        system_prompt += self.planner.prompt
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str, image_data: str | None = None) -> str:
        """Run the agent on a task."""
        if image_data:
            self.memory.add("user", task, image_data=image_data)
        else:
            self.memory.add("user", task)
        self.trajectory.initialize(task)

        # *Autonomy* loop
        for _ in range(self.planner.max_steps):
            result = self._step()
            if result is not None:
                return result

        return "Max steps reached without completion."

    def _step(self) -> str | None:
        """Perform a single step."""
        # THOUGHT: Generate response and add to memory
        self.display("thinking")
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add("assistant", response.content, tool_call=response.tool_call)

        # Tool parsing
        response = self.planner.parse(response)
        response = self.tools.parse(response)

        # ANSWER: Stopping mechanism
        if self.tools.is_done(response):
            self.display("response", response)
            self.trajectory.add(response)
            return response.content or ""

        # Show only the THOUGHT; the ACTION is displayed next
        self.display("response", replace(response, content=None))

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> None:
        """Execute a tool action."""

        # ACTION: execute tools
        self.display("tool_call", response)
        result = self.tools.execute(response)

        # OBSERVATION: add tool results to memory and display
        role, observation = self.tools.observation(result)
        tool_call_id = response.tool_call.get("id") if role == "tool" else None
        self.memory.add(role, observation, tool_call_id=tool_call_id)
        self.trajectory.add(response, observation)
        self.display("observation", observation)

        return None

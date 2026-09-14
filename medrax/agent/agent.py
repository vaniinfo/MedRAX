import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        validator: Any = None,
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        # PATCH: optional EvidenceValidator. When set, every tool result is validated
        # by code inside execute_tools -- the model cannot skip it, unlike a system
        # prompt asking for validation. Pattern borrowed from CXRAgent's _explain_func
        # (arXiv:2510.21324); the arithmetic half is computed rather than prompted.
        self.validator = validator
        self.pending_validations: List[str] = []
        self.pending_records: List[Dict[str, Any]] = []

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)
            # PATCH: a readable running transcript alongside the per-turn JSON. The JSON
            # is for machines; this is for reading why a conclusion was reached.
            self.session_log = self.log_path / (
                f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            self._write_log(f"=== MedRAX session started {datetime.now().isoformat()} ===")

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    @staticmethod
    def _last_user_text(state: AgentState) -> str:
        """Most recent human-authored text, used to infer which finding is in question."""
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", None) != "human":
                continue
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                if any(parts):
                    return " ".join(parts)
        return ""

    def _write_log(self, text: str) -> None:
        """Append to the readable session transcript. Never raises."""
        if not self.log_tools:
            return
        try:
            with open(self.session_log, "a") as handle:
                handle.write(text.rstrip() + "\n")
        except Exception:
            pass

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        # PATCH: validations are injected here rather than folded into the ToolMessage
        # content, because interface.py calls eval() on that content and would break.
        if self.pending_validations:
            messages = messages + [HumanMessage(content=(
                "Validation of the tool results above. The computed lines are facts, "
                "not suggestions: a tool marked UNINFORMATIVE must not be counted as a "
                "vote, and your stated confidence must not exceed the computed ceiling."
                "\n\n" + "\n\n".join(self.pending_validations)
            ))]
            self.pending_validations = []
            self.pending_records = []
        response = self.model.invoke(messages)

        # PATCH: record what the orchestrator concluded, so it can be compared against
        # what each tool actually reported above it in the same log.
        stamp = datetime.now().strftime("%H:%M:%S")
        if getattr(response, "tool_calls", None):
            names = ", ".join(tc["name"] for tc in response.tool_calls)
            self._write_log(f"\n[{stamp}] DIRECTOR -> calling tools: {names}")
            for tc in response.tool_calls:
                self._write_log(f"      {tc['name']}({tc['args']})")
        if response.content:
            self._write_log(f"\n[{stamp}] DIRECTOR ANSWER\n{response.content}")
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                result = self.tools[call["name"]].invoke(call["args"])

            # PATCH: forced validation -- a function call, not an instruction.
            if self.validator is not None:
                try:
                    focus = self.validator.infer_focus(
                        call.get("args", {}) or {}, self._last_user_text(state)
                    )
                    record = self.validator.assess(call, result, focus=focus)
                    self.pending_validations.append(self.validator.render_for_model(record))
                    self.pending_records.append(record)
                    self._write_log(
                        f"\n[{datetime.now().strftime('%H:%M:%S')}] TOOL VALIDATION\n"
                        + self.validator.render_for_log(record)
                    )
                except Exception as exc:
                    self.pending_validations.append(
                        f"<validation tool=\"{call['name']}\">failed: {exc}</validation>"
                    )
                    self._write_log(f"  VALIDATION FAILED for {call['name']}: {exc}")
            else:
                self._write_log(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] TOOL {call['name']} "
                    f"(validation disabled)\n  ARGS: {call['args']}\n"
                    f"  RAW OUTPUT: {str(result)[:300]}"
                )

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            # PATCH: attach the validation record for this tool, if one was produced
            for record in self.pending_records:
                if record.get("tool") == call.name:
                    log_entry["validation"] = record
                    break
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

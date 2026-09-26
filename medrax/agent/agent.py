import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .edv import EvidenceValidator

_ = load_dotenv()

# Tools whose results make no clinical claim, so EDV has nothing to validate
EDV_SKIP_TOOLS = {"image_visualizer", "dicom_processor"}


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
    # EDV critiques, one per validated tool call: {"tool_call_id", "tool", "note", "explanation"}
    edv_notes: Annotated[List[Dict[str, Any]], operator.add]


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
        edv: str = "off",
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
            edv (str, optional): CXRAgent-style evidence validation, in which every
                clinical tool result is critiqued against the image.
                "off": no validation.
                "cxragent": a final responder writes the answer from the critiques.
                "advisory": critiques are shown and logged only; the director's answer
                stands, so the critic cannot overturn the specialist tool.
                Defaults to "off".
        """
        if edv not in ("off", "cxragent", "advisory"):
            raise ValueError(f"Unknown edv mode {edv!r}")
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        self.edv = edv != "off"
        self.edv_respond = edv == "cxragent"
        self.validator = EvidenceValidator(model) if self.edv else None

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_node("respond", self.respond)
        workflow.add_conditional_edges(
            "process", self.route, {"execute": "execute", "respond": "respond", "end": END}
        )
        workflow.add_edge("execute", "process")
        workflow.add_edge("respond", END)
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

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
        response = self.model.invoke(messages)
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

    def route(self, state: AgentState) -> str:
        """Choose the next node after the director: run tools, hand over to the EDV
        responder (only if this turn produced critiques), or finish."""
        if self.has_tool_calls(state):
            return "execute"
        if self.edv_respond and self._current_notes(state):
            return "respond"
        return "end"

    @staticmethod
    def _current_turn(state: AgentState) -> List[AnyMessage]:
        """Messages after the latest user message, i.e. this turn's director/tool messages."""
        messages = state["messages"]
        last_user = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
        )
        return messages[last_user + 1 :]

    def _current_notes(self, state: AgentState) -> List[Dict[str, Any]]:
        """EDV notes for tool calls made in this turn."""
        call_ids = {
            m.tool_call_id for m in self._current_turn(state) if isinstance(m, ToolMessage)
        }
        return [n for n in state.get("edv_notes", []) if n["tool_call_id"] in call_ids]

    def respond(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """EDV final responder: answers from the image, tool conclusions and critiques."""
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        question = user_messages[-1].content if user_messages else ""
        if isinstance(question, list):
            question = " ".join(p.get("text", "") for p in question if p.get("type") == "text")

        image_paths = []
        for message in self._current_turn(state):
            for call in getattr(message, "tool_calls", None) or []:
                for path in call["args"].get("image_paths", []):
                    if path not in image_paths:
                        image_paths.append(path)

        notes = [n["note"] for n in self._current_notes(state)]
        answer = self.validator.respond(question, image_paths, notes)
        return {"messages": [AIMessage(content=answer)]}

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
        notes = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            valid = call["name"] in self.tools
            if not valid:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                result = self.tools[call["name"]].invoke(call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )

            # Forced EDV: every successful clinical tool result is critiqued. The critique
            # is kept out of the director's messages, as in CXRAgent.
            if self.edv and valid and call["name"] not in EDV_SKIP_TOOLS:
                image_paths = call["args"].get("image_paths") or [
                    p for p in [call["args"].get("image_path")] if p
                ]
                note = {"tool_call_id": call["id"], "tool": call["name"]}
                try:
                    note["explanation"] = self.validator.explain(str(result), image_paths)
                except Exception as e:
                    print(f"Evidence validation failed: {e}")
                    note["explanation"] = "Evidence validation skipped due to error."
                note["note"] = self.validator.note(str(result), note["explanation"])
                notes.append(note)

        self._save_tool_calls(results, notes)
        print("Returning to model processing!")

        return {"messages": results, "edv_notes": notes}

    def _save_tool_calls(
        self, tool_calls: List[ToolMessage], notes: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
            notes (List[Dict], optional): EDV critiques, logged with their tool call.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"
        notes_by_id = {n["tool_call_id"]: n for n in notes or []}

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            if call.tool_call_id in notes_by_id:
                log_entry["edv_explanation"] = notes_by_id[call.tool_call_id]["explanation"]
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

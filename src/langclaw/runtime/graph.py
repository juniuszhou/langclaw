"""LangGraph-based agent runtime: ReAct loop with tools and checkpointer."""

from typing import Annotated, Any, List, NotRequired, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    """State for the agent graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    rag_context: NotRequired[str]


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


def _format_rag_docs(docs: list) -> str:
    if not docs:
        return ""
    parts = []
    for i, d in enumerate(docs, start=1):
        text = getattr(d, "page_content", str(d))
        src = ""
        meta = getattr(d, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("source"):
            src = f" ({meta['source']})"
        parts.append(f"[{i}]{src}\n{text}")
    return "\n\n---\n\n".join(parts)


def create_agent_graph(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str = "You are a helpful AI assistant. Use tools when needed to answer questions.",
    checkpointer: Optional[Any] = None,
    rag_retriever: Optional[BaseRetriever] = None,
) -> StateGraph:
    """Create a compiled LangGraph agent with ReAct loop.

    Args:
        model: Chat model (must support tool calling via bind_tools).
        tools: List of tools the agent can use.
        system_prompt: System message content.
        checkpointer: Optional checkpointer (MemorySaver, PostgresSaver, etc.). If None, uses MemorySaver.
        rag_retriever: If set, a retriever node runs once per invoke and injects context into the system prompt.

    Returns:
        Compiled StateGraph (invokable agent).
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict:
        """Invoke model with messages; may return tool_calls."""
        ctx = (state.get("rag_context") or "").strip()
        sys_text = system_prompt
        if ctx:
            sys_text = (
                system_prompt.rstrip()
                + "\n\n## Retrieved context\n"
                + ctx
                + "\nAnswer using this context when relevant."
            )
        system_msg = SystemMessage(content=sys_text)
        messages = [system_msg] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Route to tools if last message has tool_calls, else end."""
        messages = state["messages"]
        if not messages:
            return "end"
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "continue"
        return "end"

    graph = StateGraph(AgentState)
    if rag_retriever is not None:
        _retriever = rag_retriever

        def retriever_node(state: AgentState) -> dict:
            """Run once per invoke: fill rag_context from last user message."""
            query = _last_human_text(state["messages"]).strip()
            if not query:
                return {"rag_context": ""}
            docs = _retriever.invoke(query)
            return {"rag_context": _format_rag_docs(list(docs or []))}

        graph.add_node("retriever", retriever_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    if rag_retriever is not None:
        graph.set_entry_point("retriever")
        graph.add_edge("retriever", "agent")
    else:
        graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


class AgentRuntime:
    """High-level agent runtime: compiles graph and provides invoke/stream."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str = "You are a helpful AI assistant. Use tools when needed.",
        checkpointer: Optional[Any] = None,
        rag_retriever: Optional[BaseRetriever] = None,
    ):
        self.graph = create_agent_graph(
            model, tools, system_prompt, checkpointer, rag_retriever
        )
        self.checkpointer = checkpointer or MemorySaver()

    def invoke(
        self,
        input: dict,
        config: Optional[dict] = None,
    ) -> dict:
        """Run agent and return final state. Input: {"messages": [BaseMessage, ...]}."""
        config = config or {}
        if "configurable" not in config:
            config["configurable"] = {}
        return self.graph.invoke(input, config=config)

    def stream(
        self,
        input: dict,
        config: Optional[dict] = None,
        stream_mode: str = "values",
    ):
        """Stream agent steps. Input: {"messages": [BaseMessage, ...]}."""
        config = config or {}
        if "configurable" not in config:
            config["configurable"] = {}
        return self.graph.stream(input, config=config, stream_mode=stream_mode)

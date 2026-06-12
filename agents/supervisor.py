"""
agents/supervisor.py
Classifies user intent and routes to the correct agent node.
Routes: rag_agent | analyst_agent | code_agent
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState

_llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

SYSTEM_PROMPT = """You are a routing supervisor for a data analysis system.
Given a user query about sales data, classify it into exactly ONE of these categories:

- rag_agent     → conceptual questions, summaries, context lookup, "what is", "explain", "tell me about"
- analyst_agent → statistical analysis, trends, rankings, charts, comparisons, "show me", "visualize", "top N", "best/worst"
- code_agent    → complex filtering, custom calculations, ad-hoc queries, "calculate", "filter where", "compute"

Reply with ONLY the category name, nothing else."""


def supervisor_node(state: AgentState) -> AgentState:
    query = state["query"]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    response = _llm.invoke(messages)
    route = response.content.strip().lower()

    valid = {"rag_agent", "analyst_agent", "code_agent"}
    if route not in valid:
        route = "analyst_agent"

    return {**state, "next_agent": route}


def router(state: AgentState) -> str:
    """Returns the next node name for LangGraph conditional edge."""
    return state.get("next_agent", "analyst_agent")

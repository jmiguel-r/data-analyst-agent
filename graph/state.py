"""
graph/state.py
Shared TypedDict state flowing through the LangGraph StateGraph.
"""
from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    query: str

    # ── Routing ──────────────────────────────────────────────────────────────
    next_agent: str       # "rag_agent" | "analyst_agent" | "code_agent"
    agent_used: str

    # ── RAG agent ────────────────────────────────────────────────────────────
    context: str
    rag_response: str

    # ── Analyst agent ────────────────────────────────────────────────────────
    analyst_insight: str
    chart_path: Optional[str]
    analyst_response: str

    # ── Code agent ───────────────────────────────────────────────────────────
    generated_code: str
    code_result: str
    code_success: bool
    code_response: str

    # ── Final ────────────────────────────────────────────────────────────────
    final_answer: str

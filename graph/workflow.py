"""
graph/workflow.py
Defines the LangGraph StateGraph connecting:
  Supervisor → [rag_agent | analyst_agent | code_agent] → consolidate → END
"""
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.supervisor import supervisor_node, router
from agents.rag_agent import rag_agent
from agents.analyst_agent import analyst_agent
from agents.code_agent import code_agent


def consolidate(state: AgentState) -> AgentState:
    """Final node — ensures final_answer is set."""
    agent = state.get("agent_used", "unknown")
    answer = state.get("final_answer", "No response generated.")
    label = {
        "rag_agent":      "📚 RAG Agent",
        "analyst_agent":  "📊 Analyst Agent",
        "code_agent":     "💻 Code Agent",
    }.get(agent, agent)
    return {**state, "final_answer": f"[{label}]\n\n{answer}"}


def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("supervisor",     supervisor_node)
    g.add_node("rag_agent",      rag_agent)
    g.add_node("analyst_agent",  analyst_agent)
    g.add_node("code_agent",     code_agent)
    g.add_node("consolidate",    consolidate)

    # Entry point
    g.set_entry_point("supervisor")

    # Conditional routing from supervisor
    g.add_conditional_edges(
        "supervisor",
        router,
        {
            "rag_agent":     "rag_agent",
            "analyst_agent": "analyst_agent",
            "code_agent":    "code_agent",
        },
    )

    # All agents → consolidate → END
    for node in ["rag_agent", "analyst_agent", "code_agent"]:
        g.add_edge(node, "consolidate")
    g.add_edge("consolidate", END)

    return g.compile()


# Singleton for import
graph = build_graph()

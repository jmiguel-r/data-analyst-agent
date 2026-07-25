"""
agents/code_agent.py
Generates Python code to answer complex queries, then executes it
in a sandboxed environment (restricted builtins, no file system writes).
"""
import pandas as pd
import io
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
from graph.state import AgentState

DATA_PATH = Path(__file__).parent.parent / "data" / "sales.csv"
_llm      = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

CODEGEN_PROMPT = """You are a Python data analyst. A pandas DataFrame called `df` is already loaded with these columns:
date (datetime), category, product, vendor, region, channel, quantity (int),
unit_price (float), discount (float), revenue (float), year (int), month (int).

Write ONLY executable Python code (no markdown, no backticks) that:
1. Answers the user's question using pandas operations on `df`
2. Prints the result clearly using print()
3. Is concise and correct

Do NOT import pandas or any library — they are already available.
Do NOT write to files. Do NOT use plt.show().
"""

EXPLAIN_PROMPT = """You are a data analyst. Given the Python code and its output, write a clear, 
concise explanation of the results for a business audience. Focus on insights, not code details."""


def _safe_exec(code: str, df: pd.DataFrame) -> tuple[str, bool]:
    """Execute code in a restricted namespace. Returns (output, success)."""
    allowed_builtins = {
        "print": print, "len": len, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "sorted": sorted,
        "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "str": str, "int": int, "float": float, "bool": bool,
        "True": True, "False": False, "None": None,
    }
    namespace = {
        "__builtins__": allowed_builtins,
        "df": df.copy(),
        "pd": pd,
    }
    captured = io.StringIO()
    import sys
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        exec(code, namespace)
        sys.stdout = old_stdout
        return captured.getvalue().strip(), True
    except Exception:
        sys.stdout = old_stdout
        return traceback.format_exc(), False


def code_agent(state: AgentState) -> AgentState:
    query = state["query"]

    # Load DataFrame with derived columns
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Step 1: Generate code
    gen_messages = [
        SystemMessage(content=CODEGEN_PROMPT),
        HumanMessage(content=query),
    ]
    code_response = _llm.invoke(gen_messages)
    code = code_response.content.strip()

    # Step 2: Execute
    result, success = _safe_exec(code, df)

    # Step 3: Explain results
    explain_messages = [
        SystemMessage(content=EXPLAIN_PROMPT),
        HumanMessage(content=f"Code:\n{code}\n\nOutput:\n{result}\n\nOriginal question: {query}"),
    ]
    explanation = _llm.invoke(explain_messages)

    final = f"{explanation.content}\n\n---\n**Execution output:**\n```\n{result}\n```"

    return {
        **state,
        "generated_code": code,
        "code_result":    result,
        "code_success":   success,
        "code_response":  explanation.content,
        "agent_used":     "code_agent",
        "final_answer":   final,
    }

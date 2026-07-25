"""
main.py
Interactive CLI for the Data Analyst Agent.

LangSmith tracing is enabled automatically when these env vars are set:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=ls__...
  LANGCHAIN_PROJECT=data-analyst-agent   (optional, defaults to project name)

Copy .env.example to .env and fill in your keys to get started.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith tracing ────────────────────────────────────────────────────────
# LangChain/LangGraph instruments all runs automatically when these are set.
# No additional code changes needed — tracing is transparent.
os.environ.setdefault("LANGCHAIN_PROJECT", "data-analyst-agent")

_tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
_langsmith_key   = os.getenv("LANGCHAIN_API_KEY", "")

from graph.workflow import graph

_langsmith_status = (
    f"🔭 LangSmith tracing ON  → project: {os.getenv('LANGCHAIN_PROJECT', 'data-analyst-agent')}"
    if _tracing_enabled and _langsmith_key
    else "⚪ LangSmith tracing OFF (set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY)"
)

BANNER = f"""
╔══════════════════════════════════════════════════════╗
║          Data Analyst Agent  —  LangGraph            ║
║   Agents: RAG · Statistical Analyst · Code REPL      ║
╚══════════════════════════════════════════════════════╝
{_langsmith_status}

Type your question about the sales dataset.
Commands: 'help' for examples  |  'exit' to quit
"""

EXAMPLES = """
Example queries:
  📚 RAG    → "What products does the company sell?"
              "Tell me about the sales channels we use"
  📊 Analyst → "Show me the monthly revenue trend"
               "Which vendor had the best performance in 2024?"
               "Compare revenue by region"
               "Top 5 products by revenue"
  💻 Code   → "Calculate average ticket per region"
               "Filter sales with discount greater than 15%"
               "Which category grew the most between 2023 and 2024?"
"""


def main():
    print(BANNER)
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not found. Copy .env.example to .env and add your key.\n")
        return

    while True:
        try:
            query = input("\n🔍 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        if query.lower() == "help":
            print(EXAMPLES)
            continue

        print("\n⏳ Processing...\n")
        try:
            result = graph.invoke({"query": query})
            print(result.get("final_answer", "No answer generated."))
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

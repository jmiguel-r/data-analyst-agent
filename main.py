"""
main.py
Interactive CLI for the Data Analyst Agent.
"""
import os
from dotenv import load_dotenv
from graph.workflow import graph

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════╗
║          Data Analyst Agent  —  LangGraph            ║
║   Agents: RAG · Statistical Analyst · Code REPL      ║
╚══════════════════════════════════════════════════════╝
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
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not found. Copy .env.example to .env and add your key.\n")
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

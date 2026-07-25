"""
agents/analyst_agent.py
Performs statistical analysis on sales.csv using Pandas.
Generates charts with Matplotlib when relevant.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
from datetime import datetime
from graph.state import AgentState

DATA_PATH   = Path(__file__).parent.parent / "data" / "sales.csv"
OUTPUT_DIR  = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

ANALYSIS_PROMPT = """You are a senior data analyst. Given a sales DataFrame summary and a user question,
produce a clear, structured analysis. Include:
1. Direct answer to the question
2. Key numbers/statistics
3. Any notable trends or insights
4. Recommendation if applicable

Be concise and professional."""


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b %Y")
    return df


def _build_summary(df: pd.DataFrame, query: str) -> str:
    lines = [
        f"Total records: {len(df)}",
        f"Date range: {df['date'].min().date()} → {df['date'].max().date()}",
        f"Total revenue: ${df['revenue'].sum():,.2f}",
        f"Avg unit price: ${df['unit_price'].mean():.2f}",
        "",
        "Revenue by category:",
        df.groupby("category")["revenue"].sum().sort_values(ascending=False).to_string(),
        "",
        "Revenue by vendor:",
        df.groupby("vendor")["revenue"].sum().sort_values(ascending=False).to_string(),
        "",
        "Revenue by region:",
        df.groupby("region")["revenue"].sum().sort_values(ascending=False).to_string(),
        "",
        "Revenue by channel:",
        df.groupby("channel")["revenue"].sum().sort_values(ascending=False).to_string(),
        "",
        "Monthly revenue (last 6 months):",
        df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().tail(6).to_string(),
    ]
    return "\n".join(lines)


def _maybe_plot(df: pd.DataFrame, query: str) -> str | None:
    q = query.lower()
    chart_path = None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    if any(w in q for w in ["trend", "monthly", "over time", "month"]):
        monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().reset_index()
        monthly["date"] = monthly["date"].dt.to_timestamp()
        ax.plot(monthly["date"], monthly["revenue"], marker="o", color="#2563EB", linewidth=2)
        ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Revenue ($)")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        chart_path = str(OUTPUT_DIR / f"trend_{datetime.now().strftime('%H%M%S')}.png")
        fig.savefig(chart_path, dpi=150)

    elif any(w in q for w in ["category", "categor", "product type"]):
        cat = df.groupby("category")["revenue"].sum().sort_values(ascending=True)
        ax.barh(cat.index, cat.values, color="#2563EB")
        ax.set_title("Revenue by Category", fontsize=14, fontweight="bold")
        ax.set_xlabel("Revenue ($)"); plt.tight_layout()
        chart_path = str(OUTPUT_DIR / f"category_{datetime.now().strftime('%H%M%S')}.png")
        fig.savefig(chart_path, dpi=150)

    elif any(w in q for w in ["vendor", "seller", "salesperson"]):
        v = df.groupby("vendor")["revenue"].sum().sort_values(ascending=True)
        ax.barh(v.index, v.values, color="#16A34A")
        ax.set_title("Revenue by Vendor", fontsize=14, fontweight="bold")
        ax.set_xlabel("Revenue ($)"); plt.tight_layout()
        chart_path = str(OUTPUT_DIR / f"vendor_{datetime.now().strftime('%H%M%S')}.png")
        fig.savefig(chart_path, dpi=150)

    elif any(w in q for w in ["region"]):
        r = df.groupby("region")["revenue"].sum().sort_values(ascending=True)
        ax.barh(r.index, r.values, color="#9333EA")
        ax.set_title("Revenue by Region", fontsize=14, fontweight="bold")
        ax.set_xlabel("Revenue ($)"); plt.tight_layout()
        chart_path = str(OUTPUT_DIR / f"region_{datetime.now().strftime('%H%M%S')}.png")
        fig.savefig(chart_path, dpi=150)

    else:
        plt.close(fig)
        return None

    plt.close(fig)
    return chart_path


def analyst_agent(state: AgentState) -> AgentState:
    query = state["query"]
    df    = _load_df()

    summary    = _build_summary(df, query)
    chart_path = _maybe_plot(df, query)

    messages = [
        SystemMessage(content=ANALYSIS_PROMPT),
        HumanMessage(content=f"Data summary:\n{summary}\n\nUser question: {query}"),
    ]
    response = _llm.invoke(messages)

    final = response.content
    if chart_path:
        final += f"\n\n📊 Chart saved to: {chart_path}"

    return {
        **state,
        "analyst_insight":  summary,
        "chart_path":       chart_path,
        "analyst_response": response.content,
        "agent_used":       "analyst_agent",
        "final_answer":     final,
    }

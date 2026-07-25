# Data Analyst Agent

A production-grade multi-agent system built with **LangGraph** that routes natural language queries about sales data to specialized agents: RAG, Statistical Analyst, and Code REPL.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Supervisor    │  ← LangGraph StateGraph
│   (Router)      │    classifies intent
└────────┬────────┘
         │
    ┌────┴─────┐──────────────┐
    ▼          ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│  RAG   │ │ Analyst  │ │  Code    │
│ Agent  │ │  Agent   │ │  Agent   │
└────────┘ └──────────┘ └──────────┘
    │          │              │
ChromaDB   Pandas +      Sandboxed
(semantic   Matplotlib    Python REPL
 search)    (charts)      (exec)
    │          │              │
    └────────┬─┘──────────────┘
             ▼
      ┌─────────────┐
      │ Consolidate │ → Final Answer
      └─────────────┘
```

## Agents

| Agent | Trigger | Tools |
|-------|---------|-------|
| **RAG Agent** | Conceptual questions, summaries | ChromaDB + sentence-transformers |
| **Analyst Agent** | Statistics, trends, rankings, charts | Pandas, Matplotlib, Seaborn |
| **Code Agent** | Complex filtering, custom calculations | Sandboxed Python REPL |

## Tech Stack

- **Orchestration:** LangGraph 0.2 (StateGraph)
- **LLM:** Gemini 2.5 Flash via `langchain-google-genai`
- **Observability:** LangSmith — full trace of every LangGraph run (supervisor → agent → consolidate)
- **Vector Store:** ChromaDB with `all-MiniLM-L6-v2` embeddings
- **Data:** Pandas, Matplotlib, Seaborn
- **Containerization:** Docker + docker-compose

## Observability with LangSmith

Every LangGraph run is automatically traced to [LangSmith](https://smith.langchain.com) when the following env vars are set:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=data-analyst-agent
```

Each trace captures the full execution tree:

```
LangGraph
  └── supervisor      → intent classification (Gemini)
       └── router     → routes to the correct agent
  └── rag_agent / analyst_agent / code_agent
       └── ChatGoogleGenerativeAI (gemini-2.5-flash)
  └── consolidate     → formats final answer
```

LangSmith records latency, token usage, cost per run, and input/output at every node — making it easy to spot slow agents, prompt regressions, or routing errors.

No code changes are needed to enable tracing — LangChain instruments LangGraph automatically.

## Project Structure

```
data-analyst-agent/
├── agents/
│   ├── supervisor.py       # Router — classifies user intent
│   ├── rag_agent.py        # Semantic search over ChromaDB
│   ├── analyst_agent.py    # Pandas stats + Matplotlib charts
│   └── code_agent.py       # Sandboxed Python code execution
├── graph/
│   ├── state.py            # Shared AgentState TypedDict
│   └── workflow.py         # LangGraph StateGraph definition
├── vectorstore/
│   └── ingest.py           # Indexes sales data into ChromaDB
├── data/
│   ├── generate_data.py    # Generates 2,000-row synthetic dataset
│   └── sales.csv           # Generated dataset (gitignored)
├── outputs/                # Charts saved here (gitignored)
├── main.py                 # Interactive CLI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/jmiguel-r/data-analyst-agent.git
cd data-analyst-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY and optionally LANGCHAIN_API_KEY to .env
# See .env.example for full list of variables
```

### 3. Generate data and index

```bash
python data/generate_data.py
python vectorstore/ingest.py
```

### 4. Run

```bash
python main.py
```

### Docker

```bash
docker compose --profile setup up   # generate data + index
docker compose up agent             # run the agent
```

## Example Queries

```
📚 RAG Agent
  "What products does the company sell?"
  "Tell me about the sales channels we use"

📊 Analyst Agent
  "Show me the monthly revenue trend"
  "Which vendor had the best performance in 2024?"
  "Top 5 products by revenue"
  "Compare revenue by region"

💻 Code Agent
  "Calculate average ticket per region"
  "Filter sales with discount greater than 15%"
  "Which category grew the most between 2023 and 2024?"
```

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Transaction date (2023–2024) |
| category | str | Electronics, Clothing, Food & Beverage, Home & Garden |
| product | str | 20 products across 4 categories |
| vendor | str | 6 sales reps |
| region | str | North, South, East, West, Central |
| channel | str | Online, Store, Phone, Partner |
| quantity | int | Units sold |
| unit_price | float | Price after discount |
| discount | float | Discount rate (0–0.20) |
| revenue | float | Total revenue with seasonality factor |

## License

MIT

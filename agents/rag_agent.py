"""
agents/rag_agent.py
Retrieves relevant sales context from ChromaDB and answers with Claude.
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
from graph.state import AgentState

CHROMA_PATH     = Path(__file__).parent.parent / "vectorstore" / "chroma_db"
COLLECTION_NAME = "sales_knowledge"
TOP_K           = 8

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
_ef  = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=_ef)


SYSTEM_PROMPT = """You are a sales data analyst assistant.
Use ONLY the provided context to answer the question.
Be concise, factual, and cite specific data points when possible.
If the context doesn't contain enough information, say so clearly."""


def rag_agent(state: AgentState) -> AgentState:
    query = state["query"]

    # Retrieve top-k relevant documents
    collection = _get_collection()
    results    = collection.query(query_texts=[query], n_results=TOP_K)
    docs       = results["documents"][0] if results["documents"] else []
    context    = "\n".join(f"- {d}" for d in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    response = _llm.invoke(messages)

    return {
        **state,
        "context":      context,
        "rag_response": response.content,
        "agent_used":   "rag_agent",
        "final_answer": response.content,
    }

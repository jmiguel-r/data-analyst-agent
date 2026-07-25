"""
vectorstore/ingest.py
Reads sales.csv, creates text summaries per row, and indexes them into ChromaDB
using local sentence-transformers embeddings (no OpenAI key needed).
"""
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path

COLLECTION_NAME = "sales_knowledge"
DATA_PATH       = Path(__file__).parent.parent / "data" / "sales.csv"
CHROMA_PATH     = Path(__file__).parent.parent / "vectorstore" / "chroma_db"

def row_to_text(row) -> str:
    discount = "no discount" if row['discount'] == 0 else f"{int(row['discount']*100)}% discount"
    return (
        f"On {row['date']}, vendor {row['vendor']} sold {row['quantity']} unit(s) of "
        f"{row['product']} ({row['category']}) via {row['channel']} channel in the "
        f"{row['region']} region at ${row['unit_price']} each ({discount}). "
        f"Total revenue: ${row['revenue']}."
    )

def ingest():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("🔧 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Drop collection if it already exists (re-ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    texts = df.apply(row_to_text, axis=1).tolist()
    ids   = [f"sale_{i}" for i in range(len(texts))]
    metas = df[["date", "category", "product", "vendor", "region", "channel"]].to_dict("records")

    # Batch insert (ChromaDB handles up to 5,000 at once)
    batch = 500
    for start in range(0, len(texts), batch):
        end = min(start + batch, len(texts))
        collection.add(documents=texts[start:end], ids=ids[start:end], metadatas=metas[start:end])
        print(f"  ✓ Indexed rows {start}–{end}")

    print(f"✅ Ingestion complete — {len(texts)} documents in '{COLLECTION_NAME}'")

if __name__ == "__main__":
    ingest()

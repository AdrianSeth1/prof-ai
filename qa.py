"""
qa.py — RAG-based question answering for live lectures.
"""

import re
import sys
from pathlib import Path

import chromadb
import ollama

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:30b-a3b"
TOP_K = 6

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _retrieve_chunks(question: str, linked_docs: list[str]) -> str:
    try:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    except Exception:
        return ""

    if not CHROMA_DIR.exists():
        return ""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return ""

    if linked_docs:
        where = {
            "$and": [
                {"file_type": {"$ne": "lecture_audio"}},
                {"source_file": {"$in": linked_docs}},
            ]
        }
    else:
        where = {"file_type": {"$ne": "lecture_audio"}}

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas"],
        where=where,
    )
    blocks = []
    for doc, meta in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
    ):
        blocks.append(f"[{meta.get('source_file', '?')}]\n{doc}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, linked_docs: list[str]) -> str:
    """Return an answer to question using RAG over linked_docs (all material if empty)."""
    chunks = _retrieve_chunks(question, linked_docs)

    context_part = f"\n\nContext:\n{chunks}" if chunks else ""
    prompt = (
        "/no_think You are answering a professor's question during a live lecture. "
        "Be concise (2-4 sentences). Use only the provided context. "
        "If the context does not answer the question, say so."
        f"{context_part}\n\n"
        f"Question: {question}"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response["message"]["content"]
    return _THINK_RE.sub("", text).strip()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this course about?"
    print(f"Question: {q}\n")
    print("Retrieving and answering...")
    ans = answer_question(q, linked_docs=[])
    print(f"\nAnswer: {ans}")

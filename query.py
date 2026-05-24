"""
query.py — ask a question against ingested course materials.
Usage:
  python query.py "question"
  python query.py --latest "question"
  python query.py --session 2026-05-21_143000 "question"
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator

import chromadb
import ollama

from manifest import load_manifest

CHROMA_DIR = Path("chroma_db")
SESSIONS_MANIFEST = Path("sessions.json")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:30b-a3b"
TOP_K = 6

SYSTEM_PROMPT = (
    "You are a research assistant helping a professor. Answer based on the provided "
    "course materials. If the materials don't contain the answer, say so plainly rather "
    "than guessing. When you make a claim, mention which source file it came from. /no_think"
)


def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def build_context(results: dict) -> tuple[str, list[str]]:
    blocks, sources = [], []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(docs, metas):
        src = meta.get("source_file", "unknown")
        label_parts = [f"Source: {src}"]

        if meta.get("file_type") == "lecture_audio" and "timestamp_seconds" in meta:
            ts = int(meta["timestamp_seconds"])
            mm, ss = divmod(ts, 60)
            label_parts.append(f"@{mm:02d}:{ss:02d}")
        elif "page_or_slide" in meta:
            key = "slide" if meta.get("file_type") == "pptx" else "page"
            label_parts.append(f"{key} {meta['page_or_slide']}")

        blocks.append(f"[{' | '.join(label_parts)}]\n{doc}")
        if src not in sources:
            sources.append(src)

    return "\n\n---\n\n".join(blocks), sources


def query_stream(question: str, session_id: str | None = None) -> Iterator[str | dict]:
    """Yield LLM response tokens, then finally yield {"sources": [list]}.

    Raises ValueError if the collection is unavailable.
    """
    if not CHROMA_DIR.exists():
        raise ValueError("No chroma_db found. Run ingest.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise ValueError(f'Collection "{COLLECTION_NAME}" not found.')
    if collection.count() == 0:
        raise ValueError("Collection is empty. Run ingest.py first.")

    query_kwargs: dict = {
        "query_embeddings": [embed(question)],
        "n_results": min(TOP_K, collection.count()),
        "include": ["documents", "metadatas"],
    }
    if session_id:
        query_kwargs["where"] = {"session_id": {"$eq": session_id}}

    results = collection.query(**query_kwargs)
    context, sources = build_context(results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Course material excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]
    for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
        yield chunk["message"]["content"]
    yield {"sources": sources}


def resolve_session(args: argparse.Namespace) -> str | None:
    if args.session:
        return args.session
    if args.latest:
        sessions = load_manifest(SESSIONS_MANIFEST)
        if not sessions:
            print("No recorded sessions found in sessions.json.")
            sys.exit(1)
        # session IDs are YYYY-MM-DD_HHMMSS, so lexicographic max == most recent
        return max(sessions.keys())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Query course materials")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--session", default=None, metavar="SESSION_ID",
                        help="Scope retrieval to this session ID")
    parser.add_argument("--latest", action="store_true",
                        help="Scope retrieval to the most recent recorded session")
    args = parser.parse_args()

    question = " ".join(args.question)
    session_id = resolve_session(args)
    if session_id:
        print(f"Scoping to session: {session_id}")

    sources: list[str] = []
    print()
    try:
        for token in query_stream(question, session_id):
            if isinstance(token, dict):
                sources = token["sources"]
            else:
                print(token, end="", flush=True)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print("\n")
    print("Sources consulted:")
    for src in sources:
        print(f"  - {src}")


if __name__ == "__main__":
    main()

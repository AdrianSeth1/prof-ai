"""
live_gap.py — background real-time gap analysis during lecture recording.
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import chromadb
import ollama

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:30b-a3b"

LIVE_GAP_INTERVAL = 60   # seconds between analysis runs
MIN_WORDS_FOR_GAP = 50   # minimum transcript words before analysing
TOP_K = 10


def _get_chunks(transcript: str, linked_docs: list[str]) -> str:
    query_text = " ".join(transcript.split()[:1500])
    try:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)["embedding"]
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


def live_gaps_stream(transcript: str, linked_docs: list[str]) -> Iterator[str]:
    """Yield gap analysis tokens quickly using /no_think."""
    chunks = _get_chunks(transcript, linked_docs)
    if not chunks:
        yield "No course material found."
        return

    doc_scope = f"Comparing against: {', '.join(linked_docs)}\n\n" if linked_docs else ""
    prompt = (
        "/no_think You are reviewing a lecture in progress. Be concise.\n\n"
        f"{doc_scope}"
        f"LECTURE SO FAR:\n{transcript}\n\n"
        f"NOTES/SLIDES:\n{chunks}\n\n"
        "List topics from the notes/slides NOT yet covered in the lecture. "
        "Bullet points only. If everything covered so far, say so."
    )
    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        yield chunk["message"]["content"]


class LiveGapWorker:
    """Runs gap analysis every LIVE_GAP_INTERVAL seconds in a background thread."""

    def __init__(
        self,
        get_session_fn: Callable,
        on_result: Callable[[str, "datetime | None"], None],
    ):
        self._get_session = get_session_fn
        self._on_result = on_result
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.wait(LIVE_GAP_INTERVAL):
            self._run_once()

    def _run_once(self) -> None:
        session = self._get_session()
        if session is None:
            return
        if not session.linked_documents:
            self._on_result("Link a document to enable live gap analysis", None)
            return
        transcript = " ".join(seg["text"] for seg in session.segments)
        if len(transcript.split()) < MIN_WORDS_FOR_GAP:
            self._on_result("Waiting for more content...", None)
            return
        result = ""
        try:
            for token in live_gaps_stream(transcript, session.linked_documents):
                result += token
        except Exception as e:
            result = f"⚠ {e}"
        session.latest_gap_analysis = result
        self._on_result(result, datetime.now())

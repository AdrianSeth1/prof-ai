"""
live_gap.py — background real-time gap analysis during lecture recording.
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import chromadb
import ollama

from gaps import parse_gap_findings

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:14b"

LIVE_GAP_INTERVAL = 60       # seconds between analysis runs
MIN_WORDS_FOR_GAP = 50       # minimum transcript words before analysing
TOP_K = 10
MAX_TRANSCRIPT_CHARS = 4000  # cap on transcript delta sent per run


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
    """Yield gap analysis tokens for a transcript delta."""
    chunks = _get_chunks(transcript, linked_docs)
    if not chunks:
        yield "No course material found."
        return

    doc_scope = f"Comparing against: {', '.join(linked_docs)}\n\n" if linked_docs else ""
    prompt = (
        "You are auditing a live lecture for coverage gaps. Be a strict fact-checker.\n\n"
        "SOURCES:\n"
        "  NOTES/SLIDES      = planned curriculum\n"
        "  RECENT TRANSCRIPT = the ONLY record of what was said in this lecture portion\n\n"
        "COVERAGE RULES:\n"
        '  "covered"   — TRANSCRIPT explicitly discusses the topic. Quote the exact words.\n'
        '  "partial"   — TRANSCRIPT mentions it but incompletely. Quote what was said.\n'
        '  "uncovered" — No evidence in TRANSCRIPT. Leave evidence as empty string.\n'
        "  When in doubt, choose uncovered. Do NOT infer coverage from the slides.\n\n"
        f"{doc_scope}"
        f"RECENT TRANSCRIPT:\n{transcript}\n\n"
        f"NOTES/SLIDES:\n{chunks}\n\n"
        "Return a JSON array (no other text). Each item must have exactly these keys:\n"
        '  "topic"    — short topic name\n'
        '  "status"   — "covered", "partial", or "uncovered"\n'
        '  "note"     — one sentence on coverage or gap\n'
        '  "evidence" — verbatim TRANSCRIPT quote for covered/partial; empty string "" for uncovered\n'
        '  "source"   — filename from the NOTES/SLIDES chunk header'
    )
    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        think=False,
        keep_alive="10m",
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
        self._last_segment_count: int = 0

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

        # Skip run if no new segments have arrived since the last analysis
        if len(session.segments) == self._last_segment_count:
            return

        full_words = sum(len(seg["text"].split()) for seg in session.segments)
        if full_words < MIN_WORDS_FOR_GAP:
            self._on_result("Waiting for more content...", None)
            return

        # Send only the delta since the last run, capped at MAX_TRANSCRIPT_CHARS
        new_segs = session.segments[self._last_segment_count:]
        delta = " ".join(seg["text"] for seg in new_segs)
        if len(delta) > MAX_TRANSCRIPT_CHARS:
            delta = delta[-MAX_TRANSCRIPT_CHARS:]
        self._last_segment_count = len(session.segments)

        full_tx = " ".join(seg.get("text", "") for seg in session.segments)
        result = ""
        t0 = time.time()
        try:
            for token in live_gaps_stream(delta, session.linked_documents):
                result += token
        except Exception as e:
            result = f"⚠ {e}"
        print(f"[GAP-LIVE] llm {time.time() - t0:.1f}s", flush=True)
        session.latest_gap_analysis = result
        findings = parse_gap_findings(result, transcript=full_tx)
        self._on_result(findings if findings is not None else result, datetime.now())

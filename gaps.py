"""
gaps.py — find what planned notes/slides weren't covered in a lecture.
Usage:
  python gaps.py --session SESSION_ID
  python gaps.py --latest
  python gaps.py --date YYYY-MM-DD
  python gaps.py --latest --show-thinking
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import chromadb
import ollama

from manifest import load_manifest

CHROMA_DIR = Path("chroma_db")
TRANSCRIPTS_DIR = Path("transcripts") / "sessions"
SESSIONS_MANIFEST = Path("sessions.json")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:14b"
TOP_K = 15
# Truncate transcript for the query embedding — very long text dilutes topical signal
EMBED_MAX_WORDS = 1500

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------

def resolve_session_id(
    session_id: str | None = None,
    latest: bool = False,
    date: str | None = None,
) -> str:
    """Importable resolver. Raises ValueError on bad input."""
    sessions = load_manifest(SESSIONS_MANIFEST)
    if not sessions:
        raise ValueError("No recorded sessions found in sessions.json.")
    if session_id:
        if session_id not in sessions:
            raise ValueError(f"Session '{session_id}' not found.")
        return session_id
    if latest:
        return max(sessions.keys())
    if date:
        matching = [sid for sid in sessions if sid.startswith(date)]
        if not matching:
            raise ValueError(f"No session found for date {date}.")
        return max(matching)
    raise ValueError("Specify session_id, latest=True, or date.")


def resolve_session(args: argparse.Namespace) -> str:
    sessions = load_manifest(SESSIONS_MANIFEST)
    if not sessions:
        print("No recorded sessions found in sessions.json.")
        sys.exit(1)

    if args.session:
        if args.session not in sessions:
            print(f"Session '{args.session}' not found in sessions.json.")
            sys.exit(1)
        return args.session

    if args.latest:
        return max(sessions.keys())  # YYYY-MM-DD_HHMMSS sorts lexicographically

    if args.date:
        matching = [sid for sid in sessions if sid.startswith(args.date)]
        if not matching:
            print(f"No session found for date {args.date}.")
            sys.exit(1)
        return max(matching)

    print("Specify --session, --latest, or --date.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}\]")


def load_transcript(session_id: str) -> str:
    path = TRANSCRIPTS_DIR / f"{session_id}.txt"
    if not path.exists():
        print(f"Transcript not found: {path}")
        sys.exit(1)

    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("--- session ended"):
            break
        # Only start collecting once we see the first [MM:SS] segment line
        if TIMESTAMP_RE.match(line) or body:
            body.append(line)

    return "\n".join(body).strip()


# ---------------------------------------------------------------------------
# ChromaDB retrieval
# ---------------------------------------------------------------------------

def build_chunks_text(results: dict) -> str:
    blocks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        src = meta.get("source_file", "unknown")
        label_parts = [f"Source: {src}"]
        if "page_or_slide" in meta:
            key = "slide" if meta.get("file_type") == "pptx" else "page"
            label_parts.append(f"{key} {meta['page_or_slide']}")
        blocks.append(f"[{' | '.join(label_parts)}]\n{doc}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Streaming with <think> suppression
#
# Qwen3 wraps its reasoning in <think>...</think>. We buffer enough bytes to
# detect tags that arrive split across chunks, then emit or discard accordingly.
# ---------------------------------------------------------------------------

def _think_filter(messages: list[dict], show_thinking: bool) -> Iterator[str]:
    """Generator: yields LLM tokens with <think> blocks suppressed or kept."""
    buf = ""
    in_think = False
    for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True, think=False):
        buf += chunk["message"]["content"]
        while buf:
            if not in_think:
                idx = buf.find(THINK_OPEN)
                if idx == -1:
                    safe = max(0, len(buf) - len(THINK_OPEN) + 1)
                    yield buf[:safe]
                    buf = buf[safe:]
                    break
                yield buf[:idx]
                if show_thinking:
                    yield THINK_OPEN
                buf = buf[idx + len(THINK_OPEN):]
                in_think = True
            else:
                idx = buf.find(THINK_CLOSE)
                if idx == -1:
                    safe = max(0, len(buf) - len(THINK_CLOSE) + 1)
                    if show_thinking:
                        yield buf[:safe]
                    buf = buf[safe:]
                    break
                if show_thinking:
                    yield buf[:idx] + THINK_CLOSE
                buf = buf[idx + len(THINK_CLOSE):]
                in_think = False
    if buf and (not in_think or show_thinking):
        yield buf


def stream_response(messages: list[dict], show_thinking: bool) -> None:
    for token in _think_filter(messages, show_thinking):
        print(token, end="", flush=True)


def gaps_stream(session_id: str, show_thinking: bool = False) -> Iterator[str]:
    """Importable: yield gap analysis tokens for the given session."""
    # Read linked_documents from sessions.json (backward-compat: old entries → [])
    entry = load_manifest(SESSIONS_MANIFEST).get(session_id, {})
    linked_docs: list[str] = entry.get("linked_documents", [])

    transcript = load_transcript(session_id)
    if not transcript:
        yield "Transcript is empty — record a lecture first."
        return

    query_text = " ".join(transcript.split()[:EMBED_MAX_WORDS])
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)["embedding"]

    if not CHROMA_DIR.exists():
        yield "No chroma_db found. Run ingest.py first."
        return
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        yield 'Collection "course_material" not found. Run ingest.py first.'
        return

    # When linked_documents is set, restrict retrieval to those specific files;
    # otherwise search all non-audio material.
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
    chunks_text = build_chunks_text(results)
    if not chunks_text:
        scope = ", ".join(linked_docs) if linked_docs else "all course material"
        yield f"No notes or slides found in: {scope}. Run ingest.py to add documents."
        return

    doc_scope = (
        f"Comparing specifically against: {', '.join(linked_docs)}\n\n"
        if linked_docs else ""
    )
    prompt = (
        "Below is a lecture transcript and the corresponding planned "
        f"notes and slides for the same topic.\n\n{doc_scope}"
        f"LECTURE TRANSCRIPT:\n{transcript}\n\n"
        f"PLANNED NOTES AND SLIDES:\n{chunks_text}\n\n"
        "Compare them carefully. Identify specific topics, concepts, "
        "definitions, or examples that appear in the planned notes/slides "
        "but were NOT covered or were only briefly mentioned in the lecture. "
        "Be specific. Quote the relevant note/slide text. If everything in "
        "the notes was covered, say so plainly.\n\n"
        "Format your response as a bulleted list of gaps, each with the "
        "source file it came from."
    )
    t0 = time.time()
    for token in _think_filter([{"role": "user", "content": prompt}], show_thinking):
        yield token
    print(f"[GAP] llm {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify topics in notes/slides not covered in the lecture"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", metavar="SESSION_ID", help="Specific session ID")
    group.add_argument("--latest", action="store_true", help="Most recent session")
    group.add_argument("--date", metavar="YYYY-MM-DD", help="Most recent session on this date")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Print Qwen3's reasoning trace (inside <think> tags)")
    args = parser.parse_args()

    session_id = resolve_session(args)
    print(f"Analyzing session: {session_id}\n")
    for token in gaps_stream(session_id, args.show_thinking):
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()

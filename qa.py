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
TRANSCRIPTS_DIR = Path("transcripts") / "sessions"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:30b-a3b"
TOP_K = 6

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _retrieve_chunks(question: str, linked_docs: list[str]) -> tuple[str, list[str]]:
    """Return (chunks_text, source_files)."""
    try:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    except Exception:
        return "", []

    if not CHROMA_DIR.exists():
        return "", []
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return "", []

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
    source_files: list[str] = []
    for doc, meta in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
    ):
        src = meta.get("source_file", "?")
        blocks.append(f"[{src}]\n{doc}")
        if src not in source_files:
            source_files.append(src)
    return "\n\n---\n\n".join(blocks), source_files


def _load_transcript(session) -> str:
    """Load the full transcript from file, falling back to in-memory segments."""
    if session is None:
        return ""
    transcript_path = TRANSCRIPTS_DIR / f"{session.session_id}.txt"
    if transcript_path.exists():
        try:
            return transcript_path.read_text(encoding="utf-8")
        except Exception:
            pass
    if session.segments:
        lines = []
        for seg in session.segments:
            mm, ss = divmod(int(seg["start_seconds"]), 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {seg['text']}")
        return "\n".join(lines)
    return ""


def answer_question(question: str, linked_docs: list[str], session=None) -> tuple[str, list[str]]:
    """Return (answer, source_files) using the full transcript and linked doc chunks."""
    transcript_text = _load_transcript(session)

    if linked_docs:
        doc_context, source_files = _retrieve_chunks(question, linked_docs)
    else:
        doc_context, source_files = "", []

    print(
        f"[Q&A] context sizes — doc:{len(doc_context)} transcript:{len(transcript_text)}",
        flush=True,
    )

    if linked_docs and doc_context:
        prompt = f"""/think You are a teaching assistant helping a professor during a live lecture.

LECTURE TRANSCRIPT (what was actually said so far):
{transcript_text if transcript_text else "(no transcript yet)"}

SOURCE MATERIAL (planned content from slides/notes):
{doc_context}

The professor just asked: "{question}"

If the question asks about what was missed, covered, forgotten, or still left to discuss, identify specific topics, concepts, or examples that appear in the SOURCE MATERIAL but NOT in the LECTURE TRANSCRIPT. Be specific. Reference the source material by name.

If the question is something else, answer using both the transcript and source material as context.

Keep responses concise (will be spoken via TTS) but specific and actionable."""
    else:
        prompt = f"""/think You are a teaching assistant helping a professor during a live lecture.

No source material is linked to this lecture session. Answer based only on the transcript.

LECTURE TRANSCRIPT (what was actually said so far):
{transcript_text if transcript_text else "(no transcript yet)"}

The professor just asked: "{question}"

Keep responses concise (will be spoken via TTS) but specific and actionable."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = _THINK_RE.sub("", response["message"]["content"]).strip()
    return answer, source_files


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this course about?"
    print(f"Question: {q}\n")
    print("Retrieving and answering...")
    ans, sources = answer_question(q, linked_docs=[])
    print(f"\nAnswer: {ans}")
    if sources:
        print(f"Sources: {', '.join(sources)}")

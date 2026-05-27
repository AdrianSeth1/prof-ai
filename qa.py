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


def _format_qa_history(session, max_turns: int = 2) -> str:
    """Return the last N Q&A turns as a formatted block, or empty string."""
    if session is None or not session.qa_history:
        return ""
    recent = session.qa_history[-max_turns:]
    lines = []
    for entry in recent:
        lines.append(f'Q: {entry["question"]}')
        lines.append(f'A: {entry["answer"]}')
    return "\n".join(lines)


def answer_question(question: str, linked_docs: list[str], session=None) -> tuple[str, list[str]]:
    """Return (answer, source_files) using the full transcript and linked doc chunks."""
    transcript_text = _load_transcript(session)
    qa_history_text = _format_qa_history(session)

    if linked_docs:
        doc_context, source_files = _retrieve_chunks(question, linked_docs)
    else:
        doc_context, source_files = "", []

    print(
        f"[Q&A] context sizes — doc:{len(doc_context)} transcript:{len(transcript_text)} history:{len(qa_history_text)}",
        flush=True,
    )

    history_block = f"""
RECENT Q&A (this conversation so far):
{qa_history_text}
""" if qa_history_text else ""

    if linked_docs and doc_context:
        prompt = f"""/think You are a teaching assistant helping a professor during a live lecture.

LECTURE TRANSCRIPT (what was actually said so far):
{transcript_text if transcript_text else "(no transcript yet)"}

SOURCE MATERIAL (planned content from slides/notes):
{doc_context}
{history_block}
The professor just asked: "{question}"

If this is a follow-up to something in the recent Q&A above (e.g. "tell me more about X" or "expand on that"), go deeper on that specific idea using the source material.

If the question asks about what was missed, covered, forgotten, or still left to discuss, identify specific topics, concepts, or examples that appear in the SOURCE MATERIAL but NOT in the LECTURE TRANSCRIPT.

Respond conversationally, as if you're a knowledgeable colleague speaking aloud — not reading a list. Group related ideas into a theme or two, give a sentence of context for why each matters, and close with a clear takeaway. Be warm, curious, and direct. No bullet points, no headers, no colons introducing lists. Two to four sentences total."""
    else:
        prompt = f"""/think You are a teaching assistant helping a professor during a live lecture.

No source material is linked to this lecture session. Answer based only on the transcript.

LECTURE TRANSCRIPT (what was actually said so far):
{transcript_text if transcript_text else "(no transcript yet)"}
{history_block}
The professor just asked: "{question}"

If this is a follow-up to something in the recent Q&A above (e.g. "tell me more about X"), go deeper on that specific idea.

Respond conversationally, as if you're a knowledgeable colleague speaking aloud — not reading a list. Be warm, curious, and direct. No bullet points, no headers, no colons introducing lists. Two to four sentences total."""

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

"""
qa.py — RAG-based question answering for live lectures.
"""

import re
import sys
import time
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


def sanitize_answer(text: str) -> str:
    """Strip inlined reasoning from the content field.

    Safety net for clients that inline the thinking trace into content
    instead of returning it in response.message.thinking. Applied after
    the dedicated thinking field is already extracted.
    If a closing </think> tag is present, keep only what follows the last one.
    Then strip any complete <think>...</think> block that might remain.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = _THINK_RE.sub("", text)
    return text.strip()


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


def _format_conversation_history(session, max_turns: int = 10) -> str:
    """Return the last N turns from session.conversation_history as Q/A lines."""
    hist = getattr(session, "conversation_history", None)
    if not hist:
        return ""
    lines = []
    for entry in hist[-max_turns:]:
        lines.append(f'Q: {entry["question"]}')
        lines.append(f'A: {entry["answer"]}')
    return "\n".join(lines)


def answer_question(question: str, linked_docs: list[str], session=None) -> tuple[str, str, list[str]]:
    """Return (answer, reasoning, source_files) using the full transcript and linked doc chunks."""
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
        prompt = f"""You are a teaching assistant feeding a professor useful information during a live lecture.

LECTURE TRANSCRIPT (what was actually said so far):
{transcript_text if transcript_text else "(no transcript yet)"}

SOURCE MATERIAL (planned content from slides/notes):
{doc_context}
{history_block}
The professor just asked: "{question}"

Decide which mode applies:

GAP MODE — only if the question is explicitly about what's missing, uncovered, or left to discuss (e.g. "what am I missing", "did I cover everything", "what's left", "what haven't I talked about"): identify specific topics in the SOURCE MATERIAL that do not appear in the LECTURE TRANSCRIPT.

CONTENT MODE — for every other question, including "can you explain", "tell me more", "what about X", "how does Y work", and any follow-up: draw primarily from SOURCE MATERIAL and RECENT Q&A. Use the transcript only as background awareness of what's been said. Do not use the gap analysis to drive the answer — it is background context only.

When the professor asks you to explain something or tell them more: provide additional substantive information from the source material that has not already been said in the recent conversation. Give the professor new on-topic content they can say to the class right now.

Do not narrate what the professor should do next. Do not say "you were about to explain", "let's circle back", or "before we dive into". Answer the question directly with information the professor can use in front of the class right now.

Avoid: "circle back", "dive into", "connect the dots", "let's", "going forward". Speak as a knowledgeable colleague.

Respond conversationally — not reading a list. Group related ideas into a theme or two, give a sentence of context per theme, and close with a clear takeaway. Be warm, curious, and direct. No bullet points, no headers, no colons introducing lists. Two to four sentences total."""
    else:
        prompt = f"""You are a teaching assistant feeding a professor useful information during a live lecture.

No source material is linked to this lecture session. Answer based only on the transcript and recent conversation.

LECTURE TRANSCRIPT (what was actually said so far):
{transcript_text if transcript_text else "(no transcript yet)"}
{history_block}
The professor just asked: "{question}"

If the question is a follow-up ("can you explain", "tell me more", "what about X"), provide additional substantive information that has not already been said in the recent conversation. Give the professor new on-topic content they can say to the class right now.

Do not narrate what the professor should do next. Do not say "you were about to explain", "let's circle back", or "before we dive into". Answer directly.

Avoid: "circle back", "dive into", "connect the dots", "let's", "going forward". Speak as a knowledgeable colleague.

Respond conversationally — not reading a list. Be warm, curious, and direct. No bullet points, no headers, no colons introducing lists. Two to four sentences total."""

    t0 = time.time()
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=True,
        keep_alive="30m",
    )
    print(f"[Q&A] llm {time.time() - t0:.1f}s", flush=True)

    # Read answer and reasoning as separate fields.
    # sanitize_answer is a safety net for clients that inline the trace into content.
    answer = sanitize_answer(response["message"]["content"])
    reasoning = response["message"].get("thinking") or ""
    if reasoning:
        print(f"[Q&A] reasoning {len(reasoning)} chars", flush=True)
    return answer, reasoning, source_files


def answer_conversation_turn(
    question: str,
    linked_docs: list[str],
    session=None,
    max_history_turns: int = 10,
    fast: bool = False,
) -> tuple[str, str, list[str]]:
    """Return (answer, reasoning, source_files) for one conversation turn.

    Uses session.conversation_history (not qa_history) so conversation context
    stays separate from single-turn Q&A context. fast=True uses think=False and
    requests a shorter spoken reply.
    """
    transcript_text = _load_transcript(session)
    history_text = _format_conversation_history(session, max_turns=max_history_turns)

    if linked_docs:
        doc_context, source_files = _retrieve_chunks(question, linked_docs)
    else:
        doc_context, source_files = "", []

    print(
        f"[CONV] context sizes — doc:{len(doc_context)} "
        f"transcript:{len(transcript_text)} history:{len(history_text)}",
        flush=True,
    )

    history_block = f"\nRECENT CONVERSATION:\n{history_text}\n" if history_text else ""
    length_instruction = (
        "Reply in one or two sentences — spoken word style, no lists."
        if fast else
        "Give a thorough answer the professor can act on. No bullet points, no headers."
    )

    if linked_docs and doc_context:
        prompt = f"""You are a teaching assistant in a multi-turn conversation with a professor.

PRIOR LECTURE TRANSCRIPT:
{transcript_text if transcript_text else "(no transcript yet)"}

SOURCE MATERIAL (slides/notes):
{doc_context}
{history_block}
The professor says: "{question}"

{length_instruction} Be direct and conversational. Speak as a knowledgeable colleague."""
    else:
        prompt = f"""You are a teaching assistant in a multi-turn conversation with a professor.

PRIOR LECTURE TRANSCRIPT:
{transcript_text if transcript_text else "(no transcript yet)"}
{history_block}
The professor says: "{question}"

{length_instruction} Be direct and conversational. Speak as a knowledgeable colleague."""

    t0 = time.time()
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=not fast,
        keep_alive="30m",
    )
    print(f"[CONV] llm {time.time() - t0:.1f}s", flush=True)

    answer = sanitize_answer(response["message"]["content"])
    reasoning = response["message"].get("thinking") or ""
    if reasoning:
        print(f"[CONV] reasoning {len(reasoning)} chars", flush=True)
    return answer, reasoning, source_files


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this course about?"
    print(f"Question: {q}\n")
    print("Retrieving and answering...")
    ans, _reasoning, sources = answer_question(q, linked_docs=[])
    print(f"\nAnswer: {ans}")
    if sources:
        print(f"Sources: {', '.join(sources)}")

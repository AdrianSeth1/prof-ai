"""
voice_qa.py — voice Q&A during live lectures.

TTS requires the Piper binary + en_US-lessac-medium voice model.
Download from: https://github.com/rhasspy/piper/releases
Place piper.exe and en_US-lessac-medium.onnx (+ .json) in a piper/ subfolder,
or put piper.exe on PATH. TTS is silently skipped if files are not found.
"""

import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

import chromadb
import numpy as np
import ollama
import sounddevice as sd

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:30b-a3b"
TOP_K = 6

PIPER_SAMPLE_RATE = 22050

_qa_lock = threading.Lock()
_qa_entries: list[dict] = []


# ---------------------------------------------------------------------------
# Q&A history
# ---------------------------------------------------------------------------

def reset_qa_history() -> None:
    with _qa_lock:
        _qa_entries.clear()


def add_qa_entry(question: str, answer: str) -> None:
    with _qa_lock:
        _qa_entries.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })


def format_qa_log() -> str:
    with _qa_lock:
        entries = list(_qa_entries)
    if not entries:
        return ""
    lines = []
    for e in entries:
        ts = datetime.fromisoformat(e["timestamp"]).strftime("%H:%M:%S")
        q = e["question"]
        a = e["answer"]
        lines.append(f"[{ts}] Q: {q}\n       A: {a}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# TTS (Piper)
# ---------------------------------------------------------------------------

def _find_piper() -> str | None:
    local = Path("piper") / "piper.exe"
    if local.exists():
        return str(local)
    return shutil.which("piper")


def _find_piper_model() -> Path | None:
    candidates = [
        Path("piper") / "en_US-lessac-medium.onnx",
        Path("en_US-lessac-medium.onnx"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def speak_text(text: str) -> None:
    """Synthesize text with Piper and play via sounddevice. No-op if unavailable."""
    binary = _find_piper()
    model = _find_piper_model()
    if not binary or not model:
        return
    try:
        proc = subprocess.Popen(
            [binary, "--model", str(model), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw, _ = proc.communicate(input=text.encode("utf-8"))
        if raw:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, samplerate=PIPER_SAMPLE_RATE, blocking=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

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


def answer_question_stream(
    question: str,
    transcript: str,
    linked_docs: list[str],
) -> Iterator[str]:
    """Stream an answer to a question using current transcript + linked docs."""
    chunks = _retrieve_chunks(question, linked_docs)
    parts = []
    if transcript.strip():
        parts.append(f"LECTURE TRANSCRIPT SO FAR:\n{transcript}")
    if chunks:
        parts.append(f"COURSE MATERIALS:\n{chunks}")
    context = "\n\n".join(parts)

    prompt = (
        "/no_think You are a professor's AI assistant answering a question "
        "about the current lecture. Be concise and accurate.\n\n"
        + (f"{context}\n\n" if context else "")
        + f"QUESTION: {question}\n\nAnswer:"
    )
    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        yield chunk["message"]["content"]

"""
batch_transcribe.py — transcribe a pre-recorded audio file into ChromaDB.
Reuses chunk_text and get_collection from ingest.py for consistency.
"""

from pathlib import Path
from typing import Iterator

import ollama
from faster_whisper import WhisperModel

from ingest import chunk_text, get_collection

EMBED_MODEL = "nomic-embed-text"

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cuda", compute_type="float16")
    return _model


def transcribe_file(path: Path) -> Iterator[str]:
    """Transcribe an audio file and store ~500-word chunks in ChromaDB.

    Yields progress strings. Shares the same collection and chunk size as
    ingest.py so batch audio and live lectures sit alongside PDF/PPTX chunks.
    """
    yield f"Transcribing {path.name}…"
    try:
        segments_gen, _ = _get_model().transcribe(str(path), language="en", beam_size=5)
        texts = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
    except Exception as e:
        yield f"  ERROR: {e}"
        return

    if not texts:
        yield f"  No speech detected in {path.name}"
        return

    chunks = chunk_text(" ".join(texts))
    collection = get_collection()

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        ids.append(f"{path.stem}_audio_{i}")
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append({
            "source_file": path.name,
            "chunk_index": i,
            "file_type": "audio_file",
        })

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    yield f"  Stored {len(ids)} chunk(s) from {path.name}"

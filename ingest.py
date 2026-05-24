"""
ingest.py — scan docs/, extract text, embed with nomic-embed-text, store in ChromaDB.
Run:  python ingest.py
"""

import sys
from pathlib import Path
from typing import Iterable, Iterator

import chromadb
import ollama
import pypdf
from docx import Document
from pptx import Presentation

from manifest import load_manifest, save_manifest

DOCS_DIR = Path("docs")
MANIFEST_PATH = Path("manifest.json")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"
CHUNK_WORDS = 500
OVERLAP_WORDS = 50


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> list[dict]:
    """Returns list of {text, page} dicts, one per page."""
    pages = []
    try:
        reader = pypdf.PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i + 1})
    except Exception as e:
        print(f"  WARNING: skipping corrupt PDF {path.name}: {e}", file=sys.stderr)
    return pages


def extract_docx(path: Path) -> list[dict]:
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "page": None}]


def extract_pptx(path: Path) -> list[dict]:
    prs = Presentation(str(path))
    slides = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = [shape.text for shape in slide.shapes if shape.has_text_frame]
        # include speaker notes when present
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text
            if notes_text.strip():
                parts.append(f"[Notes] {notes_text}")
        text = "\n".join(parts).strip()
        if text:
            slides.append({"text": text, "page": i})
    return slides


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".pptx": extract_pptx,
}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_file(path: Path, collection: chromadb.Collection, manifest: dict) -> int:
    """Ingest one file. Returns number of chunks stored (0 if skipped)."""
    key = str(path)
    mtime = path.stat().st_mtime

    if manifest.get(key) == mtime:
        print(f"  Skipping (unchanged): {path.name}")
        return

    suffix = path.suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        return

    print(f"  Processing: {path.name}")
    sections = extractor(path)
    if not sections:
        print(f"    No text extracted from {path.name}")
        return

    ids, embeddings, documents, metadatas = [], [], [], []
    chunk_index = 0

    for section in sections:
        for chunk in chunk_text(section["text"]):
            doc_id = f"{path.stem}_{chunk_index}"
            meta = {
                "source_file": path.name,
                "chunk_index": chunk_index,
                "file_type": suffix.lstrip("."),
            }
            # page_or_slide is meaningful for PDF and PPTX
            if section["page"] is not None:
                meta["page_or_slide"] = section["page"]

            ids.append(doc_id)
            embeddings.append(embed(chunk))
            documents.append(chunk)
            metadatas.append(meta)
            chunk_index += 1

    if ids:
        # upsert so re-ingesting a changed file replaces old chunks
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"    Stored {len(ids)} chunks")
        manifest[key] = mtime
        return len(ids)

    manifest[key] = mtime
    return 0


def ingest_paths(paths: Iterable[Path]) -> Iterator[str]:
    """Ingest an explicit list of paths. Yields progress strings."""
    collection = get_collection()
    manifest = load_manifest(MANIFEST_PATH)
    total = 0
    for path in paths:
        yield f"Processing {path.name}…"
        n = ingest_file(path, collection, manifest)
        yield f"  → {'skipped (unchanged)' if n == 0 else f'{n} chunks stored'}"
        total += n
    save_manifest(MANIFEST_PATH, manifest)
    yield f"Done. {total} new chunk(s) added (collection total: {collection.count()})."


def main() -> None:
    if not DOCS_DIR.exists():
        print(f"docs/ directory not found at {DOCS_DIR.resolve()}")
        sys.exit(1)

    files = sorted(DOCS_DIR.rglob("*"))
    supported = [p for p in files if p.suffix.lower() in EXTRACTORS and p.is_file()]
    print(f"Found {len(supported)} supported file(s) in {DOCS_DIR}/\n")
    for msg in ingest_paths(supported):
        print(msg)


if __name__ == "__main__":
    main()

"""
migrate_content_types.py — one-off script to backfill content_type metadata.

Sets content_type = "lecture_transcript" for live session segments and batch
audio transcripts; "source_document" for everything else.

Run once after pulling the version that writes content_type on ingest:
    python migrate_content_types.py
"""

from pathlib import Path

import chromadb

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "course_material"
BATCH_SIZE = 500


def infer_content_type(meta: dict) -> str:
    file_type = meta.get("file_type", "")
    source_file = meta.get("source_file", "")
    if file_type in ("lecture_audio", "audio_file") or source_file.startswith("lecture_"):
        return "lecture_transcript"
    return "source_document"


def migrate() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Collection not found: {e}")
        return

    total = collection.count()
    print(f"Collection has {total} documents. Scanning...")

    result = collection.get(include=["metadatas"])
    ids = result["ids"]
    metas = result["metadatas"]

    to_update_ids, to_update_metas = [], []
    already_set = 0
    for doc_id, meta in zip(ids, metas):
        if "content_type" in meta:
            already_set += 1
            continue
        ct = infer_content_type(meta)
        to_update_ids.append(doc_id)
        to_update_metas.append({**meta, "content_type": ct})

    if not to_update_ids:
        print(f"All {total} documents already have content_type. Nothing to do.")
        return

    counts = {"lecture_transcript": 0, "source_document": 0}
    for m in to_update_metas:
        counts[m["content_type"]] += 1

    print(f"  Will tag {counts['source_document']} as source_document, "
          f"{counts['lecture_transcript']} as lecture_transcript "
          f"({already_set} already had content_type).")

    for i in range(0, len(to_update_ids), BATCH_SIZE):
        batch_ids = to_update_ids[i:i + BATCH_SIZE]
        batch_metas = to_update_metas[i:i + BATCH_SIZE]
        collection.update(ids=batch_ids, metadatas=batch_metas)
        done = min(i + BATCH_SIZE, len(to_update_ids))
        print(f"  Updated {done}/{len(to_update_ids)}...")

    print(f"\nDone. {len(to_update_ids)} documents updated.")


if __name__ == "__main__":
    migrate()

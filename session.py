"""
session.py — LectureSession: manages transcript file, ChromaDB, and sessions.json.
"""

import threading
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import chromadb
import ollama

from manifest import load_manifest, save_manifest

TRANSCRIPTS_DIR = Path("transcripts") / "sessions"
SESSIONS_MANIFEST = Path("sessions.json")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "course_material"
EMBED_MODEL = "nomic-embed-text"


class Mode(Enum):
    LECTURE = auto()           # utterances → transcript
    AWAITING_QUESTION = auto() # next utterance → question capture
    PROCESSING = auto()        # RAG + TTS running; utterances discarded


class LectureSession:
    def __init__(self, name: str = "", linked_documents: list[str] | None = None):
        now = datetime.now()
        self.session_id: str = now.strftime("%Y-%m-%d_%H%M%S")
        self.start_time: datetime = now
        self.name: str = name
        self.linked_documents: list[str] = linked_documents or []
        self.is_active: bool = False
        self.segments: list[dict] = []
        self._transcript_path: Path | None = None
        self._collection = None

        self._mode: Mode = Mode.LECTURE
        self._mode_lock = threading.Lock()
        self.pending_question: str = ""
        self.qa_history: list[dict] = []
        self._qa_handler: Callable[["LectureSession"], None] | None = None

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def set_mode(self, mode: Mode) -> None:
        with self._mode_lock:
            self._mode = mode

    def current_mode(self) -> Mode:
        with self._mode_lock:
            return self._mode

    def register_qa_handler(self, handler: Callable[["LectureSession"], None]) -> None:
        """Register the function called (in a background thread) after a question is captured."""
        self._qa_handler = handler

    # ------------------------------------------------------------------
    # Q&A history
    # ------------------------------------------------------------------

    def add_qa_entry(self, question: str, answer: str) -> None:
        self.qa_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })

    def format_qa_log(self) -> str:
        if not self.qa_history:
            return ""
        lines = []
        for e in self.qa_history:
            ts = datetime.fromisoformat(e["timestamp"]).strftime("%H:%M:%S")
            lines.append(f"[{ts}] Q: {e['question']}\n       A: {e['answer']}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Transcript / ChromaDB
    # ------------------------------------------------------------------

    def start(self) -> None:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        self._transcript_path = TRANSCRIPTS_DIR / f"{self.session_id}.txt"
        header = (
            f"Lecture Session: {self.session_id}\n"
            f"Started: {self.start_time.isoformat()}\n"
            f"{'=' * 60}\n\n"
        )
        self._transcript_path.write_text(header, encoding="utf-8")

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.is_active = True

    def append_segment(self, text: str, start_s: float, end_s: float) -> None:
        mm, ss = divmod(int(start_s), 60)
        with self._transcript_path.open("a", encoding="utf-8") as f:
            f.write(f"[{mm:02d}:{ss:02d}] {text}\n")

        seg_index = len(self.segments)
        self.segments.append({"text": text, "start_seconds": start_s, "end_seconds": end_s})

        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
        self._collection.upsert(
            ids=[f"{self.session_id}_seg{seg_index}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "source_file": f"lecture_{self.session_id}",
                "file_type": "lecture_audio",
                "session_id": self.session_id,
                "segment_index": seg_index,
                "lecture_date": self.start_time.strftime("%Y-%m-%d"),
                "timestamp_seconds": start_s,
            }],
        )

    def stop(self) -> None:
        if self._transcript_path:
            with self._transcript_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- session ended {datetime.now().isoformat()} ---\n")

        self.is_active = False

        manifest = load_manifest(SESSIONS_MANIFEST)
        manifest[self.session_id] = {
            "session_id": self.session_id,
            "name": self.name,
            "linked_documents": self.linked_documents,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "segment_count": len(self.segments),
            "qa_history": self.qa_history,
        }
        save_manifest(SESSIONS_MANIFEST, manifest)
        print(f"Session {self.session_id} saved ({len(self.segments)} segments).")

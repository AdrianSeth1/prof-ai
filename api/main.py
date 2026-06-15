"""
FastAPI backend for Prof AI new frontend.
Phase 0: health endpoints + serve static build.
Phase 1: /api/sources and /api/chat (streaming NDJSON).
Phase 2: /api/ingest (streaming NDJSON) and /api/materials.
"""
import asyncio
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Project root on sys.path so we can import query.py / module_store.py ──
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from module_store import (                                                     # noqa: E402
    expand_selection, load_modules, _valid_doc_filenames,
    class_tree,
    create_class, rename_class, delete_class,
    create_module, rename_module, delete_module,
    assign_module_to_class, set_module_docs,
)
from query import query_stream, search_literature                               # noqa: E402
from ingest import ingest_paths, extract_pdf, OCR_MIN_CHARS, get_collection    # noqa: E402
from batch_transcribe import transcribe_file                                    # noqa: E402
from manifest import load_manifest                                              # noqa: E402
from gaps import gaps_stream, resolve_session_id as _resolve_session_id        # noqa: E402

app = FastAPI(title="Prof AI API", docs_url="/api/docs")

# Allow the Vite dev server to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking Ollama / ChromaDB / Whisper calls
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# File-type sets (lower-case suffixes with dot)
_DOC_EXTS   = {'.pdf', '.txt', '.docx', '.pptx'}
_AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.flac'}


# ── Health helpers ────────────────────────────────────────────────

async def _check_ollama() -> dict:
    """Hit the Ollama local API and report which model is loaded."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
        if r.status_code != 200:
            return {"status": "degraded", "info": f"HTTP {r.status_code}"}
        models = r.json().get("models", [])
        if not models:
            return {"status": "degraded", "info": "no models pulled"}
        qwen = next((m["name"] for m in models if "qwen3" in m["name"].lower()), None)
        name = qwen or models[0]["name"]
        short = name.split(":")[0] + " · loaded"
        return {"status": "up", "info": short}
    except Exception as exc:
        print(f"[HEALTH] Ollama check failed: {exc}", flush=True)
        return {"status": "down", "info": "unreachable"}


def _check_gpu() -> dict:
    """Query nvidia-smi for GPU name + utilisation on Windows."""
    candidates = [
        "nvidia-smi",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe",
    ]
    for exe in candidates:
        try:
            result = subprocess.run(
                [exe, "--query-gpu=name,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
            )
            if result.returncode != 0:
                continue
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            raw_name = parts[0] if parts else "GPU"
            util     = parts[1] if len(parts) > 1 else "?"
            name = raw_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            return {"status": "up", "info": f"{name} · {util}%"}
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return {"status": "degraded", "info": "TODO: nvidia-smi timed out"}
        except Exception as exc:
            print(f"[HEALTH] GPU check error: {exc}", flush=True)
            break
    return {"status": "degraded", "info": "TODO: nvidia-smi not found"}


# ── Health endpoint ───────────────────────────────────────────────

@app.get("/api/health")
async def health():
    ollama = await _check_ollama()
    gpu    = _check_gpu()
    print(
        f"[HEALTH] backend=up  ollama={ollama['status']}  gpu={gpu['status']}",
        flush=True,
    )
    return JSONResponse({
        "backend": {"status": "up",   "info": "127.0.0.1:8000"},
        "ollama":  ollama,
        "gpu":     gpu,
    })


# ── Sources endpoint (Phase 1) ────────────────────────────────────

@app.get("/api/sources")
def get_sources():
    """Return modules and ingested documents for the source-filter popover."""
    modules   = [{"id": m["id"], "name": m["name"]} for m in load_modules()]
    documents = sorted(_valid_doc_filenames())
    print(f"[SOURCES] {len(modules)} module(s), {len(documents)} document(s)", flush=True)
    return {"modules": modules, "documents": documents}


# ── Chat endpoint (Phase 1) ───────────────────────────────────────

class HistoryEntry(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    selected: list[str] = []        # module IDs and/or bare filenames
    literature: list[str] = []      # subset of ["pubmed","semantic_scholar","openalex"]
    history: list[HistoryEntry] = []


def _format_history(history: list[HistoryEntry]) -> str:
    """Render conversation history as a plain-text string for the LLM."""
    if not history:
        return ""
    parts: list[str] = []
    for h in history:
        label = "User" if h.role == "user" else "Assistant"
        parts.append(f"{label}: {h.content}")
    return "\n".join(parts)


@app.post("/api/chat")
async def chat(body: ChatRequest):
    """
    Stream an LLM answer as NDJSON lines:
      {"type":"token","text":"..."}   — one per token
      {"type":"done","source_details":[...],"literature":[...]}   — final metadata
      {"type":"error","message":"..."}   — on ValueError or unexpected failure
    """
    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def blocking_work() -> None:
        try:
            # Expand mixed module IDs + filenames → flat filename list
            doc_ids = expand_selection(body.selected) if body.selected else []
            conv_history = _format_history(body.history)

            # Parallel literature search (blocking; runs in this thread)
            lit_results: list[dict] = []
            if body.literature:
                lit_results = search_literature(
                    body.question,
                    body.literature,
                    conversation_history=conv_history,
                    selected_doc_titles=doc_ids,
                    max_results=5,
                )

            print(
                f"[CHAT] doc_ids={len(doc_ids) if doc_ids else 'all'}  "
                f"lit={len(lit_results)} results  "
                f"history={len(body.history)} turns",
                flush=True,
            )

            for chunk in query_stream(
                body.question,
                doc_ids=doc_ids or None,          # None → no filter → all source docs
                literature_results=lit_results or None,
                conversation_history=conv_history,
            ):
                if isinstance(chunk, dict):
                    line = json.dumps({"type": "done", **chunk}) + "\n"
                elif chunk:                        # skip empty strings
                    line = json.dumps({"type": "token", "text": chunk}) + "\n"
                else:
                    continue
                loop.call_soon_threadsafe(queue.put_nowait, line)

        except ValueError as exc:
            print(f"[CHAT] ValueError: {exc}", flush=True)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                json.dumps({"type": "error", "message": str(exc)}) + "\n",
            )
        except Exception as exc:
            import traceback
            print(f"[CHAT] Unexpected error:\n{traceback.format_exc()}", flush=True)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                json.dumps({"type": "error", "message": f"Internal error: {exc}"}) + "\n",
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    async def generate():
        future = loop.run_in_executor(_THREAD_POOL, blocking_work)
        while True:
            item = await queue.get()
            if item is None:
                try:
                    await future
                except Exception:
                    pass  # exceptions already turned into error JSON above
                return
            yield item

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── Module / Class board endpoints (Phase 3) ─────────────────────

class _NameBody(BaseModel):
    name: str

class _CreateModuleBody(BaseModel):
    name: str
    class_id: str | None = None

class _AssignClassBody(BaseModel):
    class_id: str | None   # null → Unassigned

class _SetDocsBody(BaseModel):
    filenames: list[str]


def _err(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@app.get("/api/modules/board")
def get_board():
    return {
        "classes": class_tree(),
        "library": sorted(_valid_doc_filenames()),
    }


@app.post("/api/classes")
def post_class(body: _NameBody):
    try:
        return create_class(body.name)
    except ValueError as exc:
        raise _err(exc)


@app.patch("/api/classes/{class_id}")
def patch_class(class_id: str, body: _NameBody):
    try:
        rename_class(class_id, body.name)
        return {"ok": True}
    except (ValueError, KeyError) as exc:
        raise _err(exc)


@app.delete("/api/classes/{class_id}")
def del_class(class_id: str):
    try:
        delete_class(class_id)
        return {"ok": True}
    except KeyError as exc:
        raise _err(exc)


@app.post("/api/modules")
def post_module(body: _CreateModuleBody):
    try:
        return create_module(body.name, body.class_id)
    except ValueError as exc:
        raise _err(exc)


@app.patch("/api/modules/{module_id}")
def patch_module(module_id: str, body: _NameBody):
    try:
        rename_module(module_id, body.name)
        return {"ok": True}
    except (ValueError, KeyError) as exc:
        raise _err(exc)


@app.delete("/api/modules/{module_id}")
def del_module(module_id: str):
    try:
        delete_module(module_id)
        return {"ok": True}
    except KeyError as exc:
        raise _err(exc)


@app.post("/api/modules/{module_id}/class")
def assign_class(module_id: str, body: _AssignClassBody):
    try:
        assign_module_to_class(module_id, body.class_id)
        return {"ok": True}
    except KeyError as exc:
        raise _err(exc)


@app.put("/api/modules/{module_id}/documents")
def set_docs(module_id: str, body: _SetDocsBody):
    try:
        set_module_docs(module_id, body.filenames)
        return {"ok": True}
    except KeyError as exc:
        raise _err(exc)


# ── Sessions endpoint (Phase 4) ──────────────────────────────────

@app.get("/api/sessions")
def get_sessions():
    """List all recorded sessions, newest-first, for the Gaps Analysis screen."""
    data = load_manifest(_PROJECT_ROOT / "sessions.json")
    sessions = []
    for sid, entry in sorted(data.items(), reverse=True):
        sessions.append({
            "id":               sid,
            "name":             entry.get("name") or "",
            "date":             sid[:10],           # YYYY-MM-DD from session_id
            "segment_count":    entry.get("segment_count", 0),
            "linked_documents": entry.get("linked_documents", []),
        })
    print(f"[SESSIONS] {len(sessions)} session(s)", flush=True)
    return {"sessions": sessions}


# ── Gaps endpoint (Phase 4) ───────────────────────────────────────

class _GapsRequest(BaseModel):
    session_id:    str | None = None
    latest:        bool       = False
    show_thinking: bool       = False


@app.post("/api/gaps")
async def run_gaps(body: _GapsRequest):
    """
    Stream gap-analysis tokens as NDJSON:
      {"type":"token","text":"..."}
      {"type":"done","session_id":"..."}
      {"type":"error","message":"..."}
    """
    try:
        session_id = _resolve_session_id(
            session_id=body.session_id or None,
            latest=body.latest,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    show_thinking = body.show_thinking
    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def blocking_work() -> None:
        had_error = False
        try:
            print(f"[GAPS] session={session_id} show_thinking={show_thinking}", flush=True)
            for token in gaps_stream(session_id, show_thinking):
                if token:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        json.dumps({"type": "token", "text": token}) + "\n",
                    )
        except SystemExit:
            # gaps.py calls sys.exit(1) when transcript file is missing;
            # in a non-main thread this only kills the thread, not the process.
            had_error = True
            loop.call_soon_threadsafe(
                queue.put_nowait,
                json.dumps({"type": "error",
                            "message": "Transcript file not found for this session."}) + "\n",
            )
        except Exception as exc:
            import traceback
            print(f"[GAPS] Error:\n{traceback.format_exc()}", flush=True)
            had_error = True
            loop.call_soon_threadsafe(
                queue.put_nowait,
                json.dumps({"type": "error", "message": str(exc)}) + "\n",
            )
        finally:
            if not had_error:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    json.dumps({"type": "done", "session_id": session_id}) + "\n",
                )
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    async def generate():
        future = loop.run_in_executor(_THREAD_POOL, blocking_work)
        while True:
            item = await queue.get()
            if item is None:
                try:
                    await future
                except Exception:
                    pass
                return
            yield item

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── Ingest endpoint (Phase 2) ─────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Strip path separators and control characters; keep extension."""
    return re.sub(r'[/\\?%*:|"<>\x00-\x1f]', '_', name)


def _parse_chunks(msg: str) -> int:
    """Extract chunk count from ingest/transcribe progress strings."""
    m = re.search(r'(\d+)\s+chunk', msg)
    return int(m.group(1)) if m else 0


def _save_if_changed(path: Path, content: bytes) -> bool:
    """
    Write content to path only if it differs from what's already there.
    Returns True if the file was written (new or changed), False if skipped.
    Preserving mtime when content is identical lets ingest_paths' manifest
    check correctly detect 'skipped (unchanged)'.
    """
    if path.exists():
        try:
            if path.read_bytes() == content:
                return False
        except OSError:
            pass
    path.write_bytes(content)
    return True


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """
    Multipart upload: save each file and run the appropriate ingestion generator.
    Streams progress as NDJSON:
      {"type":"progress","file":"...","message":"..."}
      {"type":"ocr_started","file":"...","message":"..."}
      {"type":"file_done","file":"...","chunks":N,"skipped":bool}
      {"type":"file_error","file":"...","message":"..."}
      {"type":"done","total_chunks":N}
    """
    # Read all file contents in the async context before handing off to thread
    files_data: list[dict] = []
    for f in files:
        content = await f.read()
        files_data.append({"filename": f.filename or "upload", "content": content})

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(obj: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps(obj) + "\n")

    def blocking_work() -> None:
        docs_dir    = _PROJECT_ROOT / "docs"
        lectures_dir = _PROJECT_ROOT / "lectures"
        docs_dir.mkdir(exist_ok=True)
        lectures_dir.mkdir(exist_ok=True)

        total_chunks = 0

        for file_data in files_data:
            raw_name  = file_data["filename"]
            safe_name = _safe_filename(raw_name)
            ext_lower = Path(safe_name).suffix.lower()
            content   = file_data["content"]

            if ext_lower not in _DOC_EXTS and ext_lower not in _AUDIO_EXTS:
                emit({"type": "file_error", "file": raw_name,
                      "message": f"Unsupported file type '{ext_lower}'"})
                continue

            # Determine save location
            if ext_lower in _DOC_EXTS:
                abs_save = docs_dir / safe_name
                rel_save = Path("docs") / safe_name   # relative path → matches manifest keys
            else:
                abs_save = lectures_dir / safe_name
                rel_save = None  # transcribe_file uses abs path; no manifest involved

            # Save file (preserve mtime for docs so manifest skip works)
            try:
                file_changed = _save_if_changed(abs_save, content)
            except OSError as exc:
                emit({"type": "file_error", "file": raw_name, "message": f"Could not save file: {exc}"})
                continue

            print(f"[INGEST] {raw_name!r} → {abs_save}  changed={file_changed}", flush=True)

            try:
                if ext_lower in _DOC_EXTS:
                    # OCR pre-check: only for new/changed PDFs
                    if ext_lower == '.pdf' and file_changed:
                        try:
                            pages = extract_pdf(abs_save)
                            total_chars = sum(len(p["text"]) for p in pages)
                            if total_chars < OCR_MIN_CHARS:
                                emit({"type": "ocr_started", "file": raw_name,
                                      "message": "PDF appears scanned — running OCR (tesseract)…"})
                        except Exception as exc:
                            print(f"[INGEST] OCR pre-check failed for {raw_name}: {exc}", flush=True)

                    chunks  = 0
                    skipped = False
                    for msg in ingest_paths([rel_save]):
                        emit({"type": "progress", "file": raw_name, "message": msg})
                        if "chunks stored" in msg:
                            chunks = _parse_chunks(msg)
                        elif "skipped (unchanged)" in msg:
                            skipped = True

                    emit({"type": "file_done", "file": raw_name,
                          "chunks": chunks, "skipped": skipped})
                    if not skipped:
                        total_chunks += chunks

                else:  # audio
                    chunks    = 0
                    had_error = False
                    for msg in transcribe_file(abs_save):
                        stripped = msg.strip()
                        if stripped.startswith("ERROR"):
                            had_error = True
                            emit({"type": "file_error", "file": raw_name, "message": stripped})
                        else:
                            emit({"type": "progress", "file": raw_name, "message": msg})
                            if "Stored" in msg and "chunk" in msg:
                                chunks = _parse_chunks(msg)

                    if not had_error:
                        emit({"type": "file_done", "file": raw_name,
                              "chunks": chunks, "skipped": False})
                        total_chunks += chunks

            except Exception as exc:
                import traceback
                print(f"[INGEST] Error on {raw_name}:\n{traceback.format_exc()}", flush=True)
                emit({"type": "file_error", "file": raw_name, "message": str(exc)})

        emit({"type": "done", "total_chunks": total_chunks})
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    async def generate():
        future = loop.run_in_executor(_THREAD_POOL, blocking_work)
        while True:
            item = await queue.get()
            if item is None:
                try:
                    await future
                except Exception:
                    pass
                return
            yield item

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── Materials endpoint (Phase 2) ──────────────────────────────────

def get_materials():
    """
    Return all ingested items from the vector store.
    Derives the list by scanning ChromaDB metadata — covers both documents
    (ingest.py) and audio transcripts (batch_transcribe.py), regardless of
    whether they appear in manifest.json.
    """
    try:
        collection = get_collection()
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
    except Exception as exc:
        print(f"[MATERIALS] ChromaDB error: {exc}", flush=True)
        return {"materials": []}

    # Group by source_file, counting chunks
    file_map: dict[str, dict] = {}
    for meta in metadatas:
        sf = (meta or {}).get("source_file", "")
        if not sf:
            continue
        if sf not in file_map:
            file_map[sf] = {
                "filename":     sf,
                "file_type":    (meta or {}).get("file_type", ""),
                "content_type": (meta or {}).get("content_type", "source_document"),
                "chunks":       0,
            }
        file_map[sf]["chunks"] += 1

    materials = sorted(file_map.values(), key=lambda x: x["filename"])
    print(f"[MATERIALS] {len(materials)} file(s) in vector store", flush=True)
    return {"materials": materials}


# Register as a sync route so FastAPI runs it in the thread pool
app.get("/api/materials")(get_materials)


# ── Static files (Vite build) ─────────────────────────────────────

_STATIC_DIR = Path(__file__).parent.parent / "web" / "dist"

if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    print(f"[STARTUP] Serving static build from {_STATIC_DIR}", flush=True)
else:
    print(
        f"[STARTUP] Static build not found at {_STATIC_DIR}. "
        "Run  cd web && npm run build  first, or use the Vite dev server.",
        flush=True,
    )


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )

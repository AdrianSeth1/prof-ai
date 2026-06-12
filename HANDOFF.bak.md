# Prof AI — Session Handoff

This document is written for a future AI session (Claude Code or Claude web) with zero prior context. Read it before touching any code.

---

## 1. What this system is

A local AI assistant for a **communications professor**. It records and transcribes lectures in real time, lets the prof ask spoken questions mid-lecture and get RAG-backed answers read aloud, and supports research queries against course materials plus academic literature databases. Everything runs on a single lab machine (RTX 4090, Windows 11). No cloud LLMs. The only external network calls are to PubMed, Semantic Scholar, and OpenAlex — all open APIs, no auth required.

Project root: `C:\prof-ai`

---

## 2. Architecture overview

### Major components

| File | Role |
|---|---|
| `app.py` | Gradio UI — four tabs: Chat, Live Lecture, Add Materials, Gaps Analysis |
| `live_transcribe.py` | Real-time audio → VAD → Whisper → transcript, plus Q&A capture state machine |
| `recorder.py` | Microphone capture via sounddevice + Silero VAD utterance segmentation |
| `batch_transcribe.py` | Transcribe pre-recorded audio files into ChromaDB |
| `ingest.py` | PDF/DOCX/PPTX/TXT → text extraction → chunk → embed → ChromaDB. OCR fallback for scanned PDFs via pytesseract |
| `session.py` | `LectureSession` class — mode state machine, transcript file, ChromaDB writes, sessions.json |
| `query.py` | RAG query backend — multi-source literature search, dedup, context assembly, LLM streaming |
| `qa.py` | Voice Q&A answer generation (RAG + synthesis) |
| `gaps.py` / `live_gap.py` | Gap analysis — compares running transcript against linked slide decks |
| `pubmed.py` | PubMed E-utilities wrapper |
| `semantic_scholar.py` | Semantic Scholar Graph API wrapper |
| `openalex.py` | OpenAlex API wrapper |
| `manifest.py` | Shared JSON manifest helpers (tracks ingested files) |
| `tts.py` | Piper TTS wrapper (`synthesize_wav` — not `synthesize`) |

### Data flows

**Live transcription path:**
```
USB mic (device 1, MME)
  → sounddevice InputStream (16 kHz, float32, blocksize=512)
  → _raw_q (thread-safe queue)
  → _vad_worker thread
      → Silero VADIterator, 512-sample windows
      → on speech-end event: emit utterance np.ndarray to _utterance_q
  → _transcription_loop (main background thread)
      → mode == LECTURE:
          transcribe_audio(audio, model, session.whisper_initial_prompt)
          → faster-whisper large-v3-turbo (CUDA float16)
          → segment text + avg_logprob
          → filter: MIN_WORDS=2, MIN_AVG_LOGPROB=-1.0
          → session.append_segment() → transcript file + ChromaDB
      → mode == AWAITING_QUESTION:
          4-second buffer window (see Q&A state machine below)
      → mode == PROCESSING:
          discard audio
```

**Document ingestion path:**
```
Upload (PDF/DOCX/PPTX/TXT) via Add Materials tab
  → ingest.py: extract_pdf / extract_docx / extract_pptx / extract_txt
      → if PDF extracts < 100 chars total → OCR fallback (pytesseract + pdf2image)
      → OCR chunks tagged: metadata["extraction_method"] = "ocr"
  → chunk_text() — 500-word chunks, 50-word overlap
  → ollama.embeddings("nomic-embed-text") per chunk
  → ChromaDB upsert (collection: "course_material", content_type: "source_document")
  → manifest.json updated (key: path, value: mtime — prevents re-ingestion)
```

**Whisper initial_prompt path (runs once per session start):**
```
session.linked_documents (list of filenames chosen by prof in UI)
  → for each doc: ChromaDB.get(where=source_file + content_type=source_document)
  → take first 2 chunks per doc, concatenate, cap at 4000 chars
  → qwen3:14b /no_think: "List 20-30 technical terms..."
  → returns comma-separated raw vocabulary
  → classify terms:
      multi-word AND starts-capital → proper noun ("David A. Kolb")
      everything else → concept term ("experiential learning")
  → _format_whisper_prompt(concepts, proper, max_chars=900)
      → "This is a university lecture. Topics may include: X, Y, and Z.
         Speakers may reference: A, B, and C."
      → drops whole items from end (concepts first) if over 900 chars
  → stored on session.whisper_initial_prompt
  → passed to every model.transcribe() call for the session duration
  → fallback when no docs linked: DEFAULT_INITIAL_PROMPT (prose, not a list)
```

**Q&A state machine (AWAITING_QUESTION mode):**
```
Prof clicks "Ask AI" → session.set_mode(AWAITING_QUESTION)
  → _transcription_loop detects mode change
  → [Q] buffer window opened — 4-second accumulation window starts
  → any VAD utterance during window: append to qa_buffer, reset last_speech timer
  → after 4s elapsed AND 1.5s tail silence:
      concatenate qa_buffer → transcribe as single audio clip
      → truncation guard: if < 4 words OR ends on dangling function word
          wait up to 2s for continuation utterance
          re-transcribe combined audio
      → validity check: MIN_WORDS, MIN_AVG_LOGPROB
      → session.pending_question = text; set_mode(PROCESSING)
      → fire _qa_handler thread (RAG + TTS)
  → if 4s elapsed with zero speech: timeout → return to LECTURE
```

**ChromaDB content types:**
```
content_type = "source_document"   → ingested PDFs, slides, docs (Chat tab queries)
content_type = "lecture_transcript" → live session transcripts (gap analysis reads)
```
Same collection (`course_material`), filtered by `content_type`. The `linked_documents` field on a session is a list of `source_file` values used to scope gap analysis and Q&A retrieval.

---

## 3. Current configuration — exact values

All values verified against the actual code as of this session. Where the original spec described a parameter that is **not in the code**, this is flagged explicitly.

### faster-whisper / transcription (live_transcribe.py and batch_transcribe.py)

| Parameter | Value | Reason |
|---|---|---|
| `WHISPER_MODEL` | `"large-v3-turbo"` | Upgraded from `"medium"`. Better accuracy on domain vocabulary, fast enough on 4090 (~500ms/utterance). ~1.6 GB VRAM. |
| `no_speech_threshold` | `0.7` | Raised from 0.6. Stricter about treating low-energy frames as silence rather than speech. Suppresses "Thank you" / YouTube-outro hallucinations on quiet input. |
| `log_prob_threshold` | `-0.8` | Raised from default `-1.0`. Discards segments where the model's own token confidence is below threshold. Hallucinations typically score poorly; genuine speech scores higher. Works alongside `no_speech_threshold` — the first catches silence mis-identified as speech, the second catches speech where Whisper is guessing. |
| `condition_on_previous_text` | `False` | Prevents transcription errors in one segment from biasing the next. Default is `True`, which causes error cascades on weak signal. |
| `beam_size` | `5` | Default value, not tuned. |
| `initial_prompt` | Session-generated prose sentence (see path above) | Biases Whisper toward domain vocabulary. Must be sentence-shaped, not a comma-separated list (see Failure Mode 3 below). |

**Parameters described in the briefing but NOT currently in the code — discrepancies:**

- **`temperature`**: The briefing described `temperature=0.0 with the fallback chain disabled` and listed "fabricated sentence continuations on real speech that ended mid-thought (the temperature-fallback issue)" as a failure mode addressed this session. **This is incorrect.** Neither `temperature` nor any fallback-disabling parameter appears in `transcribe_audio()` or `transcribe_file()`. faster-whisper's default temperature schedule (`[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`) is therefore active. This means Whisper will retry with higher temperatures when a segment triggers the compression ratio or log probability fallback thresholds. This is a **known open issue** — see Section 4.

- **`compression_ratio_threshold`**: Not set. faster-whisper default is `2.4`. A segment with a compression ratio above this triggers a temperature retry. Not tuned.

- **`prompt_reset_on_temperature`**: Not set. faster-whisper default behavior applies.

### VAD (recorder.py)

| Parameter | Value | Reason |
|---|---|---|
| `VAD_SILENCE_MS` | `600` | Raised from Silero default (~100ms). Longer silence required before VAD closes an utterance boundary. Produces longer, more contextually complete audio clips for Whisper, reducing acoustic confusion on short fragments (e.g. "different" → "deferred"). |
| `SAMPLE_RATE` | `16000` | Required by Silero VAD and faster-whisper. |
| `CHUNK_SIZE` | `512` | Silero VAD requires exactly 512 samples at 16 kHz. |
| `MAX_SPEECH_SECONDS` | `30` | Hard cap — force-emits utterance at 30s to keep Whisper latency bounded. |

### Q&A buffering (live_transcribe.py)

| Parameter | Value | Reason |
|---|---|---|
| `QA_BUFFER_SECONDS` | `4.0` | Minimum window after "Ask AI" before transcription fires. Allows prof to pause mid-question without triggering on a fragment. |
| `QA_TAIL_SECONDS` | `1.5` | Silence after last detected speech before buffer closes. Works in tandem with `QA_BUFFER_SECONDS` — both must be satisfied. |
| `QA_TRUNCATION_WAIT` | `2.0` | Additional wait if post-transcription truncation guard fires (transcript < 4 words, or ends on dangling function word). |

### Vocabulary prompt (live_transcribe.py)

| Parameter | Value | Reason |
|---|---|---|
| `VOCAB_MODEL` | `"qwen3:14b"` | Smaller model for fast vocabulary extraction. 30b is unnecessary for this structured task. |
| `_WHISPER_PROMPT_MAX_CHARS` | `900` | Conservative ceiling for Whisper's 224-token initial_prompt limit. Drops whole list items rather than slicing mid-word. |

---

## 4. Changes made this session and why

### Modules tab: three panes rendering empty (onerror bootstrap fix)

**Symptom:** The Modules tab showed the panel chrome (column headers, "Drag to reorder..." footer hint) but all three panes — module list, doc list, document library — were completely blank, even though `module_dd_choices()` returned modules correctly.

**Root cause:** Gradio's `gr.HTML` component sets content via Svelte's `{@html}` directive, which calls `innerHTML` internally. A hard rule in browsers: `<script>` tags inserted via `innerHTML` are parsed into the DOM but **never executed**. So the entire IIFE that calls `renderAll()` was silently ignored on every render and re-render.

**Fix:** Replaced the `<script>` tag with a three-part bootstrap:
1. `<script type="application/json" id="pma-data">` — stores the JSON payload. Non-standard `type` makes the browser ignore it as executable, but it's still in the DOM with readable `textContent`.
2. `<script type="text/x-pma" id="pma-init">` — stores the full JS logic as text. Same: ignored as code, present in DOM.
3. `<img src="x" onerror="...">` — `onerror` **does** fire from innerHTML-inserted elements. The handler is five lines: read `pma-init`'s text content, wrap it in `new Function()`, call it. `new Function()` executes in global scope, which has `document`, so it can read the data tag and find all the panel DOM nodes.

When Gradio replaces the panel HTML after a mutation (create/rename/delete module), the new `<img>` fires `onerror` again and re-initializes from fresh data. No stale state.

**Side effect fixed:** Added `s.onerror` to the dynamic SortableJS CDN load inside the init script so a blocked CDN now logs `[PMA] SortableJS CDN unavailable — drag reorder disabled` rather than silently failing.

**Files touched:** `app.py` (`_render_drag_panel`)

---

### Modules tab: stray "Textbox" label above the controls

**Symptom:** A freestanding "Textbox" label was visible above the Create / Rename controls in the Modules tab.

**Root cause:** The hidden bridge component `drag_state = gr.Textbox(visible=False, elem_id="drag_state")` was missing `show_label=False` and had no explicit `label=""`. Gradio's default label text ("Textbox") was rendering somewhere in the layout even though `visible=False` hides the input itself.

**Fix:** Added `label="", show_label=False` to the `drag_state` declaration.

**Files touched:** `app.py` (`build_ui`)

---

### Failure Mode 1: "Thank you" and YouTube-outro hallucinations

**Symptom:** Transcript contained phrases like "Thank you for watching", "I'll see you in the next video", random non-sequiturs — not spoken by the prof.

**Root cause:** Whisper treats very low energy audio (pauses, room tone) as speech segments and generates text from training data. The `no_speech_threshold=0.6` was too permissive.

**Fix:** Raised `no_speech_threshold` from `0.6` to `0.7` in both `live_transcribe.py` and `batch_transcribe.py`. Added `log_prob_threshold=-0.8` — Whisper's own token confidence on hallucinated phrases is typically low, so the threshold catches what `no_speech_threshold` doesn't.

**Files touched:** `live_transcribe.py`, `batch_transcribe.py`

---

### Failure Mode 2: Acoustic mistranscriptions on short fragments ("different" → "deferred")

**Symptom:** Short phrases were being transcribed with plausible-sounding but wrong words, especially on words with similar phonetics.

**Root cause:** The Silero VAD was segmenting too aggressively — every mid-sentence pause became an utterance boundary. Short audio clips give Whisper less acoustic context to disambiguate phonetically similar words.

**Fix:** Raised `VAD_SILENCE_MS` from the Silero default (~100ms) to `600ms` in `recorder.py`. Longer silence required before a boundary fires. This produces fewer, longer utterance clips, giving Whisper more surrounding phonetic context.

**Files touched:** `recorder.py`

---

### Failure Mode 3: initial_prompt content leaking into transcripts as echoes

**Symptom:** Transcript lines contained fragments like `"Kolb,David A. Kolb,David A. Kolb,David A."` and `"of the experiential learning process,Dialects,Dialects,Dialects"` — vocabulary from the Whisper prompt being echoed back.

**Root cause:** `build_whisper_prompt` was returning a raw comma-separated list of terms (e.g. `"David A. Kolb, factor analysis, dialectical, Carl Jung"`). Whisper treats the initial_prompt as text it can continue. On weak audio or silence, it extends the prompt rather than transcribing speech, repeating terms in the prompt's pattern.

**Fix:** Reformatted the prompt into sentence-shaped prose before passing to Whisper. The qwen3:14b vocabulary extraction step is unchanged — it still returns a comma-separated list internally. That list is then classified (multi-word + starts-capital → proper noun, else concept term) and assembled into:

```
"This is a university lecture. Topics may include: X, Y, and Z.
 Speakers may reference: A, B, and C."
```

Oxford-comma grammar throughout. If over 900 chars, whole items are dropped from the end (concepts first, then proper nouns) rather than truncating mid-word.

**Files touched:** `live_transcribe.py` (`_natural_list`, `_format_whisper_prompt`, `build_whisper_prompt`)

---

### Failure Mode 4: Fabricated sentence continuations (temperature fallback)

**Described in the session briefing as:** "fabricated sentence continuations on real speech that ended mid-thought — the temperature-fallback issue."

**Actual status: NOT FIXED.** This failure mode was described in the handoff prompt but no corresponding change was made during this session. The code does not set `temperature` or disable the fallback chain. faster-whisper's default temperature fallback schedule `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` is active. When a segment's compression ratio exceeds 2.4 or its log probability falls below the threshold, Whisper retries with progressively higher temperatures and picks the least-bad output. At higher temperatures, Whisper generates more varied (and sometimes fabricated) completions.

This is documented here as a **known open issue**. See Section 5.

---

## 5. Known open issues

### High priority

**Temperature fallback chain (not yet fixed)**
The faster-whisper default temperature schedule is active. To disable fabricated continuations when speech ends mid-thought:
- Add `temperature=0.0` to every `model.transcribe()` call in `live_transcribe.py` and `batch_transcribe.py`. This forces beam search at fixed temperature and disables the fallback retry loop.
- Optionally add `compression_ratio_threshold=2.4` explicitly to document the current default, or tighten it to `1.8` to be more aggressive about rejecting repetitive output.
- Risk: without temperature fallback, truly bad audio (clipping, dropout) may produce empty or garbled output rather than a plausible-but-wrong attempt. Monitor real-world error rates before committing.

**Mic hardware / gain staging**
The root cause of most transcription quality issues is SNR. The USB mic at device 1 (MME) is functional but uncontrolled. A lavalier or headset with proper gain staging would eliminate room noise and reduce hallucinations more effectively than any model tuning. If `no_speech_threshold=0.7` is still producing phantom utterances, the mic/gain path is the right fix before further parameter adjustments.

### Medium priority

**`BatchedInferencePipeline` for batch_transcribe.py**
`batch_transcribe.py` uses the standard `WhisperModel.transcribe()` which processes audio sequentially. For long pre-recorded files (full lecture recordings), faster-whisper's `BatchedInferencePipeline` would improve throughput by processing multiple segments in parallel on the 4090. The API is a drop-in replacement for long files. Not implemented — the current sequential path is fast enough for short clips.

**Streaming TTS**
Piper synthesizes the complete LLM response before playback begins. A sentence-streaming approach (synthesize and play sentence by sentence as the LLM generates) would cut perceived latency from ~15s to ~2s. Specced in notes.md, deliberately deferred — requires restructuring the Q&A handler.

**Browser-side mic capture**
The mic is server-side (sounddevice on the lab machine). For remote use (prof's office, lab machine in another building), audio capture must move to the browser via `gr.Audio(sources=["microphone"], streaming=True)`. Significant refactor.

**Cancel button during PROCESSING**
Clicking Cancel while TTS is playing should interrupt audio and return cleanly to LECTURE mode. Currently the mode gets stuck until TTS finishes.

### Low priority

**Whisper model bump for batch**
`batch_transcribe.py` uses `large-v3-turbo`. For very long pre-recorded files where latency doesn't matter, `large-v3` (non-turbo) may produce marginally better accuracy. Not worth the VRAM cost for the current use case.

**whisper-streaming**
A real-time streaming Whisper approach (word-level output during speech, not after utterance ends) was evaluated and deliberately deferred. The current VAD-segment architecture is simpler and the per-utterance latency on the 4090 (~200ms on `large-v3-turbo`) is already fast enough for the lecture use case. Revisit only if per-utterance latency becomes a user complaint.

---

## 6. Quickstart for the next session

### Read these files in this order

1. `CLAUDE.md` — project overview, stack, environment quirks, startup procedure
2. `notes.md` — design decisions and rationale
3. `live_transcribe.py` — the most complex file; understand the state machine and `build_whisper_prompt` before touching anything transcription-related
4. `recorder.py` — VAD layer; short and self-contained
5. `session.py` — `LectureSession` class; understand `linked_documents` and `whisper_initial_prompt` attributes
6. `ingest.py` — document pipeline; OCR fallback added recently
7. `query.py` — multi-source literature search; `search_literature()` is new

### Run this to verify the system works

```powershell
cd C:\prof-ai
.\venv\Scripts\Activate.ps1
$env:PATH = "$pwd\venv\Lib\site-packages\nvidia\cublas\bin;$pwd\venv\Lib\site-packages\nvidia\cudnn\bin;" + $env:PATH
python app.py
```

Then open `http://127.0.0.1:7860` in a browser.

On startup, the PowerShell window should show:
```
[Whisper] loading model='large-v3-turbo' device=cuda compute_type=float16
[Whisper] 'large-v3-turbo' ready
[Whisper] no_speech_threshold=0.7 log_prob_threshold=-0.8 vad_silence_ms=600
```

The third line is logged in `start_live_session()` each time a lecture session starts, not at app startup. To see it, go to Live Lecture tab, select a source document, and click Start Lecture.

### Verify literature search works

In the Chat tab, type any question, tick "Semantic Scholar (general academic)", and submit. The PowerShell window should show:
```
[Literature] reformulated query: '...'
[SemanticScholar] searching: '...' year=2024-2026
[SemanticScholar] N result(s)
[Literature] N after dedup (from N raw)
```

### Verify OCR works (if needed)

Upload a scanned PDF through Add Materials. The log should show:
```
  Processing: filename.pdf
    PDF appears to be scanned (0 chars extracted), running OCR...
  [OCR] page 1/N...
  [OCR] page 2/N...
  ...
    Stored N chunks
```

If it shows `[OCR] Tesseract is not installed at the OS level`, Tesseract was removed from PATH. It installs to `C:\Users\Sterling_Matthews1\AppData\Local\Programs\Tesseract-OCR` — add that to system PATH.

### Required Ollama models

```
ollama pull qwen3:30b-a3b
ollama pull qwen3:14b
ollama pull nomic-embed-text
```

Verify: `ollama list`

---

## 7. Parameter tuning reference

If transcription quality regresses, adjust in this order:

1. **Too many phantom utterances** → raise `NO_SPEECH_THRESHOLD` (max ~0.8 before dropping real quiet speech). File: `live_transcribe.py`, `batch_transcribe.py`.

2. **Real speech being dropped** → lower `LOG_PROB_THRESHOLD` toward `-1.0`. File: same.

3. **Short words still mistranscribed** → raise `VAD_SILENCE_MS` toward 800ms. File: `recorder.py`. Trade-off: higher latency per transcript line.

4. **Q&A firing on question fragments** → raise `QA_BUFFER_SECONDS`. Trade-off: more perceived latency after clicking Ask AI.

5. **Fabricated completions on speech that trails off** → add `temperature=0.0` to both `transcribe_audio()` in `live_transcribe.py` and `transcribe_file()` in `batch_transcribe.py`. This disables the temperature fallback chain entirely. Test with genuinely bad audio before deploying — you may lose some output that would otherwise be partially correct.

6. **Initial prompt still echoing** → check that `build_whisper_prompt` is returning a value that starts with "This is a university lecture." If it's returning a raw comma-separated string, the sentence formatting step broke. Search for `_format_whisper_prompt` in `live_transcribe.py`.

---

## 8. Module system rebuild — complete handoff

This section covers the module layer rewrite done in a separate session. Read it if you are touching anything in the Modules tab, the Chat doc filter, the Live Lecture source picker, or `module_store.py`.

### What a module is

A module is a named, **ordered** list of bare document filenames — Canvas-style. The real document bytes live once in `docs/`. A module is a reference, not a copy. The same file can appear in multiple modules. Order is meaningful and preserved. This is the central design decision — the old code destroyed order silently (see below).

---

### Old architecture — what existed before

**`modules.py`** was the CRUD layer. It had these problems:

1. **`add_docs_to_module` used set union.** The old implementation was:
   ```python
   mod["documents"] = sorted(existing | set(filenames))
   ```
   This silently alphabetized the whole document list on every add. There was no canonical ordering. The Modules tab UI showed docs in alphabetical order regardless of what the prof intended.

2. **`save_modules` used a dict wrapper.**
   ```python
   MODULES_PATH.write_text(json.dumps({"modules": modules}, indent=2))
   ```
   The `modules.json` format was `{"modules": [...]}`. `load_modules()` read `data.get("modules", [])`. This wrapper served no purpose and became a format incompatibility when the new store was introduced.

3. **No `set_module_docs`.** The only way to change a module's document list was `add_docs_to_module` (set union, alphabetized) or `remove_docs_from_module`. There was no function to replace the list wholesale with a specific order.

4. **No display helpers.** `unified_source_choices()` and `module_dd_choices()` lived in `app.py` and had to be maintained there. `unified_source_choices()` returned `[(filename, filename)]` tuples — bare filenames as both label and value, no `[Doc]` prefix, no `[Module]` entries with module IDs as values. This meant the Chat doc filter couldn't distinguish between "all files in a module" (one click) and "this individual file".

5. **Five handlers, each mutating inline.** Every mutation handler in `app.py` (create, rename, delete, save docs, and doc mutation) did this:
   ```python
   mods = load_modules()
   # inline mutation here
   save_modules(mods)
   mod_choices = module_dd_choices()     # duplicated
   src_choices = unified_source_choices()  # duplicated
   return (gr.update(...), gr.update(...), gr.update(...), ...)
   ```
   Adding a new dropdown to keep in sync meant editing every handler. Forgetting one caused stale UI state.

6. **`modules.json` format was `{"modules": [...]}`**, not a flat list.

---

### New architecture — what replaced it

**`module_store.py`** is the single source of truth. It is the only file that knows about modules. `app.py` imports from it and never reimplements anything.

**`modules.json` format is now a flat `[...]` list.** The dict wrapper is gone. `load_modules()` returns `data if isinstance(data, list) else []`. The existing `modules.json` was migrated manually when the rewrite was applied — all three pre-existing modules (IDs, names, document lists, created_at timestamps) were preserved.

**Key functions in `module_store.py`:**

| Function | What it does |
|---|---|
| `load_modules()` / `save_modules()` | JSON read/write with atomic temp-file replace |
| `create_module(name)` | Creates new module with UUID-based ID, empty doc list |
| `rename_module(id, name)` | Mutates name, saves |
| `delete_module(id)` | Removes by ID, never touches `docs/` |
| `add_docs_to_module(id, filenames)` | **Appends** to existing list, skips duplicates, **preserves existing order** |
| `remove_docs_from_module(id, filenames)` | Removes by name, preserves order of remainder |
| `set_module_docs(id, filenames)` | **Replaces wholesale** — deduplicates, preserves first-seen order. This is what the drag-drop UI calls for every mutation (add, remove, reorder). |
| `reorder_module_docs(id, ordered_filenames)` | Partial reorder — any current docs missing from the supplied list are appended at the end |
| `expand_selection(selection)` | Takes a mixed list of module IDs and bare filenames, returns a flat deduplicated filename list in first-seen order |
| `module_dd_choices()` | Choices for the module selector dropdown: `(name, id)` tuples |
| `unified_source_choices()` | Choices for Chat + Live Lecture source pickers: `[Module] name → id` entries first, then `[Doc] filename → filename` entries |

**`_valid_doc_filenames()`** is a private helper with mtime-based caching. It reads `manifest.json`, normalizes the keys (which are full relative paths like `docs\file.pdf`) to bare filenames using `PureWindowsPath`, and returns a set. The cache is invalidated when `manifest.json`'s mtime changes on disk — so a newly ingested document shows up in the module library without restarting the app.

---

### The `_module_refresh()` pattern in app.py

Every mutation handler now ends with one call:

```python
def _module_refresh():
    mod_choices = module_dd_choices()
    unified = unified_source_choices()
    return (
        gr.update(choices=mod_choices),     # module_dd
        gr.update(choices=unified),         # linked_docs_dd  (Live Lecture tab)
        gr.update(choices=unified),         # doc_filter_dd   (Chat tab)
        _render_drag_panel(),               # drag_panel_html (Modules tab)
    )
```

Every handler that mutates module state calls `_module_refresh()` and unpacks its 4-tuple into the return. To add a fifth synced component in the future, add it here once and every handler inherits it.

The one exception is `delete_module_stage2`, which needs to pass `value=new_val` on the `module_dd` update (pointing to the first remaining module after the deleted one). It calls `_module_refresh()[1:]` to get the last three items and manually prepends a custom `module_dd` update:

```python
return (
    gr.update(choices=mod_choices, value=new_val),   # custom module_dd
    *_module_refresh()[1:],                          # linked_docs_dd, doc_filter_dd, drag_panel
    gr.update(interactive=False),                    # confirm_delete_btn
    f"Deleted '{name}'.",                            # modules_status_tb
)
```

---

### The drag-drop UI

The Modules tab replaced the old `mod_docs_dd` multiselect + "Save Documents" button with a three-pane HTML panel built with SortableJS.

**What you see:**

- **Left pane** — module list. Click to select a module. The selected module is highlighted in blue.
- **Middle pane** — that module's documents, numbered in order. Drag to reorder within the pane. Click × to remove. Drag a document in from the library to add.
- **Right pane** — document library (all ingested files from `docs/`). Items already in the selected module are greyed out with a ✓ and are not draggable. Items that appear in more than one module show a 🔗 icon — informational only, nothing is blocked.

**How the Python/JS bridge works:**

1. `_render_drag_panel()` in `app.py` builds the entire HTML string. The module/library state is embedded as JSON in `<script type="application/json" id="pma-data">`, and the JS logic lives in `<script type="text/x-pma" id="pma-init">`. Neither executes automatically (Gradio's `gr.HTML` uses `innerHTML`, which never executes `<script>` tags). Execution is bootstrapped by an `<img src="x" onerror="...">` whose handler calls `new Function(initCode)()`. When Gradio re-renders the component after any mutation, the new `<img>` fires `onerror` again and re-initializes from fresh data — no stale state.

2. The bridge is a hidden `gr.Textbox(elem_id="drag_state")`. On every drag mutation (add, remove, reorder), the JS writes `{"module_id": "mod_xxx", "ordered_filenames": ["a.pdf", "b.pptx"]}` to that textbox and dispatches an `input` event.

3. Gradio sees the `input` event and fires `on_drag_change(payload_json)`:
   ```python
   def on_drag_change(payload_json: str):
       try:
           data = _json.loads(payload_json)
           set_module_docs(data["module_id"], data["ordered_filenames"])
       except Exception:
           pass
       return _module_refresh()
   ```
   The `except: pass` is intentional. During a Gradio hot reload, the JS can fire before a module is selected, sending a partial or empty payload. The handler silently no-ops and still returns fresh dropdown state.

4. `set_module_docs` is used for all three drag gestures (add, remove, reorder). The UI always sends the complete desired final list, so one function covers every case.

**SortableJS setup:**

- Loaded from CDN: `https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js`
- The doc list (`pma-doc-list`) and the library (`pma-lib-list`) share a named drag group (`pmadocs`).
- The library uses `pull: 'clone'` — dragging from the library clones the element into the doc list rather than moving it, so the library always stays complete.
- Library items already in the module have no `pma-item` class and use `cursor: default`, which prevents SortableJS from picking them up via the `filter` option.

**Module selection is pure JS** — clicking a module in the left pane updates the middle pane entirely client-side without a server round-trip. The local JS `MODS` array tracks the current document list for each module. When you switch modules, the JS commits any pending state to the current module's entry before switching. A mutation only reaches the server when a drag fires `push()`.

**After any server-side mutation** (create, rename, delete, drag save), `_module_refresh()` returns a fresh `_render_drag_panel()` HTML string, which replaces the `gr.HTML` content. The JS re-initializes from scratch with the updated state. This means module selection resets to the first module after any mutation. That is the current behavior — it is a known UX rough edge, not a bug.

---

### The create/rename/delete controls

These are kept as standard Gradio textbox + button rows below the drag panel. The module selector dropdown (`module_dd`) in this section is only for rename/delete — it does not affect or reflect the drag panel's selection state. They are independent: the drag panel has its own internal JS selection state, and the dropdown has Gradio's state.

**Outputs wiring for each handler:**

| Handler | outputs list |
|---|---|
| `create_mod_btn.click` | `[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html, new_mod_tb, modules_status_tb]` |
| `rename_mod_btn.click` | `[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html, modules_status_tb]` |
| `delete_mod_btn.click` (stage 1) | `[confirm_delete_btn, modules_status_tb]` (no mutation, no refresh) |
| `confirm_delete_btn.click` (stage 2) | `[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html, confirm_delete_btn, modules_status_tb]` |
| `drag_state.input` | `[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html]` |

The first four positions are always the `_module_refresh()` tuple, in that order.

---

### Files changed and deleted

| File | Status | What happened |
|---|---|---|
| `module_store.py` | **New** | The canonical module layer. Dropped in as a replacement for `modules.py`. |
| `modules.py` | **Deleted** | Old CRUD layer. Replaced entirely. |
| `modules.json` | **Migrated** | Format changed from `{"modules": [...]}` to flat `[...]`. All existing modules preserved. The migration was done manually in-session — no backup was made because the file was untracked in git. |
| `app.py` | **Modified** | Import swapped to `module_store`. Removed duplicate `unified_source_choices()` and `module_dd_choices()` helpers. Added `_render_drag_panel()`, `_module_refresh()`, `on_drag_change()`. Replaced 5 verbose handlers with streamlined versions. Replaced Modules tab UI. |
| `qa.py` | **Fixed** | `answer_question()` docstring and `__main__` unpacking corrected. The function already returned a 3-tuple `(answer, reasoning, source_files)` — the docstring said 2. |
| `voice_qa.py` | **Unchanged** | Flagged as potentially unused in the audit but explicitly kept per user instruction. Do not delete. |

---

### What to watch for when testing

**The drag panel re-renders on every mutation.** When you create, rename, or delete a module, the drag panel HTML is replaced and the JS re-initializes. The selected module in the drag panel resets to the first in the list. This is expected.

**SortableJS loads from CDN.** On first load, the library drag handles won't initialize until the CDN script loads. `renderMods/Docs/Lib` run immediately (renders the list items), and `initDocSort()` / `initLibSort()` run in the `onload` callback. If the lab machine has no internet access, SortableJS fails silently with a console warning `[PMA] SortableJS CDN unavailable — drag reorder disabled`. The panes still render correctly — you just can't drag to reorder or drag-add. Click × still removes items. To fix offline: download `sortablejs@1.15.2/Sortable.min.js` into the project, serve it from Gradio's static path or inline it in `_render_drag_panel()`.

**The `[Doc]` prefix on unified_source_choices.** The Chat doc filter and the Live Lecture source picker now show items as `[Doc] filename.pdf` rather than bare `filename.pdf`. The dropdown value is still the bare filename — `expand_selection()` receives and returns bare filenames. Only the display label changed.

**Module IDs in linked_documents.** If a session was saved before the rebuild with bare filenames in `linked_documents` (from sessions.json), those still work — `expand_selection()` treats anything that isn't a recognized module ID as a bare filename and passes it through directly, filtering against the manifest for validity.

**If modules.json is ever corrupted or deleted.** `load_modules()` returns `[]` on any read error — the app starts with no modules, no crash. Everything that calls `expand_selection()` with a module ID will get an empty list back and behave as if no documents are linked.

# Prof AI — Session Handoff

This document is written for a future AI session (Claude Code or Claude web) with zero prior context. Read it before touching any code. Code is ground truth — if this doc and the code disagree, the code wins.

---

## Change log

| Date | Summary |
|---|---|
| 2026-06-24 | Literature search rate-limiting fixes. `semantic_scholar.py`: `SEMANTIC_SCHOLAR_API_KEY` now read from `os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")` (was hardcoded `""`); `_RETRY_DELAYS` shortened to `(2, 4)` — 2 retries max, ≤6 s total wait (was `(2,4,8,16)` = 30 s). `query.py`: `_SOURCE_TIMEOUT = 8` constant added; `as_completed(futures, timeout=_SOURCE_TIMEOUT)` wraps the parallel literature fetch, catching `FuturesTimeoutError` and logging which sources were abandoned — completed sources still contribute results; `ollama.chat` in `query_stream` gains `keep_alive="30m"` so the Chat LLM stays resident between queries. |
| 2026-06-24 | LLM pre-warm + clean TTS cancel/interrupt (React/FastAPI app). `api/main.py` lifespan: after Whisper+Piper pre-warm, fires a throwaway `ollama.chat(model="qwen3:30b-a3b", messages=[{"role":"user","content":"hi"}], think=False, keep_alive="30m", options={"num_predict":1})` in `_THREAD_POOL` to load the answer LLM into VRAM before the first real question. `_qa_cancel = threading.Event()` added inside the `ws_lecture` closure. `qa_handler`: calls `_qa_cancel.clear()` at entry; checks `_qa_cancel.is_set()` after the LLM call and after `tts.synthesize()` — if set, returns early skipping answer emit and audio bytes. `cancel` message handler: calls `_qa_cancel.set()` and forces `session.set_mode(Mode.LECTURE)` regardless of current mode (was previously guarded by `mode == AWAITING_QUESTION` only). `LiveLecture.tsx`: added `ttsSourceRef` (AudioBufferSourceNode | null) and `ttsMutedRef` (boolean) refs. New `interruptTts()` callback stops and nulls the source node, resets `playTimeRef.current=0`. `handleTtsAudio`: stores source in `ttsSourceRef`, checks `ttsMutedRef.current` before and after the async `decodeAudioData` await — drops bytes if muted. `cancelAsk`: sets `ttsMutedRef.current=true`, calls `interruptTts()`, restores gate threshold, then sends WS cancel. `question` message handler: resets `ttsMutedRef.current=false` to unmute the next Q&A cycle. Cancel button: was shown only in `uiMode==='awaiting'`, now shown for `awaiting \| processing \| speaking`; "Done speaking" button stays awaiting-only. Files touched: `api/main.py`, `web/src/screens/LiveLecture.tsx`. |
| 2026-06-24 | Question-capture flush: Done Speaking button + adaptive silence + 20s hard cap. `app.py`: Added `ask_ai_done_btn` ("✓ Done Speaking") to the main controls row, enabled only while mode is AWAITING_QUESTION and not in conversation — polled by the existing 2s timer via new `poll_ask_ai_done_btn()`. New `ask_ai_done_fn()` force-flushes `state["qa_buffer"]` via `push_question_audio()` immediately. Replaced the old fixed 5.5s capture window with adaptive silence detection: `ambient_rms` is tracked during LECTURE mode via a slow asymmetric EMA (α=0.05, only updates when chunk_rms < ambient×3 to exclude speech); in AWAITING_QUESTION (single-turn) the silence threshold is `max(0.005, ambient_rms × 2.5)` instead of the fixed `_SPEECH_ENERGY_THRESHOLD`; `qa_silence_samples` counter flushes after 600ms of sub-threshold audio but only after ≥1.5s of buffer accumulated (`_QA_MIN_WINDOW_S`). Hard cap raised to `_QA_MAX_WINDOW_S=20.0s` (was implicit 5.5s). State dict gained two fields: `ambient_rms` (float, init 0.005) and `qa_silence_samples` (int, init 0). All state init sites updated (`start_recording`, `stop_recording_audio`, `handle_pause`, `handle_resume`, `handle_audio_chunk` null-init). `stop_recording` return and `stop_btn.click` outputs updated to include `ask_ai_done_btn`. Conversation mode (in_conversation=True AWAITING path) is unchanged — still accumulates until "Done speaking" / 60s cap. Cancel still aborts to LECTURE at any time. |
| 2026-06-15 | Structured gap findings + question-capture robustness. `gaps.py`: added `parse_gap_findings(text) -> list[dict]|None` (exported, strips code fences, finds first JSON array). Prompt changed to request JSON array `{topic, status:"covered"|"partial"|"uncovered", note, source}`. `live_gap.py`: imports `parse_gap_findings` from gaps.py; `live_gaps_stream` prompt same JSON format; `_run_once` calls `parse_gap_findings` and delivers `list[dict]|str` to `on_result`. `api/main.py`: `on_gap` now emits `{type:"gap", findings:[...]}` or `{type:"gap", raw:"..."}`. `/api/gaps` endpoint buffers content tokens, streams think tokens, parses findings at end, emits `{type:"done", findings:[...], raw:"..."}`. `web/src/components/FindingCard.tsx`: new shared card component with green/amber/neutral status colours. `LiveLecture.tsx`: gap state changed to `gapFindings:Finding[]|null` + `gapRaw:string|null`; gap panel renders FindingCards or raw fallback text. `GapsAnalysis.tsx`: `findings` state populated from `done` message; renders FindingCard grid; raw text fallback via FindingsBlock. `LiveLecture.tsx` question-capture robustness: AudioWorklet gains `force_flush` message (drains partial buffer immediately); ambient RMS tracked via slow EMA (α=0.005); `askAI` raises gate threshold to `max(gate, ambient×1.3)` before entering AWAITING; `doneSpeaking` restores threshold + calls `force_flush`; `cancelAsk` also restores threshold; 20 s auto-flush timeout via useEffect; "Done speaking" button visible in side-panel awaiting card AND control bar. Build: 0 TypeScript errors. |
| 2026-06-15 | Phase 5B — Live Lecture screen wired in new React/FastAPI app. `api/main.py`: added `/ws/lecture` WebSocket endpoint with full mode state machine (LECTURE/AWAITING_QUESTION/PROCESSING + speaking/paused derived at WS layer), asyncio lifespan for Whisper+TTS pre-warming, asyncio.Queue drain task to bridge bg-thread QA handler to WS, LiveGapWorker integration, `answer_question`/`answer_conversation_turn` dispatch on `session.in_conversation`, TTS WAV sent as binary WS frames. `web/src/screens/LiveLecture.tsx`: full screen replacing placeholder — pre-start panel (name, doc picker), AudioWorklet blob URL capture (48kHz→16kHz, RMS gate, 700ms flush), state bar (5 visual states, color-coded), transcript pane (JetBrains Mono 13.5px), answer card with citation chips, live gaps panel, Q&A history, control bar with Ask AI/Pause/Cancel/Conv mode/gate slider. `web/src/App.tsx`: passes `onToast` to `<LiveLecture />`. `npm run build` clean — 0 TypeScript errors. Gradio `app.py` unchanged. |
| 2026-06-12 | Whisper fixes after first SPEC 2 field test. (1) `temperature=0.0` caused an un-escapable repetition loop ("university university ..." for a full segment) because temperature retries are also Whisper's escape hatch for repetition. Changed to `temperature=[0.0, 0.2, 0.4]` in live_transcribe.py and batch_transcribe.py — keeps loop escape, still avoids the high-temperature fabrication zone. (2) Vocab term filtering in `build_whisper_prompt()`: drops terms under 3 chars, bare numbers/dates, and duplicates before classification, because the 14b extractor shredded a doc title into junk tokens ("September", "30", "2014") that polluted the initial prompt. |
| 2026-06-12 | Modules drag panel keeps selection across re-renders. app.py `_render_drag_panel()` JS: selection (class + module) is persisted to localStorage in `__pmaSelClass` / `__pmaSel` via new `saveSel()`, and the init IIFE restores it on every panel rebuild instead of always selecting the first class/module. Restore prefers the stored module and locates whichever class currently contains it. Fixes: adding several documents to one module bounced the user back to Unassigned after every drop. |
| 2026-06-12 | SPEC 2 implemented (commit 5bdba9b, via aider + qwen3.6:27b, reviewed and patched by Claude). live_transcribe.py and batch_transcribe.py: `vad_filter=True` with `min_silence_duration_ms=250`, `temperature=0.0` (disables the fallback chain), and a `HALLUCINATION_PHRASES` filter that drops segments matching a known-hallucination phrase AND `no_speech_prob > 0.3`, logged with `[Whisper]` prefix. Vocab extraction: `/no_think` prompt prefix removed, replaced with `think=False`. NOTE: qwen originally wrote `options={"think": False}`, which Ollama silently ignores — corrected to the top-level `think=False` kwarg. Section 3 tables and section 5 open issues updated accordingly. |
| 2026-06-11 | Doc audit, no code changes. Verified every file against code. Fixed in this doc: Q&A retrieval filter is `file_type != lecture_audio` (was wrongly stated as content_type based), gap analysis reads the transcript from file not from ChromaDB `lecture_transcript` chunks (nothing queries those back out), added `language="en"` row to the Whisper table, noted qa.py never reads `session.latest_gap_analysis`. Added dead-code entries: `reformulate_for_pubmed()` and the `NCBI_EMAIL` placeholder. CLAUDE.md got a larger rewrite: stack section (large-v3-turbo not "small", browser mic not sounddevice, three literature APIs not just PubMed), file layout (added module_store, live_gap, semantic_scholar, openalex, ui_theme, classes.json, modules.json; removed nonexistent text_utils.py; flagged voice_qa.py as dead), thinking modes (think= API param, not prompt prefixes), state machine (set_mode only sets the enum; the described flush/timestamp behavior never existed in current code), Q&A context assembly (full transcript, last 2 turns, no gap-analysis block), Chat tab (multi-source literature), pending-issues list pruned of three completed items. UPGRADE_SPEC.md corrected to match: removed already-done `language="en"` item, fixed slide-chunking claims (pptx already chunks per slide with page_or_slide metadata). |
| 2026-06-11 | Dark UI overhaul. New `ui_theme.py` holds THEME (gr.themes.Soft with emerald/cyan/slate dark variables), CSS (defines `--pa-*` CSS variables plus header, monospace live panes, hidden-bridge, scrollbar styles), FORCE_DARK_JS (reloads once with `?__theme=dark`), HEADER_HTML (brand header). app.py: all hardcoded hex colors in inline HTML (drag panel, literature cards) replaced with `var(--pa-*)` references so the palette is controlled in one place. Layout: Chat retrieval controls moved into a collapsed "Search scope and literature sources" accordion, Live Lecture top controls grouped (lecture name + linked docs + refresh in one row inside gr.Group), Modules tab module/class control stacks now side by side in two columns. elem_ids added: pa-status, pa-transcript, pa-gap, pa-qa-log, pa-conv. launch() now passes theme=THEME, css=CSS, js=FORCE_DARK_JS (in Gradio 6 these are launch params, not Blocks params — the old code passed theme to launch already but Soft() default light look was unstyled). No event wiring changed. Verified: `build_ui()` constructs 143 blocks without error under the project venv. |
| 2026-05-28 | Fix: drag bridge never reached Python (doc AND class drags). Root cause: the bridge was a `gr.Textbox(elem_id="drag_state")` and JS set its `.value` then dispatched a synthetic `input` event — but Gradio's Svelte frontend ignores programmatic value-sets + synthetic events, so the bound `drag_state.input(...)` handler never fired. Confirmed by a DIAG test: JS logged the value-set, Python stayed silent. Fix: a hidden `gr.Button(elem_id="drag_trigger")` whose `.click(fn=on_drag_change, inputs=[drag_state], js="read #drag_state value and return it")` routes through Gradio's real event system. JS now writes `#drag_state` then `.click()`s `#drag_trigger`. The dead `drag_state.input()` binding was removed. `on_drag_change` routes by payload shape: `ordered_filenames` → `set_module_docs`, `class_id` → `assign_module_to_class`. Cleanup: removed the inert `class_drag_state` textbox + `on_class_drag_change` stub (the failed second bridge) and all diagnostic logs. |
| 2026-05-28 | Modules tab: four-pane layout. Replaced three-pane (class+module tree / docs / library) with four independent panes: Classes (pane 1, click-select + module drop target) → Modules in selected class (pane 2, click-select + drag source for class reassignment) → Documents in selected module (pane 3, reorder + library drops) → Document Library (pane 4). Each pane `flex:1 1 0`. Module reorder within pane 2 disabled. Drop-target highlighting via CSS classes on `#profai-mod-manager`. `__pmaToggleClass` removed; replaced by `__pmaSelClass`. |
| 2026-05-28 | Fix: drag panel JS silent crash. Python `\'` in f-string is a Python escape (backslash consumed), producing bare `'` that closed JS string literals inside onclick attributes early. `new Function(content)` threw SyntaxError, caught silently, both panes stuck at "Loading...". Fixed by replacing `\'...\' ` with `&#39;...&#39;` (HTML entity decoded by browser before JS runs). Three sites: pmaToggleClass, pmaSel, pmaRm. |
| 2026-05-28 | Class layer above modules. module_store.py: CLASSES_PATH, load/save_classes, get_class, create/rename/delete_class, assign_module_to_class, class_tree(), class_dd_choices(). create_module() gains optional class_id param. app.py: _render_drag_panel() rewritten for two-level tree, class CRUD handlers, class_drag_state bridge, class controls in Modules tab UI. |
| 2026-05-28 | Conversation mode Phase 1. session.py: in_conversation, conversation_history, write_transcript_marker. qa.py: answer_conversation_turn (fast/deep), _format_conversation_history. app.py: _conversation_handler, start/end/speak/done handlers, LECTURE guard in handle_audio_chunk, variable-length AWAITING_QUESTION capture, polling functions, Conversation accordion UI. |
| 2026-05-28 | Full audit against code. Corrected live transcription path (browser mic, not server-side sounddevice). Corrected Q&A state machine (browser audio handler, _transcription_loop deleted). Fixed tts.py API name. Documented Q&A statefulness and answer_question() return shape. Added LLM model table. Removed resolved "browser mic" open issue. Noted dead code. |

---

## 1. What this system is

A local AI assistant for a **communications professor**. It transcribes lectures in real time from the professor's **browser microphone**, lets the prof ask spoken questions mid-lecture and get RAG-backed answers read aloud, and supports research queries against course materials plus academic literature databases. Everything runs on a single lab machine (RTX 4090, Windows 11). No cloud LLMs. The only external network calls are to PubMed, Semantic Scholar, and OpenAlex — all open APIs, no auth required.

Project root: `C:\prof-ai`

---

## 2. Architecture overview

### Major components

| File | Role |
|---|---|
| `app.py` | Gradio UI — five tabs: Chat, Live Lecture, Add Materials, Modules, Gaps Analysis |
| `ui_theme.py` | Visual identity: THEME (dark Gradio theme), CSS (`--pa-*` palette variables referenced by inline HTML in app.py), FORCE_DARK_JS, HEADER_HTML. Imported by app.py and passed to `launch()`. |
| `live_transcribe.py` | Exposes `push_audio()` and `push_question_audio()` called from `app.py` browser streaming handler. Manages Whisper model singleton, builds initial prompts, manages active session singleton. No audio capture or VAD. |
| `recorder.py` | Defines audio constants (`SAMPLE_RATE`, `VAD_SILENCE_MS`, `INPUT_DEVICE_NAME`) and `resolve_input_device()`. **Not in the live recording path.** Browser mic replaced server-side sounddevice. Still imported by live_transcribe.py for constants. |
| `batch_transcribe.py` | Transcribe pre-recorded audio files into ChromaDB |
| `ingest.py` | PDF/DOCX/PPTX/TXT → text extraction → chunk → embed → ChromaDB. OCR fallback for scanned PDFs via pytesseract |
| `session.py` | `LectureSession` class — mode state machine, transcript file, ChromaDB writes, sessions.json |
| `module_store.py` | Module CRUD layer. Manages named ordered document lists. Provides `expand_selection()`, `module_dd_choices()`, `unified_source_choices()`. Replaced the old `modules.py`. |
| `query.py` | RAG query backend — multi-source literature search, dedup, context assembly, LLM streaming |
| `qa.py` | Voice Q&A answer generation (RAG + LLM with thinking enabled). Returns `(answer, reasoning, source_files)`. |
| `gaps.py` / `live_gap.py` | Gap analysis — compares running transcript against linked slide decks |
| `pubmed.py` | PubMed E-utilities wrapper |
| `semantic_scholar.py` | Semantic Scholar Graph API wrapper |
| `openalex.py` | OpenAlex API wrapper |
| `manifest.py` | Shared JSON manifest helpers (tracks ingested files) |
| `tts.py` | Piper TTS wrapper. Public method: `PiperTTS.synthesize(text) -> bytes` (returns WAV bytes). Internally calls `self._voice.synthesize_wav()` on the PiperVoice object — callers do NOT call synthesize_wav directly. |

### Data flows

**Live transcription path (browser mic):**
```
Prof's browser mic
  → Gradio gr.Audio(sources=["microphone"], streaming=True)
  → handle_audio_chunk(state, audio_chunk) in app.py
      → normalize: stereo→mono, resample to 16kHz (scipy.signal.resample), float32
      → state["paused"] == True → return early, skip everything
      → mode check:

      mode == LECTURE:
          RMS noise gate:
              chunk_rms = sqrt(mean(audio^2))
              chunk_rms >= _SPEECH_ENERGY_THRESHOLD (0.02)
                  → speech: append to state["buffer"], reset state["silence_samples"] = 0
              chunk_rms <  _SPEECH_ENERGY_THRESHOLD (0.02)
                  → noise/silence: discard chunk, state["silence_samples"] += len(chunk)
          ambient noise tracking (passive, used by Q&A path only):
              if chunk_rms < state["ambient_rms"] × 3.0:
                  ambient = (1 - 0.05) × ambient + 0.05 × chunk_rms  (clamped 0.001–0.1)
              speech chunks do not raise the ambient estimate
          flush condition:
              silence_samples * 1000 / 16000 >= _VAD_SILENCE_MS (600ms) AND len(buffer) > 0
              OR len(buffer) / 16000 >= _MAX_BUFFER_S (15.0s)
          on flush:
              push_audio(buf) in live_transcribe.py
                → process_utterance(buf, model, session, session_start)
                    → transcribe_audio() → faster-whisper large-v3-turbo (CUDA float16)
                    → filter: MIN_WORDS=2, MIN_AVG_LOGPROB=-1.0
                    → session.append_segment() → transcript file + ChromaDB

      mode == AWAITING_QUESTION (single-turn Ask AI, not in_conversation):
          all audio accumulated in state["qa_buffer"]
          adaptive silence threshold = max(0.005, state["ambient_rms"] × 2.5)
              (ambient_rms was calibrated during LECTURE; 2.5× lifts threshold
               above room noise but below speech — so end-of-question silence
               registers even in noisy rooms)
          state["qa_silence_samples"] increments when chunk_rms < adaptive_thresh,
              resets to 0 on speech
          flush conditions (whichever fires first):
              A. buf_s >= _QA_MAX_WINDOW_S (20.0s) — hard cap, always fires
              B. buf_s >= _QA_MIN_WINDOW_S (1.5s) AND qa_silence_ms >= 600ms
                    — adaptive silence detected after minimum window
          "✓ Done Speaking" button (ask_ai_done_btn) — also calls push_question_audio
              directly; primary reliable path, independent of silence detection
          on any flush → push_question_audio(qa_buf) in live_transcribe.py
              → transcribe_audio()
              → validity check: MIN_WORDS, MIN_AVG_LOGPROB
                  fail → session.set_mode(LECTURE)
                  pass → session.pending_question = text
                       → session.set_mode(PROCESSING)
                       → fire _qa_handler thread in app.py

      mode == AWAITING_QUESTION (conversation mode, in_conversation=True):
          all audio accumulated in state["qa_buffer"] — no silence detection
          "Done speaking" button (done_btn, in Conversation accordion) is
              the primary flush path
          safety cap: buf_s >= _CONV_MAX_TURN_S (60.0s) → auto-flush

      mode == PROCESSING:
          discard all audio while _qa_handler runs

  → mic.stop_recording event:
      stop_recording_audio(state) — flushes remaining state["buffer"] to push_audio()
```

Note: `_transcription_loop`, `AudioRecorder`, and Silero VAD are **not used** in the live path. They were replaced by this browser streaming architecture.

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
  → qwen3:14b with think=False (API param, set since 2026-06-12; prior `/no_think`
    prefix was removed)
  → returns comma-separated raw vocabulary
  → classify terms:
      multi-word AND starts-capital → proper noun ("David A. Kolb")
      everything else → concept term ("experiential learning")
  → _format_whisper_prompt(concepts, proper, max_chars=900)
      → "This is a university lecture. Topics may include: X, Y, and Z.
         Speakers may reference: A, B, and C."
      → drops whole items from end (concepts first) if over 900 chars
  → stored on session.whisper_initial_prompt
  → passed to every transcribe_audio() call for the session duration
  → fallback when no docs linked: DEFAULT_INITIAL_PROMPT (prose, not a list)
```

**Voice Q&A answer generation (qa.py):**
```
answer_question(question, linked_docs, session) -> tuple[str, str, list[str]]
  returns (answer, reasoning, source_files)

Context assembly:
  1. RAG chunks from ChromaDB, top 6, where file_type != "lecture_audio" AND
     source_file in linked_docs (NOTE: the filter is file_type based — it does NOT
     use content_type, unlike the Chat path in query.py)
  2. Full lecture transcript from file (via _load_transcript)
  3. Last 2 Q&A turns from session.qa_history (via _format_qa_history — STATEFUL)

session.latest_gap_analysis is NOT used here. The live gap worker populates it for
the UI panel, but qa.py builds its own GAP MODE / CONTENT MODE comparison from the
raw transcript and source blocks.

LLM call:
  model=qwen3:30b-a3b, think=True, keep_alive="30m"
  Response fields read separately:
    answer    = sanitize_answer(response["message"]["content"])
    reasoning = response["message"].get("thinking") or ""

sanitize_answer() is a safety net for clients that inline reasoning into content:
  if "</think>" in text: keep only text after last </think>
  strip any remaining <think>...</think> blocks
  strip whitespace

What goes where:
  answer    → tts.synthesize(answer), sessions.json "answer" field, Q&A log display
  reasoning → sessions.json "reasoning" field only (never to TTS, never displayed)
```

**Q&A statefulness:**
Q&A is stateful within a session. `_format_qa_history(session, max_turns=2)` assembles the last 2 turns from `session.qa_history` into the LLM prompt as a "RECENT Q&A" block. This means the LLM can see prior questions and answers when formulating the next response. History lives on `session.qa_history`: list of dicts with keys `question`, `answer`, `reasoning` (optional), `sources`, `timestamp`.

**ChromaDB content types:**
```
content_type = "source_document"    → ingested PDFs, slides, docs (Chat tab filters on this)
content_type = "lecture_transcript"  → live session segments + batch audio (written, never queried back)
```
Same collection (`course_material`). Only Chat (`query.py`) filters by `content_type`. Voice Q&A (qa.py) and gap analysis (gaps.py, live_gap.py) filter by `file_type != "lecture_audio"` instead, plus `source_file in linked_documents` when docs are linked. Gap analysis reads the lecture transcript from the transcript FILE (`load_transcript`), not from ChromaDB — nothing currently retrieves `lecture_transcript` chunks. The `linked_documents` field on a session is a list of `source_file` values used to scope gap analysis and Q&A retrieval.

---

## 3. Current configuration — exact values

All values verified against the actual code.

### faster-whisper / transcription (live_transcribe.py and batch_transcribe.py)

| Parameter | Value | Reason |
|---|---|---|
| `WHISPER_MODEL` | `"large-v3-turbo"` | Better accuracy on domain vocabulary, fast enough on 4090 (~200ms/utterance). |
| `no_speech_threshold` | `0.7` | Stricter silence detection. Suppresses hallucinations on quiet input. |
| `log_prob_threshold` | `-0.8` | Discards segments where Whisper's token confidence is below threshold. Hallucinations score poorly. |
| `condition_on_previous_text` | `False` | Prevents error cascades across utterances on weak signal. |
| `beam_size` | `5` | Default, not tuned. |
| `language` | `"en"` | Skips per-utterance language detection. Set in both live and batch paths. |
| `temperature` | `[0.0, 0.2, 0.4]` | Short retry ladder. 0.0-only (tried 2026-06-12) caused un-escapable repetition loops; the full default ladder up to 1.0 fabricates completions. This is the middle ground. |
| `vad_filter` | `True` (`min_silence_duration_ms=250`) | Silero VAD inside faster-whisper, second gate behind the app.py RMS gate. |
| `initial_prompt` | Session-generated prose (see path above) | Biases Whisper toward domain vocabulary. Batch path uses DEFAULT_INITIAL_PROMPT. |
| `HALLUCINATION_PHRASES` | module constant, both files | Segments matching a phrase AND `no_speech_prob > 0.3` are dropped and logged. |

**Not set — faster-whisper defaults apply:**
- `compression_ratio_threshold`: default `2.4`
- `prompt_reset_on_temperature`: default behavior

### Conversation mode (app.py, qa.py, session.py)

| Constant / attribute | Location | Value / type |
|---|---|---|
| `_CONV_MAX_TURN_S` | `app.py` | `60.0` — auto-flush cap for forgotten "Done speaking" clicks |
| `session.in_conversation` | `session.py` | `bool`, default `False` |
| `session.conversation_history` | `session.py` | `list[dict]` — `{question, answer, reasoning, sources, timestamp}` |
| `_conv_pending_answer` | `app.py` | module-level `str`; non-empty when a Deep mode answer awaits "Speak this" |

**Turn flow:** "Speak to AI" → registers `_conversation_handler(fast=...)` on session, sets `AWAITING_QUESTION`. Browser audio accumulates in `state["qa_buffer"]` with no fixed timeout. "Done speaking" flushes the buffer to `push_question_audio()`. The 60s cap fires if "Done speaking" is never clicked. `push_question_audio` transcribes and fires the registered handler.

**Fast path:** `_conversation_handler` calls `answer_conversation_turn(fast=True)` (think=False, short answer), auto-plays TTS via `_set_qa_audio`, fires `_resume_after_conv_turn` timer. Mode returns to LECTURE (conversation idle) after TTS.

**Deep path:** `_conversation_handler` calls `answer_conversation_turn(fast=False)` (think=True, full answer). Answer shown in conv_display_box. Mode stays PROCESSING. `_conv_pending_answer` set. "Speak this" button becomes active. Prof clicks it to play TTS.

**Escape hatch:** `end_conversation_fn()` calls `session.set_mode(Mode.LECTURE)` unconditionally — force-exits any stuck PROCESSING or AWAITING_QUESTION state. Always available.

**History separation:** Conversation turns go to `session.conversation_history` only — NOT to `session.qa_history`. Single-turn "Ask AI" uses `_format_qa_history` which reads `qa_history`; conversation uses `_format_conversation_history` which reads `conversation_history`. No cross-contamination.

**Transcript markers:** `write_transcript_marker()` writes `--- [CONVERSATION START HH:MM:SS] ---` and `--- [CONVERSATION END HH:MM:SS] ---` directly into the transcript file (not as segments, not in ChromaDB).

**Persistence:** `conversation_history` is saved to `sessions.json` under `"conversation_history"` key when the session closes.

### Browser audio processing (app.py — active noise gate)

| Constant | Value | Role |
|---|---|---|
| `_SPEECH_ENERGY_THRESHOLD` | `0.02` | RMS gate: chunks below this are noise and discarded in LECTURE mode. Raise if background noise bleeds through; lower if quiet speakers get clipped. Not used in AWAITING_QUESTION path (uses adaptive threshold instead). |
| `_VAD_SILENCE_MS` | `600` | Consecutive noise/silence milliseconds required to flush the speech buffer (LECTURE) or the question buffer (single-turn Ask AI). Matches `recorder.VAD_SILENCE_MS`. |
| `_MAX_BUFFER_S` | `15.0` | Hard cap on LECTURE mode speech buffer: flush at 15s even without silence detection (non-stop talker). |
| `_QA_MAX_WINDOW_S` | `20.0` | Hard cap on single-turn Ask AI question buffer. Flushes automatically after 20s regardless of silence — prevents indefinite hangs. |
| `_QA_MIN_WINDOW_S` | `1.5` | Minimum seconds of audio that must accumulate before silence detection can trigger a flush. Prevents immediate flush before the prof starts speaking. |
| `_AMBIENT_SPEECH_MULTIPLIER` | `2.5` | Adaptive silence threshold = `ambient_rms × 2.5`. Raise if end-of-question is being detected too aggressively; lower if silence detection fails in noisier rooms. |
| `_AMBIENT_ALPHA` | `0.05` | EMA weight for ambient noise tracking during LECTURE mode. Lower = slower calibration. |

**State dict keys** (carried on `gr.State`):

| Key | Type | Role |
|---|---|---|
| `buffer` | `np.ndarray float32` | Accumulated speech for LECTURE mode transcription |
| `qa_buffer` | `np.ndarray float32` | Accumulated audio for AWAITING_QUESTION |
| `silence_samples` | `int` | Consecutive silence sample count (LECTURE mode) |
| `paused` | `bool` | When True, all audio is dropped |
| `ambient_rms` | `float` | Rolling ambient noise estimate, updated during LECTURE mode |
| `qa_silence_samples` | `int` | Consecutive sub-threshold sample count (AWAITING_QUESTION single-turn) |

### recorder.py constants (imported but Silero VAD not used in live path)

| Parameter | Value | Note |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Required by faster-whisper. Used for resampling target in app.py. |
| `VAD_SILENCE_MS` | `600` | Imported into live_transcribe.py and app.py as the silence counter threshold. |
| `CHUNK_SIZE` | `512` | Silero requirement. Not used in live path (Silero VAD not called). |
| `MAX_SPEECH_SECONDS` | `30` | Defined in recorder.py. Replaced by `_MAX_BUFFER_S=15.0` in app.py. |

### Q&A buffering (app.py + live_transcribe.py)

The capture window for single-turn Ask AI is driven entirely by `app.py`'s `handle_audio_chunk()`. The `live_transcribe.py` constants are now unused in the single-turn path.

| Parameter | Location | Value | Note |
|---|---|---|---|
| `_QA_MAX_WINDOW_S` | `app.py` | `20.0` | Hard cap — single-turn ask AI auto-flushes after 20s. Primary safety net. |
| `_QA_MIN_WINDOW_S` | `app.py` | `1.5` | Silence detection doesn't fire until this many seconds of audio are buffered. |
| `_AMBIENT_SPEECH_MULTIPLIER` | `app.py` | `2.5` | Adaptive threshold multiplier for silence detection during question capture. |
| `_CONV_MAX_TURN_S` | `app.py` | `60.0` | Hard cap for conversation mode turns (Done speaking forgotten). |
| `QA_BUFFER_SECONDS` | `live_transcribe.py` | `4.0` | **Unused in active paths.** Defined but not referenced by `push_question_audio()` or `handle_audio_chunk()`. |
| `QA_TAIL_SECONDS` | `live_transcribe.py` | `1.5` | **Unused in active paths.** Was the old fixed-window tail; replaced by adaptive silence. |
| `QA_TRUNCATION_WAIT` | `live_transcribe.py` | `2.0` | **Dead code.** Only used by `_handle_question_utterance()` which is also dead. |

### Vocabulary prompt (live_transcribe.py)

| Parameter | Value | Reason |
|---|---|---|
| `VOCAB_MODEL` | `"qwen3:14b"` | Smaller model for fast vocabulary extraction. |
| `_WHISPER_PROMPT_MAX_CHARS` | `900` | Conservative ceiling for Whisper's 224-token initial_prompt limit. |

### LLM models and think settings

| Path | File | Model | think setting | keep_alive |
|---|---|---|---|---|
| Voice Q&A answer | `qa.py` | `qwen3:30b-a3b` | `think=True` (reasoning wanted) | `"30m"` |
| Whisper vocab extraction | `live_transcribe.py` | `qwen3:14b` | `think=False` | not set |
| Gap analysis batch | `gaps.py` | `qwen3:14b` | `think=False` | not set |
| Gap analysis live | `live_gap.py` | `qwen3:14b` | `think=False` | `"10m"` |
| Literature reformulation | `query.py` | `qwen3:14b` | `think=False` | not set |
| PubMed reformulation | `pubmed.py` | `qwen3:14b` | `think=False` | not set |
| Chat/research streaming | `query.py` | `qwen3:30b-a3b` | not set (thinking on by default) | not set |

---

## 4. Changes made in prior sessions (history)

### Browser mic refactor
Server-side sounddevice capture replaced by `gr.Audio(sources=["microphone"], streaming=True)`. `_transcription_loop` deleted. `pause_recording()` and `resume_recording()` are now no-ops — PROCESSING mode discards audio instead. `recorder.py` remains but is not in the live path.

### Noise gate
`handle_audio_chunk()` in app.py now applies an RMS energy gate per chunk before buffering. Chunks below `_SPEECH_ENERGY_THRESHOLD=0.02` are discarded and increment a silence counter rather than going into the speech buffer. Silence-gated flush replaced the old fixed 5s timer.

### Pause/resume via state flag
`gr.State` now carries a `"paused"` boolean. "Pause Mic" / "Resume Mic" buttons flip it. When paused, `handle_audio_chunk()` returns early and discards audio.

### Q&A thinking enabled, reasoning separated
`answer_question()` now uses `think=True`. Response is split into `answer` (content field, sanitized) and `reasoning` (thinking field). Only `answer` reaches TTS and sessions.json "answer". Reasoning stored in separate "reasoning" field of the qa_history entry for debugging.

### Qwen3 think=False on fast paths
Gap analysis (gaps.py, live_gap.py), literature reformulation (query.py), and PubMed reformulation (pubmed.py) all have `think=False` set via the API parameter. The `/no_think`/`/think` prompt prefixes were removed from those paths.

### Gap analysis model corrected
gaps.py and live_gap.py changed from `qwen3:30b-a3b` to `qwen3:14b`.

### Gap analysis delta transcript
`LiveGapWorker._run_once()` now sends only the delta since the last run (new segments since `_last_segment_count`) capped at `MAX_TRANSCRIPT_CHARS=4000`, instead of the growing full transcript. Skips the run entirely if no new segments arrived.

### Module system rebuild
`modules.py` deleted. Replaced by `module_store.py` with ordered document lists, atomic save, and `expand_selection()`. Modules tab rebuilt with a three-pane drag-drop HTML panel (SortableJS).

### Failure modes fixed in earlier sessions
1. Hallucination spam ("Thank you for watching"): raised `no_speech_threshold` to 0.7, added `log_prob_threshold=-0.8`
2. Short-word mistranscriptions: raised `VAD_SILENCE_MS` to 600ms in recorder.py
3. Prompt echoing in transcripts: reformatted `build_whisper_prompt` from comma list to sentence prose

---

## 4b. New React/FastAPI app (parallel to Gradio, in `web/` and `api/`)

A full replacement UI is being built alongside the Gradio app. Both can run simultaneously: Gradio on port 7860, FastAPI on port 8000. The new app serves the built Vite SPA from `web/dist/` and exposes a REST + WebSocket API.

### Stack
- **Frontend**: Vite + React 18 + TypeScript + Tailwind v3, in `web/src/`
- **Backend**: FastAPI in `api/main.py`, port 8000
- **Build**: `cd web && npm run build` → `web/dist/`
- **Dev**: `npm run dev` in `web/` (Vite dev server on 5173), `python -m uvicorn api.main:app --port 8000` separately
- **Run prod**: `python api/main.py` (serves the built Vite dist)

### Phases completed (commits 0cb1dff, b9f1799, 283abee)
- **Phase 0**: App shell, sidebar, command palette, toast system, FastAPI skeleton
- **Phase 1**: Chat screen — RAG + literature streaming NDJSON
- **Phase 2**: Add Materials — multipart ingest, NDJSON progress
- **Phase 3**: Modules — dnd-kit 4-column drag board
- **Phase 4**: Gaps Analysis — session selector, gaps_stream NDJSON
- **Phase 5A**: Standalone audio spike at `spike/` (port 8100) — proves Whisper+TTS transport
- **Phase 5B**: Live Lecture screen fully wired — see change log entry 2026-06-15

### `/ws/lecture` WebSocket protocol (api/main.py)
```
Client → Server (text JSON):
  {type:"start", name:"...", linked_documents:[...]}
  {type:"flush"}
  {type:"ask_ai"}
  {type:"cancel"}
  {type:"pause_mic"}
  {type:"resume_mic"}
  {type:"set_conversation_mode", enabled:true|false}
  {type:"stop"}
Client → Server (binary):
  Int16 PCM frames (20ms chunks, 16kHz equivalent after 48→16kHz decimation)

Server → Client (text JSON):
  {type:"started", session_id:"...", name:"..."}
  {type:"state", mode:"lecture"|"awaiting"|"processing"|"speaking"|"paused"}
  {type:"transcript", text:"...", confidence:0.95, ms:120}
  {type:"question", text:"..."}
  {type:"answer", text:"...", sources:[...], reasoning:"..."}
  {type:"tts_start"}
  {type:"tts_end"}
  {type:"gap", text:"...", ts:"2026-06-15T..."}
  {type:"stopped"}
  {type:"error", message:"..."}
Server → Client (binary):
  WAV bytes (Piper libritts-high, 22050 Hz mono 16-bit)
```

### Key design decisions
- **No push_audio()**: `push_audio()` returns None — can't get transcript text. WS handler calls `transcribe_audio()` + `session.append_segment()` directly, same filtering as `process_utterance()` (`_MIN_WORDS=2`, `_MIN_LOGPROB=-1.0`).
- **QA handler bridge**: QA runs in bg thread, bridges to WS via `asyncio.Queue` + `loop.call_soon_threadsafe`, drained by an async task. Same pattern as NDJSON streaming in `/api/chat`.
- **TTS**: `PiperTTS.synthesize(answer)` → WAV bytes sent as WS binary frame, played via Web Audio API queued playback (`playTime` cursor).
- **5 UI states from 3 backend modes**: `speaking` = between `tts_start` and `tts_end`; `paused` = client sent `pause_mic`. Backend modes are `LECTURE/AWAITING_QUESTION/PROCESSING`.
- **Conversation mode**: `session.in_conversation` toggled via `set_conversation_mode` message; dispatches `answer_conversation_turn` instead of `answer_question`. Frontend appends to `session.conversation_history` after each turn.
- **Level meter**: Updated via direct DOM ref mutation (not React state) to avoid 50Hz re-renders.
- **Startup pre-warming**: Whisper and TTS pre-loaded in asyncio lifespan before accepting connections. ~30-60s startup time.

### New app run instructions
```powershell
cd C:\prof-ai
.\venv\Scripts\Activate.ps1
python api\main.py
```
Then open `http://127.0.0.1:8000` in a browser. Startup log shows:
```
[STARTUP] pre-warming Whisper (large-v3-turbo · CUDA float16)…
[STARTUP] Whisper ready
[STARTUP] pre-warming Piper TTS…
[STARTUP] TTS ready — sample_rate=22050 Hz
[STARTUP] Serving static build from ...\web\dist
```

---

## 5. Known open issues

### High priority

**Mic hardware / gain staging**
RMS noise gate helps but the root cause of transcription quality issues is SNR. A lavalier or headset with proper gain staging eliminates room noise more effectively than any software threshold.

### Medium priority

**`BatchedInferencePipeline` for batch_transcribe.py**
`batch_transcribe.py` uses sequential `WhisperModel.transcribe()`. For long pre-recorded files, faster-whisper's `BatchedInferencePipeline` would improve throughput. Not implemented — sequential path is fast enough for short clips.

**Streaming TTS**
Piper synthesizes the complete LLM response before playback begins. A sentence-streaming approach (synthesize and play sentence by sentence as the LLM generates) would cut perceived latency from ~15s to ~2s. Implementation spec now lives in UPGRADE_SPEC.md (SPEC 1); earlier notes in notes.md. Requires restructuring the Q&A handler.

**Cancel button during PROCESSING**
Clicking Cancel while TTS is playing should interrupt audio and return cleanly to LECTURE mode. Currently the mode gets stuck until TTS finishes.

### Low priority

**Whisper model bump for batch**
`batch_transcribe.py` uses `large-v3-turbo`. For very long pre-recorded files where latency doesn't matter, `large-v3` (non-turbo) may produce marginally better accuracy.

**whisper-streaming**
Real-time streaming Whisper (word-level output during speech) was evaluated and deliberately deferred. Current per-utterance latency on the 4090 (~200ms) is fast enough for the lecture use case.

**`_handle_question_utterance()` is dead code**
This function still exists in `live_transcribe.py` (with a broken `"AudioRecorder"` type hint since that import was removed). It is not called by any active path. `push_question_audio()` replaced it. Safe to delete in a cleanup pass.

**`reformulate_for_pubmed()` in pubmed.py is dead code**
The shared `reformulate_for_search()` in query.py replaced it — `search_literature()` reformulates once and passes the finished query string to `search_pubmed()`. Nothing imports `reformulate_for_pubmed`. Safe to delete in a cleanup pass.

**`NCBI_EMAIL` placeholder**
pubmed.py still sends `your-email@example.com` as the NCBI contact email. NCBI asks for a real address on E-utilities requests. One-line fix.

---

## 6. Quickstart for the next session

### Read these files in this order

1. `CLAUDE.md` — project overview, stack, environment quirks, startup procedure
2. `notes.md` — design decisions and rationale
3. `app.py` — `handle_audio_chunk()` is now the live transcription entry point; read the browser streaming section before touching anything transcription-related
4. `live_transcribe.py` — `push_audio()`, `push_question_audio()`, `build_whisper_prompt()`; understand these before touching transcription logic
5. `session.py` — `LectureSession` class; understand `linked_documents`, `whisper_initial_prompt`, `qa_history`
6. `qa.py` — `answer_question()` returns `(answer, reasoning, sources)`; understand `sanitize_answer()` and `_format_qa_history()`
7. `ingest.py` — document pipeline; OCR fallback
8. `query.py` — multi-source literature search; `search_literature()` and `query_stream()`
9. `recorder.py` — read for constants only; NOT the live audio path
10. `ui_theme.py` — only if touching visuals; defines the `--pa-*` CSS variables that inline HTML in app.py depends on

### Run this to verify the system works

```powershell
cd C:\prof-ai
.\venv\Scripts\Activate.ps1
$env:PATH = "$pwd\venv\Lib\site-packages\nvidia\cublas\bin;$pwd\venv\Lib\site-packages\nvidia\cudnn\bin;" + $env:PATH
python app.py
```

Then open `http://127.0.0.1:7860` in a browser. The page reloads itself once with `?__theme=dark` appended — that is FORCE_DARK_JS in ui_theme.py doing its job, not a bug.

On startup the PowerShell window should show:
```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://....gradio.live
```

The Whisper model loads lazily on the first `push_audio()` call (first flush after speech), not at app startup. When that happens:
```
[Whisper] loading model='large-v3-turbo' device=cuda compute_type=float16
[Whisper] 'large-v3-turbo' ready
```

The config log line:
```
[Whisper] no_speech_threshold=0.7 log_prob_threshold=-0.8 vad_silence_ms=600
```
is printed by `start_live_session()` when the prof clicks Start Lecture, not at app startup.

### Verify literature search works

In the Chat tab, type any question, tick "Semantic Scholar (general academic)", and submit. The PowerShell window should show:
```
[Literature] llm 0.Xs
[Literature] reformulated query: '...'
[SemanticScholar] N result(s)
[Literature] N after dedup (from N raw)
```

### Verify OCR works (if needed)

Upload a scanned PDF through Add Materials. The log should show:
```
  Processing: filename.pdf
    PDF appears to be scanned (0 chars extracted), running OCR...
  [OCR] page 1/N...
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

1. **Too much noise transcribed** → raise `_SPEECH_ENERGY_THRESHOLD` in app.py (toward `0.04`). Trade-off: quiet speakers may get clipped.

2. **Too many speech fragments dropped** → lower `_SPEECH_ENERGY_THRESHOLD` (toward `0.01`).

3. **Too many phantom utterances from Whisper** → raise `NO_SPEECH_THRESHOLD` (max ~0.8). File: `live_transcribe.py`, `batch_transcribe.py`.

4. **Real speech being dropped by Whisper** → lower `LOG_PROB_THRESHOLD` toward `-1.0`. File: same.

5. **Silence flush not triggering** (transcript not updating during speech pauses) → lower `_SPEECH_ENERGY_THRESHOLD` so more frames count as "speech" and fewer advance the silence counter. Or lower `_VAD_SILENCE_MS` to flush sooner.

6. **Q&A flush fires before question is complete** → raise `_QA_MIN_WINDOW_S` (more audio required before silence can trigger) or raise `_AMBIENT_SPEECH_MULTIPLIER` (harder for silence to register as silence). "✓ Done Speaking" button is always the reliable fallback.

   **Q&A flush never fires on silence (noisy room)** → lower `_AMBIENT_SPEECH_MULTIPLIER` toward 2.0. Check the PowerShell log line `[ASK AI] auto-flush (silence, Xs, thresh=Y)` to see what threshold is being used. If ambient_rms is high (e.g. 0.04), the multiplier needs to stay below `speech_rms / ambient_rms`.

   **Q&A hangs with no flush at all** → 20s hard cap will always fire. If even that seems too slow, lower `_QA_MAX_WINDOW_S`.

7. **Fabricated completions on speech that trails off** → add `temperature=0.0` to both `transcribe_audio()` in `live_transcribe.py` and `transcribe_file()` in `batch_transcribe.py`. Disables the temperature fallback chain entirely. Test with genuinely bad audio first.

8. **Initial prompt still echoing** → check that `build_whisper_prompt` returns a value starting with "This is a university lecture." Search for `_format_whisper_prompt` in `live_transcribe.py`.

---

## 8. Module system rebuild — complete handoff

This section covers the module layer rewrite. Read it if you are touching anything in the Modules tab, the Chat doc filter, the Live Lecture source picker, or `module_store.py`.

### What a module is

A module is a named, **ordered** list of bare document filenames — Canvas-style. The real document bytes live once in `docs/`. A module is a reference, not a copy. The same file can appear in multiple modules. Order is meaningful and preserved.

---

### Old architecture — what existed before

**`modules.py`** was the CRUD layer. It had these problems:

1. **`add_docs_to_module` used set union.** The old implementation was:
   ```python
   mod["documents"] = sorted(existing | set(filenames))
   ```
   This silently alphabetized the whole document list on every add.

2. **`save_modules` used a dict wrapper.**
   The `modules.json` format was `{"modules": [...]}`. `load_modules()` read `data.get("modules", [])`.

3. **No `set_module_docs`.** No function to replace the list wholesale with a specific order.

4. **No display helpers.** `unified_source_choices()` and `module_dd_choices()` lived in `app.py`.

5. **Five handlers, each mutating inline.**

6. **`modules.json` format was `{"modules": [...]}`**, not a flat list.

---

### New architecture — what replaced it

**`module_store.py`** is the single source of truth. It is the only file that knows about modules. `app.py` imports from it and never reimplements anything.

**`modules.json` format is now a flat `[...]` list.** `load_modules()` returns `data if isinstance(data, list) else []`.

**Key functions in `module_store.py`:**

| Function | What it does |
|---|---|
| `load_modules()` / `save_modules()` | JSON read/write with atomic temp-file replace |
| `create_module(name)` | Creates new module with UUID-based ID, empty doc list |
| `rename_module(id, name)` | Mutates name, saves |
| `delete_module(id)` | Removes by ID, never touches `docs/` |
| `add_docs_to_module(id, filenames)` | Appends to existing list, skips duplicates, preserves existing order |
| `remove_docs_from_module(id, filenames)` | Removes by name, preserves order of remainder |
| `set_module_docs(id, filenames)` | Replaces wholesale — deduplicates, preserves first-seen order. Used by drag-drop UI. |
| `reorder_module_docs(id, ordered_filenames)` | Partial reorder — current docs missing from the list are appended at the end |
| `expand_selection(selection)` | Takes a mixed list of module IDs and bare filenames, returns a flat deduplicated filename list |
| `module_dd_choices()` | Choices for the module selector dropdown: `(name, id)` tuples |
| `unified_source_choices()` | Choices for Chat + Live Lecture pickers: `[Module] name → id` entries first, then `[Doc] filename → filename` |

**`_valid_doc_filenames()`** is a private helper with mtime-based caching. It reads `manifest.json`, normalizes keys (full relative paths like `docs\file.pdf`) to bare filenames via `PureWindowsPath`, and returns a set. Cache is invalidated when `manifest.json`'s mtime changes.

---

### The `_module_refresh()` pattern in app.py

Every mutation handler ends with one call:

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

The one exception is `delete_module_stage2`, which passes `value=new_val` on the `module_dd` update and calls `_module_refresh()[1:]`.

---

### The drag-drop UI

Four-pane HTML panel built with SortableJS:
- **Pane 1 (Classes)** — flat list of class names ("Unassigned" first). Click to select. Each class row is a SortableJS drop target (`group: pmaclasses, put:true, sort:false`) for modules dragged from pane 2.
- **Pane 2 (Modules)** — modules belonging to the selected class. Click to select. The module list is a SortableJS drag source (`group: pmaclasses, pull:true, sort:false`). No reordering within pane 2.
- **Pane 3 (Documents)** — the selected module's docs, numbered. Drag to reorder (`group: pmadocs`), drag from library to add, click × to remove. Drop-target highlight via `.pma-lib-dragging` CSS class on `#profai-mod-manager`.
- **Pane 4 (Document Library)** — all ingested docs, filter input. SortableJS `pull:'clone'` source. Items already in the selected module greyed out with ✓.

All four panes use `flex:1 1 0;min-width:0` — equal widths, text truncates with ellipsis.

**Selection state rules:**
- Selecting a class always resets module selection to the first module in that class (or null if empty).
- On load: first class (Unassigned) is selected; its first module is auto-selected.
- JS variables: `SEL_CLASS` (class_id string, `""` for Unassigned), `SEL_MOD` (module_id or null), `DOCS` (live doc list for selected module).

**How the Python/JS bridge works:**

1. `_render_drag_panel()` in `app.py` builds the entire HTML string. Module/library state embedded as JSON in `<script type="application/json" id="pma-data">`. JS logic in `<script type="text/x-pma" id="pma-init">`. Execution bootstrapped by `<img src="x" onerror="...">` (onerror fires from innerHTML-inserted elements; plain script tags do not execute via innerHTML). When Gradio re-renders after a mutation, the new `<img>` fires onerror again — no stale state.

2. Both drag types share one bridge: a hidden `gr.Textbox(elem_id="drag_state")` paired with a hidden `gr.Button(elem_id="drag_trigger")`. JS writes the payload to `#drag_state textarea`, then calls `.click()` on `#drag_trigger`. The button's `.click(fn=on_drag_change, inputs=[drag_state], js="() => read #drag_state value and return it")` is what actually reaches Python. Doc drags write `{"module_id": "mod_xxx", "ordered_filenames": [...]}`; class-assignment drags write `{"module_id": "...", "class_id": "cls_..."|null}`. `on_drag_change` discriminates by key: `ordered_filenames` → `set_module_docs`, `class_id` → `assign_module_to_class`.

3. Why the button, not `drag_state.input()`: Gradio's Svelte frontend ignores a programmatic textarea value-set + synthetic `input` event, so a `.input()` binding on the textbox never fires. A real `.click()` on a Gradio Button does fire, and its `js=` return value is fed to the Python `fn` as the input. There is no separate class bridge — an earlier `class_drag_state` textbox + `on_class_drag_change` handler were removed as dead code.

4. When a module is dropped on a class row in pane 1 (native HTML5 DnD — `drop` event on the row), the module id is read from `dataTransfer` (set on `dragstart` in pane 2) and `pushClass()` fires through the shared `drag_state` / `#drag_trigger` bridge. Python calls `assign_module_to_class()` and returns `_module_refresh()`.

**SortableJS:** CDN `https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js`. If CDN is unavailable, drag reorder is disabled with a console warning; panes still render, × still removes items.

---

### Files changed and deleted

| File | Status | What happened |
|---|---|---|
| `module_store.py` | **New** | The canonical module layer. |
| `modules.py` | **Deleted** | Old CRUD layer. Replaced entirely. |
| `modules.json` | **Migrated** | Format changed from `{"modules": [...]}` to flat `[...]`. All existing modules preserved. |
| `app.py` | **Modified** | Import swapped to `module_store`. Added `_render_drag_panel()`, `_module_refresh()`, `on_drag_change()`. Replaced Modules tab UI. |
| `voice_qa.py` | **Unchanged** | Flagged as potentially unused in the audit but explicitly kept per user instruction. Do not delete. |

---

### What to watch for when testing

**The drag panel re-renders on every mutation, but selection is preserved.** The selected class and module ids are saved to browser localStorage (`pma-sel-class`, `pma-sel-mod`) by `saveSel()` and restored by the init IIFE on every rebuild. If the stored module was dragged to a different class, the restore follows the module to its new class. Falls back to first class / first module when the stored ids no longer exist. (Before 2026-06-12 the selection reset to the first class on every drop, which made multi-document adds painful.)

**SortableJS loads from CDN.** On first load, drag handles won't initialize until the CDN script loads. If the lab machine has no internet access, SortableJS fails silently. Download `sortablejs@1.15.2/Sortable.min.js` and inline it in `_render_drag_panel()` for offline use.

**The `[Doc]` prefix on unified_source_choices.** The Chat doc filter and the Live Lecture source picker show items as `[Doc] filename.pdf`. The dropdown value is still the bare filename.

**Module IDs in linked_documents.** `expand_selection()` treats anything that is not a recognized module ID as a bare filename and passes it through directly.

**If modules.json is corrupted or deleted.** `load_modules()` returns `[]` on any read error — no crash.

---

## 9. Class layer — complete handoff

A class is a named container that owns modules. Modules belong to at most one class. Classes do not contain documents directly.

### Data model

| File | Format | Notes |
|---|---|---|
| `classes.json` | Flat `[{id, name, created_at}]` | `id` format: `cls_<12 hex>`. Created on first `create_class()` call. |
| `modules.json` | Same flat list, now with `class_id` field | `class_id` defaults to `null` on old records — backward compatible. |

### Key functions added to module_store.py

| Function | What it does |
|---|---|
| `load_classes()` / `save_classes()` | JSON read/write with atomic temp-file replace |
| `create_class(name)` | Creates class record, returns dict |
| `rename_class(class_id, new_name)` | Mutates name, saves |
| `delete_class(class_id)` | Removes class record; sets `class_id=None` on all owned modules. Never deletes modules or documents. |
| `assign_module_to_class(module_id, class_id)` | Sets `class_id` on the module. `None` = Unassigned. |
| `class_tree()` | Returns `[{class_id, name, modules}]`. First entry is always the synthetic "Unassigned" group (`class_id=None`) containing every module whose `class_id` is null or points at a deleted class. Remaining entries are real classes in stored order. |
| `class_dd_choices()` | `(name, id)` tuples for the class selector dropdown |

`create_module()` now accepts an optional `class_id=None` kwarg.

### Four-pane layout (replaces the old left-pane tree)

`_render_drag_panel()` calls `class_tree()` and splits the view across four equal-width panes. Pane 1 shows only class names as a flat click list (no nested tree, no collapse). Pane 2 shows modules for the currently selected class. This is a cleaner separation than the old collapsible tree.

The `class_id` for Unassigned is encoded as `""` in the HTML `data-class-id` attribute and in the JS `SEL_CLASS` variable. `pushClass` converts falsy values to `null`. Python receives `null` as `None` and calls `assign_module_to_class(module_id, None)`.

`window.__pmaSelClass(cid)` is the click handler for class rows (replaced old `__pmaToggleClass`). It sets `SEL_CLASS`, resets module selection to the first module in that class, and re-renders all four panes.

### Class drags use the single shared bridge

There is no separate class bridge. Class-assignment drags write their
`{module_id, class_id}` payload to the same `#drag_state` textbox and fire the same
`#drag_trigger` button as doc drags; `on_drag_change` routes by payload key. An
earlier `class_drag_state` textbox + `on_class_drag_change` handler were tried, found
inert (Gradio ignores synthetic `input` events — see Section 8), and removed.

### Class controls in the Modules tab

Below the existing module controls, a "Classes" section provides:
- New class name textbox + "Create Class" button
- Class dropdown (`class_dd`) for rename/delete
- Rename textbox + "Rename" button
- Two-stage delete: "Delete Class" → warn → "Confirm Delete Class"
- `classes_status_tb` for status messages

Handlers: `create_class_fn`, `rename_class_fn`, `delete_class_stage1`, `delete_class_stage2`. All return `(*_module_refresh(), gr.update(choices=class_dd_choices()), ...)` so both the drag panel and the class dropdown stay in sync.

### What is NOT changed

- Middle pane (module documents) and right pane (document library) are identical.
- `expand_selection()`, `unified_source_choices()`, `module_dd_choices()` are unchanged.
- `ingest.py`, `manifest.json`, `docs/` are untouched.
- Classes are not orderable.
- A module cannot belong to two classes simultaneously.

### JS-in-f-string quoting trap

When writing JavaScript inside a Python f-string, `\'` is a **Python** escape sequence — the backslash is consumed, and the output is a bare `'`. If that `'` appears inside a JS single-quoted string literal (e.g. an onclick attribute building `'string'`), it closes the string early. `new Function(content)` throws a SyntaxError, the `try/catch` in the onerror catches it silently, and nothing renders.

**Rule:** For single quotes inside onclick attribute values being built by JS string concatenation inside a Python f-string, use `&#39;` (HTML entity). The browser decodes it to `'` before evaluating the JavaScript.

```python
# WRONG — Python strips the backslash, bare ' breaks JS string
+ ' onclick="window.__pmaFoo(\''+escA(x)+'\')"'

# CORRECT — &#39; is decoded by browser before JS runs
+ ' onclick="window.__pmaFoo(&#39;'+escA(x)+'&#39;)"'
```

This applies to any inline event handler attribute (`onclick`, `oninput`, etc.) where you need a JS string argument containing a single quote, inside a Python f-string that builds the HTML via `+` string concatenation.

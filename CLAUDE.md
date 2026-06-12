# Prof AI

Local AI assistant for a Baylor neuroscience professor. The system records and transcribes lectures in real time, answers spoken questions during a lecture, and supports research queries against course materials plus recent academic literature.

Everything runs locally on a lab machine with an RTX 4090, Windows 11. External calls: PubMed E-utilities, Semantic Scholar Graph API, and OpenAlex API for literature search (all free, no auth), plus the SortableJS CDN (jsdelivr) loaded by the Modules drag panel, plus the gradio.live tunnel while `share=True`.

## Stack

- **LLMs**: `qwen3:30b-a3b` via Ollama for chat and Q&A; `qwen3:14b` for fast tasks (query reformulation, gap analysis, Whisper vocabulary extraction)
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector store**: ChromaDB, in-process, no server
- **Transcription**: `faster-whisper` `large-v3-turbo` on CUDA float16
- **Audio capture**: browser-side mic via `gr.Audio(sources=["microphone"], streaming=True)` plus an RMS noise gate in app.py. `sounddevice` and `silero-vad` exist only in recorder.py, which is NOT in the live path.
- **TTS**: Piper via the piper-tts Python package, voice `en_US-libritts-high` (tts.py loads only this one; the lessac-medium fallback exists only in dead code voice_qa.py)
- **UI**: Gradio 6, dark theme from ui_theme.py, served at `http://127.0.0.1:7860`, currently exposed via gradio.live tunnel
- **Project root**: `C:\prof-ai` on the lab machine (the dev copy may live elsewhere)

## File layout

```
C:\prof-ai\
├── venv\                       # Python virtual environment
├── docs\                       # source PDFs, docx, pptx ingested into ChromaDB
├── lectures\                   # pre-recorded audio for batch processing
├── transcripts\sessions\       # live lecture session transcripts
├── transcripts\qa_audio\       # synthesized TTS audio files (per Q&A turn)
├── chroma_db\                  # vector store on disk
├── models\piper\               # Piper voice .onnx and .json files
├── manifest.json               # tracks ingested documents (path → mtime)
├── sessions.json               # session records: id, name, linked_documents, start/end times, segment_count, qa_history, conversation_history
├── modules.json                # module records: flat list with id, name, class_id, ordered documents
├── classes.json                # class records: flat list with id, name, created_at
├── ingest.py                   # processes documents into ChromaDB (OCR fallback for scanned PDFs)
├── batch_transcribe.py         # processes pre-recorded audio
├── session.py                  # LectureSession class plus Mode enum
├── recorder.py                 # server-side capture + Silero VAD — NOT in the live path, kept for constants and the legacy CLI
├── live_transcribe.py          # Whisper singleton, vocab initial_prompt, push_audio / push_question_audio entry points
├── query.py                    # RAG query backend for Chat tab plus multi-source literature search
├── gaps.py                     # batch gap analysis (transcript vs linked docs)
├── live_gap.py                 # LiveGapWorker — background gap analysis every 60s during recording
├── module_store.py             # module + class CRUD layer (modules.json, classes.json)
├── manifest.py                 # shared manifest helpers
├── pubmed.py                   # PubMed E-utilities wrapper
├── semantic_scholar.py         # Semantic Scholar Graph API wrapper
├── openalex.py                 # OpenAlex API wrapper
├── tts.py                      # Piper TTS wrapper (PiperTTS.synthesize → WAV bytes)
├── qa.py                       # Voice Q&A and conversation-turn answer generation
├── voice_qa.py                 # DEAD CODE — old piper.exe implementation, not imported anywhere, kept per user instruction
├── migrate_content_types.py    # one-off backfill script for content_type metadata
├── ui_theme.py                 # dark theme, CSS variables, force-dark JS, header HTML
└── app.py                      # Gradio UI: Chat, Live Lecture, Add Materials, Modules, Gaps Analysis tabs
```

There is no text_utils.py yet. It is planned as part of streaming TTS (see UPGRADE_SPEC.md).

## Environment quirks

These will bite you if you don't know about them.

1. **NVIDIA libraries for faster-whisper aren't on Windows PATH by default.** Both `live_transcribe.py` and `batch_transcribe.py` now prepend the nvidia wheel DLL dirs to PATH at import time (the `_nvidia_dirs` block at the top of each), so this is handled automatically. The manual PowerShell version still works as a belt-and-braces step:
   ```powershell
   $env:PATH = "$pwd\venv\Lib\site-packages\nvidia\cublas\bin;$pwd\venv\Lib\site-packages\nvidia\cudnn\bin;" + $env:PATH
   ```
   The libraries are installed via `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. They exist on disk but ctranslate2 can't find them without the PATH entry.

2. **Mic device indexes only matter for the legacy server-side path.** Live recording uses the browser mic, so the OS device selection happens in the browser, not in Python. The old guidance (USB mic at device index 1, MME not WASAPI because WASAPI throws `PaErrorCode -9997`) applies only if you revive `recorder.py` or use `tts.py speak()` server-side. There is no input device dropdown in the current UI.

3. **Qwen3 thinking is controlled via the `think=` parameter on `ollama.chat()`, not prompt prefixes.** Voice Q&A (`qa.py answer_question`) uses `think=True` and reads the trace from `response["message"]["thinking"]`, separate from content. Conversation Fast mode, gap analysis, and literature reformulation use `think=False`. Chat streaming in `query.py` sets nothing (model default). Vocabulary extraction in `build_whisper_prompt()` also uses `think=False` (was a `/no_think` prompt prefix until 2026-06-12). `sanitize_answer()` in qa.py strips any `<think>` blocks that leak into content as a safety net.

4. **Gradio dropdown choices are static at component creation.** Always refresh via explicit `gr.update(choices=...)` calls on relevant events. A browser refresh will NOT pick up new options.

5. **Piper-tts API**: the method to write a WAV file is `synthesize_wav(text, wav_file)`, not `synthesize(text, wav_file)`. The older name returns audio chunks in the current package version and silently does nothing with the wav_file argument. The symptom of getting this wrong is 0-byte audio output (just a 44-byte WAV header).

6. **Audio output is browser-side.** TTS is delivered to the browser via `gr.Audio(autoplay=True)` polled by a 2-second timer. `sounddevice` is NOT used in the Voice Q&A path. The lab machine's audio device is irrelevant for response playback.

7. **Microphone capture is browser-side.** Audio streams from the browser through `mic_audio.stream()` into `handle_audio_chunk()`, gets resampled to 16kHz with scipy, and is gated by RMS energy before reaching Whisper. Remote deployment therefore needs only an HTTPS URL (browsers require a secure context for `getUserMedia`), which is what the Tailscale spec in UPGRADE_SPEC.md provides. The browser typically delivers 48kHz audio, hence the resample.

## State machine

`session.mode` cycles through three states:

- `LECTURE` — utterances are transcribed and appended to the live transcript
- `AWAITING_QUESTION` — set by Ask AI click (or Speak to AI in conversation mode); browser audio accumulates in `state["qa_buffer"]` instead of the transcript path
- `PROCESSING` — Q&A handler is running, incoming audio is discarded, TTS playing

`set_mode()` just sets the enum under a lock, nothing else. The mechanics live in `handle_audio_chunk()` in app.py: in AWAITING_QUESTION the audio accumulates until the fixed window fires (`QA_BUFFER_SECONDS + QA_TAIL_SECONDS = 5.5s` for single-turn Ask AI, or until "Done speaking" / the 60s cap in conversation mode), then `push_question_audio()` transcribes it and fires the registered handler thread, which sets PROCESSING. The old flush-and-timestamp behavior described here previously does not exist in the current code.

## Q&A context assembly

`answer_question()` in qa.py assembles three context blocks before calling the LLM:

1. **Source material chunks** from ChromaDB — top 6, filtered by `source_file in linked_documents` plus `file_type != lecture_audio` (note: the filter is file_type based, not content_type based)
2. **Full lecture transcript** — read from the transcript file via `_load_transcript()`, falling back to in-memory segments. Not truncated.
3. **Last 2 Q&A turns** from `session.qa_history`

The prompt instructs the model to pick GAP MODE (for "what did I miss" questions, comparing transcript against source material itself) or CONTENT MODE (everything else). `session.latest_gap_analysis` is populated by the live gap worker but qa.py never reads it — the model does its own gap comparison from the raw blocks. Conversation mode uses `answer_conversation_turn()` instead, with `session.conversation_history` (last 10 turns) and a fast/deep switch.

## Chat tab features

- Multi-select source filter accepting modules and bare documents (empty selection = search all source documents); module IDs expand via `expand_selection()`
- Literature sources checkbox group: PubMed, Semantic Scholar, OpenAlex, searched in parallel with per-source max results and DOI/title dedup
- One shared query reformulation via qwen3:14b (`reformulate_for_search` in query.py) — broad queries for exploratory questions, no LLM-generated dates
- Recency filter computed in Python using `date.today()`: PubMed gets `mindate`/`maxdate` (2 years), Semantic Scholar and OpenAlex get year ranges
- Citations rendered as `[Doc: filename]`, `[PubMed: PMID]`, `[Semantic Scholar: ID]`, `[OpenAlex: ID]`
- Conversation history (last 6 turns) passed to the LLM for follow-up questions

## ChromaDB metadata and filters

Two values for the `content_type` field: `source_document` (ingested materials) and `lecture_transcript` (live session segments and batch audio). There is also a `file_type` field (`pdf`, `pptx`, `docx`, `txt`, `lecture_audio`, `audio_file`).

Who filters on what:

- Chat (`query.py`) filters `content_type = source_document`
- Voice Q&A (`qa.py`), gap analysis (`gaps.py`, `live_gap.py`): filter `file_type != lecture_audio` plus `source_file in linked_documents` when docs are linked
- Gap analysis reads the lecture transcript from the transcript FILE, not from ChromaDB. Nothing currently queries the `lecture_transcript` chunks back out of ChromaDB.

## Working style

User runs Sonnet via Claude Code for implementation. The model giving specs is Opus, in a separate chat. Specs are pasted into Claude Code as-is.

Writing style:
- Direct, conversational, naturally loose structure
- Short sentences mixed with longer ones
- No em dashes, no semicolons, no over-formalization
- Explain the WHY when there's a learning moment, skip basic CS

When the user is wrong, say so plainly. The user is a recent neuroscience grad (Baylor, May 2026), weak at coding but actively learning, and wants honesty over coddling.

## Implementation conventions

- Use `flush=True` on print statements that need to appear immediately in the PowerShell window (logs in background threads especially)
- Tag log lines with bracketed prefixes: `[Q&A]`, `[PubMed]`, `[ASK AI]`, `[MODE]`, `[STREAM]` etc.
- Background work that touches Gradio components should update state on the session object, and let a polled component on the UI side pick it up. Direct cross-thread Gradio updates are flaky.
- When refactoring, leave the old code paths in place until the new ones are tested. Don't replace working code with untested code in the same commit.

## Known issues and improvements pending

- **Streaming TTS** for low-latency response playback (sentence-by-sentence). Specced in UPGRADE_SPEC.md, not implemented.
- **Tailscale access** for production deployment. Currently using gradio.live tunnel, which times out on long sessions. Specced in UPGRADE_SPEC.md.
- **RAG quality pass** (eval harness, smaller chunks, query expansion, reranking). Specced in UPGRADE_SPEC.md.
- **Cancel button behavior during PROCESSING** — should interrupt TTS and resume lecture mode cleanly. Currently the mode is stuck until TTS finishes.
- **NCBI_EMAIL placeholder** in pubmed.py is still `your-email@example.com`. NCBI asks for a real contact address.

Done and removed from this list: browser-side mic capture, Whisper model bump (now large-v3-turbo), hallucination suppression (thresholds, `language="en"`, and as of 2026-06-12 `temperature=0.0`, Silero `vad_filter`, and a hallucination phrase filter), vocab extraction `think=False`.

## Keeping HANDOFF.md accurate (mandatory)

HANDOFF.md is the source of truth for future zero-context sessions. It must not drift from the code.

For every session that changes code:
1. Code is ground truth. If HANDOFF.md and the code disagree, correct the doc.
2. Before declaring any task complete, update HANDOFF.md as the final step:
   - Update every factual section the change touches (architecture tables, config values, data flows, function tables).
   - Add a dated entry under "## Change log" at the top of that section, newest first, stating what changed, why, and which files were touched.
3. If a change fixes or obsoletes a documented behavior or known issue, update or remove that entry. Do not leave a fixed issue listed as open.
4. If a claim cannot be verified against the code, mark it UNVERIFIED rather than asserting it.
5. Preserve the zero-context framing. A new session must be able to read HANDOFF.md cold and orient.

## Where to look when things break

- **TTS silent**: confirm tts.py internally calls `self._voice.synthesize_wav` (the old `synthesize` method name on PiperVoice silently produces 0-byte WAVs). Confirm voice files exist in `models\piper\`. Check that the duration print shows greater than 0.0s. App code calls `PiperTTS.synthesize(text)`, which wraps this.
- **Q&A answers are vague**: confirm `linked_documents` is non-empty (Ask AI is disabled without it). Read the `[Q&A] context sizes` diagnostic print — a doc context of 0 chars means retrieval found nothing for the linked docs.
- **Live Transcript shows gibberish or YouTube outros**: wrong mic selected in the BROWSER (capture is browser-side now), mic signal too weak for the RMS gate, or Whisper hallucinating on near-silence. Classic hallucination phrases: "Thank you for watching", anime references, random non-sequiturs. Knobs: `_SPEECH_ENERGY_THRESHOLD` in app.py, `NO_SPEECH_THRESHOLD` / `LOG_PROB_THRESHOLD` in live_transcribe.py.
- **Literature search returns nothing useful**: read the `[Literature] reformulated query` line in the log. Likely over-narrow or has invented constraints not in the user question.
- **Frontend shows "Unexpected token '<'"**: backend threw an unhandled exception and Gradio returned HTML. Read the PowerShell terminal for the Python traceback.
- **Mode stuck in AWAITING_QUESTION or PROCESSING**: should auto-recover via Cancel or after TTS completes. If not, restart `app.py`.
- **gradio.live tunnel timeouts**: tunnel is unreliable for long sessions. Use `http://127.0.0.1:7860` when on the lab machine, or set up Tailscale.

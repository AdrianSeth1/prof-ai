# Prof AI — Project Notes

Running notes on the local AI assistant for the professor. What's built, what's pending, decisions and their rationale, and a troubleshooting cheatsheet.

For Sonnet/Claude Code, see `CLAUDE.md` in this same directory.

## What this is

A local AI assistant that:
- Records and transcribes lectures in real time
- Answers "did I miss anything?" during a lecture by comparing what's been said against the linked slides
- Lets the prof do research queries against course materials plus recent PubMed literature
- Runs entirely on the lab machine (RTX 4090, Windows 11), no cloud LLMs

Project root: `C:\prof-ai`

## What's working

### Document handling
- Ingestion of PDFs, docx, pptx via the Add Materials tab
- Documents stored in ChromaDB with a `content_type` metadata field (`source_document` vs `lecture_transcript`)
- Manifest tracks ingested documents
- Multi-select document filter in both Live Lecture and Chat tabs

### Live lecture pipeline
- Real-time audio capture via USB mic (device 1, MME)
- Silero VAD for utterance detection
- Faster-whisper "small" transcribes on GPU (~50-200ms per utterance)
- Session names plus linked documents persisted to `sessions.json`
- Live Transcript panel updates incrementally
- Live Gap Analysis panel runs every 60 seconds, lists slide topics not yet covered

### Voice Q&A
- Ask AI button flips state to AWAITING_QUESTION and flushes the audio buffer
- Next utterance is captured as the question, not appended to the lecture transcript
- Mic pauses while TTS plays so the system doesn't record its own voice
- Q&A handler pulls three context blocks: source docs, lecture transcript, current gap analysis output
- Piper synthesizes via `en_US-libritts-high`
- Audio plays in the browser via `gr.Audio(autoplay=True)`, polled by a 2-second timer
- Q&A history persisted to `sessions.json` per session

### Chat tab (research mode)
- Multi-select document filter
- PubMed toggle for recent literature
- Query reformulator (qwen3:14b) converts vague user questions into proper PubMed search queries
- Date range computed in Python (avoids stale dates from LLM training cutoff)
- Conversation history (last 6 turns) passed into LLM prompt
- Citations: `[Doc: filename]` and `[PubMed: PMID]`

### Gap analysis
- Background thread compares the running transcript to linked documents every 60 seconds
- Output displayed in Live Gap Analysis panel
- Same output stored on `session.latest_gap_analysis` and fed into Q&A handler

## What's pending

### Real-time and quality
- **Streaming TTS** — speak the answer sentence by sentence as the LLM generates it, instead of waiting for the full response. Spec written, not yet built. Would drop perceived latency from ~15s to ~2s.
- **Browser-side mic capture** — currently the mic is on the lab machine. For true remote use (prof's office in one building, lab machine in another), audio capture needs to move to the browser via `gr.Audio(sources=["microphone"], streaming=True)`. Bigger refactor.
- **Whisper "medium" model** — better transcription quality than "small". Negligible speed cost on the 4090.
- **Whisper hallucination suppression** — set `no_speech_threshold=0.6` and `condition_on_previous_text=False` to reduce phantom utterances on weak input.

### Deployment
- **Tailscale or LAN access** instead of gradio.live tunnel. The tunnel adds latency and exposes content through third-party infrastructure.
- **Startup script** so the prof doesn't have to manage Python and Ollama himself.
- **Browser mic for remote setup** (depends on the above).

### Polish
- Cancel button during PROCESSING — interrupt TTS and resume lecture mode cleanly
- Visual indicator when Ask AI is armed but waiting for question
- Audio device selector in the UI (currently the dropdown shows the right device but it's still up to the user to select it before starting)
- Wake on lab machine or always-running daemon mode

## Decisions and rationale

### Why en_US-libritts-high for TTS
More natural prosody than the default lessac-medium. ~100MB on disk. Synthesis is fast enough on the 4090 that the quality is worth the slight size cost. Lessac is kept as a fallback in `models/piper/`.

### Why PubMed instead of arxiv
Prof is neuroscience focused. PubMed is the right index for biomedical literature. No API key required. Sortable by date, filterable by `mindate`/`maxdate`. Arxiv would be a good add for preprints but PubMed covers the main need.

### Why qwen3:14b for query reformulation
Reformulating a user question into a PubMed query is a small, structured task. 14b is faster than 30b and the output quality is fine for this scope. Keeps 30b available for the main chat/Q&A generation.

### Why Python computes the date range, not the LLM
LLMs have stale training data and can't reliably know what "recent" means in real time. We compute `mindate` and `maxdate` in Python using `date.today()` and pass them as parameters to PubMed esearch. Cleaner separation of concerns.

### Why mic pauses during TTS
Without it, the system records its own voice through the room speakers and treats it as more lecture content or another question. Pause duration is estimated from `len(audio_bytes) / (sample_rate * 2)` with a 0.5-1.0s buffer for browser playback latency.

### Why content_type metadata in ChromaDB
Lecture transcripts and source documents share the same collection but serve different purposes. The Chat tab filters to `source_document` only (so prof's queries don't get answered with his own past lectures). The gap analysis flow reads `lecture_transcript` for the current session. Same store, different filters.

### Why three context blocks in the Q&A handler
"Did I miss anything?" needs to compare transcript against slides. A normal RAG query against just the docs can't answer it. By including transcript + gap analysis + doc chunks, the same handler can answer both gap-style questions and content-style questions. The prompt instructions tell the model which context to prioritize for which question type.

### Why browser audio output (and not server-side sounddevice)
The prof's deployment plan is to access from his laptop, with the lab machine in another room. Server-side sounddevice playback would play on the lab machine and never reach him. Browser audio via `gr.Audio(autoplay=True)` works regardless of where the server is physically.

### Why polling-based audio update instead of direct event chaining
Background threads can't reliably update Gradio components mid-flight. The cleanest pattern is: background thread updates state on the session object, and a polled component on the UI side reads from session state. Slight latency cost (1-2s for the poll cycle), but reliable across browsers.

## Quirks worth knowing

### Whisper hallucinations on silence
When the mic signal is weak or quiet, Whisper generates plausible-sounding phrases from its training data. The classic offenders are YouTube-style outros ("Thank you for watching, I'll see you in the next video"), anime fragments, and random non-sequiturs ("Vancouver furry convention", "Aranyan"). If you see these in your transcript, the fix is better mic signal, not a better model.

### Gradio dropdown updates
Choices baked in at component creation. Always update via explicit `gr.update(choices=...)` on the relevant event handlers. Refreshing the page won't pick up new options.

### Mode-aware audio buffer flush
The state machine flushes the audio buffer when entering AWAITING_QUESTION so any in-flight utterance from before the click gets dropped. Without this, you get instant flips from AWAITING to PROCESSING because the recorder finished processing audio captured during LECTURE mode.

### gradio.live tunnel and latency
Audio paths through gradio.live add ~200-800ms of latency and the tunnel sometimes returns HTML error pages when busy (causing the dreaded "Unexpected token '<'" error in the browser). Production deployment should use Tailscale or direct LAN access. The tunnel is fine for development testing.

### piper-tts API change
The package renamed `synthesize` to `synthesize_wav` for the method that writes WAV files. The old `synthesize` now returns audio chunks for streaming. If you ever see TTS audio that's 44 bytes (just the WAV header) and the duration logs as "0.0s", you're calling the old method.

## Where to look when things break

- **TTS silent or "audio ready (0.0s)"**: `tts.py` is calling `synthesize` instead of `synthesize_wav`, or the voice file is missing or corrupt.
- **Q&A answers are vague or generic**: `linked_documents` is empty, or `session.latest_gap_analysis` hasn't been populated yet. Check the diagnostic prints of context sizes.
- **Live Transcript shows random English phrases**: input device is wrong (should be 1, MME) or mic signal too weak. Whisper is hallucinating.
- **PubMed returns nothing useful**: read the reformulated query in the log. Likely over-narrow or has invented constraints not in the user question.
- **Frontend shows "Unexpected token '<'"**: backend threw an unhandled exception that returned HTML. Find the traceback in the PowerShell window.
- **Mode stuck in AWAITING_QUESTION or PROCESSING**: should self-recover. If not, click Cancel or restart `app.py`.
- **Ollama model errors (404)**: a required model isn't pulled. Run `ollama list` to confirm `qwen3:30b-a3b`, `qwen3:14b`, and `nomic-embed-text` are all present.
- **NVIDIA library errors at startup**: the PATH env var wasn't set before launching. See `CLAUDE.md` for the PowerShell snippet.

## Startup procedure

From a fresh PowerShell window:

```powershell
cd C:\prof-ai
.\venv\Scripts\Activate.ps1
$env:PATH = "$pwd\venv\Lib\site-packages\nvidia\cublas\bin;$pwd\venv\Lib\site-packages\nvidia\cudnn\bin;" + $env:PATH
python app.py
```

Then open `http://127.0.0.1:7860` in a browser.

For a lecture session:
1. Live Lecture tab
2. Set Input device to device 1 (MME USB mic)
3. Select the deck(s) for this lecture in Source material
4. Optionally enter a session name
5. Click Start Lecture
6. Wait ~60 seconds for the gap analysis thread to populate before clicking Ask AI

## Ollama models required

```
ollama pull qwen3:30b-a3b
ollama pull qwen3:14b
ollama pull nomic-embed-text
```

Verify with `ollama list`.

## Piper voices

Currently installed in `C:\prof-ai\models\piper\`:
- `en_US-libritts-high.onnx` (default) — natural, audiobook-trained voice
- `en_US-lessac-medium.onnx` — fallback, smaller and faster

To add a new voice:
```powershell
cd C:\prof-ai\models\piper
python -m piper.download_voices <voice_name>
```

Or download `.onnx` and `.onnx.json` files directly from `https://huggingface.co/rhasspy/piper-voices`.

# Upgrade specs: streaming TTS, Whisper, RAG quality, remote deployment

Written 2026-06-11. Four independent specs. Paste ONE spec at a time into Claude Code
along with this preamble. Do not paste all four at once, each is a full work session.

Preamble for every paste: "Read CLAUDE.md and HANDOFF.md first. Code is ground truth.
Leave old code paths in place until the new ones are tested. Update HANDOFF.md
(change log entry plus every factual section touched) before declaring done."

Current reality check, verified against code on 2026-06-11 (CLAUDE.md and HANDOFF.md
were corrected against code the same day, so the three docs now agree):

- Whisper is already `large-v3-turbo` on CUDA float16 with `language="en"`,
  `no_speech_threshold=0.7`, `log_prob_threshold=-0.8`, `condition_on_previous_text=False`,
  `beam_size=5`, and a vocabulary-based `initial_prompt` built from linked docs
  (live_transcribe.py lines 34-61, 188-196).
- Mic capture is already browser-side via `gr.Audio(sources=["microphone"], streaming=True)`
  and `handle_audio_chunk` in app.py.
- TTS is whole-answer synthesis: `_qa_handler` calls `answer_question()` (blocking,
  full generation), then `tts.synthesize(answer)` (full WAV), then one file is polled
  by a 2 second timer. Latency is full LLM time plus full TTS time before any audio.
- Voice Q&A retrieval (qa.py) filters `file_type != lecture_audio`, not content_type,
  and includes no page/slide labels in chunk headers. Chat retrieval (query.py) filters
  `content_type = source_document` and DOES label page/slide. Relevant to SPEC 3.

---

## SPEC 1 — Streaming TTS (sentence-by-sentence playback)

### Why

The professor asks a question mid-lecture and waits in silence for the entire answer
to generate and synthesize. With qwen3:30b thinking enabled that is easily 10-25
seconds of dead air in front of a class. Streaming the answer sentence by sentence
gets the first audio out as soon as the first sentence exists, roughly 3-5 seconds.

### Architecture

Pipeline with three stages connected by the existing polling pattern. Background
thread does LLM streaming and TTS. UI timer polls a queue. No direct cross-thread
Gradio updates (that rule already exists in CLAUDE.md and it stays).

```
ollama stream → sentence splitter → Piper per sentence → audio queue → UI poll timer
```

### Changes, file by file

**1. New file `text_utils.py`** (planned in CLAUDE.md's file layout, does not exist
yet, create it):

```python
import re

_ABBREV = {"dr", "mr", "mrs", "ms", "prof", "vs", "etc", "e.g", "i.e", "fig", "et al"}
_TERMINAL = re.compile(r'([.!?])(\s|$)')

def pop_sentences(buffer: str, min_chars: int = 25) -> tuple[list[str], str]:
    """Split complete sentences off the front of buffer.
    Returns (complete_sentences, remaining_buffer).
    A sentence is complete when terminal punctuation is followed by whitespace
    and the candidate is at least min_chars long and does not end in a known
    abbreviation. min_chars stops Piper being fed fragments like 'Yes.'
    which sound clipped and waste a synthesis call."""
```

Implementation notes for the body: scan with `_TERMINAL`, check the word before the
punctuation against `_ABBREV` (lowercase, strip trailing period), check digit-period-digit
(decimals like 3.5 must not split). Return whatever full sentences exist, keep the
remainder. Write 8-10 unit-style asserts at the bottom under `if __name__ == "__main__":`
covering abbreviations, decimals, multiple sentences in one push, and no-split cases.

**2. `qa.py`: add a streaming variant, do not modify `answer_question`**

```python
def answer_question_stream(question, linked_docs, session=None):
    """Yields ("token", str) for each content token, then ("done", (answer, reasoning, sources)).
    Same prompt assembly as answer_question. Uses ollama.chat(stream=True, think=True)."""
```

Critical detail: with `think=True` and `stream=True`, ollama yields chunks where
reasoning arrives in `chunk["message"]["thinking"]` and answer text arrives in
`chunk["message"]["content"]`. Only yield content tokens to the caller. Accumulate
thinking separately for the final tuple. Run `sanitize_answer` on the accumulated
content at the end as the existing safety net, but do NOT feed `<think>` text to TTS,
so also guard the token path: if a content token stream ever contains `<think>`,
suppress output until after `</think>`.

**3. `app.py`: replace the single audio slot with a queue**

Replace `_qa_audio_path` (single path plus lock) with:

```python
_qa_audio_queue: collections.deque[str] = collections.deque()
_qa_audio_lock = threading.Lock()          # guards the deque
_qa_busy_until: float = 0.0                # time.time() when current clip finishes
_qa_generation_done: bool = False          # producer finished
```

New `_qa_handler` flow (keep the old one as `_qa_handler_legacy` until tested):

1. Set status, pause mic, exactly as now.
2. Iterate `answer_question_stream`. Feed tokens into a buffer, call
   `pop_sentences(buffer)` after each token. For each complete sentence call
   `tts.synthesize(sentence)`, save with `_save_qa_audio`, append path to the queue.
3. On "done": synthesize any non-empty remainder, call `session.add_qa_entry` with
   the full answer exactly as now, set `_qa_generation_done = True`.
4. Do NOT start the resume timer here. Resume logic moves to the poll (step 5).

`poll_qa_audio` becomes the scheduler. Poll it on a NEW dedicated `gr.Timer(value=0.5)`
(leave the existing 2 second timer for everything else):

```python
def poll_qa_audio():
    now = time.time()
    with _qa_audio_lock:
        if _qa_audio_queue and now >= _qa_busy_until:
            path = _qa_audio_queue.popleft()
            duration = _wav_duration_s(path)   # wave module, frames / framerate
            _set_busy_until(now + duration + 0.3)
            return gr.update(value=path)
        if _qa_generation_done and not _qa_audio_queue and now >= _qa_busy_until:
            _finish_qa_playback()   # resume mic, mode → LECTURE, status, reset flags
    return gr.update()
```

The 0.3 second gap between clips covers browser load time. If clips audibly overlap
raise it, if pauses feel long lower it. Print the chosen duration per clip with a
`[STREAM]` log prefix.

`cancel_qa` must also: clear the queue, set `_qa_busy_until = 0`, set
`_qa_generation_done = False`. The in-flight ollama stream should be abandoned by
checking a cancel flag inside the token loop each iteration.

**4. Same pattern for conversation Fast mode** in `_conversation_handler`. Deep mode
keeps whole-answer synthesis since the prof reads it first anyway.

### Acceptance

- Ask a question with a linked doc. First audio starts in under 5 seconds. Console
  shows `[STREAM]` lines per sentence with durations.
- Sentences play in order with no overlap and no clip cut off mid-word.
- Mic resumes only after the last clip finishes. Status returns to recording.
- Cancel during playback stops further clips and resumes the mic.
- Decimals and "Dr. Smith" do not cause mid-sentence splits (run text_utils.py asserts).

### Do not

- Do not use `gr.Audio(streaming=True)` for output. The producer is a background
  thread, not the event handler, so generator-based streaming does not fit this
  architecture. The queue keeps the existing polling pattern.
- Do not remove the legacy handler in the same commit.

---

## SPEC 2 — Whisper polish (small, most of this is already done)

### Why

The big items (model bump, English pin, hallucination thresholds, conditioning off)
are already in live_transcribe.py. What remains is a last hallucination net plus two
small correctness items HANDOFF.md lists as open issues.

### Changes

In `live_transcribe.py`, the `model.transcribe(...)` call:

1. Add `vad_filter=True, vad_parameters={"min_silence_duration_ms": 250}`. The app
   already gates on RMS energy in app.py, but Silero VAD inside faster-whisper
   catches breath noise and mic rumble that pass an energy gate.
2. Add a hallucination phrase filter where segments are accepted (the same place
   MIN_WORDS and MIN_AVG_LOGPROB are checked). Module-level constant:

```python
HALLUCINATION_PHRASES = (
    "thank you for watching", "thanks for watching", "please subscribe",
    "see you in the next video", "subtitles by", "like and subscribe",
)
```

Drop any segment whose lowercased text contains one of these AND has
`no_speech_prob > 0.3`. The probability guard means a professor who literally says
"thanks for watching" in a real sentence still gets transcribed. Log drops with a
`[Whisper]` prefix so false positives are visible.

3. Consider `temperature=0.0` on both `transcribe()` calls (live_transcribe.py and
   batch_transcribe.py). The default fallback schedule retries bad segments at higher
   temperatures and can fabricate plausible completions. Known risk, documented in
   HANDOFF.md section 5: truly bad audio then produces empty output instead of
   plausible-but-wrong. Test with deliberately bad audio before committing.

4. Fix the vocab extraction think setting: `build_whisper_prompt()` uses a `/no_think`
   prompt prefix on the qwen3:14b call, which is unreliable. Set `think=False` on the
   `ollama.chat()` call and drop the prefix. (HANDOFF.md section 5, medium priority.)

5. Update HANDOFF.md: mark the fixed items resolved, remove them from section 5.

### Acceptance

- Lecture transcription still works end to end.
- With the mic on and nobody speaking for 60 seconds, zero phantom segments appear.
- Vocab extraction log no longer shows any `<think>` leakage and runs no slower.

---

## SPEC 3 — RAG answer quality

### Why

Both answer paths (qa.py for voice, query.py for chat) do single-embedding top-6
retrieval over 500-word chunks with no reranking. Three weaknesses, in order of
impact: chunks are too big so the embedding averages away slide-level detail,
one query embedding misses paraphrases, and there is no second-pass relevance check.

### Measure first, then change

**Step 0, build the eval harness before touching retrieval.** New file `eval_rag.py`
plus `golden_qa.json`. The JSON holds 15-25 entries, written by hand from real course
materials: `{"question": ..., "expected_source_file": ..., "expected_keywords": [...]}`.
The script runs retrieval only (no LLM), reports hit rate (expected file in top k)
and keyword recall, prints a per-question table. Run it, commit the baseline numbers
in a comment at the top of the file. Every change below gets re-run against it.
The reason: retrieval tuning without measurement is guessing, and a 4090 makes the
eval loop fast enough to be painless.

**Step 1, chunking.** In `ingest.py`: `CHUNK_WORDS = 500, OVERLAP_WORDS = 50` becomes
`CHUNK_WORDS = 180, OVERLAP_WORDS = 40`. This mainly affects PDFs, docx, txt, and
batch audio transcripts (batch_transcribe.py reuses chunk_text). PPTX is already in
good shape: extract_pptx yields one section per slide with `page_or_slide` metadata,
and chunk_text only splits a slide that exceeds the cap, so slides are effectively
per-slide chunks today. Do not change the pptx path. PDFs chunk per page section the
same way, so the win is within long pages and unpaged formats. Re-ingest everything
afterwards (delete chroma_db, re-run ingest — manifest skips by mtime, so also delete
manifest.json or the re-ingest will skip unchanged files). Re-run eval.

**Step 2, query expansion.** In both `qa.py:_retrieve_chunks` and the equivalent in
`query.py`: generate one reformulation of the question with qwen3:14b, `think=False`,
prompt "Rewrite this question as a statement of the answer's likely content, using
domain vocabulary. One line." Embed both the original and the rewrite, query ChromaDB
with both embeddings (n_results=12 each), union by chunk id. The voice path may skip
this when the 14b call would add latency, gate it behind a module constant
`EXPAND_QUERIES = True` so it can be flipped per path. Re-run eval.

**Step 3, reranking.** Retrieve 24 candidates (union from step 2), rerank, keep 6.
Use a cross-encoder, it runs locally and fits the no-cloud constraint:

```
pip install sentence-transformers
model: BAAI/bge-reranker-v2-m3, device cuda
```

Wrap in a new `rerank.py` with lazy singleton load (same pattern as the TTS and
Whisper singletons) and `rerank(question, chunks, top_k=6) -> list[int]`. Apply in
chat always. In voice Q&A apply it too (one cross-encoder pass on 24 pairs is tens of
milliseconds on a 4090), but skip it in conversation Fast mode where every millisecond
counts. Re-run eval, expect the biggest gain here.

**Step 4, context assembly hygiene** in both paths:

- Drop near-duplicate chunks: if two retrieved chunks share more than 60 percent of
  their word set, keep the higher ranked one. Overlapping windows from step 1 make
  duplicates likely.
- Port query.py's location labeling into qa.py. `build_context()` in query.py already
  emits `[Source: file | slide 12]` headers from the `page_or_slide` metadata. qa.py's
  `_retrieve_chunks()` only emits `[source_file]`. Give the voice path the same labeled
  headers so spoken answers can reference slides precisely.
- Order final chunks by (source_file, position) not by score. The model reads
  coherent document order better than relevance-shuffled fragments.

### Acceptance

- eval_rag.py hit rate improves over the committed baseline at each step, or the step
  is reverted (the harness exists precisely to allow honest reverts).
- Voice Q&A latency to first token does not regress by more than 0.5 s (log it).
- Voice Q&A chunk headers now carry page/slide labels (chat already had them).

### Do not

- Do not swap the embedding model in the same pass. One variable at a time, the eval
  can justify that experiment later.
- Do not raise final context beyond 6 chunks for the voice path. Spoken answers are
  2-4 sentences, more context just slows thinking.

---

## SPEC 4 — Remote deployment with Tailscale (free)

### Why

gradio.live tunnels time out on long sessions (documented in CLAUDE.md) and expose a
public URL guarded by one shared password. Tailscale gives a stable private HTTPS URL
reachable only by enrolled devices, free for up to 3 users and 100 devices. HTTPS
matters specifically because browsers only allow microphone capture in a secure
context, and the mic is browser-side.

### One-time setup (human steps, not code)

1. Create a free Tailscale account (Google or Microsoft login works).
2. Install Tailscale on the lab machine (Windows installer from tailscale.com),
   sign in. Install it on the professor's laptop, sign in to the SAME account
   (or invite his account to the tailnet, 3 users are free).
3. On the lab machine in PowerShell:

```powershell
tailscale serve --bg 7860
tailscale serve status
```

   This publishes `https://<machine-name>.<tailnet>.ts.net/` with a certificate
   Tailscale provisions automatically, proxying to localhost:7860. `--bg` keeps it
   across reboots.

### Code changes in `app.py` launch()

```python
build_ui().launch(
    server_name="127.0.0.1",   # was 0.0.0.0, see why below
    server_port=7860,
    theme=THEME,
    css=CSS,
    js=FORCE_DARK_JS,
    share=False,               # gradio.live tunnel no longer needed
    auth=("prof", "<NEW PASSWORD, not the one in git history>"),
)
```

Why `127.0.0.1`: with Tailscale serve proxying from the tailnet, the app no longer
needs to listen on all interfaces. Binding loopback means the ONLY ways in are the
lab machine itself and the tailnet. Anyone on the campus LAN scanning ports finds
nothing. Keep `auth` anyway as a second layer.

### Use

The professor opens `https://<machine-name>.<tailnet>.ts.net/` on any enrolled
device, logs in with the auth credentials, grants mic permission once. The URL is
stable across sessions and reboots.

### Acceptance

- Mic recording works from a different machine on the tailnet over the https URL
  (this proves the secure-context requirement is satisfied).
- The app is NOT reachable via `http://<lab-machine-LAN-ip>:7860` from another
  machine on the same LAN.
- A lecture runs 60+ minutes without the connection dropping (the gradio.live
  failure mode this replaces).

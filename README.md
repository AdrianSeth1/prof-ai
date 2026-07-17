# prof-ai

**A local, voice-interactive lecture assistant that listens to class, answers questions out loud, and stays grounded in the professor's own materials, with no student data ever leaving the room.**

prof-ai runs entirely on local consumer hardware. There are no per-query API costs and nothing is sent to a third party, which is what made it adoptable with zero budget and no procurement process. A Baylor University professor is using it in class right now, and the university is looking at wider adoption.

---

## Why it exists

Students fall behind in lectures for boring, fixable reasons: a term goes undefined, a step gets skipped, a question feels too small to interrupt for. A cloud AI assistant could help, but pointing one at a live classroom means streaming students' voices and a professor's unpublished materials to someone else's servers. That's a non-starter for most instructors, and rightly so.

prof-ai takes the other path. Everything runs locally. The professor's slides and readings are the knowledge base, the microphone audio is transcribed on-device, and answers are spoken back through a simple web interface. Nothing leaves the machine.

## What it does

- **Live transcription.** Captures the lecture in real time with `faster-whisper`.
- **Gap detection.** Flags moments where an explanation jumps ahead or leaves a concept undefined, so they can be surfaced or reviewed.
- **Grounded Q&A.** Answers questions using retrieval over the professor's own course materials (RAG), not the open internet, so responses stay on-syllabus.
- **Voice answers.** Speaks responses aloud through `Piper` TTS for a hands-free, in-class experience.
- **Multi-turn conversation mode.** Supports follow-up questions in a single thread, not just one-shot lookups.
- **Optional literature lookup.** Can pull supporting references from PubMed, OpenAlex, and Semantic Scholar when a question reaches beyond the course materials.

The system is organized as a lecture / question / processing state machine, designed and iterated on directly with the professor using it.

## How it works

```
Microphone ──▶ faster-whisper ──▶ live transcript
                                        │
              course materials ──▶ ChromaDB (embeddings / retrieval)
                                        │
                    student question ──▶ retrieve context ──▶ local LLM (Ollama)
                                        │
                              answer ──▶ Piper TTS ──▶ spoken response
```

Everything in that pipeline runs on local hardware. The LLM is served through **Ollama**; retrieval is backed by **ChromaDB** over embeddings of the professor's uploaded content.

## Tech stack

| Layer | Choice |
| --- | --- |
| Language model | Local LLM via **Ollama** |
| Retrieval | **ChromaDB** vector store over course materials |
| Speech-to-text | **faster-whisper** |
| Text-to-speech | **Piper TTS** |
| Backend | **Python** |
| Interface | Web UI (TypeScript) |
| External evidence | PubMed, OpenAlex, Semantic Scholar |

## Getting started

> Requires a machine capable of running a local LLM through Ollama, plus Python 3.11+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull a local model with Ollama (example)
ollama pull qwen3:30b

# 3. Ingest the course materials into the vector store
python ingest.py

# 4. Start the assistant
python app.py
```

Then open the web interface and start a lecture session.

## Status

In active classroom use. The current roadmap focuses on tightening gap-detection accuracy, refining the live-lecture layout, and expanding multi-turn "collaborate" mode. See `HANDOFF.md` for the running build log.

---

Built by [Aryamaan Seth](https://www.linkedin.com/in/aryamaanseth/), a neuroscience grad who builds local-first AI and teaches the people around him to use it.

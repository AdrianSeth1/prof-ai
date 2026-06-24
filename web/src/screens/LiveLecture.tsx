import { useCallback, useEffect, useRef, useState } from 'react'
import SourceFilter from '../components/SourceFilter'
import type { SourceData } from '../components/SourceFilter'
import FindingCard from '../components/FindingCard'
import type { Finding } from '../components/FindingCard'

interface Props {
  onToast: (title: string, desc?: string) => void
}

type UIMode = 'lecture' | 'awaiting' | 'processing' | 'speaking' | 'paused'

interface SessionInfo {
  session_id: string
  name: string
}

interface QAEntry {
  question: string
  answer: string
  sources: string[]
  reasoning: string
}

interface TxSegment {
  text: string
  ms: number
  confidence: number
}

// ── AudioWorklet source — inlined as blob (same pattern as spike) ─
const WORKLET_SRC = `
class AudioProc extends AudioWorkletProcessor {
  constructor(options) {
    super(options);
    this._buf        = [];
    this._silCount   = 0;
    this._hasSpeech  = false;
    this._threshold  = 0.012;
    this._SIL_FRAMES = 35; // 35 x 20ms = 700ms silence -> flush

    this.port.onmessage = (ev) => {
      const d = ev.data;
      if (d.type === 'set_threshold') {
        this._threshold = d.value;
      } else if (d.type === 'force_flush') {
        // Drain any partial buffer immediately and signal flush
        if (this._buf.length > 0) {
          const rem = this._buf.splice(0);
          const i16 = new Int16Array(rem.length);
          for (let j = 0; j < rem.length; j++)
            i16[j] = Math.max(-32768, Math.min(32767, (rem[j] * 32767) | 0));
          this.port.postMessage({ type: 'audio', buf: i16.buffer }, [i16.buffer]);
        }
        this._hasSpeech = false;
        this._silCount  = 0;
        this.port.postMessage({ type: 'flush' });
      }
    };
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    const dec = Math.round(sampleRate / 16000); // 3 for 48kHz
    for (let i = 0; i < ch.length; i += dec) {
      this._buf.push(ch[i]);
      if (this._buf.length >= 320) {
        const chunk = this._buf.splice(0, 320);
        let sum = 0;
        for (const s of chunk) sum += s * s;
        const rms = Math.sqrt(sum / chunk.length);
        this.port.postMessage({ type: 'rms', rms });
        if (rms > this._threshold) {
          this._hasSpeech = true;
          this._silCount  = 0;
          const i16 = new Int16Array(chunk.length);
          for (let j = 0; j < chunk.length; j++)
            i16[j] = Math.max(-32768, Math.min(32767, (chunk[j] * 32767) | 0));
          this.port.postMessage({ type: 'audio', buf: i16.buffer }, [i16.buffer]);
        } else if (this._hasSpeech) {
          this._silCount++;
          if (this._silCount >= this._SIL_FRAMES) {
            this._hasSpeech = false;
            this._silCount  = 0;
            this.port.postMessage({ type: 'flush' });
          }
        }
      }
    }
    return true;
  }
}
registerProcessor('audio-proc', AudioProc);
`

const MODE_COLOR: Record<UIMode, string> = {
  lecture:    'var(--rec)',
  awaiting:   'var(--await)',
  processing: 'var(--accent)',
  speaking:   'var(--accent)',
  paused:     '#5a5e66',
}

const MODE_LABEL: Record<UIMode, string> = {
  lecture:    '● Recording',
  awaiting:   '◎ Listening for question…',
  processing: '◌ Processing…',
  speaking:   '▶ Speaking',
  paused:     '⏸ Paused',
}

function SourceChip({ src }: { src: string }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '2px 8px', borderRadius: 4,
      background: 'rgba(94,106,210,0.12)', border: '1px solid rgba(94,106,210,0.3)',
      fontSize: 11, color: 'var(--fg-muted)',
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', flexShrink: 0 }} />
      <span style={{ color: 'var(--accent)', fontWeight: 600, marginRight: 2 }}>Doc</span>
      {src}
    </span>
  )
}

export default function LiveLecture({ onToast }: Props) {
  // Audio / WS refs — mutations here don't need re-renders
  const wsRef          = useRef<WebSocket | null>(null)
  const audioCtxRef    = useRef<AudioContext | null>(null)
  const workletRef     = useRef<AudioWorkletNode | null>(null)
  const micStreamRef   = useRef<MediaStream | null>(null)
  const sourceNodeRef  = useRef<MediaStreamAudioSourceNode | null>(null)
  const playTimeRef    = useRef<number>(0)
  const ttsSourceRef   = useRef<AudioBufferSourceNode | null>(null)  // currently-playing TTS node
  const ttsMutedRef    = useRef<boolean>(false)                      // true after cancel, until next question
  const txEndRef       = useRef<HTMLDivElement | null>(null)
  const levelBarRef    = useRef<HTMLDivElement | null>(null)
  const curQuestionRef = useRef<string>('')
  // Question-capture robustness refs
  const ambientRmsRef  = useRef<number>(0.012)   // rolling EMA of mic RMS (α=0.005)
  const qaTimeoutRef   = useRef<ReturnType<typeof setTimeout> | null>(null)
  const uiModeRef      = useRef<UIMode>('lecture') // mirrors uiMode for use in callbacks

  // React state
  const [sessionInfo, setSessionInfo]         = useState<SessionInfo | null>(null)
  const [uiMode, setUiMode]                   = useState<UIMode>('lecture')
  const [segments, setSegments]               = useState<TxSegment[]>([])
  const [currentAnswer, setCurrentAnswer]     = useState<QAEntry | null>(null)
  const [currentQuestion, setCurrentQuestion] = useState<string>('')
  const [qaHistory, setQaHistory]             = useState<QAEntry[]>([])
  const [gapFindings, setGapFindings]         = useState<Finding[] | null>(null)
  const [gapRaw, setGapRaw]                   = useState<string | null>(null)
  const [gapTs, setGapTs]                     = useState<string | null>(null)
  const [sessionName, setSessionName]         = useState<string>('')
  const [selection, setSelection]             = useState<Set<string>>(new Set())
  const [sources, setSources]                 = useState<SourceData>({ modules: [], documents: [] })
  const [gapsOpen, setGapsOpen]               = useState<boolean>(true)
  const [historyOpen, setHistoryOpen]         = useState<boolean>(false)
  const [convMode, setConvMode]               = useState<boolean>(false)
  const [connecting, setConnecting]           = useState<boolean>(false)
  const [threshold, setThreshold]             = useState<number>(0.012)
  const [showReasoning, setShowReasoning]     = useState<boolean>(false)

  // Fetch modules + docs for the pre-start picker
  useEffect(() => {
    fetch('/api/sources')
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setSources(d))
      .catch(() => {})
  }, [])

  const toggleItem = useCallback((id: string) => {
    setSelection(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])

  // Keep uiModeRef in sync (used in worklet message handler and timeout callbacks)
  useEffect(() => {
    uiModeRef.current = uiMode
  }, [uiMode])

  // 20s auto-flush timeout when in AWAITING mode
  useEffect(() => {
    if (uiMode !== 'awaiting') return
    const t = setTimeout(() => {
      if (uiModeRef.current === 'awaiting') {
        workletRef.current?.port.postMessage({ type: 'force_flush' })
      }
    }, 20000)
    qaTimeoutRef.current = t
    return () => clearTimeout(t)
  }, [uiMode])

  // Auto-scroll transcript pane when new segments arrive
  useEffect(() => {
    txEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [segments])

  // ── TTS interrupt — stops any in-flight audio node immediately ──
  const interruptTts = useCallback(() => {
    const src = ttsSourceRef.current
    if (src) {
      try { src.stop() } catch { /* already stopped */ }
      ttsSourceRef.current = null
    }
    playTimeRef.current = 0
  }, [])

  // ── TTS playback (queued, never overlapping) ────────────────────
  const handleTtsAudio = useCallback(async (buf: ArrayBuffer) => {
    const ctx = audioCtxRef.current
    if (!ctx) return
    // If cancelled while the LLM/TTS was computing, the backend already skipped
    // sending bytes — but as a belt-and-braces guard, drop anything that arrives
    // while muted.
    if (ttsMutedRef.current) return
    try {
      const decoded = await ctx.decodeAudioData(buf)
      if (ttsMutedRef.current) return  // re-check after await (cancel can arrive during decode)
      const src = ctx.createBufferSource()
      src.buffer = decoded
      src.connect(ctx.destination)
      const startAt = Math.max(ctx.currentTime, playTimeRef.current)
      src.start(startAt)
      ttsSourceRef.current = src
      playTimeRef.current = startAt + decoded.duration
    } catch (e) {
      console.error('TTS decode error:', e)
    }
  }, [])

  // ── WS message handler ──────────────────────────────────────────
  const handleWsMessage = useCallback((ev: MessageEvent) => {
    if (ev.data instanceof ArrayBuffer) {
      handleTtsAudio(ev.data)
      return
    }
    let msg: Record<string, unknown>
    try { msg = JSON.parse(ev.data as string) } catch { return }

    const type = msg.type as string

    if (type === 'started') {
      setSessionInfo({ session_id: msg.session_id as string, name: msg.name as string })
      setConnecting(false)
      onToast('Session started', msg.name as string)

    } else if (type === 'state') {
      setUiMode(msg.mode as UIMode)

    } else if (type === 'transcript') {
      const text = (msg.text as string) || ''
      if (text) {
        setSegments(prev => [...prev, {
          text,
          ms: msg.ms as number,
          confidence: msg.confidence as number,
        }])
      }

    } else if (type === 'question') {
      ttsMutedRef.current = false  // new Q&A cycle — allow TTS playback
      const q = (msg.text as string) || ''
      curQuestionRef.current = q
      setCurrentQuestion(q)
      setCurrentAnswer(null)

    } else if (type === 'answer') {
      const entry: QAEntry = {
        question: curQuestionRef.current,
        answer:   (msg.text as string) || '',
        sources:  (msg.sources as string[]) || [],
        reasoning: (msg.reasoning as string) || '',
      }
      setCurrentAnswer(entry)
      setQaHistory(prev => [...prev, entry])

    } else if (type === 'gap') {
      if (Array.isArray(msg.findings)) {
        setGapFindings(msg.findings as Finding[])
        setGapRaw(null)
      } else {
        setGapFindings(null)
        setGapRaw((msg.raw as string) || (msg.text as string) || '')
      }
      setGapTs((msg.ts as string | null) ?? null)

    } else if (type === 'stopped') {
      stopAudio()
      setSessionInfo(null)
      setSegments([])
      setCurrentAnswer(null)
      setCurrentQuestion('')
      setQaHistory([])
      setGapFindings(null)
      setGapRaw(null)
      setGapTs(null)
      setUiMode('lecture')
      onToast('Session ended')

    } else if (type === 'error') {
      onToast('Error', (msg.message as string) || 'Unknown error')
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handleTtsAudio, onToast])

  // ── Audio teardown ──────────────────────────────────────────────
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const stopAudio = useCallback(() => {
    workletRef.current?.disconnect()
    workletRef.current = null
    sourceNodeRef.current?.disconnect()
    sourceNodeRef.current = null
    micStreamRef.current?.getTracks().forEach(t => t.stop())
    micStreamRef.current = null
    audioCtxRef.current?.close()
    audioCtxRef.current = null
    playTimeRef.current = 0
    if (levelBarRef.current) levelBarRef.current.style.width = '0%'
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudio()
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [stopAudio])

  // ── Start session ───────────────────────────────────────────────
  const startSession = useCallback(async () => {
    setConnecting(true)
    const wsUrl = `ws://${location.host}/ws/lecture`
    const ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws
    ws.onmessage = handleWsMessage

    ws.onerror = () => {
      onToast('WebSocket error', 'Could not connect to lecture backend')
      setConnecting(false)
    }

    ws.onclose = () => {
      wsRef.current = null
    }

    ws.onopen = async () => {
      const name = sessionName.trim() ||
        `Lecture ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
      ws.send(JSON.stringify({ type: 'start', name, selection: [...selection] }))

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
        micStreamRef.current = stream

        const ctx = new AudioContext()
        audioCtxRef.current = ctx
        playTimeRef.current = 0

        const blob = new Blob([WORKLET_SRC], { type: 'application/javascript' })
        const blobUrl = URL.createObjectURL(blob)
        await ctx.audioWorklet.addModule(blobUrl)
        URL.revokeObjectURL(blobUrl)

        const worklet = new AudioWorkletNode(ctx, 'audio-proc')
        workletRef.current = worklet

        worklet.port.onmessage = (ev) => {
          const m = ev.data
          if (m.type === 'audio') {
            if (ws.readyState === WebSocket.OPEN) ws.send(m.buf)
          } else if (m.type === 'flush') {
            if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'flush' }))
          } else if (m.type === 'rms') {
            if (levelBarRef.current) {
              levelBarRef.current.style.width = `${Math.min(100, m.rms * 3500)}%`
            }
            // Track ambient noise level: slow EMA (α=0.005) regardless of mode
            // In practice converges to background+voice average, good enough for threshold calibration
            ambientRmsRef.current = 0.995 * ambientRmsRef.current + 0.005 * m.rms
          }
        }

        const source = ctx.createMediaStreamSource(stream)
        sourceNodeRef.current = source
        source.connect(worklet)
      } catch (e) {
        onToast('Mic error', (e as Error).message)
        ws.close()
        setConnecting(false)
      }
    }
  }, [sessionName, selection, handleWsMessage, onToast])

  // ── Controls ────────────────────────────────────────────────────
  const _restoreThreshold = useCallback(() => {
    workletRef.current?.port.postMessage({ type: 'set_threshold', value: threshold })
  }, [threshold])

  const askAI = useCallback(() => {
    // Raise silence gate above ambient noise so end-of-question can be detected
    // in noisy rooms. ambient * 1.3 is above floor noise but below speech level.
    const adapted = Math.max(threshold, ambientRmsRef.current * 1.3)
    if (adapted > threshold) {
      workletRef.current?.port.postMessage({ type: 'set_threshold', value: adapted })
    }
    wsRef.current?.send(JSON.stringify({ type: 'ask_ai' }))
  }, [threshold])

  const doneSpeaking = useCallback(() => {
    if (qaTimeoutRef.current) { clearTimeout(qaTimeoutRef.current); qaTimeoutRef.current = null }
    _restoreThreshold()
    workletRef.current?.port.postMessage({ type: 'force_flush' })
  }, [_restoreThreshold])

  const cancelAsk = useCallback(() => {
    if (qaTimeoutRef.current) { clearTimeout(qaTimeoutRef.current); qaTimeoutRef.current = null }
    _restoreThreshold()
    // Stop any in-flight TTS audio immediately and mute future bytes for this
    // cycle (handles the race where TTS bytes are already in-flight when cancel
    // is clicked). ttsMutedRef is cleared when the next 'question' message arrives.
    ttsMutedRef.current = true
    interruptTts()
    wsRef.current?.send(JSON.stringify({ type: 'cancel' }))
  }, [_restoreThreshold, interruptTts])

  const togglePause = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: uiMode === 'paused' ? 'resume_mic' : 'pause_mic' }))
  }, [uiMode])

  const endSession = useCallback(() => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
    } else {
      stopAudio()
      ws?.close()
      wsRef.current = null
      setSessionInfo(null)
    }
  }, [stopAudio])

  const toggleConvMode = useCallback(() => {
    const next = !convMode
    setConvMode(next)
    wsRef.current?.send(JSON.stringify({ type: 'set_conversation_mode', enabled: next }))
  }, [convMode])

  const updateThreshold = useCallback((val: number) => {
    setThreshold(val)
    workletRef.current?.port.postMessage({ type: 'set_threshold', value: val })
  }, [])

  const canAskAI = sessionInfo !== null && (uiMode === 'lecture' || uiMode === 'paused')

  // ── Pre-start panel ─────────────────────────────────────────────
  if (!sessionInfo) {
    return (
      <div style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 32,
      }}>
        <div style={{
          width: '100%', maxWidth: 540,
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 12, padding: 32,
          display: 'flex', flexDirection: 'column', gap: 22,
        }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--fg-base)', marginBottom: 4 }}>
              Start a lecture session
            </div>
            <div style={{ fontSize: 12, color: 'var(--fg-muted)' }}>
              Browser mic → Whisper CUDA → live transcript. Ask AI anytime during the lecture.
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <label style={{
              fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
              textTransform: 'uppercase', color: 'var(--fg-muted)',
            }}>
              Session name
            </label>
            <input
              type="text"
              value={sessionName}
              onChange={e => setSessionName(e.target.value)}
              placeholder={`Lecture ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`}
              onKeyDown={e => e.key === 'Enter' && !connecting && startSession()}
              style={{
                background: 'var(--bg-base)', border: '1px solid var(--border)',
                borderRadius: 6, padding: '8px 12px',
                color: 'var(--fg-base)', fontSize: 13, fontFamily: 'inherit', outline: 'none',
              }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <label style={{
              fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
              textTransform: 'uppercase', color: 'var(--fg-muted)',
            }}>
              Link materials
            </label>
            <SourceFilter
              sources={sources}
              selected={selection}
              onToggle={toggleItem}
              emptyLabel="No materials linked"
              note="No selection = transcript only (no RAG)."
            />
            {sources.modules.length === 0 && sources.documents.length === 0 && (
              <div style={{ fontSize: 11, color: 'var(--fg-muted)', fontStyle: 'italic' }}>
                No documents ingested yet — gaps analysis and RAG Q&amp;A will be unavailable.
              </div>
            )}
          </div>

          <button
            onClick={startSession}
            disabled={connecting}
            style={{
              background: connecting ? 'rgba(94,106,210,0.4)' : 'var(--accent)',
              border: 'none', borderRadius: 6, padding: '10px 20px',
              color: '#fff', fontSize: 13, fontWeight: 600,
              cursor: connecting ? 'default' : 'pointer',
              fontFamily: 'inherit', transition: 'background 120ms',
            }}
          >
            {connecting ? 'Connecting…' : '▶ Start Recording'}
          </button>
        </div>
      </div>
    )
  }

  // ── In-session layout ───────────────────────────────────────────
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* State bar */}
      <div style={{
        flexShrink: 0,
        background: MODE_COLOR[uiMode],
        padding: '5px 16px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        transition: 'background 300ms',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: '#fff', letterSpacing: '0.03em' }}>
            {MODE_LABEL[uiMode]}
          </span>
          <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)' }}>
            {sessionInfo.name}
          </span>
        </div>
        {/* Level meter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.6)' }}>Level</span>
          <div style={{
            width: 120, height: 6,
            background: 'rgba(0,0,0,0.25)', borderRadius: 3, overflow: 'hidden',
          }}>
            <div
              ref={levelBarRef}
              style={{ width: '0%', height: '100%', background: 'rgba(255,255,255,0.8)', transition: 'width 80ms linear' }}
            />
          </div>
        </div>
      </div>

      {/* Main content area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* Transcript pane */}
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column',
          overflow: 'hidden', borderRight: '1px solid var(--border)',
        }}>
          <div style={{
            flex: 1, overflowY: 'auto',
            padding: '18px 20px',
            fontFamily: "'JetBrains Mono','Courier New',monospace",
            fontSize: 13.5, lineHeight: 1.85,
            color: 'var(--fg-base)',
          }}>
            {segments.length === 0 && (
              <span style={{ color: 'var(--fg-muted)', fontSize: 12, fontStyle: 'italic' }}>
                Transcript will appear here as you speak…
              </span>
            )}
            {segments.map((seg, i) => (
              <span key={i}>{seg.text} </span>
            ))}
            {(uiMode === 'lecture') && (
              <span style={{
                display: 'inline-block', width: 8, height: 16,
                background: 'var(--accent)',
                verticalAlign: 'text-bottom', marginLeft: 2,
              }} className="animate-blink" />
            )}
            {(uiMode === 'awaiting') && (
              <span style={{
                display: 'inline-block', width: 8, height: 16,
                background: 'var(--await)',
                verticalAlign: 'text-bottom', marginLeft: 2,
                opacity: 0.8,
              }} className="animate-blink" />
            )}
            <div ref={txEndRef} />
          </div>
          <div style={{
            flexShrink: 0, padding: '5px 20px',
            borderTop: '1px solid var(--border)',
            fontSize: 10, color: 'var(--fg-muted)',
            display: 'flex', gap: 12,
          }}>
            <span>{segments.length} segment{segments.length !== 1 ? 's' : ''}</span>
            <span>·</span>
            <span>{segments.reduce((n, s) => n + s.text.split(' ').length, 0)} words</span>
          </div>
        </div>

        {/* Side panel */}
        <div style={{ width: 364, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>

          {/* Awaiting indicator */}
          {uiMode === 'awaiting' && !currentQuestion && (
            <div style={{
              margin: 12, padding: 16,
              background: 'rgba(233,162,59,0.08)',
              border: '1px solid rgba(233,162,59,0.3)', borderRadius: 8,
              display: 'flex', flexDirection: 'column', gap: 10,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--await)' }}>
                ◎ Speak your question
              </div>
              <div style={{ fontSize: 11, color: 'var(--fg-muted)' }}>
                Press <strong style={{ color: 'var(--text)' }}>Done speaking</strong> when finished,
                or wait for 700 ms of silence. Auto-flushes after 20 s.
              </div>
              <button
                onClick={doneSpeaking}
                style={{
                  alignSelf: 'flex-start',
                  background: 'var(--await)', color: '#fff',
                  border: 'none', borderRadius: 6, padding: '6px 14px',
                  fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                ✓ Done speaking
              </button>
            </div>
          )}

          {/* Processing indicator */}
          {uiMode === 'processing' && (
            <div style={{ margin: 12, padding: 16, display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{
                width: 18, height: 18, borderRadius: '50%',
                border: '2px solid var(--accent)', borderTopColor: 'transparent',
              }} className="animate-spin-slow" />
              <div style={{ fontSize: 12, color: 'var(--fg-muted)', overflow: 'hidden' }}>
                {currentQuestion
                  ? `"${currentQuestion.slice(0, 55)}${currentQuestion.length > 55 ? '…' : ''}"`
                  : 'Generating answer…'}
              </div>
            </div>
          )}

          {/* Answer card */}
          {currentAnswer && (
            <div style={{
              margin: 12,
              background: 'var(--bg-surface)', border: '1px solid var(--border)',
              borderRadius: 8, overflow: 'hidden', flexShrink: 0,
            }}>
              <div style={{
                padding: '9px 14px', borderBottom: '1px solid var(--border)',
                fontSize: 11, color: 'var(--fg-muted)', fontStyle: 'italic',
              }}>
                Q: {currentAnswer.question}
              </div>
              <div style={{
                padding: '12px 14px',
                fontSize: 13, lineHeight: 1.6, color: 'var(--fg-base)',
              }}>
                {currentAnswer.answer}
              </div>
              {currentAnswer.sources.length > 0 && (
                <div style={{
                  padding: '8px 14px', borderTop: '1px solid var(--border)',
                  display: 'flex', flexWrap: 'wrap', gap: 6,
                }}>
                  {currentAnswer.sources.map(s => <SourceChip key={s} src={s} />)}
                </div>
              )}
              {currentAnswer.reasoning && (
                <div style={{ padding: '6px 14px', borderTop: '1px solid var(--border)' }}>
                  <button
                    onClick={() => setShowReasoning(v => !v)}
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      fontSize: 10, color: 'var(--fg-muted)', padding: 0, fontFamily: 'inherit',
                    }}
                  >
                    {showReasoning ? '▲ Hide reasoning' : '▼ Show reasoning'}
                  </button>
                  {showReasoning && (
                    <div style={{
                      marginTop: 8, fontSize: 11, lineHeight: 1.6, color: 'var(--fg-muted)',
                      fontFamily: "'JetBrains Mono',monospace",
                      whiteSpace: 'pre-wrap', maxHeight: 180, overflowY: 'auto',
                    }}>
                      {currentAnswer.reasoning}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Gaps panel */}
          <div style={{
            margin: currentAnswer ? '0 12px 0 12px' : '12px 12px 0 12px',
            border: '1px solid var(--border)', borderRadius: 8,
            overflow: 'hidden', flexShrink: 0,
          }}>
            <button
              onClick={() => setGapsOpen(v => !v)}
              style={{
                width: '100%', padding: '9px 14px',
                background: 'var(--bg-surface)', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                fontSize: 11, fontFamily: 'inherit',
              }}
            >
              <span style={{ fontWeight: 600, color: 'var(--fg-base)' }}>Live Gaps</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--fg-muted)' }}>
                {gapTs && (
                  <span style={{ fontSize: 10 }}>
                    {new Date(gapTs).toLocaleTimeString()}
                  </span>
                )}
                <span>{gapsOpen ? '▲' : '▼'}</span>
              </span>
            </button>
            {gapsOpen && (
              <div style={{
                padding: '10px 14px', borderTop: '1px solid var(--border)',
                maxHeight: 200, overflowY: 'auto',
              }}>
                {gapFindings ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {gapFindings.map((f, i) => <FindingCard key={i} finding={f} />)}
                  </div>
                ) : gapRaw ? (
                  <div style={{
                    fontSize: 11.5, lineHeight: 1.7, color: 'var(--fg-muted)',
                    fontFamily: "'JetBrains Mono',monospace", whiteSpace: 'pre-wrap',
                  }}>
                    {gapRaw}
                  </div>
                ) : (
                  <span style={{ fontSize: 11.5, fontStyle: 'italic', color: 'var(--fg-muted)' }}>
                    Gap analysis runs every 60 s after 50+ words are spoken.
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Q&A history */}
          <div style={{
            margin: '8px 12px 12px 12px',
            border: '1px solid var(--border)', borderRadius: 8,
            overflow: 'hidden', flex: 1, minHeight: 0,
            display: 'flex', flexDirection: 'column',
          }}>
            <button
              onClick={() => setHistoryOpen(v => !v)}
              style={{
                flexShrink: 0, width: '100%', padding: '9px 14px',
                background: 'var(--bg-surface)', border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                fontSize: 11, fontFamily: 'inherit',
              }}
            >
              <span style={{ fontWeight: 600, color: 'var(--fg-base)' }}>Q&amp;A History</span>
              <span style={{ color: 'var(--fg-muted)' }}>
                {qaHistory.length} · {historyOpen ? '▲' : '▼'}
              </span>
            </button>
            {historyOpen && (
              <div style={{ flex: 1, overflowY: 'auto', background: 'var(--bg-base)' }}>
                {qaHistory.length === 0 ? (
                  <div style={{
                    padding: '12px 14px', borderTop: '1px solid var(--border)',
                    fontSize: 11, color: 'var(--fg-muted)', fontStyle: 'italic',
                  }}>
                    No Q&amp;A turns yet.
                  </div>
                ) : (
                  [...qaHistory].reverse().map((entry, i) => (
                    <div key={i} style={{
                      padding: '10px 14px',
                      borderTop: '1px solid var(--border)',
                    }}>
                      <div style={{ fontSize: 10, color: 'var(--fg-muted)', fontStyle: 'italic', marginBottom: 4 }}>
                        Q: {entry.question}
                      </div>
                      <div style={{ fontSize: 11.5, color: 'var(--fg-base)', lineHeight: 1.5 }}>
                        {entry.answer}
                      </div>
                      {entry.sources.length > 0 && (
                        <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                          {entry.sources.map(s => <SourceChip key={s} src={s} />)}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Control bar */}
      <div style={{
        flexShrink: 0,
        borderTop: '1px solid var(--border)',
        padding: '10px 16px',
        display: 'flex', alignItems: 'center', gap: 8,
        background: 'var(--bg-surface)',
      }}>
        <button
          onClick={askAI}
          disabled={!canAskAI}
          style={{
            background: canAskAI ? 'var(--accent)' : 'rgba(94,106,210,0.2)',
            color: canAskAI ? '#fff' : 'var(--fg-muted)',
            border: 'none', borderRadius: 6, padding: '6px 14px',
            fontSize: 12, fontWeight: 600,
            cursor: canAskAI ? 'pointer' : 'default',
            fontFamily: 'inherit',
          }}
        >
          ◎ Ask AI
        </button>

        {uiMode === 'awaiting' && (
          <button
            onClick={doneSpeaking}
            style={{
              background: 'var(--await)', color: '#fff',
              border: 'none', borderRadius: 6,
              padding: '6px 14px',
              fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            ✓ Done speaking
          </button>
        )}

        {(uiMode === 'awaiting' || uiMode === 'processing' || uiMode === 'speaking') && (
          <button
            onClick={cancelAsk}
            style={{
              background: 'rgba(239,77,86,0.10)',
              border: '1px solid rgba(239,77,86,0.3)', borderRadius: 6,
              padding: '6px 14px', color: 'var(--rec)',
              fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            Cancel
          </button>
        )}

        <button
          onClick={togglePause}
          disabled={uiMode === 'processing' || uiMode === 'speaking' || uiMode === 'awaiting'}
          style={{
            background: 'var(--bg-base)', border: '1px solid var(--border)',
            borderRadius: 6, padding: '6px 14px',
            color: 'var(--fg-base)', fontSize: 12,
            cursor: (uiMode === 'processing' || uiMode === 'speaking' || uiMode === 'awaiting') ? 'default' : 'pointer',
            fontFamily: 'inherit',
            opacity: (uiMode === 'processing' || uiMode === 'speaking' || uiMode === 'awaiting') ? 0.35 : 1,
          }}
        >
          {uiMode === 'paused' ? '▶ Resume' : '⏸ Pause'}
        </button>

        <button
          onClick={toggleConvMode}
          style={{
            background: convMode ? 'rgba(94,106,210,0.14)' : 'var(--bg-base)',
            border: convMode ? '1px solid rgba(94,106,210,0.4)' : '1px solid var(--border)',
            borderRadius: 6, padding: '6px 14px',
            color: convMode ? 'var(--accent)' : 'var(--fg-muted)',
            fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
          }}
        >
          {convMode ? '⇄ Conv.' : '⇄ Single'}
        </button>

        {/* RMS gate slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 4 }}>
          <span style={{ fontSize: 10, color: 'var(--fg-muted)', whiteSpace: 'nowrap' }}>Gate</span>
          <input
            type="range" min={2} max={60} value={Math.round(threshold * 1000)}
            onChange={e => updateThreshold(Number(e.target.value) / 1000)}
            style={{ accentColor: 'var(--accent)', width: 72 }}
          />
          <span style={{ fontSize: 10, color: 'var(--fg-muted)', fontFamily: 'monospace', minWidth: 36 }}>
            {threshold.toFixed(3)}
          </span>
        </div>

        <div style={{ flex: 1 }} />

        <button
          onClick={endSession}
          style={{
            background: 'rgba(239,77,86,0.08)',
            border: '1px solid rgba(239,77,86,0.3)', borderRadius: 6,
            padding: '6px 14px', color: 'var(--rec)',
            fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
          }}
        >
          ■ End Session
        </button>
      </div>
    </div>
  )
}

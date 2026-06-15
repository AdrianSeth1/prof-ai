import { useCallback, useEffect, useRef, useState } from 'react'
import FindingCard from '../components/FindingCard'
import type { Finding } from '../components/FindingCard'

// ── Types ─────────────────────────────────────────────────────────

interface Session {
  id: string
  name: string
  date: string
  segment_count: number
  linked_documents: string[]
}

interface Props { onToast?: (title: string, desc?: string) => void }

type RunState = 'idle' | 'streaming' | 'done' | 'error'

interface TextSegment { type: 'content' | 'thinking'; text: string }

// ── Helpers ───────────────────────────────────────────────────────

function parseSegments(text: string): TextSegment[] {
  const segs: TextSegment[] = []
  let rest = text
  while (rest) {
    const si = rest.indexOf('<think>')
    if (si === -1) {
      if (rest) segs.push({ type: 'content', text: rest })
      break
    }
    if (si > 0) segs.push({ type: 'content', text: rest.slice(0, si) })
    const ei = rest.indexOf('</think>', si + 7)
    if (ei === -1) {
      segs.push({ type: 'thinking', text: rest.slice(si + 7) })
      break
    }
    segs.push({ type: 'thinking', text: rest.slice(si + 7, ei) })
    rest = rest.slice(ei + 8)
  }
  return segs
}

function sessionLabel(s: Session): string {
  const time = s.id.slice(11, 13) + ':' + s.id.slice(13, 15)
  const base  = s.name ? `${s.name} · ${s.date} ${time}` : `${s.date} ${time}`
  return `${base} (${s.segment_count} seg)`
}

// ── Inline renderer with bold + file chips ────────────────────────

function FileChip({ filename }: { filename: string }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '1px 7px',
      border: '1px solid rgba(255,255,255,0.10)',
      borderRadius: 6,
      background: 'var(--surface-pop)',
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: 10.5,
      color: 'var(--text-soft)',
      verticalAlign: 'middle',
      margin: '0 2px',
      flexShrink: 0,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', flexShrink: 0 }} />
      <span style={{ color: 'var(--text-faint)' }}>Doc</span>
      {filename}
    </span>
  )
}

// Matches **bold**, *italic*, and bare filenames
const INLINE_RE = /(\*\*([^*]+)\*\*)|(\*([^*]+)\*)|(\b([\w\-_.]+\.(?:pdf|pptx|docx|txt))\b)/gi

function InlineParts({ text }: { text: string }) {
  const nodes: React.ReactNode[] = []
  let lastIndex = 0
  const re = new RegExp(INLINE_RE.source, 'gi')
  let match: RegExpExecArray | null
  while ((match = re.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(<span key={`t${match.index}`}>{text.slice(lastIndex, match.index)}</span>)
    }
    if (match[1]) {
      nodes.push(<strong key={`b${match.index}`} style={{ color: 'var(--text)', fontWeight: 600 }}>{match[2]}</strong>)
    } else if (match[3]) {
      nodes.push(<em key={`i${match.index}`} style={{ color: 'var(--text-soft)' }}>{match[4]}</em>)
    } else if (match[5]) {
      nodes.push(<FileChip key={`f${match.index}`} filename={match[6]} />)
    }
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) nodes.push(<span key="tail">{text.slice(lastIndex)}</span>)
  return <>{nodes}</>
}

// ── Markdown findings renderer ────────────────────────────────────

function FindingsBlock({ text, streaming }: { text: string; streaming: boolean }) {
  const lines = text.split('\n')
  const elements: React.ReactNode[] = []
  let bulletBuf: string[] = []
  let keyCounter = 0

  function nextKey() { return String(keyCounter++) }

  function flushBullets() {
    if (bulletBuf.length === 0) return
    elements.push(
      <div key={`ul${nextKey()}`} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {bulletBuf.map((line, i) => (
          <div key={i} style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 10,
            background: 'var(--surface)',
            padding: '12px 14px',
          }}>
            <span style={{ color: 'var(--accent)', flexShrink: 0, fontSize: 14, lineHeight: 1.6 }}>·</span>
            <span style={{ fontSize: 13.5, color: 'var(--text-body)', lineHeight: 1.6, minWidth: 0 }}>
              <InlineParts text={line} />
            </span>
          </div>
        ))}
      </div>
    )
    bulletBuf = []
  }

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) { flushBullets(); continue }

    const bulletMatch = trimmed.match(/^[-*•]\s+(.+)/)
    if (bulletMatch) {
      bulletBuf.push(bulletMatch[1])
      continue
    }

    flushBullets()

    const h2Match = trimmed.match(/^#{1,3}\s+(.+)/)
    if (h2Match) {
      elements.push(
        <div key={`h${nextKey()}`} style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)', marginTop: 10, marginBottom: 4 }}>
          <InlineParts text={h2Match[1]} />
        </div>
      )
    } else {
      elements.push(
        <p key={`p${nextKey()}`} style={{ fontSize: 13.5, color: 'var(--text-body)', lineHeight: 1.65, margin: '4px 0' }}>
          <InlineParts text={trimmed} />
        </p>
      )
    }
  }

  flushBullets()

  if (streaming) {
    elements.push(
      <span key="cursor" className="animate-blink" style={{
        display: 'inline-block',
        width: 8, height: 16,
        background: 'var(--accent)',
        verticalAlign: 'text-bottom',
        marginLeft: 4,
      }} />
    )
  }

  return <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>{elements}</div>
}

// ── Thinking block (collapsible) ─────────────────────────────────

function ThinkBlock({ text, open, onToggle }: { text: string; open: boolean; onToggle: () => void }) {
  return (
    <div style={{
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: 10,
      background: 'var(--bg-sidebar)',
      marginBottom: 18,
    }}>
      <button
        onClick={onToggle}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          padding: '10px 14px',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
          color: 'var(--text-faint)',
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 10,
          letterSpacing: '0.04em',
        }}
      >
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"
          strokeLinecap="round"
          style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 140ms ease', flexShrink: 0 }}>
          <path d="M6 4l4 4-4 4" />
        </svg>
        REASONING TRACE
        <span style={{ marginLeft: 'auto', color: 'var(--text-ghost)' }}>{open ? 'collapse' : 'expand'}</span>
      </button>
      {open && (
        <div style={{
          padding: '2px 14px 12px',
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 11.5,
          color: 'var(--text-ghost)',
          lineHeight: 1.75,
          borderTop: '1px solid rgba(255,255,255,0.05)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          maxHeight: 280,
          overflowY: 'auto',
        }}>
          {text || '…'}
        </div>
      )}
    </div>
  )
}

// ── Checkbox ──────────────────────────────────────────────────────

function Checkbox({ checked, onChange, disabled = false }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <span
      onClick={() => !disabled && onChange(!checked)}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 15, height: 15,
        border: checked ? '1px solid var(--accent)' : '1px solid rgba(255,255,255,0.20)',
        borderRadius: 4,
        background: checked ? 'var(--accent)' : 'transparent',
        cursor: disabled ? 'default' : 'pointer',
        flexShrink: 0,
        opacity: disabled ? 0.4 : 1,
        transition: 'background 140ms ease, border-color 140ms ease',
      }}
    >
      {checked && (
        <svg width="9" height="9" viewBox="0 0 16 16" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 8.5l3.2 3.2L13 4.5" />
        </svg>
      )}
    </span>
  )
}

// ── Read-only scope chip (no × button) ───────────────────────────

function ScopeChip({ filename }: { filename: string }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      height: 24, padding: '0 9px',
      borderRadius: 6,
      background: 'var(--surface-pop)',
      border: '1px solid rgba(255,255,255,0.10)',
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: 11,
      color: 'var(--text-soft)',
    }}>
      {filename}
    </span>
  )
}

// ── Main component ────────────────────────────────────────────────

export default function GapsAnalysis({ onToast }: Props) {
  const [sessions,     setSessions]     = useState<Session[]>([])
  const [sessLoading,  setSessLoading]  = useState(true)
  const [selId,        setSelId]        = useState<string | null>(null)
  const [useLatest,    setUseLatest]    = useState(false)
  const [showThinking, setShowThinking] = useState(false)
  const [runState,     setRunState]     = useState<RunState>('idle')
  const [accumulated,  setAccumulated]  = useState('')
  const [findings,     setFindings]     = useState<Finding[] | null>(null)
  const [rawContent,   setRawContent]   = useState<string>('')
  const [errorMsg,     setErrorMsg]     = useState<string | null>(null)
  const [thinkOpen,    setThinkOpen]    = useState(false)
  const resultsRef = useRef<HTMLDivElement>(null)
  const abortRef   = useRef<AbortController | null>(null)

  // ── Load sessions ─────────────────────────────────────────────
  useEffect(() => {
    fetch('/api/sessions')
      .then(r => r.ok ? r.json() as Promise<{ sessions: Session[] }> : Promise.reject(r.status))
      .then(data => {
        setSessions(data.sessions)
        if (data.sessions.length > 0) setSelId(data.sessions[0].id)
      })
      .catch(e => console.error('[GAPS] sessions load error', e))
      .finally(() => setSessLoading(false))
  }, [])

  // ── Derived ───────────────────────────────────────────────────
  const latestSession    = sessions[0] ?? null
  const selectedSession  = useLatest
    ? latestSession
    : sessions.find(s => s.id === selId) ?? null

  const segments    = parseSegments(accumulated)
  const thinkText   = segments.filter(s => s.type === 'thinking').map(s => s.text).join('')
  const hasThink    = thinkText.trim().length > 0
  const canRun      = (useLatest ? !!latestSession : !!selId) && runState !== 'streaming'

  // ── Auto-scroll during streaming ──────────────────────────────
  useEffect(() => {
    if (runState === 'streaming' && resultsRef.current) {
      resultsRef.current.scrollTop = resultsRef.current.scrollHeight
    }
  }, [accumulated, runState])

  // ── Run / Stop ────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    if (runState === 'streaming') {
      abortRef.current?.abort()
      return
    }
    if (!canRun) return

    setRunState('streaming')
    setAccumulated('')
    setFindings(null)
    setRawContent('')
    setErrorMsg(null)
    setThinkOpen(true)   // expand reasoning panel as it streams in

    const ctrl = new AbortController()
    abortRef.current = ctrl

    const bodyObj = useLatest
      ? { latest: true, show_thinking: showThinking }
      : { session_id: selId, show_thinking: showThinking }

    let localState: RunState = 'streaming'

    try {
      const r = await fetch('/api/gaps', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyObj),
        signal: ctrl.signal,
      })

      if (!r.ok) {
        const errBody = await r.json().catch(() => ({}))
        const msg = (errBody as { detail?: string }).detail ?? `HTTP ${r.status}`
        setErrorMsg(msg)
        setRunState('error')
        return
      }

      const reader = r.body!.getReader()
      const dec    = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          if (localState === 'streaming') {
            localState = 'done'
            setRunState('done')
          }
          break
        }
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const msg = JSON.parse(line) as {
              type: string; text?: string; message?: string;
              session_id?: string; findings?: Finding[]; raw?: string
            }
            if (msg.type === 'token' && msg.text) {
              setAccumulated(prev => prev + msg.text)
            } else if (msg.type === 'error') {
              localState = 'error'
              setErrorMsg(msg.message ?? 'Unknown error')
              setRunState('error')
            } else if (msg.type === 'done') {
              localState = 'done'
              setRunState('done')
              if (Array.isArray(msg.findings) && msg.findings.length > 0) {
                setFindings(msg.findings)
              }
              if (msg.raw) setRawContent(msg.raw)
              const sessId = msg.session_id ?? selId ?? ''
              onToast?.('Analysis complete', `Session ${sessId.slice(0, 10)}`)
            }
          } catch {
            // ignore parse errors on partial NDJSON lines
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setRunState('idle')
        setAccumulated('')
        setFindings(null)
        setRawContent('')
      } else {
        setErrorMsg(String(err))
        setRunState('error')
      }
    }
  }, [runState, canRun, useLatest, selId, showThinking, onToast])

  // ── Render ────────────────────────────────────────────────────
  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      <div style={{ maxWidth: 880, margin: '0 auto', padding: '24px 24px 40px' }}>

        {/* Header */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: '-0.02em' }}>Gap analysis</div>
          <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
            batch comparison of a lecture transcript against its linked documents
          </div>
        </div>

        {/* ── Config card ── */}
        <div style={{
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 11,
          background: 'var(--surface)',
          padding: 15,
          marginBottom: 18,
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}>

          {/* Session row */}
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div style={{ flex: 1, minWidth: 240 }}>
              <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.04em', marginBottom: 6 }}>SESSION</div>
              <div style={{ position: 'relative' }}>
                <select
                  disabled={useLatest || sessLoading || runState === 'streaming'}
                  value={selId ?? ''}
                  onChange={e => setSelId(e.target.value || null)}
                  style={{
                    display: 'block',
                    width: '100%',
                    height: 34,
                    padding: '0 28px 0 11px',
                    background: 'var(--bg-base)',
                    border: '1px solid rgba(255,255,255,0.10)',
                    borderRadius: 8,
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: 12,
                    color: (useLatest || runState === 'streaming') ? 'var(--text-faint)' : 'var(--text-soft)',
                    cursor: (useLatest || runState === 'streaming') ? 'default' : 'pointer',
                    outline: 'none',
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    opacity: useLatest ? 0.5 : 1,
                  }}
                >
                  {sessLoading && <option value="">Loading…</option>}
                  {!sessLoading && sessions.length === 0 && <option value="">No sessions recorded</option>}
                  {sessions.map(s => (
                    <option key={s.id} value={s.id}>{sessionLabel(s)}</option>
                  ))}
                </select>
                {/* chevron icon */}
                <svg
                  width="11" height="11" viewBox="0 0 16 16" fill="none"
                  stroke="#6b7079" strokeWidth="1.5" strokeLinecap="round"
                  style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}
                >
                  <path d="M4 6l4 4 4-4" />
                </svg>
              </div>
            </div>

            {/* Use latest checkbox */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 9, height: 34, cursor: runState === 'streaming' ? 'default' : 'pointer', color: 'var(--text-soft)', fontSize: 12.5, userSelect: 'none' }}>
              <Checkbox
                checked={useLatest}
                onChange={v => {
                  setUseLatest(v)
                  if (v && sessions[0]) setSelId(sessions[0].id)
                }}
                disabled={runState === 'streaming'}
              />
              Use latest session
            </label>
          </div>

          {/* Document scope row (read-only) */}
          <div>
            <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.04em', marginBottom: 6 }}>DOCUMENT SCOPE</div>
            <div style={{ padding: '9px 11px', background: 'var(--bg-base)', border: '1px solid rgba(255,255,255,0.10)', borderRadius: 8, minHeight: 38, display: 'flex', alignItems: 'center' }}>
              {!selectedSession ? (
                <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 11, color: 'var(--text-ghost)' }}>select a session above</span>
              ) : selectedSession.linked_documents.length > 0 ? (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                  {selectedSession.linked_documents.map(doc => (
                    <ScopeChip key={doc} filename={doc} />
                  ))}
                </div>
              ) : (
                <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 11, color: 'var(--text-faint)' }}>
                  all course material
                </span>
              )}
            </div>
            <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 10, color: 'var(--text-ghost)', marginTop: 5 }}>
              scope is set when linking documents to a session in Live Lecture
            </div>
          </div>

          {/* Controls row */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 9, cursor: runState === 'streaming' ? 'default' : 'pointer', color: 'var(--text-soft)', fontSize: 12.5, userSelect: 'none' }}>
              <Checkbox checked={showThinking} onChange={setShowThinking} disabled={runState === 'streaming'} />
              Show reasoning trace
            </label>

            <button
              onClick={handleRun}
              disabled={!canRun && runState !== 'streaming'}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                height: 36,
                padding: '0 18px',
                borderRadius: 8,
                border: 'none',
                background: runState === 'streaming'
                  ? 'rgba(239,77,86,0.12)'
                  : canRun ? 'var(--accent)' : 'rgba(94,106,210,0.28)',
                color: runState === 'streaming' ? 'var(--rec-text)' : '#fff',
                fontSize: 13.5,
                fontWeight: 500,
                cursor: (canRun || runState === 'streaming') ? 'pointer' : 'default',
                fontFamily: 'inherit',
                transition: 'background 140ms ease',
                boxShadow: canRun ? '0 1px 0 rgba(255,255,255,0.10) inset' : 'none',
              }}
            >
              {runState === 'streaming' ? (
                <>
                  <span className="animate-spin-slow" style={{ display: 'inline-flex' }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
                      <circle cx="8" cy="8" r="5.5" strokeDasharray="25" strokeDashoffset="10" />
                    </svg>
                  </span>
                  Stop
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="7" cy="7" r="4.3" /><path d="M13.5 13.5l-3-3" />
                  </svg>
                  Find Gaps
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── Results area ── */}
        <div ref={resultsRef}>

          {/* Error */}
          {runState === 'error' && errorMsg && (
            <div style={{
              border: '1px solid var(--rec-line)',
              borderRadius: 10,
              background: 'var(--rec-08)',
              padding: '13px 15px',
              fontSize: 13,
              color: 'var(--rec-text)',
              marginBottom: 16,
            }}>
              <strong>Error:</strong> {errorMsg}
            </div>
          )}

          {/* Reasoning block */}
          {showThinking && hasThink && (
            <ThinkBlock
              text={thinkText}
              open={thinkOpen}
              onToggle={() => setThinkOpen(o => !o)}
            />
          )}

          {/* Findings header */}
          {(findings !== null || rawContent || runState === 'streaming') && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <span style={{ fontSize: 12.5, fontWeight: 600 }}>Findings</span>
              {selectedSession && (
                <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 10.5, color: 'var(--text-faint)' }}>
                  {selectedSession.date}
                  {selectedSession.linked_documents.length > 0
                    ? ` · ${selectedSession.linked_documents.length} linked doc${selectedSession.linked_documents.length !== 1 ? 's' : ''}`
                    : ' · all course material'}
                </span>
              )}
              {runState === 'done' && (
                <span style={{
                  marginLeft: 'auto',
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: 10,
                  color: 'var(--ok)',
                  display: 'flex', alignItems: 'center', gap: 5,
                }}>
                  <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--ok)' }} />
                  done
                </span>
              )}
            </div>
          )}

          {/* Structured findings cards */}
          {findings !== null && findings.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {findings.map((f, i) => <FindingCard key={i} finding={f} />)}
            </div>
          )}

          {/* Fallback: raw text when structured parse failed */}
          {findings === null && rawContent && runState === 'done' && (
            <FindingsBlock text={rawContent} streaming={false} />
          )}

          {/* Streaming — LLM is reasoning, findings not yet available */}
          {runState === 'streaming' && (
            <div style={{
              border: '1px solid rgba(255,255,255,0.07)',
              borderRadius: 10,
              background: 'var(--surface)',
              padding: '18px 16px',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: 11,
              color: 'var(--text-faint)',
            }}>
              <span className="animate-spin-slow" style={{ display: 'inline-flex' }}>
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="var(--accent)" strokeWidth="1.6">
                  <circle cx="8" cy="8" r="5.5" strokeDasharray="25" strokeDashoffset="10" />
                </svg>
              </span>
              {hasThink ? 'reasoning through transcript and notes…' : 'analysing…'}
            </div>
          )}

          {/* Idle empty state */}
          {runState === 'idle' && (
            <div style={{
              border: '1px dashed rgba(255,255,255,0.07)',
              borderRadius: 10,
              padding: '36px 24px',
              textAlign: 'center',
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: 11,
              color: 'var(--text-ghost)',
            }}>
              select a session and click Find Gaps to compare the transcript against linked documents
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

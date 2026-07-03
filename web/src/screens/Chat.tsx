import React, { useCallback, useEffect, useRef, useState } from 'react'
import SourceFilter from '../components/SourceFilter'
import type { SourceData } from '../components/SourceFilter'
import ThinkBlock from '../components/ThinkBlock'

// ── Local types ───────────────────────────────────────────────────

type ChatMode = 'research' | 'collaborate'

interface SourceDetail { source_file: string; location: string; preview: string }
interface LitItem {
  source: 'pubmed' | 'semantic_scholar' | 'openalex'
  pmid?: string
  id?: string
  title?: string
}

interface Chip {
  kind: 'Doc' | 'PubMed' | 'Semantic Scholar' | 'OpenAlex'
  label: string
  url?: string
  internal: boolean
}

type MsgKind = 'user' | 'thinking' | 'streaming' | 'answer' | 'error'

interface ChatMsg {
  id: string
  role: 'user' | 'assistant'
  kind: MsgKind
  text: string
  time: string
  chips?: Chip[]
  mode: ChatMode
  reasoning?: string
  reasoningOpen?: boolean
}

interface HistoryEntry { role: 'user' | 'assistant'; content: string }

// ── Helpers ───────────────────────────────────────────────────────

function stamp(): string {
  return new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

function uid(): string { return Math.random().toString(36).slice(2, 10) }

function buildChips(details: SourceDetail[], literature: LitItem[]): Chip[] {
  const chips: Chip[] = []
  const seen = new Set<string>()
  for (const d of details) {
    if (!seen.has(d.source_file)) {
      seen.add(d.source_file)
      chips.push({ kind: 'Doc', label: d.source_file, internal: true })
    }
  }
  for (const a of literature) {
    if (a.source === 'pubmed' && a.pmid) {
      chips.push({
        kind: 'PubMed', label: a.pmid, internal: false,
        url: `https://pubmed.ncbi.nlm.nih.gov/${a.pmid}/`,
      })
    } else if (a.source === 'semantic_scholar' && a.id) {
      chips.push({
        kind: 'Semantic Scholar', label: a.id, internal: false,
        url: `https://www.semanticscholar.org/paper/${a.id}`,
      })
    } else if (a.source === 'openalex' && a.id) {
      chips.push({
        kind: 'OpenAlex', label: a.id, internal: false,
        url: `https://openalex.org/${a.id}`,
      })
    }
  }
  return chips
}

function historyFrom(msgs: ChatMsg[]): HistoryEntry[] {
  return msgs
    .filter(m => m.kind === 'user' || m.kind === 'answer')
    .slice(-12)
    .map(m => ({ role: m.role as 'user' | 'assistant', content: m.text }))
}

// ── Citation chip ─────────────────────────────────────────────────

function CitationChip({ chip }: { chip: Chip }) {
  const [hov, setHov] = useState(false)
  const go = () => {
    if (chip.url) window.open(chip.url, '_blank', 'noopener,noreferrer')
  }
  return (
    <span
      onClick={go}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '1px 8px',
        border: `1px solid ${hov ? 'rgba(94,106,210,0.4)' : 'rgba(255,255,255,0.10)'}`,
        borderRadius: 6,
        background: hov ? '#181a1f' : 'var(--surface-pop)',
        fontFamily: '"JetBrains Mono", monospace', fontSize: 10.5,
        color: '#b9bcc4',
        cursor: chip.url ? 'pointer' : 'default',
        verticalAlign: 'middle',
        transition: 'border-color 140ms ease, background 140ms ease',
        flexShrink: 0,
        userSelect: 'none',
      }}
    >
      {chip.internal
        ? <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', flexShrink: 0 }} />
        : <span style={{ width: 6, height: 6, borderRadius: '50%', border: '1px solid rgba(255,255,255,0.3)', flexShrink: 0 }} />
      }
      <span style={{ color: '#6b7079' }}>{chip.kind}</span>
      {chip.label}
    </span>
  )
}

// ── Skeleton shimmer ──────────────────────────────────────────────

function ShimmerLine({ w }: { w: string }) {
  return (
    <div className="animate-shimmer" style={{ height: 11, width: w, borderRadius: 5 }} />
  )
}

// ── Message row ───────────────────────────────────────────────────

function MsgRow({ msg, onToggleReasoning }: { msg: ChatMsg; onToggleReasoning: (id: string) => void }) {
  const isUser = msg.role === 'user'
  const msgMode = msg.mode
  const hasReasoning = !!msg.reasoning && msg.reasoning.trim().length > 0
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {isUser
          ? (
            <div style={{
              width: 22, height: 22, borderRadius: 6,
              background: '#22242a',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 11, fontWeight: 600, color: '#c4c8cf', flexShrink: 0,
            }}>P</div>
          ) : (
            <div style={{
              width: 22, height: 22, borderRadius: 6,
              background: msgMode === 'collaborate'
                ? 'linear-gradient(150deg,#e9a23b,#c47d10)'
                : 'linear-gradient(150deg,#6e79e0,#5059bd)',
              flexShrink: 0,
              boxShadow: '0 0 0 1px rgba(255,255,255,0.06) inset',
            }} />
          )
        }
        <span style={{ fontSize: 12.5, fontWeight: 600 }}>
          {isUser ? 'You' : msgMode === 'collaborate' ? 'Collaborator' : 'Prof AI'}
        </span>
        <span style={{ fontFamily: '"JetBrains Mono",monospace', fontSize: 10.5, color: 'var(--text-faint)' }}>
          {msg.time}
        </span>
      </div>

      {/* Body */}
      <div style={{ paddingLeft: 30 }}>
        {msg.kind === 'user' && (
          <div style={{ fontSize: 14, color: 'var(--text-body)', lineHeight: 1.6 }}>{msg.text}</div>
        )}

        {hasReasoning && (
          <ThinkBlock
            text={msg.reasoning ?? ''}
            open={msg.reasoningOpen ?? true}
            onToggle={() => onToggleReasoning(msg.id)}
            streaming={msg.kind === 'thinking'}
          />
        )}

        {msg.kind === 'thinking' && !hasReasoning && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <ShimmerLine w="92%" />
            <ShimmerLine w="78%" />
            <ShimmerLine w="40%" />
          </div>
        )}

        {(msg.kind === 'streaming' || msg.kind === 'answer') && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
            <div style={{ fontSize: 14, color: 'var(--text-body)', lineHeight: 1.68, whiteSpace: 'pre-wrap' }}>
              {msg.text}
            </div>
            {/* Only show citation chips in research mode — collaborate sends none */}
            {msg.kind === 'answer' && msgMode === 'research' && msg.chips && msg.chips.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, alignItems: 'center' }}>
                {msg.chips.map((c, i) => <CitationChip key={i} chip={c} />)}
              </div>
            )}
          </div>
        )}

        {msg.kind === 'error' && (
          <div style={{
            fontSize: 13, color: 'var(--rec-text)', lineHeight: 1.5,
            padding: '9px 12px',
            border: '1px solid var(--rec-line)', borderRadius: 8,
            background: 'var(--rec-08)',
          }}>
            {msg.text}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Literature toggle ─────────────────────────────────────────────

function LitToggle({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 7,
        height: 28, padding: '0 11px', borderRadius: 8,
        fontSize: 12, cursor: 'pointer',
        transition: 'all 140ms ease',
        border: `1px solid ${active
          ? 'rgba(94,106,210,0.5)'
          : hov ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.10)'}`,
        background: active ? 'rgba(94,106,210,0.13)' : 'transparent',
        color: active ? 'var(--accent-text)' : hov ? 'var(--text-soft)' : 'var(--text-muted)',
        userSelect: 'none',
      }}
    >
      {active && (
        <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--accent)', flexShrink: 0 }} />
      )}
      {label}
    </div>
  )
}

// ── Mode toggle (segmented control) ──────────────────────────────

function ModeToggle({ mode, onChange }: { mode: ChatMode; onChange: (m: ChatMode) => void }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center',
      padding: 3, gap: 2,
      background: 'rgba(255,255,255,0.05)',
      borderRadius: 9, flexShrink: 0,
    }}>
      {(['research', 'collaborate'] as ChatMode[]).map(m => (
        <button
          key={m}
          onClick={() => onChange(m)}
          style={{
            height: 24, padding: '0 11px',
            borderRadius: 6, border: 'none',
            background: mode === m ? 'var(--bg-base)' : 'transparent',
            color: mode === m ? 'var(--text)' : 'var(--text-faint)',
            fontSize: 11.5, fontWeight: mode === m ? 500 : 400,
            cursor: 'pointer', fontFamily: 'inherit',
            boxShadow: mode === m ? '0 1px 3px rgba(0,0,0,0.35)' : 'none',
            transition: 'background 120ms ease, color 120ms ease, box-shadow 120ms ease',
            whiteSpace: 'nowrap',
          }}
        >
          {m === 'research' ? 'Research' : 'Collaborate'}
        </button>
      ))}
    </div>
  )
}

// ── Main Chat screen ──────────────────────────────────────────────

export default function Chat() {
  const [mode, setModeRaw] = useState<ChatMode>(() => {
    try {
      const stored = localStorage.getItem('profai_chat_mode')
      return (stored === 'collaborate' || stored === 'research') ? stored : 'research'
    } catch { return 'research' }
  })

  const setMode = useCallback((m: ChatMode) => {
    setModeRaw(m)
    try { localStorage.setItem('profai_chat_mode', m) } catch { /* private browsing */ }
  }, [])

  const [sources, setSources]       = useState<SourceData>({ modules: [], documents: [] })
  const [selected, setSelected]     = useState<Set<string>>(new Set())
  const [lit, setLit]               = useState({ pubmed: true, semantic_scholar: true, openalex: false })
  const [messages, setMessages]     = useState<ChatMsg[]>([])
  const [chatInput, setChatInput]   = useState('')
  const [streaming, setStreaming]   = useState(false)
  const [chunkCount, setChunkCount] = useState<number | null>(null)
  const [composerFocus, setComposerFocus] = useState(false)

  const threadRef   = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Load modules + documents for the filter popover
  useEffect(() => {
    fetch('/api/sources')
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setSources(d))
      .catch(() => {})
  }, [])

  // Auto-scroll thread
  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight
    }
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }, [chatInput])

  const toggleItem = useCallback((id: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])

  const toggleReasoning = useCallback((id: string) => {
    setMessages(prev => prev.map(m =>
      m.id === id ? { ...m, reasoningOpen: !(m.reasoningOpen ?? true) } : m
    ))
  }, [])

  // ── Send message ────────────────────────────────────────────────
  const send = useCallback(async () => {
    const q = chatInput.trim()
    if (!q || streaming) return

    const history = historyFrom(messages)
    const activeLit = mode === 'research'
      ? Object.entries(lit).filter(([, v]) => v).map(([k]) => k)
      : []

    setChatInput('')
    setStreaming(true)
    setChunkCount(null)

    const t    = stamp()
    const umid = uid()
    const amid = uid()
    setMessages(prev => [
      ...prev,
      { id: umid, role: 'user',      kind: 'user',     text: q, time: t, mode },
      { id: amid, role: 'assistant', kind: 'thinking', text: '', time: t, mode },
    ])

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question:   q,
          selected:   [...selected],
          literature: activeLit,
          history,
          mode,
        }),
      })

      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf       = ''
      let text      = ''
      let transitioned = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''

        for (const raw of lines) {
          const line = raw.trim()
          if (!line) continue
          let parsed: Record<string, unknown>
          try { parsed = JSON.parse(line) } catch { continue }

          if (parsed.type === 'reasoning') {
            const tok = (parsed.text as string) ?? ''
            if (!tok) continue
            setMessages(prev => prev.map(m =>
              m.id === amid ? { ...m, reasoning: (m.reasoning ?? '') + tok, reasoningOpen: true } : m
            ))
          } else if (parsed.type === 'token') {
            const tok = (parsed.text as string) ?? ''
            if (!tok) continue
            text += tok
            if (!transitioned) {
              transitioned = true
              // The answer has started — auto-collapse a reasoning trace, if any.
              setMessages(prev => prev.map(m =>
                m.id === amid ? { ...m, kind: 'streaming', text: tok, reasoningOpen: false } : m
              ))
            } else {
              setMessages(prev => prev.map(m =>
                m.id === amid ? { ...m, text: m.text + tok } : m
              ))
            }
          } else if (parsed.type === 'done') {
            const details   = (parsed.source_details as SourceDetail[]) ?? []
            const lit_items = (parsed.literature     as LitItem[])      ?? []
            const chips = buildChips(details, lit_items)
            setChunkCount(details.length)
            setMessages(prev => prev.map(m =>
              m.id === amid ? { ...m, kind: 'answer', text, chips } : m
            ))
          } else if (parsed.type === 'error') {
            setMessages(prev => prev.map(m =>
              m.id === amid
                ? { ...m, kind: 'error', text: (parsed.message as string) ?? 'Unknown error' }
                : m
            ))
          }
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === amid
          ? { ...m, kind: 'error', text: `Request failed: ${err instanceof Error ? err.message : String(err)}` }
          : m
      ))
    } finally {
      setStreaming(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatInput, streaming, selected, lit, messages, mode])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const placeholder = mode === 'collaborate'
    ? 'Ask for ideas, examples, or to be taught…'
    : 'Ask about the course materials or literature…'

  const footerRight = chunkCount !== null
    ? `${mode === 'collaborate' ? 'Collaborate' : 'RAG'} · qwen3 · ${chunkCount} chunk${chunkCount !== 1 ? 's' : ''}`
    : mode === 'collaborate' ? 'Collaborate · qwen3' : 'RAG · qwen3'

  // ── Render ──────────────────────────────────────────────────────
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>

      {/* ── Source filter bar ── */}
      <div style={{
        flexShrink: 0,
        borderBottom: '1px solid var(--line)',
        padding: '11px 0',
        background: 'var(--bg-sidebar)',
        position: 'relative',
        zIndex: 4,
      }}>
        <div style={{
          maxWidth: 840, margin: '0 auto', width: '100%',
          padding: '0 24px',
          display: 'flex', alignItems: 'center', gap: 9, flexWrap: 'wrap',
        }}>

          {/* Mode toggle — leftmost */}
          <ModeToggle mode={mode} onChange={setMode} />

          {/* Divider */}
          <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.08)', flexShrink: 0 }} />

          {/* Scope dropdown */}
          <SourceFilter
            sources={sources}
            selected={selected}
            onToggle={toggleItem}
          />

          {/* Literature toggles — research mode only */}
          {mode === 'research' && (
            <>
              <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.08)', flexShrink: 0 }} />
              <span style={{
                fontFamily: '"JetBrains Mono",monospace',
                fontSize: 10.5, color: 'var(--text-faint)',
              }}>literature</span>
              <LitToggle label="PubMed"           active={lit.pubmed}           onClick={() => setLit(l => ({ ...l, pubmed: !l.pubmed }))} />
              <LitToggle label="Semantic Scholar"  active={lit.semantic_scholar} onClick={() => setLit(l => ({ ...l, semantic_scholar: !l.semantic_scholar }))} />
              <LitToggle label="OpenAlex"          active={lit.openalex}         onClick={() => setLit(l => ({ ...l, openalex: !l.openalex }))} />
            </>
          )}
        </div>
      </div>

      {/* ── Message thread ── */}
      <div ref={threadRef} style={{ flex: 1, overflowY: 'auto' }}>
        {messages.length === 0
          ? (mode === 'collaborate' ? <CollaborateEmptyState /> : <ResearchEmptyState />)
          : (
            <div style={{
              maxWidth: 840, margin: '0 auto', width: '100%',
              padding: '26px 24px 8px',
              display: 'flex', flexDirection: 'column', gap: 26,
            }}>
              {messages.map(m => <MsgRow key={m.id} msg={m} onToggleReasoning={toggleReasoning} />)}
            </div>
          )
        }
      </div>

      {/* ── Composer ── */}
      <div style={{ flexShrink: 0, padding: '0 24px 20px' }}>
        <div style={{ maxWidth: 840, margin: '0 auto', width: '100%' }}>
          <div
            onFocus={() => setComposerFocus(true)}
            onBlur={() => setComposerFocus(false)}
            style={{
              display: 'flex', alignItems: 'flex-end', gap: 9,
              border: `1px solid ${composerFocus
                ? mode === 'collaborate' ? 'rgba(233,162,59,0.45)' : 'rgba(94,106,210,0.5)'
                : 'rgba(255,255,255,0.12)'}`,
              borderRadius: 12,
              background: '#101114',
              padding: '9px 9px 9px 14px',
              transition: 'border-color 140ms ease',
            }}
          >
            <textarea
              ref={textareaRef}
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              style={{
                flex: 1, background: 'transparent', border: 'none',
                outline: 'none', resize: 'none',
                color: 'var(--text)',
                fontFamily: 'inherit', fontSize: 14, lineHeight: 1.5,
                padding: '5px 0', maxHeight: 120,
                overflow: chatInput.split('\n').length > 3 ? 'auto' : 'hidden',
              }}
            />
            <SendButton
              onClick={send}
              disabled={!chatInput.trim() || streaming}
              collaborate={mode === 'collaborate'}
            />
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            marginTop: 8, padding: '0 2px',
            fontFamily: '"JetBrains Mono",monospace',
            fontSize: 10.5, color: 'var(--text-faint)',
          }}>
            <KbdChip>⏎</KbdChip> send
            <KbdChip style={{ marginLeft: 4 }}>⇧⏎</KbdChip> newline
            <span style={{ marginLeft: 'auto' }}>{footerRight}</span>
          </div>
        </div>
      </div>

    </div>
  )
}

// ── Small sub-components ──────────────────────────────────────────

function SendButton({ onClick, disabled, collaborate }: { onClick: () => void; disabled: boolean; collaborate?: boolean }) {
  const [hov, setHov] = useState(false)
  const accent = collaborate ? '#c47d10' : 'var(--accent)'
  const accentHov = collaborate ? '#d48f20' : 'var(--accent-hover)'
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        width: 34, height: 34, flexShrink: 0,
        borderRadius: 8, border: 'none',
        cursor: disabled ? 'not-allowed' : 'pointer',
        background: disabled
          ? collaborate ? 'rgba(196,125,16,0.30)' : 'rgba(94,106,210,0.35)'
          : hov ? accentHov : accent,
        color: '#fff',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'background 140ms ease',
      }}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor"
        strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2.5 8h10M8 3.5L12.5 8 8 12.5" />
      </svg>
    </button>
  )
}

function KbdChip({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <span style={{
      border: '1px solid rgba(255,255,255,0.09)',
      borderRadius: 4, padding: '0px 5px',
      ...style,
    }}>{children}</span>
  )
}

function ResearchEmptyState() {
  return (
    <div style={{
      height: '100%', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
      padding: 32,
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: 'var(--accent-08)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 14px',
        }}>
          <svg width="20" height="20" viewBox="0 0 16 16" fill="none" stroke="var(--accent-light)"
            strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="2.5" width="12" height="9" rx="2.2" />
            <path d="M5 11.5v2l2.4-2" />
          </svg>
        </div>
        <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-soft)', marginBottom: 6 }}>
          Ask a question
        </div>
        <div style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.6, maxWidth: 300 }}>
          Search course materials and recent literature. Select sources above to scope retrieval.
        </div>
      </div>
    </div>
  )
}

function CollaborateEmptyState() {
  return (
    <div style={{
      height: '100%', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
      padding: 32,
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: 'rgba(196,125,16,0.12)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 14px',
        }}>
          <svg width="20" height="20" viewBox="0 0 16 16" fill="none" stroke="var(--await)"
            strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2.5 8.5l3.5-5.5 2.5 4 2-2.5 3 4H2.5z" />
            <circle cx="12" cy="4" r="1" fill="var(--await)" stroke="none" />
          </svg>
        </div>
        <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-soft)', marginBottom: 8 }}>
          Teaching collaborator
        </div>
        <div style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.75, maxWidth: 320 }}>
          Ask for a fresh analogy or example · get a concept explained a different way · brainstorm how to teach a hard section · explore ideas beyond what's in the slides
        </div>
      </div>
    </div>
  )
}

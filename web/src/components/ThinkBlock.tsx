import { useEffect, useRef } from 'react'

// Collapsible, auto-scrolling reasoning-trace block. Shared by GapsAnalysis and Chat.
export default function ThinkBlock({
  text, open, onToggle, streaming = false,
}: {
  text: string
  open: boolean
  onToggle: () => void
  streaming?: boolean
}) {
  const bodyRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (streaming && open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight
    }
  }, [text, streaming, open])

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
        {streaming ? 'THINKING…' : 'REASONING TRACE'}
        <span style={{ marginLeft: 'auto', color: 'var(--text-ghost)' }}>{open ? 'collapse' : 'expand'}</span>
      </button>
      {open && (
        <div
          ref={bodyRef}
          style={{
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
          }}
        >
          {text || '…'}
        </div>
      )}
    </div>
  )
}

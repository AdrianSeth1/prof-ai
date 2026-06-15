import { useEffect, useRef, useState } from 'react'
import type { PaletteCommand } from '../types'

interface Props {
  query: string
  onQueryChange: (q: string) => void
  commands: PaletteCommand[]
  onClose: () => void
}

export default function CommandPalette({ query, onQueryChange, commands, onClose }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [activeIdx, setActiveIdx] = useState(0)

  // Filter commands by case-insensitive substring match on label or group
  const q = query.toLowerCase()
  const filtered = commands.filter(c =>
    c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q)
  )

  // Reset index when filter changes
  useEffect(() => setActiveIdx(0), [query])

  // Focus input on open
  useEffect(() => {
    setTimeout(() => inputRef.current?.focus(), 20)
  }, [])

  // Arrow-key + enter navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setActiveIdx(i => (i + 1) % Math.max(1, filtered.length))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setActiveIdx(i => (i - 1 + Math.max(1, filtered.length)) % Math.max(1, filtered.length))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        const cmd = filtered[activeIdx]
        if (cmd) {
          cmd.run()
          onClose()
        }
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [filtered, activeIdx, onClose])

  // Build rows with group headers
  type Row =
    | { kind: 'group'; label: string; key: string }
    | { kind: 'cmd'; cmd: PaletteCommand; idx: number; key: string }

  const rows: Row[] = []
  let lastGroup = ''
  let cmdIdx = 0
  for (const cmd of filtered) {
    if (cmd.group !== lastGroup) {
      rows.push({ kind: 'group', label: cmd.group, key: 'g-' + cmd.group })
      lastGroup = cmd.group
    }
    rows.push({ kind: 'cmd', cmd, idx: cmdIdx, key: cmd.label })
    cmdIdx++
  }

  return (
    /* Backdrop */
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(4,5,6,0.55)',
        backdropFilter: 'blur(2px)',
        zIndex: 90,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '14vh',
      }}
    >
      {/* Panel */}
      <div
        onClick={e => e.stopPropagation()}
        className="animate-fade-up"
        style={{
          width: 560,
          maxWidth: '92vw',
          border: '1px solid var(--line-input)',
          borderRadius: 13,
          background: 'var(--surface-pop)',
          boxShadow: '0 24px 70px rgba(0,0,0,0.6)',
          overflow: 'hidden',
        }}
      >
        {/* Search row */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 11,
          padding: '13px 15px',
          borderBottom: '1px solid var(--line)',
        }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#6b7079" strokeWidth="1.5" strokeLinecap="round">
            <circle cx="7" cy="7" r="4.3" />
            <path d="M13.5 13.5l-3-3" />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={e => onQueryChange(e.target.value)}
            placeholder="Search commands, screens, documents…"
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: 'var(--text)',
              fontFamily: 'inherit',
              fontSize: 15,
            }}
          />
          <span style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 10,
            color: '#6b7079',
            border: '1px solid rgba(255,255,255,0.10)',
            borderRadius: 4,
            padding: '1px 6px',
          }}>esc</span>
        </div>

        {/* Results */}
        <div style={{ maxHeight: 340, overflowY: 'auto', padding: 7 }}>
          {filtered.length === 0 ? (
            <div style={{ padding: 24, textAlign: 'center', fontSize: 13, color: 'var(--text-faint)' }}>
              No matches
            </div>
          ) : (
            rows.map(row => {
              if (row.kind === 'group') {
                return (
                  <div key={row.key} style={{
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: 10,
                    color: 'var(--text-faint)',
                    letterSpacing: '0.04em',
                    padding: '8px 9px 4px',
                  }}>{row.label.toUpperCase()}</div>
                )
              }

              const active = row.idx === activeIdx
              return (
                <div
                  key={row.key}
                  onClick={() => { row.cmd.run(); onClose() }}
                  onMouseEnter={() => setActiveIdx(row.idx)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 11,
                    padding: '9px 10px',
                    borderRadius: 8,
                    cursor: 'pointer',
                    background: active ? 'rgba(94,106,210,0.14)' : 'transparent',
                    transition: 'background 100ms ease',
                  }}
                >
                  {/* Icon tile */}
                  <span style={{
                    width: 22, height: 22,
                    borderRadius: 6,
                    background: 'rgba(255,255,255,0.05)',
                    flexShrink: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <CmdIcon group={row.cmd.group} label={row.cmd.label} />
                  </span>
                  <span style={{ fontSize: 13.5, color: 'var(--text)', flex: 1 }}>{row.cmd.label}</span>
                  {row.cmd.hint && (
                    <span style={{
                      fontFamily: '"JetBrains Mono", monospace',
                      fontSize: 10,
                      color: '#6b7079',
                      border: '1px solid rgba(255,255,255,0.10)',
                      borderRadius: 4,
                      padding: '1px 6px',
                    }}>{row.cmd.hint}</span>
                  )}
                </div>
              )
            })
          )}
        </div>

        {/* Footer */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          padding: '9px 15px',
          borderTop: '1px solid var(--line)',
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 10,
          color: 'var(--text-faint)',
        }}>
          <span>↑↓ navigate</span>
          <span>⏎ select</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  )
}

function CmdIcon({ group, label }: { group: string; label: string }) {
  if (group === 'Navigation') {
    return (
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#8a8f98" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 4h12M2 8h8M2 12h5" />
      </svg>
    )
  }
  if (label.includes('Ask AI')) {
    return (
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#8b95ea" strokeWidth="1.5" strokeLinecap="round">
        <path d="M8 2.5a2.4 2.4 0 012.4 2.4v3a2.4 2.4 0 01-4.8 0v-3A2.4 2.4 0 018 2.5z" />
        <path d="M4 7.5a4 4 0 008 0M8 11.5v2" />
      </svg>
    )
  }
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#8a8f98" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8" cy="8" r="5" />
      <path d="M8 5v3l2 2" />
    </svg>
  )
}

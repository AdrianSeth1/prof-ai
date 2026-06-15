import { useEffect, useRef, useState } from 'react'

export interface Module { id: string; name: string }
export interface SourceData { modules: Module[]; documents: string[] }

interface Props {
  sources: SourceData
  selected: Set<string>
  onToggle: (id: string) => void
  emptyLabel?: string
  note?: string
}

function ScopeTrigger({ label, open, onClick }: { label: string; open: boolean; onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        height: 30, padding: '0 11px', borderRadius: 8,
        border: `1px solid ${open || hov ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.10)'}`,
        cursor: 'pointer', fontSize: 12.5, color: '#c4c8cf',
        transition: 'border-color 140ms ease',
        userSelect: 'none',
      }}
    >
      <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M2 4h12M4 8h8M6 12h4" />
      </svg>
      <span>{label}</span>
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="#6b7079" strokeWidth="1.5" strokeLinecap="round">
        <path d="M4 6l4 4 4-4" />
      </svg>
    </div>
  )
}

function GroupLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontFamily: '"JetBrains Mono",monospace',
      fontSize: 10, color: 'var(--text-faint)',
      letterSpacing: '0.04em',
      padding: '6px 8px 4px',
    }}>{children}</div>
  )
}

function CheckRow({
  checked, label, mono, onClick,
}: { checked: boolean; label: string; mono?: boolean; onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 9,
        padding: '7px 8px', borderRadius: 6,
        cursor: 'pointer', fontSize: 12.5, color: '#c4c8cf',
        background: hov ? 'rgba(255,255,255,0.04)' : 'transparent',
        transition: 'background 140ms ease',
        userSelect: 'none',
      }}
    >
      <span style={{
        width: 15, height: 15, borderRadius: 4,
        flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: `1px solid ${checked ? 'var(--accent)' : 'rgba(255,255,255,0.2)'}`,
        background: checked ? 'var(--accent)' : 'transparent',
        transition: 'all 140ms ease',
      }}>
        {checked && (
          <svg width="9" height="9" viewBox="0 0 16 16" fill="none"
            stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 8.5l3.2 3.2L13 4.5" />
          </svg>
        )}
      </span>
      <span style={mono ? { fontFamily: '"JetBrains Mono",monospace', fontSize: 11 } : {}}>
        {label}
      </span>
    </div>
  )
}

export default function SourceFilter({
  sources,
  selected,
  onToggle,
  emptyLabel = 'All materials',
  note = 'Empty selection searches everything.',
}: Props) {
  const [open, setOpen]   = useState(false)
  const [query, setQuery] = useState('')
  const rootRef           = useRef<HTMLDivElement>(null)

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const h = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node))
        setOpen(false)
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [open])

  const count = selected.size
  const label = count === 0 ? emptyLabel : `${count} selected`

  const q           = query.toLowerCase()
  const filteredMods = sources.modules.filter(m => m.name.toLowerCase().includes(q))
  const filteredDocs = sources.documents.filter(d => d.toLowerCase().includes(q))
  const anyResults  = filteredMods.length > 0 || filteredDocs.length > 0

  return (
    <div ref={rootRef} style={{ position: 'relative' }}>
      <ScopeTrigger label={label} open={open} onClick={() => setOpen(o => !o)} />

      {open && (
        <div
          className="animate-fade-up"
          style={{
            position: 'absolute', top: 36, left: 0, width: 320, zIndex: 20,
            border: '1px solid rgba(255,255,255,0.10)',
            borderRadius: 10,
            background: 'var(--surface-pop)',
            boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
            padding: 7,
          }}
        >
          {/* Search */}
          <div style={{ padding: '0 2px 6px' }}>
            <input
              autoFocus
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Filter modules and documents…"
              style={{
                width: '100%', background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.10)', borderRadius: 6,
                padding: '5px 9px', color: '#c4c8cf',
                fontFamily: 'inherit', fontSize: 12, outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>

          {!anyResults && (
            <div style={{ padding: '6px 8px', fontSize: 12, color: 'var(--text-faint)', fontStyle: 'italic' }}>
              No matches for "{query}"
            </div>
          )}

          {/* Modules group */}
          {filteredMods.length > 0 && (
            <>
              <GroupLabel>MODULES</GroupLabel>
              {filteredMods.map(m => (
                <CheckRow
                  key={m.id}
                  checked={selected.has(m.id)}
                  label={m.name}
                  onClick={() => onToggle(m.id)}
                />
              ))}
            </>
          )}

          {/* Documents group */}
          {filteredDocs.length > 0 && (
            <div style={{
              borderTop: filteredMods.length > 0 ? '1px solid rgba(255,255,255,0.06)' : undefined,
              marginTop: filteredMods.length > 0 ? 4 : 0,
              paddingTop: filteredMods.length > 0 ? 4 : 0,
            }}>
              <GroupLabel>DOCUMENTS</GroupLabel>
              {filteredDocs.map(fn => (
                <CheckRow
                  key={fn}
                  checked={selected.has(fn)}
                  label={fn}
                  mono
                  onClick={() => onToggle(fn)}
                />
              ))}
            </div>
          )}

          {note && (
            <div style={{
              fontSize: 10.5, color: 'var(--text-faint)',
              padding: '8px 8px 4px',
              borderTop: '1px solid rgba(255,255,255,0.06)',
              marginTop: 4,
            }}>
              {note}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

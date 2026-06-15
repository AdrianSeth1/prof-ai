import type { Mode } from '../types'

interface Props {
  title: string
  sub: string
  mode: Mode
  onOpenPalette: () => void
}

interface ModeMeta {
  color: string
  soft: string
  line: string
  label: string
  pulse: boolean
}

function getModeMeta(mode: Mode): ModeMeta {
  switch (mode) {
    case 'lecture':
      return { color: 'var(--rec)', soft: 'var(--rec-08)', line: 'var(--rec-line)', label: 'REC', pulse: true }
    case 'awaiting':
      return { color: 'var(--await)', soft: 'var(--await-08)', line: 'var(--await-line)', label: 'LISTENING', pulse: false }
    case 'processing':
      return { color: 'var(--accent)', soft: 'var(--accent-08)', line: 'var(--accent-line)', label: 'THINKING', pulse: true }
    case 'speaking':
      return { color: 'var(--accent)', soft: 'var(--accent-08)', line: 'var(--accent-line)', label: 'SPEAKING', pulse: false }
  }
}

export default function TopBar({ title, sub, mode, onOpenPalette }: Props) {
  const mm = getModeMeta(mode)

  return (
    <header style={{
      height: 48,
      flexShrink: 0,
      borderBottom: '1px solid var(--line)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 16px',
      background: 'rgba(8,9,10,0.7)',
      backdropFilter: 'blur(10px)',
      position: 'relative',
      zIndex: 5,
    }}>
      {/* Left: screen title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, minWidth: 0 }}>
        <span style={{ fontWeight: 600, fontSize: 14, letterSpacing: '-0.02em' }}>{title}</span>
        <span style={{ color: 'var(--text-ghost)' }}>/</span>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 11.5,
          color: 'var(--text-muted)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>{sub}</span>
      </div>

      {/* Right: mode pill + search */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {/* Mode pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          height: 26,
          padding: '0 10px',
          borderRadius: 7,
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 11,
          fontWeight: 500,
          letterSpacing: '0.02em',
          border: `1px solid ${mm.line}`,
          background: mm.soft,
          color: mm.color,
        }}>
          <span style={{
            width: 7, height: 7,
            borderRadius: '50%',
            background: mm.color,
            animation: mm.pulse ? 'pulseRec 1.6s infinite' : 'none',
          }} />
          <span>{mm.label}</span>
        </div>

        {/* Divider */}
        <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.08)' }} />

        {/* Search / ⌘K button */}
        <div
          onClick={onOpenPalette}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            height: 28,
            padding: '0 9px',
            borderRadius: 7,
            border: '1px solid rgba(255,255,255,0.09)',
            cursor: 'pointer',
            color: 'var(--text-muted)',
            fontSize: 12,
            transition: 'border-color 140ms ease, color 140ms ease',
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.16)'
            ;(e.currentTarget as HTMLDivElement).style.color = 'var(--text)'
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.09)'
            ;(e.currentTarget as HTMLDivElement).style.color = 'var(--text-muted)'
          }}
        >
          <span>Search</span>
          <span style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 10,
            color: '#6b7079',
            border: '1px solid rgba(255,255,255,0.10)',
            borderRadius: 4,
            padding: '0px 5px',
          }}>⌘K</span>
        </div>
      </div>
    </header>
  )
}

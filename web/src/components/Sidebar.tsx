import type { Screen, Mode, HealthData, HealthState } from '../types'

interface Props {
  screen: Screen
  mode: Mode
  health: HealthData
  onNavigate: (s: Screen) => void
  onOpenPalette: () => void
}

function navItemStyle(active: boolean): React.CSSProperties {
  return {
    display: 'flex',
    alignItems: 'center',
    gap: 11,
    height: 34,
    padding: '0 10px',
    borderRadius: 7,
    cursor: 'pointer',
    position: 'relative',
    fontSize: 13,
    transition: 'background 140ms ease, color 140ms ease',
    color: active ? 'var(--text)' : 'var(--text-muted)',
    background: active ? 'rgba(255,255,255,0.055)' : 'transparent',
    fontWeight: active ? 500 : 450,
    userSelect: 'none',
  }
}

function dotColor(s: HealthState): string {
  return s === 'up' ? 'var(--ok)' : s === 'degraded' ? 'var(--await)' : 'var(--rec)'
}

function dotGlow(s: HealthState): string {
  if (s === 'up')       return '0 0 5px rgba(63,185,80,0.6)'
  if (s === 'degraded') return '0 0 5px rgba(233,162,59,0.5)'
  return '0 0 5px rgba(239,77,86,0.5)'
}

export default function Sidebar({ screen, mode, health, onNavigate, onOpenPalette }: Props) {
  const isLive = screen === 'live'
  const isRec  = mode === 'lecture'

  return (
    <aside style={{
      width: 236,
      flexShrink: 0,
      borderRight: '1px solid var(--line)',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-sidebar)',
    }}>
      {/* ── Brand header ── */}
      <div style={{
        height: 48,
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '0 14px',
        borderBottom: '1px solid var(--line)',
      }}>
        <div style={{
          width: 24, height: 24,
          borderRadius: 6,
          background: 'linear-gradient(150deg, #6e79e0, #5059bd)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontWeight: 600, fontSize: 13, color: '#fff',
          flexShrink: 0,
          boxShadow: '0 0 0 1px rgba(255,255,255,0.06) inset',
        }}>P</div>

        <div style={{ fontWeight: 600, fontSize: 13.5, letterSpacing: '-0.02em' }}>Prof AI</div>

        <div style={{
          marginLeft: 'auto',
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 10,
          color: 'var(--text-faint)',
          border: '1px solid var(--line-strong)',
          borderRadius: 4,
          padding: '1px 5px',
        }}>local</div>
      </div>

      {/* ── Nav ── */}
      <nav style={{ display: 'flex', flexDirection: 'column', gap: 2, padding: '10px 8px' }}>
        <div style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 10,
          color: 'var(--text-faint)',
          letterSpacing: '0.04em',
          padding: '6px 10px 4px',
        }}>WORKSPACE</div>

        <NavItem
          active={screen === 'chat'}
          label="Chat"
          hint="1"
          onClick={() => onNavigate('chat')}
          icon={<ChatIcon />}
        />
        <NavItem
          active={isLive}
          label="Live Lecture"
          onClick={() => onNavigate('live')}
          icon={<LiveIcon />}
          badge={isLive && isRec ? <RecBadge /> : <ShortcutBadge>2</ShortcutBadge>}
        />
        <NavItem
          active={screen === 'materials'}
          label="Add Materials"
          hint="3"
          onClick={() => onNavigate('materials')}
          icon={<MaterialsIcon />}
        />
        <NavItem
          active={screen === 'modules'}
          label="Modules"
          hint="4"
          onClick={() => onNavigate('modules')}
          icon={<ModulesIcon />}
        />
        <NavItem
          active={screen === 'gaps'}
          label="Gaps Analysis"
          hint="5"
          onClick={() => onNavigate('gaps')}
          icon={<GapsIcon />}
        />
      </nav>

      <div style={{ flex: 1 }} />

      {/* ── Search trigger ── */}
      <div style={{ padding: '8px', borderTop: '1px solid var(--line)' }}>
        <SearchTrigger onClick={onOpenPalette} />
      </div>

      {/* ── Session card + status ── */}
      <div style={{
        padding: 10,
        borderTop: '1px solid var(--line)',
        background: 'var(--bg-base)',
      }}>
        {/* Session card */}
        <div style={{
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 8,
          padding: '9px 10px',
          background: 'var(--surface-2)',
          cursor: 'pointer',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <div style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: 10,
              color: 'var(--text-faint)',
              letterSpacing: '0.04em',
            }}>SESSION</div>
            <ChevronIcon style={{ marginLeft: 'auto' }} />
          </div>
          <div style={{ fontSize: 12.5, fontWeight: 500, marginTop: 3 }}>Spring '26 · Lecture 12</div>
          <div style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 10.5,
            color: 'var(--text-muted)',
            marginTop: 1,
          }}>NSC 4310 · Cellular Neuro</div>
        </div>

        {/* System status */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 10, padding: '0 2px' }}>
          <StatusRow label="Backend" info={health.backend.info} status={health.backend.status} />
          <StatusRow label="Ollama"  info={health.ollama.info}  status={health.ollama.status} />
          <StatusRow label="GPU"     info={health.gpu.info}     status={health.gpu.status} />
        </div>
      </div>
    </aside>
  )
}

/* ── Sub-components ── */

interface NavItemProps {
  active: boolean
  label: string
  hint?: string
  onClick: () => void
  icon: React.ReactNode
  badge?: React.ReactNode
}

function NavItem({ active, label, hint, onClick, icon, badge }: NavItemProps) {
  return (
    <div
      style={navItemStyle(active)}
      onClick={onClick}
      onMouseEnter={e => {
        if (!active) {
          (e.currentTarget as HTMLDivElement).style.background = 'rgba(255,255,255,0.04)'
          ;(e.currentTarget as HTMLDivElement).style.color = 'var(--text)'
        }
      }}
      onMouseLeave={e => {
        if (!active) {
          (e.currentTarget as HTMLDivElement).style.background = 'transparent'
          ;(e.currentTarget as HTMLDivElement).style.color = 'var(--text-muted)'
        }
      }}
    >
      {active && (
        <div style={{
          position: 'absolute',
          left: -8,
          top: 9,
          width: 2,
          height: 16,
          borderRadius: 2,
          background: 'var(--accent)',
        }} />
      )}
      {icon}
      <span style={{ flex: 1 }}>{label}</span>
      {badge ?? (hint ? <ShortcutBadge>{hint}</ShortcutBadge> : null)}
    </div>
  )
}

function ShortcutBadge({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: 10,
      color: 'var(--text-faint)',
    }}>{children}</span>
  )
}

function RecBadge() {
  return (
    <span style={{
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: 9,
      fontWeight: 600,
      color: 'var(--rec)',
      border: '1px solid rgba(239,77,86,0.35)',
      borderRadius: 4,
      padding: '0px 4px',
      letterSpacing: '0.03em',
    }}>REC</span>
  )
}

function SearchTrigger({ onClick }: { onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        height: 30,
        padding: '0 9px',
        borderRadius: 7,
        cursor: 'pointer',
        color: 'var(--text-muted)',
        fontSize: 12.5,
        transition: 'background 140ms ease, color 140ms ease',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.background = 'rgba(255,255,255,0.04)'
        ;(e.currentTarget as HTMLDivElement).style.color = 'var(--text)'
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.background = 'transparent'
        ;(e.currentTarget as HTMLDivElement).style.color = 'var(--text-muted)'
      }}
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
        <circle cx="7" cy="7" r="4.3" />
        <path d="M13.5 13.5l-3-3" />
      </svg>
      <span style={{ flex: 1 }}>Search &amp; commands</span>
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 10,
        color: '#6b7079',
        border: '1px solid rgba(255,255,255,0.10)',
        borderRadius: 4,
        padding: '1px 5px',
      }}>⌘K</span>
    </div>
  )
}

function StatusRow({ label, info, status }: { label: string; info: string; status: HealthState }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 7,
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: 10.5,
      color: 'var(--text-muted)',
    }}>
      <span style={{
        width: 6, height: 6,
        borderRadius: '50%',
        background: dotColor(status),
        boxShadow: dotGlow(status),
        flexShrink: 0,
      }} />
      <span style={{ color: '#b9bcc4' }}>{label}</span>
      <span style={{ color: 'var(--text-faint)', marginLeft: 'auto' }}>{info}</span>
    </div>
  )
}

function ChevronIcon({ style }: { style?: React.CSSProperties }) {
  return (
    <svg style={style} width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="#6b7079" strokeWidth="1.5" strokeLinecap="round">
      <path d="M4 6l4 4 4-4" />
    </svg>
  )
}

/* ── Nav icons (16×16 stroke) ── */
function ChatIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <rect x="2" y="2.5" width="12" height="9" rx="2.2" />
      <path d="M5 11.5v2l2.4-2" />
    </svg>
  )
}

function LiveIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <circle cx="8" cy="8" r="2" fill="currentColor" stroke="none" />
      <path d="M4.2 4.2a5.4 5.4 0 000 7.6M11.8 4.2a5.4 5.4 0 010 7.6" />
    </svg>
  )
}

function MaterialsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M4 2h5l3 3v9H4z" />
      <path d="M9 2v3h3M6 8.5h4M6 11h4" />
    </svg>
  )
}

function ModulesIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <rect x="2.5" y="2.5" width="11" height="3" rx="1" />
      <rect x="2.5" y="6.8" width="11" height="3" rx="1" />
      <rect x="2.5" y="11.1" width="11" height="2.4" rx="1" />
    </svg>
  )
}

function GapsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M2.5 13.5h11" />
      <rect x="3.5" y="8" width="2.4" height="4" />
      <rect x="7" y="4.5" width="2.4" height="7.5" />
      <rect x="10.5" y="9.5" width="2.4" height="2.5" />
    </svg>
  )
}

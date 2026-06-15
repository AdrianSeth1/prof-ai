interface Props {
  title: string
  detail: string
  phase: string
}

export default function PlaceholderScreen({ title, detail, phase }: Props) {
  return (
    <div style={{
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 32,
    }}>
      <div style={{
        border: '1px solid var(--line)',
        borderRadius: 11,
        background: 'var(--surface)',
        padding: '28px 32px',
        maxWidth: 480,
        width: '100%',
        textAlign: 'center',
      }}>
        {/* Phase badge */}
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 10,
          fontWeight: 500,
          color: 'var(--accent-light)',
          border: '1px solid var(--accent-line)',
          borderRadius: 4,
          padding: '2px 8px',
          marginBottom: 16,
          letterSpacing: '0.04em',
        }}>{phase.toUpperCase()}</div>

        <div style={{
          fontSize: 15,
          fontWeight: 600,
          letterSpacing: '-0.02em',
          marginBottom: 8,
        }}>{title}</div>

        <div style={{
          fontSize: 12.5,
          color: 'var(--text-muted)',
          lineHeight: 1.6,
          marginBottom: 20,
        }}>{detail}</div>

        <div style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 11,
          color: 'var(--text-faint)',
          borderTop: '1px solid var(--line)',
          paddingTop: 14,
        }}>coming soon — scaffold only</div>
      </div>
    </div>
  )
}

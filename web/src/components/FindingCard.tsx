export type FindingStatus = 'covered' | 'partial' | 'uncovered'

export interface Finding {
  topic: string
  status: FindingStatus
  note: string
  source: string
}

const STATUS_DOT: Record<FindingStatus, string> = {
  covered:   'var(--ok)',
  partial:   'var(--await)',
  uncovered: 'rgba(255,255,255,0.22)',
}

const STATUS_BORDER: Record<FindingStatus, string> = {
  covered:   'rgba(63,185,80,0.22)',
  partial:   'rgba(233,162,59,0.22)',
  uncovered: 'rgba(255,255,255,0.08)',
}

export default function FindingCard({ finding }: { finding: Finding }) {
  const status = finding.status in STATUS_DOT ? finding.status : 'uncovered' as FindingStatus
  const dot    = STATUS_DOT[status]
  const border = STATUS_BORDER[status]

  return (
    <div style={{
      border: `1px solid ${border}`,
      borderRadius: 10,
      background: 'var(--surface)',
      padding: '10px 13px',
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: dot, flexShrink: 0,
        }} />
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>
          {finding.topic}
        </span>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 9.5, color: dot, opacity: 0.9, flexShrink: 0,
        }}>
          {finding.status}
        </span>
      </div>
      {finding.note && (
        <div style={{
          fontSize: 12.5, color: 'var(--text-body)', lineHeight: 1.55, paddingLeft: 15,
        }}>
          {finding.note}
        </div>
      )}
      {finding.source && (
        <div style={{ paddingLeft: 15 }}>
          <span style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 10, color: 'var(--text-faint)',
          }}>
            {finding.source}
          </span>
        </div>
      )}
    </div>
  )
}
